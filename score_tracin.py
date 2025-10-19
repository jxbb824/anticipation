import os
import json
import argparse
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, default_data_collator

from dattri.algorithm.tracin import TracInAttributor
from dattri.task import AttributionTask
import torch.nn as nn


class TextDataset(Dataset):
    def __init__(self, file_path: str, max_length: int = 1024, num_samples: Optional[int] = None):
        self.examples = []
        self.max_length = max_length

        print(f"Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                self.examples.append(line.strip())

        print(f"Loaded {len(self.examples)} samples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        safe_text = " ".join(self.examples[idx].split()[:-1])
        input_ids = np.fromstring(safe_text, dtype=int, sep=" ")
        input_ids = input_ids[:self.max_length]

        tensor_ids = torch.tensor(input_ids, dtype=torch.long)
        # Return tuple to satisfy TracIn's expectation (data.to(device) over tuple items)
        return tensor_ids, tensor_ids.clone()


def parse_args():
    parser = argparse.ArgumentParser(description='Compute TracIn attribution using dattri TracInAttributor.')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data.')
    parser.add_argument('--valid_file', type=str, required=True, help='Path to validation data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing checkpoints and for saving results.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for dataloaders.')
    parser.add_argument('--num_checkpoints', type=int, default=10, help='Number of epoch checkpoints to use (0..N-1 directories).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--valid_num_samples', type=int, default=500, help='Number of validation samples to attribute.')
    parser.add_argument('--output_filename', type=str, default='score_TracIn.pt', help='Output filename to save attribution scores.')
    parser.add_argument('--normalized_grad', action='store_true', help='Use gradient normalization (CosIn). Default False (TracIn).')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TextDataset(args.train_file)
    eval_dataset = TextDataset(args.valid_file, num_samples=args.valid_num_samples)

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        print("Error: Dataset is empty or failed to load. Exiting.")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found. Exiting.")
        return

    # Expected checkpoints layout: output_dir/{0,1,2,...,num_checkpoints-1}
    checkpoints = [os.path.join(args.output_dir, str(i)) for i in range(args.num_checkpoints)]
    missing = [p for p in checkpoints if not os.path.isdir(p)]
    if missing:
        print(f"Error: Missing checkpoint directories: {missing}")
        return

    # Load final model (architecture) — TracIn uses checkpoints and LRs
    model_path = os.path.join(args.output_dir, 'full_model')
    if not os.path.isdir(model_path):
        print(f"Error: Model directory {model_path} not found. Exiting.")
        return

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
    model.eval()

    # Load learning rates recorded during training
    lrs_json = os.path.join(args.output_dir, 'lrs.json')
    if not os.path.isfile(lrs_json):
        print(f"Error: Learning rate file {lrs_json} not found. Exiting.")
        return
    with open(lrs_json, 'r') as f:
        lr_obj = json.load(f)
    epoch_lrs = lr_obj.get('lrs', [])
    if len(epoch_lrs) < len(checkpoints):
        print(f"Error: lrs length {len(epoch_lrs)} < checkpoints {len(checkpoints)}; please ensure LR is recorded for each checkpoint.")
        return
    # Ensure the same length as checkpoints
    epoch_lrs = epoch_lrs[:len(checkpoints)]
    weight_list = torch.tensor(epoch_lrs, dtype=torch.float32)

    # Define loss function f(params, data) as in the official example
    def ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    def f(params, data):
        if isinstance(data, dict):
            input_ids = ensure_batch_dim(data["input_ids"]).to(device)
            attention_mask = ensure_batch_dim(data.get("attention_mask", torch.ones_like(data["input_ids"]))).to(device)
            labels = ensure_batch_dim(data.get("labels", data["input_ids"]).clone()).to(device)
        elif isinstance(data, (list, tuple)):
            # Fallback for (input, label)
            input_ids = ensure_batch_dim(data[0]).to(device)
            labels = ensure_batch_dim(data[1]).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            raise ValueError("Unsupported data type for TracIn loss function.")

        outputs = torch.func.functional_call(
            model,
            params,
            (input_ids,),
            kwargs={"attention_mask": attention_mask, "labels": labels},
        )
        return outputs.loss

    def checkpoints_load_func(model_in, checkpoint_dir):
        ckpt_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, attn_implementation="eager").to(device)
        ckpt_model.eval()
        return ckpt_model

    task = AttributionTask(
        loss_func=f,
        model=model,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
    )


    projector_kwargs = {
        "proj_dim": 512,
        "proj_max_batch_size": 32,
        "proj_seed": 42,
        "device": device,
        "use_half_precision": True,
    }
    attributor = TracInAttributor(
        task=task,
        weight_list=weight_list,
        normalized_grad=args.normalized_grad,
        device=str(device),
        projector_kwargs=projector_kwargs,
    )

    print("Attributing scores with TracIn...")
    with torch.no_grad():
        score = attributor.attribute(train_dataloader, eval_dataloader)

    output_score_file = os.path.join(args.output_dir, args.output_filename)
    torch.save(score, output_score_file)
    print(f"Results saved to {output_score_file}")
    print(f"Score shape: {score.shape}")


if __name__ == "__main__":
    main()


