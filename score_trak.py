import torch
from transformers import AutoModelForCausalLM, default_data_collator
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
from dattri.func.utils import flatten_func, flatten_params
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask
from anticipation.vocab import AUTOREGRESS

class TextDataset(Dataset):
    def __init__(self, file_path, max_length=1024, num_samples=None, is_generated: bool = False):
        self.examples = []
        self.max_length = max_length
        self.is_generated = is_generated
        
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
        if self.is_generated:
            # generated_samples.txt: first/last tokens missing → prepend AUTOREGRESS; keep all tokens
            arr = np.fromstring(self.examples[idx], dtype=int, sep=" ")
            if arr.size == 0:
                input_ids = np.array([AUTOREGRESS], dtype=int)
            else:
                input_ids = np.concatenate([np.array([AUTOREGRESS], dtype=int), arr])
        else:
            # standard format: drop last token (file id)
            safe_text = " ".join(self.examples[idx].split()[:-1])
            input_ids = np.fromstring(safe_text, dtype=int, sep=" ")
        input_ids = input_ids[:self.max_length]
        
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones_like(torch.tensor(input_ids, dtype=torch.long))}


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate attribution scores using TRAK.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data.')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to validation data.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing checkpoints and for saving results.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for dataloaders.')
    parser.add_argument('--num_checkpoints', type=int, default=10,
                        help='Number of checkpoints to use.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--valid_is_generated', action='store_true',
                        help='If set, treat valid_file as generated samples: prepend AUTOREGRESS and do not drop last token.')
    parser.add_argument('--output_filename', type=str, default='score_TRAK_8192_generated.pt',
                        help='Output filename to save attribution scores.')
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
    eval_dataset = TextDataset(args.valid_file, num_samples=500, is_generated=args.valid_is_generated)
    
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        print(f"Error: Dataset is empty or failed to load. Exiting.")
        return
    
    print(f"The training dataset length: {len(train_dataset)}.")
    print(f"The eval dataset length: {len(eval_dataset)}.")
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )

    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found. Exiting.")
        return
    
    checkpoints = [os.path.join(args.output_dir, str(i)) for i in range(args.num_checkpoints)]
    
    if not os.path.isdir(checkpoints[0]):
        print(f"Error: Checkpoint directory {checkpoints[0]} not found. Exiting.")
        return
    
    # Load model from output_dir/full_model
    model_path = os.path.join(args.output_dir, 'full_model')
    
    if not os.path.isdir(model_path):
        print(f"Error: Model directory {model_path} not found. Exiting.")
        return
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
    model.eval()
    
    def f(params, batch):
        outputs = torch.func.functional_call(model, params, batch["input_ids"].to(device),
                                             kwargs={"attention_mask": batch["attention_mask"].to(device),
                                                     "labels": batch["labels"].to(device)})
        logp = -outputs.loss
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, batch):
        outputs = torch.func.functional_call(model, params, batch["input_ids"].to(device),
                                             kwargs={"attention_mask": batch["attention_mask"].to(device),
                                                     "labels": batch["labels"].to(device)})
        p = torch.exp(-outputs.loss)
        return p
    
    def checkpoints_load_func(model, checkpoint):
        model = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="eager").to(device) # TODO: error if attn_implementation is not eager, don't know why
        model.eval()
        return model
    
    task = AttributionTask(loss_func=f, model=model,
                           checkpoints=checkpoints,
                           checkpoints_load_func=checkpoints_load_func)
    
    projector_kwargs = {
        "device": device,
        "proj_dim": 8192,
        "use_half_precision": False,
    }
    
    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=device,
        projector_kwargs=projector_kwargs,
        regularization=0.01,
    )
    
    print("Caching train dataloader...")
    attributor.cache(train_dataloader)
    
    print("Attributing scores...")
    with torch.no_grad():
        score = attributor.attribute(eval_dataloader)
    
    output_score_file = os.path.join(args.output_dir, args.output_filename)
    torch.save(score, output_score_file)
    print(f"Results saved to {output_score_file}")
    print(f"Score shape: {score.shape}")

if __name__ == "__main__":
    main()