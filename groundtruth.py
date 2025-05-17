import torch
from transformers import AutoModelForCausalLM, default_data_collator
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
# Assuming dattri.func.utils are available in the environment
from dattri.func.utils import flatten_func, flatten_params

class TextDataset(Dataset):
    def __init__(self, file_path, max_length=1024, num_samples=None):
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
        
        # For Causal LM, labels are typically the same as input_ids
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones_like(torch.tensor(input_ids, dtype=torch.long))}


def parse_args():
    parser = argparse.ArgumentParser(description='Run ground truth analysis task on model checkpoints.')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to validation data.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing checkpoints (0-49) and for saving results.')
    parser.add_argument('--batch_size', type=int, default=1, # Snippet implies batch_size=1 for eval_dataloader
                        help='Batch size for evaluation dataloader.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_dataset = TextDataset(args.valid_file, num_samples=100) 
    if len(eval_dataset) == 0:
        print(f"Error: Validation dataset from {args.valid_file} is empty or failed to load. Exiting.")
        return
    
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )

    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found. Exiting.")
        return
        
    checkpoints = [os.path.join(args.output_dir, str(i)) for i in range(50)]
    result_list = []

    for checkpoint_path in checkpoints:
        if not os.path.isdir(checkpoint_path):
            print(f"Warning: Checkpoint directory {checkpoint_path} not found. Skipping.")
            continue
        
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
        model.eval()

        params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        if not params:
            print(f"Warning: No trainable parameters found in model from {checkpoint_path}. Skipping.")
            continue
        
        # Define f faithfully to the snippet
        @flatten_func(model)
        def f(flat_params, batch): # flat_params is used by flatten_func
            # Move batch data to the same device as the model
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # The `params` argument for functional_call should be the flattened ones provided to `f`
            outputs = torch.func.functional_call(model, flat_params, (input_ids,),
                                                kwargs={"attention_mask": attention_mask,
                                                        "labels": labels})
            logp = -outputs.loss
            # Adding epsilon for numerical stability, a necessary modification
            return logp - torch.log(1 - torch.exp(logp))

        result_iter = []
        for _, batch in enumerate(eval_dataloader):
            result_iter.append(f(flatten_params(params), batch).detach())
        
        result_iter_cpu = [res.cpu() for res in result_iter]
        result_iter_stacked = torch.stack(result_iter_cpu)
        result_list.append(result_iter_stacked)

    final_result = torch.stack(result_list)

    print(f"Final result: {final_result}")
    output_gt_file = os.path.join(args.output_dir, "gt.pt")
    torch.save(final_result, output_gt_file)
    print(f"Results saved to {output_gt_file}")
    print(f"Result shape: {final_result.shape}")

    print("Processing completed.")

if __name__ == "__main__":
    main()
