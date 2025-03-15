import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import numpy as np
from torch.utils.data import Dataset
import logging
import os

logging.basicConfig(level=logging.INFO)

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, max_length=5420, num_samples=None):
        self.examples = []
        self.max_length = max_length
        
        logging.info(f"Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                # Assume each line is a sequence of numbers separated by spaces
                tokens = [int(token) for token in line.strip().split()]
                # Truncate or pad to max length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                self.examples.append({
                    "input_ids": torch.tensor(tokens, dtype=torch.long),
                    "attention_mask": torch.ones(len(tokens), dtype=torch.long)
                })
        
        logging.info(f"Loaded {len(self.examples)} samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Custom data collator
class CustomDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples):
        # Get maximum length in the batch
        max_length = max([len(example["input_ids"]) for example in examples])
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for example in examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids and attention_mask
            padded_input_ids = torch.cat([
                input_ids, 
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ]) if padding_length > 0 else input_ids
            
            padded_attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ]) if padding_length > 0 else attention_mask
            
            # Add to batch
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(padded_attention_mask)
            batch["labels"].append(padded_input_ids.clone())  # For language models, labels are the same as inputs
        
        # Convert lists to tensors
        batch = {k: torch.stack(v) for k, v in batch.items()}
        return batch

def main():
    # Define model configuration
    model_config = GPT2Config.from_pretrained(
        "gpt2",  # Use standard GPT-2 configuration
        vocab_size=55028,
        n_positions=5420,   # Maximum sequence length
        n_ctx=1024,         # Context length
        n_embd=768,        # Embedding dimension
        n_layer=12,        # Number of transformer layers
        n_head=12,         # Number of attention heads
    )
    
    # Initialize model from configuration
    model = GPT2LMHeadModel(model_config)
    
    # Data paths
    train_file = "/home/xiruij/anticipation/datasets/lakhmidi/train.txt"
    valid_file = "/home/xiruij/anticipation/datasets/lakhmidi/valid.txt"
    
    # Load datasets
    train_dataset = TextDataset(train_file, num_samples=200000)
    valid_dataset = TextDataset(valid_file, num_samples=10000)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./mytraintest/gpt2_output",          # Output directory
        overwrite_output_dir=True,           # Overwrite output directory
        num_train_epochs=1,                  # Number of training epochs
        per_device_train_batch_size=8,       # Training batch size
        per_device_eval_batch_size=8,        # Evaluation batch size
        eval_steps=1000,                      # Evaluate every 500 steps
        save_steps=2000,                     # Save every 1000 steps
        logging_dir="./mytraintest/logs",                # Logging directory
        logging_steps=200,                   # Log every 100 steps
        evaluation_strategy="steps",         # Evaluation strategy
        save_total_limit=2,                  # Maximum number of checkpoints to save
        load_best_model_at_end=True,         # Load best model at the end
        fp16=True,
    )
    
    # Use custom data collator instead of DataCollatorForLanguageModeling
    data_collator = CustomDataCollator(pad_token_id=0)  # Assume 0 is the padding token
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    # Start training
    logging.info("start...")
    trainer.train()
    
    # Save model
    logging.info("save...")
    trainer.save_model("./mytraintest/gpt2_final_model")
    logging.info("done!")

    # # Test set evaluation
    # if os.path.exists("test.txt"):
    #     logging.info("testing...")
    #     test_dataset = TextDataset("test.txt")
    #     results = trainer.evaluate(eval_dataset=test_dataset)
    #     logging.info(f"result: {results}")

if __name__ == "__main__":
    main()