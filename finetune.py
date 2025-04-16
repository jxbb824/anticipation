import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
import numpy as np
from torch.utils.data import Dataset
import logging
import os
import argparse
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)

# PassthroughTokenizer implementation
class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        self._eos = 55025 # self._vocab_size - 1
        self._eos_token = str(self._eos)
        
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = np.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


class TextDataset(Dataset):
    def __init__(self, file_path, max_length=1024, num_samples=None):
        self.examples = []
        self.max_length = max_length
        
        logging.info(f"Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                self.examples.append(line.strip())
        
        logging.info(f"Loaded {len(self.examples)} samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = np.fromstring(self.examples[idx], dtype=int, sep=" ")
        input_ids = input_ids[:self.max_length]
        
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 model')
    parser.add_argument('--pretrained_model_path', type=str, default='/home/xiruij/anticipation/mytraintest/gpt2_final_model/pytorch_model.bin',
                        help='Path to pretrained model')
    parser.add_argument('--train_file', type=str, 
                        default='/home/xiruij/anticipation/datasets/finetune/train.txt',
                        help='Path to training data')
    parser.add_argument('--valid_file', type=str, 
                        default='/home/xiruij/anticipation/datasets/finetune/test.txt',
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, 
                        default='./finetune_output',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for fine-tuning')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = PassthroughTokenizer(vocab_size=55028)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load pretrained model
    logging.info(f"Loading pretrained model from {args.pretrained_model_path}")
    try:
        model = GPT2LMHeadModel.from_pretrained('stanford-crfm/music-small-800k').cuda()
        logging.info("Successfully loaded pretrained model")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Fallback to creating a model with default config if loading fails
        logging.warning("Falling back to default configuration")
        model_config = GPT2Config(
            vocab_size=55028,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        model = GPT2LMHeadModel(model_config)
    
    # Load datasets
    train_dataset = TextDataset(args.train_file)
    valid_dataset = TextDataset(args.valid_file, num_samples=100)  # Reduced validation samples for faster evaluation
    
    # Fine-tuning arguments - lower learning rate, fewer epochs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_steps=5000,                      # More frequent evaluation
        save_steps=10000,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=500,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        learning_rate=args.learning_rate,     # Lower learning rate for fine-tuning
        warmup_ratio=0.1,                     # Use ratio instead of steps
        lr_scheduler_type="linear",           # Linear is often better for fine-tuning
        weight_decay=0.01,                    # Add some regularization
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Use command-line argument
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    # Start fine-tuning
    logging.info("Starting fine-tuning...")
    trainer.train()
    
    # Save fine-tuned model
    final_output_path = f"{args.output_dir}/final_model"
    logging.info(f"Saving fine-tuned model to {final_output_path}")
    trainer.save_model(final_output_path)
    logging.info("Fine-tuning completed!")

if __name__ == "__main__":
    main()
