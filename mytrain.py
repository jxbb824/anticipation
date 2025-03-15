import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
import numpy as np
from torch.utils.data import Dataset
import logging
import os
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)

# PassthroughTokenizer implementation
class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        self._eos = self._vocab_size - 1
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

def main():
    # Define model configuration
    model_config = GPT2Config.from_pretrained(
        "gpt2",  # Use standard GPT-2 configuration
        vocab_size=55028,
        n_positions=1024,   # Maximum sequence length
        n_ctx=1024,         # Context length
        n_embd=768,        # Embedding dimension
        n_layer=12,        # Number of transformer layers
        n_head=12,         # Number of attention heads
    )
    
    # Initialize tokenizer and model
    tokenizer = PassthroughTokenizer(vocab_size=55028)
    tokenizer.pad_token = tokenizer.eos_token  # Add pad_token = eos_token
    model = GPT2LMHeadModel(model_config)
    
    # Data paths
    train_file = "/home/xiruij/anticipation/datasets/lakhmidi/train.txt"
    valid_file = "/home/xiruij/anticipation/datasets/lakhmidi/valid.txt"
    
    # Load datasets with tokenizer
    train_dataset = TextDataset(train_file, num_samples=20000)
    valid_dataset = TextDataset(valid_file, num_samples=1000)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./mytraintest/gpt2_output",          # Output directory
        overwrite_output_dir=True,           # Overwrite output directory
        num_train_epochs=1,                  # Number of training epochs
        per_device_train_batch_size=8,       # Training batch size
        per_device_eval_batch_size=8,        # Evaluation batch size
        eval_steps=1000,                     # Evaluate every 1000 steps
        save_steps=2000,                     # Save every 2000 steps
        logging_dir="./mytraintest/logs",    # Logging directory
        logging_steps=200,                   # Log every 200 steps
        evaluation_strategy="steps",         # Evaluation strategy
        save_total_limit=2,                  # Maximum number of checkpoints to save
        load_best_model_at_end=True,         # Load best model at the end
        fp16=True,
        learning_rate=5e-5,
        warmup_steps=500,
    )
    
    # Use standard DataCollatorForLanguageModeling
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