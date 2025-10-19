import torch
from transformers import AutoModelForCausalLM
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from anticipation.sample import add_token
from anticipation.vocab import AUTOREGRESS, TIME_OFFSET

def parse_args():
    parser = argparse.ArgumentParser(description='Generate token sequences for evaluation.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path for generated samples.')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to generate.')
    parser.add_argument('--sequence_length', type=int, default=1023,
                        help='Target length of each generated sequence (must be divisible by 3).')
    parser.add_argument('--top_p', type=float, default=0.98,
                        help='Top-p for nucleus sampling.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Prepare output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check that sequence_length is divisible by 3 (tokens come in triplets)
    if args.sequence_length % 3 != 0:
        print(f"Warning: sequence_length {args.sequence_length} is not divisible by 3.")
        # Round down to nearest multiple of 3
        args.sequence_length = (args.sequence_length // 3) * 3
        print(f"Adjusted to {args.sequence_length} tokens.")
    
    num_triplets = args.sequence_length // 3
    
    generated_samples = []
    
    print(f"Generating {args.num_samples} samples with {args.sequence_length} tokens ({num_triplets} triplets) each...")
    print(f"Using autoregressive generation with top_p={args.top_p}")
    
    z = [AUTOREGRESS]  # Autoregressive mode
    
    for sample_idx in tqdm(range(args.num_samples), desc="Generating samples"):
        tokens = []
        current_time = 0
        
        # Generate num_triplets triplets (each triplet is 3 tokens: time, duration, note)
        for _ in range(num_triplets):
            try:
                # add_token generates one triplet (3 tokens) at a time
                new_triplet = add_token(model, z, tokens, args.top_p, current_time, debug=False)
                tokens.extend(new_triplet)
                
                # Update current_time from the time token (first token in triplet)
                current_time = new_triplet[0] - TIME_OFFSET
                
            except Exception as e:
                print(f"\nError generating triplet for sample {sample_idx + 1}: {e}")
                # If error, fill remaining with zeros
                remaining_tokens = args.sequence_length - len(tokens)
                tokens.extend([0] * remaining_tokens)
                break
        
        # Ensure exact length
        tokens = tokens[:args.sequence_length]
        
        # Convert to space-separated string
        token_str = ' '.join(map(str, tokens))
        generated_samples.append(token_str)
    
    # Save to file
    print(f"\nSaving {len(generated_samples)} samples to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(sample + '\n')
    
    print(f"Successfully generated {len(generated_samples)} samples!")
    print(f"Output saved to {args.output_file}")
    
    # Verify the output
    with open(args.output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"\nVerification:")
    print(f"  Total lines: {len(lines)}")
    if lines:
        first_line_tokens = len(lines[0].strip().split())
        last_line_tokens = len(lines[-1].strip().split())
        print(f"  First line: {first_line_tokens} tokens")
        print(f"  Last line: {last_line_tokens} tokens")

if __name__ == "__main__":
    main()

