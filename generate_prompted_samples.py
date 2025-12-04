import torch
from transformers import AutoModelForCausalLM
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

from anticipation.sample import add_token
from anticipation.vocab import AUTOREGRESS, ANTICIPATE, TIME_OFFSET
from anticipation import ops


def parse_args():
    parser = argparse.ArgumentParser(description='Generate continuations conditioned on test prompts (clean 1023-token outputs).')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model.')
    parser.add_argument('--source_file', type=str, required=True,
                        help='Source prompts file (tokenized, first token is mode flag, last field is an identifier).')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path for generated samples (each line has exactly sequence_length tokens).')
    parser.add_argument('--num_prompts', type=int, default=500,
                        help='How many prompt lines to read from source_file.')
    parser.add_argument('--sequence_length', type=int, default=1023,
                        help='Length of the generated continuation per prompt (must be divisible by 3).')
    parser.add_argument('--top_p', type=float, default=0.98,
                        help='Top-p for nucleus sampling.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything
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

    # Ensure sequence_length is multiple of 3
    if args.sequence_length % 3 != 0:
        print(f"Warning: sequence_length {args.sequence_length} is not divisible by 3.")
        args.sequence_length = (args.sequence_length // 3) * 3
        print(f"Adjusted to {args.sequence_length} tokens.")

    num_triplets = args.sequence_length // 3

    # Read prompts
    print(f"Reading up to {args.num_prompts} prompts from {args.source_file} ...")
    prompts = []  # list of (z, tokens)
    with open(args.source_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.num_prompts:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                # Need at least a mode flag and an id
                continue
            # Last field is an identifier (non-token); drop it
            token_strs = parts[:-1]
            try:
                token_ints = [int(x) for x in token_strs]
            except ValueError:
                # Malformed line; skip
                continue

            if len(token_ints) == 0:
                # No mode flag present
                continue

            z_val = token_ints[0]
            if z_val not in (AUTOREGRESS, ANTICIPATE):
                # Fallback to AR if unexpected
                z_val = AUTOREGRESS
            z = [z_val]

            prefix_tokens = token_ints[1:]
            if len(prefix_tokens) < 3:
                # Too short to provide a meaningful context; still allow generation
                prefix_tokens = []
            if len(prefix_tokens) % 3 != 0:
                trimmed = (len(prefix_tokens) // 3) * 3
                print(f"Warning: prompt #{i+1} tokens not divisible by 3 ({len(prefix_tokens)}); trimming to {trimmed}.")
                prefix_tokens = prefix_tokens[:trimmed]

            prompts.append((z, prefix_tokens))

    print(f"Loaded {len(prompts)} prompts.")

    generated_lines = []
    for idx, (z, prefix_tokens) in enumerate(tqdm(prompts, desc="Generating continuations")):
        # Prepare history and current time from the prompt
        tokens = prefix_tokens.copy()
        try:
            current_time = ops.max_time(tokens, seconds=False) if len(tokens) >= 3 else 0
        except Exception:
            current_time = 0

        continuation = []
        for _ in range(num_triplets):
            try:
                new_triplet = add_token(model, z, tokens, args.top_p, current_time, debug=False)
                tokens.extend(new_triplet)
                continuation.extend(new_triplet)
                current_time = new_triplet[0] - TIME_OFFSET
            except Exception as e:
                print(f"\nError generating triplet for prompt {idx+1}: {e}")
                remain = args.sequence_length - len(continuation)
                continuation.extend([0] * remain)
                break

        # Ensure exact length
        continuation = continuation[:args.sequence_length]

        # Normalize event times so that the first time equals 0
        if len(continuation) >= 3:
            first_time = continuation[0]
            if first_time >= TIME_OFFSET:
                dt = first_time - TIME_OFFSET
                if dt > 0:
                    for k in range(0, len(continuation), 3):
                        continuation[k] = continuation[k] - dt
        generated_lines.append(' '.join(map(str, continuation)))

    # Save
    print(f"\nSaving {len(generated_lines)} samples to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in generated_lines:
            f.write(line + '\n')

    print(f"Successfully generated {len(generated_lines)} samples!")
    print(f"Output saved to {args.output_file}")

    # Verify
    if len(generated_lines) > 0:
        first_len = len(generated_lines[0].split())
        last_len = len(generated_lines[-1].split())
        print(f"Verification: first={first_len} tokens, last={last_len} tokens")


if __name__ == "__main__":
    main()

