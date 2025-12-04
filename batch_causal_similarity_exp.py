"""
Batch experiment for causal vs similarity analysis.

For a given seed and test_sample_index:
1. Train full model (shared across K=100 and K=500)
2. Compute LoGra attribution scores (shared across K=100 and K=500)
3. For each K in [100, 500]:
   - Find HA-LS top-K training samples
   - Find random top-K training samples  
   - Train removed_hals model
   - Train removed_random model
4. Evaluate all models on the test sample

Usage:
    python batch_causal_similarity_exp.py --seed 0 --test_idx 123 --output_dir ./exp_outputs
"""

import torch
import numpy as np
import os
import argparse
import random
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, default_data_collator

# Import from existing files
from anticipation.vocab import AUTOREGRESS


@dataclass
class PairItem:
    train_idx: int
    rank: int
    score: float
    sim: float
    delta: float


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_hals_for_test_col(score_m: torch.Tensor,
                           sim_m: torch.Tensor,
                           test_idx: int,
                           top_k: int = 10) -> List[PairItem]:
    """Find High-Attribution Low-Similarity samples for a test column.
    
    Simply sort by delta = Z(sim) - Z(score), take the most negative (high score, low sim).
    """
    s_col = score_m[:, test_idx]
    m_col = sim_m[:, test_idx]

    mu_s, std_s = s_col.float().mean().item(), (s_col.float().std(unbiased=False).item() or 1e-6)
    mu_m, std_m = m_col.float().mean().item(), (m_col.float().std(unbiased=False).item() or 1e-6)
    Zs = (s_col.float() - mu_s) / std_s
    Zm = (m_col.float() - mu_m) / std_m
    delta = (Zm - Zs)  # negative = high attribution, low similarity

    # Rank by score (for reporting)
    idx_desc = torch.argsort(s_col, descending=True)
    rank = torch.empty_like(idx_desc, dtype=torch.int32)
    rank[idx_desc] = torch.arange(1, idx_desc.size(0) + 1, dtype=torch.int32)

    # Sort by delta ascending (most negative first = HA-LS)
    sorted_indices = torch.argsort(delta)[:top_k]

    items: List[PairItem] = []
    for i in sorted_indices.tolist():
        items.append(PairItem(
            train_idx=int(i),
            rank=int(rank[i].item()),
            score=float(s_col[i].item()),
            sim=float(m_col[i].item()),
            delta=float(delta[i].item()),
        ))
    return items


def find_random_samples(n_train: int, top_k: int, seed: int, exclude: Optional[set] = None) -> List[int]:
    """Find random training samples."""
    rng = random.Random(seed)
    all_indices = list(range(n_train))
    if exclude:
        all_indices = [i for i in all_indices if i not in exclude]
    return rng.sample(all_indices, min(top_k, len(all_indices)))


def load_generated_sample(file_path: str, sample_index: int, device: torch.device, max_length: int = 1024):
    """Load a generated sample for evaluation."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert 0 <= sample_index < len(lines), f"sample_index={sample_index} out of range"
    arr = np.fromstring(lines[sample_index].strip(), dtype=int, sep=" ")
    if arr.size == 0:
        input_ids = np.array([AUTOREGRESS], dtype=int)
    else:
        input_ids = np.concatenate([np.array([AUTOREGRESS], dtype=int), arr])
    input_ids = input_ids[:max_length]
    input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    batch = {
        "input_ids": input_ids_t.to(device),
        "attention_mask": torch.ones_like(input_ids_t, dtype=torch.long).to(device),
        "labels": input_ids_t.to(device),
    }
    return batch, int(input_ids_t.size(1))


def compute_metrics(model, batch: dict, num_tokens: int):
    """Compute generation probability metrics."""
    with torch.no_grad():
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        nll_mean = float(outputs.loss.item())
    return {
        "n_tokens": num_tokens,
        "nll_mean": nll_mean,
        "avg_log_prob": -nll_mean,
        "sum_log_prob": -nll_mean * num_tokens,
        "ppl": float(np.exp(nll_mean)),
    }


def run_finetune(output_dir: str, train_file: str, valid_file: str, 
                 exclude_indices: str, epochs: int, seed: int):
    """Run finetune.py as subprocess."""
    cmd = [
        sys.executable, "finetune.py",
        "--output_dir", output_dir,
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--exclude_indices", exclude_indices,
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in finetune: {result.stderr}")
        raise RuntimeError(f"Finetune failed: {result.stderr}")
    print(result.stdout)
    return result.returncode == 0


def run_score(train_file: str, valid_file: str, model_path: str, 
              output_dir: str, output_filename: str, valid_is_generated: bool = True):
    """Run score.py (LoGra) as subprocess."""
    cmd = [
        sys.executable, "score.py",
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--output_filename", output_filename,
    ]
    if valid_is_generated:
        cmd.append("--valid_is_generated")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in score: {result.stderr}")
        raise RuntimeError(f"Score failed: {result.stderr}")
    print(result.stdout)
    return result.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(description='Batch causal vs similarity experiment')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--test_idx', type=int, required=True, help='Test sample index')
    parser.add_argument('--output_dir', type=str, default='./exp_causal_similarity',
                        help='Base output directory for all experiments')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Training data file')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Validation data file')
    parser.add_argument('--gen_file', type=str, required=True,
                        help='Generated samples file for LoGra and evaluation')
    parser.add_argument('--sim_pt', type=str, required=True,
                        help='Pre-computed similarity matrix')
    parser.add_argument('--layer_index', type=int, default=25, help='Layer index for similarity (1-based)')
    parser.add_argument('--pooling_index', type=int, default=0, help='Pooling index (0=mean, 1=max)')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--ks', type=str, default='100,500', help='Comma-separated K values')
    parser.add_argument('--skip_train_full', action='store_true', help='Skip full model training if exists')
    parser.add_argument('--skip_logra', action='store_true', help='Skip LoGra computation if exists')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    ks = [int(k.strip()) for k in args.ks.split(',')]
    
    # Create output directory structure
    # Include test_idx in path to avoid conflicts when running multiple test samples in parallel
    base_dir = Path(args.output_dir)
    exp_dir = base_dir / f"seed_{args.seed}_test_{args.test_idx}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}, Test Index: {args.test_idx}, Ks: {ks}")
    
    # Paths
    full_model_path = exp_dir / "full_model"
    score_file = exp_dir / "score_LoGra.pt"
    
    # ========== Step 1: Train full model ==========
    if args.skip_train_full and full_model_path.exists():
        print(f"Skipping full model training, using existing: {full_model_path}")
    else:
        print("=" * 60)
        print("Step 1: Training full model...")
        print("=" * 60)
        run_finetune(
            output_dir=str(full_model_path),
            train_file=args.train_file,
            valid_file=args.valid_file,
            exclude_indices="",
            epochs=args.epochs,
            seed=args.seed,
        )
    
    # ========== Step 2: Compute LoGra attribution scores ==========
    if args.skip_logra and score_file.exists():
        print(f"Skipping LoGra computation, using existing: {score_file}")
    else:
        print("=" * 60)
        print("Step 2: Computing LoGra attribution scores...")
        print("=" * 60)
        run_score(
            train_file=args.train_file,
            valid_file=args.gen_file,
            model_path=str(full_model_path),
            output_dir=str(exp_dir),
            output_filename="score_LoGra.pt",
            valid_is_generated=True,
        )
    
    # ========== Step 3: Load scores and similarity ==========
    print("=" * 60)
    print("Step 3: Loading scores and similarity...")
    print("=" * 60)
    
    score = torch.load(score_file, map_location="cpu", weights_only=False)
    sim = torch.load(args.sim_pt, map_location="cpu", weights_only=False)
    
    L, P, N_train, N_test = sim.shape
    layer_idx0 = max(0, min(L - 1, args.layer_index - 1))
    sim_selected = sim[layer_idx0, args.pooling_index]
    
    print(f"Score shape: {score.shape}, Sim shape: {sim_selected.shape}")
    
    # ========== Step 4: For each K, find samples and train models ==========
    all_results = {
        "seed": args.seed,
        "test_idx": args.test_idx,
        "ks": ks,
        "results": {}
    }
    
    for K in ks:
        print("=" * 60)
        print(f"Step 4: Processing K={K}...")
        print("=" * 60)
        
        k_dir = exp_dir / f"k_{K}"
        k_dir.mkdir(parents=True, exist_ok=True)
        
        # Find HA-LS samples (sorted by delta, always returns exactly K)
        hals_items = find_hals_for_test_col(score, sim_selected, args.test_idx, top_k=K)
        hals_indices = [item.train_idx for item in hals_items]
        
        # Find random samples (different from HA-LS)
        random_indices = find_random_samples(N_train, K, args.seed + K, exclude=set(hals_indices))
        
        # Save sample info
        sample_info = {
            "hals_indices": hals_indices,
            "hals_details": [asdict(item) for item in hals_items],
            "random_indices": random_indices,
        }
        with open(k_dir / "sample_info.json", "w") as f:
            json.dump(sample_info, f, indent=2)
        
        print(f"HA-LS indices ({len(hals_indices)}): {hals_indices[:10]}...")
        print(f"Random indices ({len(random_indices)}): {random_indices[:10]}...")
        
        # Train removed_hals model
        removed_hals_path = k_dir / "removed_hals_model"
        if not removed_hals_path.exists():
            print(f"Training removed_hals model for K={K}...")
            run_finetune(
                output_dir=str(removed_hals_path),
                train_file=args.train_file,
                valid_file=args.valid_file,
                exclude_indices=",".join(map(str, hals_indices)),
                epochs=args.epochs,
                seed=args.seed,
            )
        
        # Train removed_random model
        removed_random_path = k_dir / "removed_random_model"
        if not removed_random_path.exists():
            print(f"Training removed_random model for K={K}...")
            run_finetune(
                output_dir=str(removed_random_path),
                train_file=args.train_file,
                valid_file=args.valid_file,
                exclude_indices=",".join(map(str, random_indices)),
                epochs=args.epochs,
                seed=args.seed,
            )
        
        # ========== Step 5: Evaluate all models ==========
        print(f"Evaluating models for K={K}...")
        
        models_to_eval = {
            "full": str(full_model_path),
            "removed_hals": str(removed_hals_path),
            "removed_random": str(removed_random_path),
        }
        
        k_results = {}
        batch, n_tokens = load_generated_sample(args.gen_file, args.test_idx, device)
        
        for name, mpath in models_to_eval.items():
            if not os.path.isdir(mpath):
                print(f"Skip {name}: path not found -> {mpath}")
                continue
            print(f"Loading {name}...")
            model = AutoModelForCausalLM.from_pretrained(mpath, attn_implementation="eager").to(device)
            model.eval()
            metrics = compute_metrics(model, batch, n_tokens)
            k_results[name] = metrics
            print(f"  {name}: avg_log_prob={metrics['avg_log_prob']:.6f}, ppl={metrics['ppl']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        all_results["results"][f"k_{K}"] = k_results
    
    # Save final results
    results_file = exp_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for K in ks:
        print(f"\nK={K}:")
        k_res = all_results["results"].get(f"k_{K}", {})
        for name, metrics in k_res.items():
            print(f"  {name:15s}: avg_log_prob={metrics['avg_log_prob']:.6f}, ppl={metrics['ppl']:.4f}")


if __name__ == "__main__":
    main()
