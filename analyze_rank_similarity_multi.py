"""
Analyze a (L, 2, N_train, N_test) similarity tensor produced by MERT.

For every layer (L) and pooling type (0 = mean, 1 = max) we

1. rank each test column by a 2-D **score matrix** of shape (N_train, N_test);
2. collect the similarity at each rank position;
3. average across all columns → (rank, avg-similarity) curve;
4. save an individual scatter + trend-line plot;
5. after all features are processed, draw **one figure** that shows
   *smoothed* trend-lines for every feature (no scatter).

The final stacked similarity tensor is stored as:

    similarity_tensor[layer, pooling, train, test]

Usage example
-------------
python analyze_mert_tensor.py \
    --score /path/score.pt \
    --sim   /path/audio_similarity_all_layers.pt \
    --out   /path/plots \
    --max_ranks 28000 \
    --sort_abs
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# -----------------------------------------------------------------------------#
#                         similarity → rank-curve utils                        #
# -----------------------------------------------------------------------------#
def compute_avg_similarity_per_rank(
    score_m: torch.Tensor,
    sim_m: torch.Tensor,
    max_ranks: int,
    sort_by_abs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given 2-D tensors (N_train, N_test), compute average similarity vs rank.

    Returns
    -------
    ranks : np.ndarray, shape (K,)
    avg_sim : np.ndarray, shape (K,)
        K == max_ranks (may be truncated if < max_ranks rows).
    """
    score_m = score_m.cpu()
    sim_m = sim_m.cpu()

    n_rows, n_cols = score_m.shape
    max_ranks = min(max_ranks, n_rows)

    # bucket[rank] → list of similarity values across all columns
    buckets: List[List[float]] = [[] for _ in range(max_ranks)]

    for c in range(n_cols):
        scores = score_m[:, c]
        sims = sim_m[:, c]

        if sort_by_abs:
            idx = torch.argsort(torch.abs(scores), descending=True)
        else:
            idx = torch.argsort(scores, descending=True)

        for r in range(max_ranks):
            buckets[r].append(float(sims[idx[r]]))

    ranks = np.arange(1, max_ranks + 1)
    avg_sim = np.array([np.mean(b) for b in buckets], dtype=np.float32)
    return ranks, avg_sim


def smooth_curve(y: np.ndarray, window: int = 201) -> np.ndarray:
    """Moving-average smoothing with reflection padding to avoid edge drop."""
    n = int(len(y))
    if n < 3:
        return y
    # ensure odd window and not longer than the series
    window = int(window)
    window = max(3, min(window, n))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y
    half = window // 2
    box = np.ones(window, dtype=np.float32) / window
    # reflect padding (no zero-padding) to eliminate start/end dips
    y_pad = np.pad(y.astype(np.float32), (half, half), mode="reflect")
    return np.convolve(y_pad, box, mode="valid")


# -----------------------------------------------------------------------------#
#                                    main                                      #
# -----------------------------------------------------------------------------#
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, help="score matrix .pt (N×M)")
    ap.add_argument("--sim", required=True, help="similarity tensor .pt (L×2×N×M)")
    ap.add_argument("--out", required=True, help="output directory for plots")
    ap.add_argument("--max_ranks", type=int, default=500)
    ap.add_argument("--sort_abs", action="store_true")
    ap.add_argument("--smooth_win", type=int, default=201,
                    help="window size for moving-average smoothing")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading score matrix …")
    score = torch.load(args.score, map_location="cpu")         # (N, M)
    print("Loading similarity tensor …")
    sim_tensor = torch.load(args.sim, map_location="cpu")      # (L, 2, N, M)

    if score.ndim != 2 or sim_tensor.ndim != 4:
        raise ValueError("score must be 2-D and similarity 4-D")

    L, P, N, M = sim_tensor.shape
    assert score.shape == (N, M), "score / similarity shape mismatch"

    pooling_tag = {0: "mean", 1: "max"}

    # Store all curves for the final combined figure
    all_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for l in range(L):
        for p in range(P):
            tag = f"layer{l}_{pooling_tag[p]}"
            print(f"Processing {tag} …")
            ranks, avg_sim = compute_avg_similarity_per_rank(
                score, sim_tensor[l, p], max_ranks=args.max_ranks,
                sort_by_abs=args.sort_abs,
            )

            # plt.figure(figsize=(12, 8))
            # plt.scatter(ranks, avg_sim, s=15, alpha=0.6, label="samples")
            # z = np.polyfit(ranks, avg_sim, 1)
            # plt.plot(ranks, np.poly1d(z)(ranks), "r--",
            #          label=f"trend (slope={z[0]:.6f})")
            # plt.title(f"{tag}: Rank vs Avg Similarity")
            # plt.xlabel("Rank")
            # plt.ylabel("Average Similarity")
            # plt.grid(alpha=0.4)
            # plt.legend()
            # plt.tight_layout()
            # fname = os.path.join(args.out, f"{tag}.png")
            # plt.savefig(fname, dpi=300)
            # plt.close()
            # print(f"  → saved {fname}")

            all_curves.append((tag, ranks, avg_sim))

    # ------------------------------------------------------------------#
    #                  combined smoothed trend-line figure              #
    # ------------------------------------------------------------------#
    plt.figure(figsize=(14, 10))
    for tag, ranks, avg_sim in all_curves:
        smoothed = smooth_curve(avg_sim, window=args.smooth_win)
        plt.plot(ranks, smoothed, alpha=0.5, linewidth=1.2, label=tag)

    plt.title("All Features: Smoothed Rank–Similarity Trends")
    plt.xlabel("Rank")
    plt.ylabel("Average Similarity")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    comb_path = os.path.join(args.out, "all_features_trendlines.png")
    plt.savefig(comb_path, dpi=300)
    plt.close()
    print(f"\nCombined trend-line figure saved to {comb_path}")


if __name__ == "__main__":
    main()
