#!/usr/bin/env python3
"""
Extract and save top-K elements from LDS matrices
"""
import torch
from pathlib import Path

# ==================== Configuration ====================
SCORE_PT = "/home/xiruij/anticipation/checkpoints_subset_large/score_LoGra_4096_gen.pt"
SIM_PT = "/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers_gen_prompted.pt"
LAYER_INDEX = 25  # Layer to use (1-based indexing), only for 4D similarity matrix
POOLING_INDEX = 0  # 0=mean, 1=max, only for 4D similarity matrix
TOP_KS = [28000]  # List of Top-K values to extract
OUTPUT_DIR = Path("./checkpoints_subset_large/lds_matrices")
USE_SIMILARITY_RANKING = True  # True: rank by similarity values, False: rank by score values

# ==================== Main Logic ====================
def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    score = torch.load(SCORE_PT, map_location="cpu", weights_only=False)
    sim = torch.load(SIM_PT, map_location="cpu", weights_only=False)
    
    print(f"Score shape: {tuple(score.shape)}, Similarity shape: {tuple(sim.shape)}")
    
    # Handle different dimensions of similarity matrix
    if sim.dim() == 4:
        # 4D: (layers, pooling_types, N_train, N_test)
        original_shape = sim.shape
        print(f"Detected 4D similarity matrix with shape {original_shape}")
        layer_idx = LAYER_INDEX - 1  # Convert to 0-based indexing
        sim_matrix = sim[layer_idx, POOLING_INDEX]
        print(f"Using layer {LAYER_INDEX} (0-based idx={layer_idx}), pooling={POOLING_INDEX}")
        print(f"Extracted similarity matrix shape: {tuple(sim_matrix.shape)}")
    elif sim.dim() == 2:
        # 2D: (N_train, N_test) - use directly
        print(f"Detected 2D similarity matrix, using directly")
        sim_matrix = sim
    else:
        raise ValueError(f"Unsupported similarity matrix dimension: {sim.dim()}D. Expected 2D or 4D.")
    
    # Verify shape match
    if score.shape != sim_matrix.shape:
        raise ValueError(f"Shape mismatch! Score: {score.shape}, Similarity: {sim_matrix.shape}")
    
    # Sort each column in descending order
    if USE_SIMILARITY_RANKING:
        print("Computing sorted indices by similarity values...")
        idx_desc = torch.argsort(sim_matrix, dim=0, descending=True)  # (N_train, N_test)
        ranking_method = "similarity"
    else:
        print("Computing sorted indices by score values...")
        idx_desc = torch.argsort(score, dim=0, descending=True)  # (N_train, N_test)
        ranking_method = "score"
    
    # Extract input similarity matrix name for output filename
    sim_name = Path(SIM_PT).stem  # Get filename without extension
    
    # Build LDS masked matrix for each K value and save
    N_train, N_test = score.shape
    for K in TOP_KS:
        print(f"\nProcessing K={K}...")
        K = min(K, N_train)
        
        # Build mask: keep top-K per column
        mask = torch.zeros(N_train, N_test, dtype=torch.bool)
        for j in range(N_test):
            top_indices = idx_desc[:K, j]
            mask[top_indices, j] = True
        
        # Apply mask: keep top-K, set others to zero
        lds_matrix = torch.where(mask, sim_matrix, torch.zeros_like(sim_matrix))
        
        # Save with ranking method and input matrix name in filename
        output_path = OUTPUT_DIR / f"lds_masked_{sim_name}_{ranking_method}_K{K}.pt"
        torch.save(lds_matrix, output_path)
        print(f"✓ Saved to: {output_path}")
        print(f"  Non-zero elements: {mask.sum().item()} / {N_train * N_test}")

    print(f"\n✅ Done! All matrices saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

