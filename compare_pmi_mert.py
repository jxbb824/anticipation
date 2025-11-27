import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load matrices
print("Loading PMI similarity matrix...")
pmi_matrix = torch.load("/home/xiruij/anticipation/checkpoints_subset_large/melody_similarity_pmi_ti.pt", 
                        map_location=torch.device('cpu'))
print(f"PMI matrix shape: {pmi_matrix.shape}")

print("Loading MERT similarity matrix...")
mert_matrix = torch.load("/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers.pt",
                         map_location=torch.device('cpu'))
print(f"MERT matrix shape (original): {mert_matrix.shape}")

# Extract last layer mean and max pooling from MERT
mert_mean = mert_matrix[-1, 0, :, :]  # Last layer, mean pooling
mert_max = mert_matrix[-1, 1, :, :]   # Last layer, max pooling
print(f"MERT mean shape: {mert_mean.shape}")
print(f"MERT max shape: {mert_max.shape}")

# Check shapes match
assert pmi_matrix.shape == mert_mean.shape == mert_max.shape, "Matrix shapes must match"

num_rows, num_cols = pmi_matrix.shape
max_ranks = min(28000, num_rows)  # Analyze top 500 ranks

print(f"\nAnalyzing top {max_ranks} ranks...")

# Store MERT similarity values for each PMI rank position
rank_mert_mean = [[] for _ in range(max_ranks)]
rank_mert_max = [[] for _ in range(max_ranks)]

# For each column (test sample)
for col in range(num_cols):
    pmi_col = pmi_matrix[:, col]
    mert_mean_col = mert_mean[:, col]
    mert_max_col = mert_max[:, col]
    
    # Sort by PMI similarity in descending order
    sorted_indices = torch.argsort(pmi_col, descending=True)
    
    # Extract corresponding MERT values based on PMI rank
    for rank in range(max_ranks):
        if rank < len(sorted_indices):
            idx = sorted_indices[rank]
            rank_mert_mean[rank].append(mert_mean_col[idx].item())
            rank_mert_max[rank].append(mert_max_col[idx].item())

# Calculate average MERT similarity for each PMI rank
ranks = []
avg_mert_mean = []
avg_mert_max = []

for rank in range(max_ranks):
    if rank_mert_mean[rank]:
        ranks.append(rank + 1)  # Rank starts from 1
        avg_mert_mean.append(np.mean(rank_mert_mean[rank]))
        avg_mert_max.append(np.mean(rank_mert_max[rank]))

print(f"Calculated {len(ranks)} rank positions")

# Also prepare data for naive scatter plots (all points)
print("\nPreparing data for naive scatter plots...")
pmi_all = pmi_matrix.flatten().numpy()
mert_mean_all = mert_mean.flatten().numpy()
mert_max_all = mert_max.flatten().numpy()
print(f"Total data points: {len(pmi_all)}")

# Create figure with 4 subplots (2 rows x 2 cols)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ========== Top Left: PMI Rank vs MERT Mean Pooling ==========
ax1 = axes[0, 0]
ax1.scatter(ranks, avg_mert_mean, alpha=0.7, s=20, color='blue')

# Add trend line
if len(ranks) > 1:
    z = np.polyfit(ranks, avg_mert_mean, 1)
    p = np.poly1d(z)
    ax1.plot(ranks, p(ranks), "r--", alpha=0.8, linewidth=2, 
             label=f'Trend: slope={z[0]:.6f}')
    
    # Calculate correlation
    corr_mean = np.corrcoef(ranks, avg_mert_mean)[0, 1]
    spearman_mean = stats.spearmanr(ranks, avg_mert_mean)[0]
    ax1.text(0.05, 0.95, f'Pearson: {corr_mean:.4f}\nSpearman: {spearman_mean:.4f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('PMI Rank', fontsize=14)
ax1.set_ylabel('Average MERT Similarity (Mean Pooling)', fontsize=14)
ax1.set_title('PMI Rank vs MERT Mean Pooling', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# ========== Top Right: PMI Rank vs MERT Max Pooling ==========
ax2 = axes[0, 1]
ax2.scatter(ranks, avg_mert_max, alpha=0.7, s=20, color='green')

# Add trend line
if len(ranks) > 1:
    z = np.polyfit(ranks, avg_mert_max, 1)
    p = np.poly1d(z)
    ax2.plot(ranks, p(ranks), "r--", alpha=0.8, linewidth=2,
             label=f'Trend: slope={z[0]:.6f}')
    
    # Calculate correlation
    corr_max = np.corrcoef(ranks, avg_mert_max)[0, 1]
    spearman_max = stats.spearmanr(ranks, avg_mert_max)[0]
    ax2.text(0.05, 0.95, f'Pearson: {corr_max:.4f}\nSpearman: {spearman_max:.4f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('PMI Rank', fontsize=14)
ax2.set_ylabel('Average MERT Similarity (Max Pooling)', fontsize=14)
ax2.set_title('PMI Rank vs MERT Max Pooling', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# ========== Bottom Left: Naive Scatter - PMI vs MERT Mean ==========
ax3 = axes[1, 0]
ax3.scatter(pmi_all, mert_mean_all, alpha=0.3, s=1, color='blue')

# Add trend line
z3 = np.polyfit(pmi_all, mert_mean_all, 1)
p3 = np.poly1d(z3)
pmi_range = np.array([pmi_all.min(), pmi_all.max()])
ax3.plot(pmi_range, p3(pmi_range), "r-", alpha=0.8, linewidth=2,
         label=f'y = {z3[0]:.4f}x + {z3[1]:.4f}')

# Calculate correlation
corr_naive_mean = np.corrcoef(pmi_all, mert_mean_all)[0, 1]
spearman_naive_mean = stats.spearmanr(pmi_all, mert_mean_all)[0]
ax3.text(0.05, 0.95, f'Pearson: {corr_naive_mean:.4f}\nSpearman: {spearman_naive_mean:.4f}',
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax3.set_xlabel('PMI Similarity', fontsize=14)
ax3.set_ylabel('MERT Similarity (Mean Pooling)', fontsize=14)
ax3.set_title('PMI vs MERT Mean (All Points)', fontsize=16)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()

# ========== Bottom Right: Naive Scatter - PMI vs MERT Max ==========
ax4 = axes[1, 1]
ax4.scatter(pmi_all, mert_max_all, alpha=0.3, s=1, color='green')

# Add trend line
z4 = np.polyfit(pmi_all, mert_max_all, 1)
p4 = np.poly1d(z4)
ax4.plot(pmi_range, p4(pmi_range), "r-", alpha=0.8, linewidth=2,
         label=f'y = {z4[0]:.4f}x + {z4[1]:.4f}')

# Calculate correlation
corr_naive_max = np.corrcoef(pmi_all, mert_max_all)[0, 1]
spearman_naive_max = stats.spearmanr(pmi_all, mert_max_all)[0]
ax4.text(0.05, 0.95, f'Pearson: {corr_naive_max:.4f}\nSpearman: {spearman_naive_max:.4f}',
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_xlabel('PMI Similarity', fontsize=14)
ax4.set_ylabel('MERT Similarity (Max Pooling)', fontsize=14)
ax4.set_title('PMI vs MERT Max (All Points)', fontsize=16)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend()

plt.tight_layout()

# Save figure
output_path = "/home/xiruij/anticipation/pmi_vs_mert_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Print statistics
print(f"\n=== Statistics ===")

print(f"\n[Rank-based Analysis]")
print(f"\nMERT Mean Pooling (by PMI Rank):")
print(f"  PMI Rank 1 → MERT: {avg_mert_mean[0]:.6f}")
print(f"  PMI Rank {len(ranks)} → MERT: {avg_mert_mean[-1]:.6f}")
print(f"  Change: {avg_mert_mean[-1] - avg_mert_mean[0]:.6f}")
if len(ranks) > 1:
    print(f"  Pearson correlation: {corr_mean:.6f}")
    print(f"  Spearman correlation: {spearman_mean:.6f}")

print(f"\nMERT Max Pooling (by PMI Rank):")
print(f"  PMI Rank 1 → MERT: {avg_mert_max[0]:.6f}")
print(f"  PMI Rank {len(ranks)} → MERT: {avg_mert_max[-1]:.6f}")
print(f"  Change: {avg_mert_max[-1] - avg_mert_max[0]:.6f}")
if len(ranks) > 1:
    print(f"  Pearson correlation: {corr_max:.6f}")
    print(f"  Spearman correlation: {spearman_max:.6f}")

print(f"\n[Naive Correlation - All Points]")
print(f"\nPMI vs MERT Mean Pooling:")
print(f"  Pearson correlation: {corr_naive_mean:.6f}")
print(f"  Spearman correlation: {spearman_naive_mean:.6f}")
print(f"  Linear regression: y = {z3[0]:.6f}x + {z3[1]:.6f}")

print(f"\nPMI vs MERT Max Pooling:")
print(f"  Pearson correlation: {corr_naive_max:.6f}")
print(f"  Spearman correlation: {spearman_naive_max:.6f}")
print(f"  Linear regression: y = {z4[0]:.6f}x + {z4[1]:.6f}")

plt.close()

