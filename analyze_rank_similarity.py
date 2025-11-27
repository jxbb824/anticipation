import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def analyze_rank_similarity(score_matrix_path, similarity_matrix_path, output_plot_path="rank_similarity_scatter.png", max_ranks=500, sort_by_abs=False, filter_zeros=False, print_examples=False, break_y=False):
    """
    For each column, sort based on score_matrix, find the corresponding similarity_matrix value for each rank,
    calculate the average similarity for each rank, and draw a scatter plot.
    
    Args:
        score_matrix_path: path to the score matrix file
        similarity_matrix_path: path to the similarity matrix file
        output_plot_path: output plot path
        max_ranks: maximum number of ranks to analyze
        sort_by_abs: whether to sort by absolute values
        filter_zeros: whether to filter out similarity values that are 0
        print_examples: whether to print random 5 (train_idx, test_idx) for rank 1,
            last rank, and the highest-average-similarity rank
        break_y: whether to use broken y-axis
    """
    print(f"Loading score matrix: {score_matrix_path}")
    try:
        score_matrix = torch.load(score_matrix_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found {score_matrix_path}")
        return
    except Exception as e:
        print(f"Error loading score matrix: {e}")
        return

    print(f"Loading similarity matrix: {similarity_matrix_path}")
    try:
        similarity_matrix = torch.load(similarity_matrix_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found {similarity_matrix_path}")
        return
    except Exception as e:
        print(f"Error loading similarity matrix: {e}")
        return

    # Handle multi-dimensional similarity matrices (e.g., shape: (layers, pooling_types, a, b))
    # Automatically extract the last layer's mean pooling result
    if similarity_matrix.dim() == 4:
        original_shape = similarity_matrix.shape
        print(f"Detected 4D similarity matrix with shape {original_shape}")
        print(f"Extracting last layer (index -1) and mean pooling (index 1)...")
        similarity_matrix = similarity_matrix[-1, 0, :, :]  # Last layer, mean pooling
        print(f"Extracted similarity matrix shape: {similarity_matrix.shape}")
    elif similarity_matrix.dim() == 3:
        original_shape = similarity_matrix.shape
        print(f"Detected 3D similarity matrix with shape {original_shape}")
        print(f"Extracting last dimension slice [−1, :, :]...")
        similarity_matrix = similarity_matrix[-1, :, :]
        print(f"Extracted similarity matrix shape: {similarity_matrix.shape}")

    # Limit the last dimension (columns) to 500
    if similarity_matrix.shape[1] > 500:
        print(f"Limiting similarity matrix columns from {similarity_matrix.shape[1]} to 500")
        similarity_matrix = similarity_matrix[:, :500]
        score_matrix = score_matrix[:, :500]
    
    if score_matrix.shape != similarity_matrix.shape:
        print(f"Error: Matrix shapes do not match. Score shape: {score_matrix.shape}, Similarity shape: {similarity_matrix.shape}")
        return

    num_rows, num_cols = score_matrix.shape
    max_ranks = min(max_ranks, num_rows)
    
    print(f"Matrix shape: {num_rows} x {num_cols}")
    
    # Store similarity values for each rank position
    rank_similarities = [[] for _ in range(max_ranks)]
    # Example collections
    rank1_examples = []      # (train_idx, test_idx)
    last_rank_examples = []  # (train_idx, test_idx)
    
    print(f"Analyzing ranks and corresponding similarity values for each column...")
    for col in range(num_cols):
        score_col = score_matrix[:, col]
        similarity_col = similarity_matrix[:, col]
        
        # Sort by score in descending order to get sorted indices
        if sort_by_abs:
            sorted_indices = torch.argsort(torch.abs(score_col), descending=True)
        else:
            sorted_indices = torch.argsort(score_col, descending=True)
        
        # Extract corresponding similarity values based on rank
        for rank in range(max_ranks):
            if rank < len(sorted_indices):
                idx = sorted_indices[rank]
                similarity_value = similarity_col[idx].item()
                # Filter out zero values if filter_zeros is enabled
                if filter_zeros and similarity_value == 0:
                    continue
                rank_similarities[rank].append(similarity_value)
                if print_examples:
                    train_idx = int(idx.item() if hasattr(idx, 'item') else int(idx))
                    test_idx = col
                    if rank == 0:
                        rank1_examples.append((train_idx, test_idx))
                    if rank == max_ranks - 1:
                        last_rank_examples.append((train_idx, test_idx))
    
    # Calculate the average similarity for each rank
    ranks = []
    avg_similarities = []
    
    for rank in range(max_ranks):
        if rank_similarities[rank]:  # Ensure there is data
            avg_sim = np.mean(rank_similarities[rank])
            ranks.append(rank + 1)  # Rank starts from 1
            avg_similarities.append(avg_sim)
    
    if not ranks:
        print("Error: No valid data for plotting")
        return
    
    print(f"Calculation complete. Analyzed {len(ranks)} rank positions.")
    
    title_map = {
        'mert': 'MERT vs TDA Rank',
        'clap': 'CLAP vs TDA Rank', 
        'pmi': 'PMI vs TDA Rank'
    }
    title = 'Similarity vs TDA Rank'
    for key, val in title_map.items():
        if key in output_plot_path.lower():
            title = val
            break
    
    if max_ranks <= 50:
        step = 10
    elif max_ranks <= 200:
        step = 50
    elif max_ranks <= 1000:
        step = 200
    else:
        step = 5000
    
    if break_y:
        # Broken axis plot - separate top 5 outliers from normal values
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [1, 3]})
        fig.subplots_adjust(hspace=0.05)
        
        # Find top 8 values as outliers
        sorted_sims = sorted(avg_similarities, reverse=True)
        n_outliers = 8
        outlier_threshold = sorted_sims[n_outliers - 1]  # 8th largest value
        normal_max = sorted_sims[n_outliers]  # 9th largest value (max of normal)
        
        y_min = min(avg_similarities)
        y_max = max(avg_similarities)
        
        # Plot same data on both axes
        ax1.scatter(ranks, avg_similarities, alpha=0.7, s=50)
        ax2.scatter(ranks, avg_similarities, alpha=0.7, s=50)
        
        # Top axis: outliers (top 8 values)
        ax1.set_ylim(outlier_threshold, y_max)
        # Bottom axis: normal values
        ax2.set_ylim(y_min, normal_max)
        
        # Hide spines between axes
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(labeltop=False, bottom=False)
        ax2.xaxis.tick_bottom()
        
        # Add break marks
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        # Labels and title
        fig.suptitle(title, fontsize=40, fontweight='bold')
        ax2.set_xlabel('Rank', fontsize=36, fontweight='bold')
        fig.text(0.02, 0.5, 'Average Similarity Score', va='center', rotation='vertical', 
                 fontsize=36, fontweight='bold')
        
        ax1.tick_params(axis='y', labelsize=30)
        ax2.tick_params(axis='y', labelsize=30)
        ax2.tick_params(axis='x', labelsize=30)
        ax1.grid(True, linestyle='--', alpha=0.7, linewidth=3)
        ax2.grid(True, linestyle='--', alpha=0.7, linewidth=3)
        ax2.set_xticks(range(0, max_ranks + 1, step))
        
        plt.tight_layout()
        fig.subplots_adjust(left=0.15)
    else:
        # Normal plot
        plt.figure(figsize=(12, 8))
        plt.scatter(ranks, avg_similarities, alpha=0.7, s=50)
        
        plt.title(title, fontsize=40, fontweight='bold')
        plt.xlabel('Rank', fontsize=36, fontweight='bold')
        plt.ylabel('Average Similarity Score', fontsize=36, fontweight='bold')
        plt.yticks(fontsize=30)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=3)
        plt.xticks(range(0, max_ranks + 1, step), fontsize=30)
        
        plt.tight_layout()
    
    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {output_plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Average similarity for rank 1: {avg_similarities[0]:.6f}")
    print(f"Average similarity for rank {len(ranks)}: {avg_similarities[-1]:.6f}")
    print(f"Change in similarity: {avg_similarities[-1] - avg_similarities[0]:.6f}")
    
    # Print example indices if requested
    if print_examples:
        print("\n" + "=" * 60)
        print("Example Samples (random up to 5 each):")
        print("=" * 60)

        def print_random_examples(title, examples):
            if not examples:
                print(f"\n{title}: (no examples)")
                return
            k = min(5, len(examples))
            sampled = random.sample(examples, k)
            print(f"\n{title} (train_idx, test_idx):")
            for train_idx, test_idx in sampled:
                print(f"  ({train_idx}, {test_idx})")

        # Rank 1 examples
        print_random_examples("Rank 1 examples", rank1_examples)

        # Last rank examples
        last_rank_label = ranks[-1] if ranks else None
        print_random_examples(f"Last rank (rank {last_rank_label}) examples", last_rank_examples)

        # Highest average similarity rank examples
        highest_rank_examples = []
        if avg_similarities:
            max_sim_idx = int(np.argmax(avg_similarities))
            max_sim_rank = int(ranks[max_sim_idx])
            max_sim_rank_zero_indexed = max_sim_rank - 1

            # Recompute the (train_idx, test_idx) pairs for the highest similarity rank
            for col in range(num_cols):
                score_col = score_matrix[:, col]
                similarity_col = similarity_matrix[:, col]
                if sort_by_abs:
                    sorted_indices = torch.argsort(torch.abs(score_col), descending=True)
                else:
                    sorted_indices = torch.argsort(score_col, descending=True)
                if max_sim_rank_zero_indexed < len(sorted_indices):
                    idx = sorted_indices[max_sim_rank_zero_indexed]
                    sim_val = similarity_col[idx].item()
                    if filter_zeros and sim_val == 0:
                        continue
                    train_idx = int(idx.item() if hasattr(idx, 'item') else int(idx))
                    test_idx = col
                    highest_rank_examples.append((train_idx, test_idx))

            print_random_examples(
                f"Highest similarity rank (rank {max_sim_rank}, avg similarity: {avg_similarities[max_sim_idx]:.6f}) examples",
                highest_rank_examples,
            )
        print("=" * 60 + "\n")
    
    return ranks, avg_similarities


def main():
    base_path = "/home/xiruij/anticipation/checkpoints_subset_large"
    
    score_matrix_paths = [
        # f"{base_path}/score_LoGra_4096.pt",
        # f"{base_path}/score_LoGra_4096.pt",
        f"{base_path}/score_LoGra_4096.pt",
        # f"{base_path}/score_LoGra_4096_gen_prompted.pt",
        # f"{base_path}/score_LoGra_4096_gen_prompted.pt",
        f"{base_path}/score_LoGra_4096_gen_prompted.pt"
        # f"{base_path}/score_LoGra_4096.pt"
    ]
    
    similarity_matrix_paths = [
        # f"{base_path}/lds_matrices/lds_masked_audio_similarity_all_layers.pt",
        # f"{base_path}/audio_similarity_clap.pt",
        f"{base_path}/melody_similarity_pmi_ti.pt",
        # f"{base_path}/lds_matrices/lds_masked_audio_similarity_all_layers_gen_prompted.pt",
        # f"{base_path}/audio_similarity_clap_gen_prompted.pt",
        f"{base_path}/melody_similarity_pmi_ti_gen_prompted.pt"
        # "/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers_test_improved.pt"
    ]
    
    output_plot_paths = [
        # "/home/xiruij/anticipation/rank_similarity_mert_test.pdf",
        # "/home/xiruij/anticipation/rank_similarity_clap_test.pdf",
        "/home/xiruij/anticipation/rank_similarity_pmi_test_break.pdf",
        # "/home/xiruij/anticipation/rank_similarity_mert_gen.pdf",
        # "/home/xiruij/anticipation/rank_similarity_clap_gen.pdf",
        "/home/xiruij/anticipation/rank_similarity_pmi_gen_break.pdf"
        # "/home/xiruij/anticipation/improved.png"
    ]
    
    for score_path, sim_path, output_path in zip(score_matrix_paths, similarity_matrix_paths, output_plot_paths):
        print(f"\n{'='*80}")
        print(f"Processing: {output_path}")
        print(f"{'='*80}")
        analyze_rank_similarity(
            score_path,
            sim_path, 
            output_path,
            max_ranks=28000,
            sort_by_abs=False,
            filter_zeros=True,
            print_examples=False,
            break_y=True
        )

if __name__ == "__main__":
    main()
