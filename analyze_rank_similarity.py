import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_rank_similarity(score_matrix_path, similarity_matrix_path, output_plot_path="rank_similarity_scatter.png", max_ranks=500):
    """
    For each column, sort based on score_matrix, find the corresponding similarity_matrix value for each rank,
    calculate the average similarity for each rank, and draw a scatter plot.
    
    Args:
        score_matrix_path: path to the score matrix file
        similarity_matrix_path: path to the similarity matrix file
        output_plot_path: output plot path
        max_ranks: maximum number of ranks to analyze
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

    if score_matrix.shape != similarity_matrix.shape:
        print(f"Error: Matrix shapes do not match. Score shape: {score_matrix.shape}, Similarity shape: {similarity_matrix.shape}")
        return

    num_rows, num_cols = score_matrix.shape
    max_ranks = min(max_ranks, num_rows)
    
    print(f"Matrix shape: {num_rows} x {num_cols}")
    
    # Store similarity values for each rank position
    rank_similarities = [[] for _ in range(max_ranks)]
    
    print(f"Analyzing ranks and corresponding similarity values for each column...")
    for col in range(num_cols):
        score_col = score_matrix[:, col]
        similarity_col = similarity_matrix[:, col]
        
        # Sort by score in descending order to get sorted indices
        sorted_indices = torch.argsort(score_col, descending=True)
        
        # Extract corresponding similarity values based on rank
        for rank in range(max_ranks):
            if rank < len(sorted_indices):
                idx = sorted_indices[rank]
                similarity_value = similarity_col[idx].item()
                rank_similarities[rank].append(similarity_value)
    
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
    
    # Plot scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(ranks, avg_similarities, alpha=0.7, s=20)
    plt.title('Rank vs. Average Similarity Score', fontsize=16)
    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Average Similarity Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(ranks) > 1:
        z = np.polyfit(ranks, avg_similarities, 1)
        p = np.poly1d(z)
        plt.plot(ranks, p(ranks), "r--", alpha=0.8, label=f'Trend Line (slope: {z[0]:.6f})')
        plt.legend()
    
    # Set x-axis ticks
    if max_ranks <= 50:
        plt.xticks(range(1, max_ranks + 1, 5))
    elif max_ranks <= 200:
        plt.xticks(range(1, max_ranks + 1, 20))
    else:
        plt.xticks(range(1, max_ranks + 1, 50))
    
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
    
    return ranks, avg_similarities


def main():
    # Define file paths
    score_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/score.pt"
    similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/audio_similarity_matrix.pt"
    output_plot_path = "/home/xiruij/anticipation/rank_similarity_scatter.png"
    
    # Analyze the relationship between rank and similarity
    analyze_rank_similarity(
        score_matrix_path,
        similarity_matrix_path, 
        output_plot_path,
        max_ranks=3000  # Analyze top 3000 ranks
    )

if __name__ == "__main__":
    main()
