import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_and_visualize_correlations(matrix_file1, matrix_file2, output_visualization_path, column_to_visualize_scatter=None, scatter_plot_output_path="scatter_plot.png"):
    """
    Loads two matrices, calculates column-wise Pearson correlations,
    computes the average correlation, visualizes the distribution of correlations,
    and optionally creates a scatter plot for a specified column.
    """
    print(f"Loading tensor from: {matrix_file1}")
    try:
        tensor1 = torch.load(matrix_file1, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {matrix_file1}")
        return
    except Exception as e:
        print(f"Error loading {matrix_file1}: {e}")
        return

    print(f"Loading tensor from: {matrix_file2}")
    try:
        tensor2 = torch.load(matrix_file2, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {matrix_file2}")
        return
    except Exception as e:
        print(f"Error loading {matrix_file2}: {e}")
        return

    if tensor1.shape != tensor2.shape:
        print(f"Error: Tensor shapes do not match. Shape 1: {tensor1.shape}, Shape 2: {tensor2.shape}")
        return

    num_rows = tensor1.shape[0]
    num_cols = tensor1.shape[1]
    column_correlations = []

    print(f"Calculating Pearson correlation for {num_cols} columns...")
    for i in range(num_cols):
        col1 = tensor1[:, i].numpy()
        col2 = tensor2[:, i].numpy()
        
        # Pearson correlation requires variance. If a column has no variance, pearsonr can return NaN.
        if np.std(col1) == 0 or np.std(col2) == 0:
            # print(f"Warning: Column {i} has zero variance in one or both tensors. Skipping correlation for this column.")
            # Appending NaN or a specific value like 0, or simply skipping, are options.
            # For averaging, it's better to skip NaNs.
            correlation = np.nan 
        else:
            correlation, _ = pearsonr(col1, col2)
        
        if not np.isnan(correlation):
            column_correlations.append(correlation)
        # else:
            # print(f"NaN correlation for column {i}. Skipping.")


    if not column_correlations:
        print("Error: No valid column correlations could be calculated.")
        return

    average_correlation = np.mean(column_correlations)
    print(f"Average column-wise Pearson correlation: {average_correlation:.4f}")

    # Visualize the distribution of correlations
    plt.figure(figsize=(10, 6))
    plt.hist(column_correlations, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Column-wise Pearson Correlations')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(average_correlation, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {average_correlation:.2f}')
    plt.legend()
    
    try:
        plt.savefig(output_visualization_path)
        print(f"Correlation distribution visualization saved to {output_visualization_path}")
        plt.close() # Close the figure to free memory
    except Exception as e:
        print(f"Error saving correlation distribution visualization: {e}")
    
    # Scatter plot for a specific column
    if column_to_visualize_scatter is not None:
        if 0 <= column_to_visualize_scatter < num_cols:
            col_data1 = tensor1[:, column_to_visualize_scatter].numpy()
            col_data2 = tensor2[:, column_to_visualize_scatter].numpy()

            plt.figure(figsize=(8, 8))
            plt.scatter(col_data1, col_data2, alpha=0.5)
            plt.title(f'Scatter Plot for Column {column_to_visualize_scatter}')
            plt.xlabel(f'Values from Tensor 1 (Column {column_to_visualize_scatter})')
            plt.ylabel(f'Values from Tensor 2 (Column {column_to_visualize_scatter})')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a line y=x for reference if scales are similar, or a regression line
            # For simplicity, let's just plot the points.
            # If you want a regression line:
            # m, b = np.polyfit(col_data1, col_data2, 1)
            # plt.plot(col_data1, m*col_data1 + b, color='red', label='Linear Fit')
            # plt.legend()

            try:
                plt.savefig(scatter_plot_output_path)
                print(f"Scatter plot for column {column_to_visualize_scatter} saved to {scatter_plot_output_path}")
                plt.close() # Close the figure
            except Exception as e:
                print(f"Error saving scatter plot: {e}")
        else:
            print(f"Error: column_to_visualize_scatter index {column_to_visualize_scatter} is out of bounds for {num_cols} columns.")


def calculate_and_visualize_correlations_by_rank(matrix_file1, matrix_file2, output_visualization_path):
    """
    Loads two matrices, and for each column, sorts the rows based on the score matrix.
    Then, it calculates Pearson correlations for different rank groups of scores
    (Top 20, 20-50, 50-100, 100-200, 200-500).
    Finally, it visualizes the distribution of these correlations for each rank group using a box plot.
    """
    print(f"Loading tensor from: {matrix_file1}")
    try:
        tensor1 = torch.load(matrix_file1, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {matrix_file1}")
        return
    except Exception as e:
        print(f"Error loading {matrix_file1}: {e}")
        return

    print(f"Loading tensor from: {matrix_file2}")
    try:
        tensor2 = torch.load(matrix_file2, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {matrix_file2}")
        return
    except Exception as e:
        print(f"Error loading {matrix_file2}: {e}")
        return

    if tensor1.shape != tensor2.shape:
        print(f"Error: Tensor shapes do not match. Shape 1: {tensor1.shape}, Shape 2: {tensor2.shape}")
        return

    num_rows = tensor1.shape[0]
    num_cols = tensor1.shape[1]

    rank_labels = ['Top 50', '50-100', '100-200', '200-500', '500+']
    rank_bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, num_rows)]
    
    if num_rows < rank_bins[-1][1]:
        print(f"Error: Number of rows ({num_rows}) is less than the maximum required rank (500).")
        return

    correlations_by_rank = {label: [] for label in rank_labels}

    print(f"Calculating Pearson correlation for {num_cols} columns across different rank bins...")
    for i in range(num_cols):
        col1 = tensor1[:, i]
        col2 = tensor2[:, i]

        # Sort based on col2 (score) in descending order
        sorted_indices = torch.argsort(col2, descending=True)
        sorted_col1 = col1[sorted_indices].numpy()
        sorted_col2 = col2[sorted_indices].numpy()

        for label, (start, end) in zip(rank_labels, rank_bins):
            slice1 = sorted_col1[start:end]
            slice2 = sorted_col2[start:end]

            if np.std(slice1) == 0 or np.std(slice2) == 0:
                correlation = np.nan
            else:
                correlation, _ = pearsonr(slice1, slice2)
            
            if not np.isnan(correlation):
                correlations_by_rank[label].append(correlation)

    # Visualization
    plt.figure(figsize=(12, 8))
    
    data_to_plot = [correlations_by_rank[label] for label in rank_labels]
    
    plot_data = []
    plot_labels = []
    for label, data in zip(rank_labels, data_to_plot):
        if data:
            plot_data.append(data)
            plot_labels.append(f"{label}\n(n={len(data)})")
        else:
            print(f"Warning: No valid correlations for rank group '{label}'. Skipping in plot.")

    if not plot_data:
        print("Error: No data to plot. All correlation calculations resulted in NaN.")
        return

    plt.boxplot(plot_data, labels=plot_labels)
    
    plt.title('Distribution of Pearson Correlations by Score Rank')
    plt.xlabel('Score Rank Group')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    try:
        plt.savefig(output_visualization_path)
        print(f"Box plot visualization saved to {output_visualization_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving box plot visualization: {e}")


def main():
    # Define file paths
    similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/audio_similarity_matrix.pt"
    score_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/score.pt"
    visualization_output_path = "/home/xiruij/anticipation/correlations_by_rank_boxplot.png"
    
    calculate_and_visualize_correlations_by_rank(
        similarity_matrix_path, 
        score_matrix_path, 
        visualization_output_path
    )

if __name__ == "__main__":
    main()
