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


def main():
    # Define file paths
    similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_clap/audio_similarity_matrix.pt"
    score_matrix_path = "/home/xiruij/anticipation/checkpoints_clap/score.pt"
    visualization_output_path = "/home/xiruij/anticipation/column_correlations_visualization.png"
    
    # Specify which column to create a scatter plot for (e.g., 0 for the first column)
    # Set to None if you don't want a scatter plot for any specific column.
    column_for_scatter = 35
    scatter_plot_file_path = f"/home/xiruij/anticipation/scatter_plot_column_{column_for_scatter}.png" if column_for_scatter is not None else None
    
    # Create checkpoints_clap directory if it doesn't exist, for dummy file creation if needed for testing
    # os.makedirs(os.path.dirname(similarity_matrix_path), exist_ok=True)
    
    # Example: Create dummy files for testing if they don't exist
    # This part is for demonstration; in a real scenario, these files would already exist.
    # if not os.path.exists(similarity_matrix_path):
    #     print(f"Creating dummy file: {similarity_matrix_path}")
    #     torch.save(torch.rand(100, 50), similarity_matrix_path) # Example shape: 100 rows, 50 columns
    # if not os.path.exists(score_matrix_path):
    #     print(f"Creating dummy file: {score_matrix_path}")
    #     # Create a tensor that has some correlation with the first one
    #     dummy_tensor1 = torch.load(similarity_matrix_path)
    #     noise = torch.rand(100,50) * 0.5
    #     torch.save(dummy_tensor1 * 0.8 + noise, score_matrix_path)


    calculate_and_visualize_correlations(
        similarity_matrix_path, 
        score_matrix_path, 
        visualization_output_path,
        column_to_visualize_scatter=column_for_scatter,
        scatter_plot_output_path=scatter_plot_file_path
    )

if __name__ == "__main__":
    main()
