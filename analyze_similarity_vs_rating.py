import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path

def load_similarity_data(csv_path):
    """Load similarity data from CSV"""
    df = pd.read_csv(csv_path)
    similarity_data = {}
    
    for _, row in df.iterrows():
        test_id = row['test_sample_id']
        similarity_data[test_id] = {
            'rank_1': row['rank_1_similarity_score'],
            'rank_10': row['rank_10_similarity_score'], 
            'rank_100': row['rank_100_similarity_score'],
            'rank_1000': row['rank_1000_similarity_score'],
            'rank_3000': row['rank_3000_similarity_score']
        }
    
    return similarity_data

def get_human_ratings():
    """Get hardcoded human rating data"""
    # Rater 1 data
    rater1_data = [
        (1, 5, 3, 2, 1, 1),
        (2, 5, 3, 4, 1, 2),
        (3, 2, 2, 4, 4, 5),
        (4, 5, 2, 1, 3, 1),
        (5, 3, 5, 3, 2, 2),
        (6, 4, 3, 1, 2, 3),
        (7, 5, 2, 1, 1, 1),
        (8, 1, 2, 4, 4, 1),
        (9, 2, 3, 2, 1, 1),
        (10, 5, 2, 4, 1, 3)
    ]
    
    # Rater 2 additional data
    rater2_additional = [
        (1, 4, 1, 2, 1, 1),
        (2, 5, 2, 2, 1, 2),
        (3, 3, 5, 4, 1, 4),
        (4, 4, 3, 2, 2, 1),
        (5, 5, 1, 2, 1, 2),
        (6, 4, 2, 1, 1, 2),
        (7, 4, 2, 3, 2, 1),
        (8, 3, 2, 2, 4, 5),
        (9, 3, 5, 1, 2, 1),
        (10, 5, 2, 3, 1, 4)
    ]
    
    # Rater 3 complete 30 samples
    rater3_data = [
        (1, 5, 4, 3, 2, 3), (2, 5, 3, 4, 2, 2), (3, 4, 4, 5, 3, 3),
        (4, 4, 4, 3, 3, 1), (5, 4, 3, 3, 2, 4), (6, 5, 2, 2, 1, 2),
        (7, 5, 1, 2, 3, 4), (8, 4, 5, 3, 2, 3), (9, 5, 3, 1, 2, 4),
        (10, 5, 4, 3, 2, 3), (11, 3, 4, 4, 3, 1), (12, 2, 4, 3, 2, 2),
        (13, 5, 4, 3, 1, 3), (14, 3, 2, 3, 4, 2), (15, 2, 3, 3, 4, 2),
        (16, 4, 2, 3, 2, 3), (17, 4, 4, 2, 2, 2), (18, 4, 3, 3, 1, 2),
        (19, 2, 4, 4, 2, 3), (20, 5, 3, 2, 2, 3), (21, 2, 3, 3, 4, 5),
        (22, 3, 4, 2, 4, 3), (23, 4, 3, 2, 2, 4), (24, 4, 1, 3, 2, 2),
        (25, 5, 2, 3, 3, 3), (26, 3, 2, 4, 4, 3), (27, 4, 3, 2, 1, 3),
        (28, 5, 3, 3, 4, 5), (29, 2, 4, 3, 2, 2), (30, 4, 3, 3, 2, 4)
    ]
    
    # Rater 4 complete 30 samples
    rater4_data = [
        (1, 3, 1, 2, 1, 1), (2, 4, 2, 3, 1, 1), (3, 3, 3, 3, 1, 2),
        (4, 1, 3, 3, 2, 1), (5, 4, 4, 3, 1, 1), (6, 4, 3, 1, 1, 2),
        (7, 3, 4, 2, 1, 2), (8, 2, 1, 2, 1, 2), (9, 3, 4, 2, 3, 3),
        (10, 4, 3, 1, 2, 2), (11, 2, 3, 2, 1, 1), (12, 2, 3, 3, 1, 1),
        (13, 4, 2, 1, 2, 1), (14, 3, 2, 1, 2, 1), (15, 1, 2, 3, 3, 2),
        (16, 1, 3, 2, 1, 3), (17, 3, 2, 1, 3, 1), (18, 2, 2, 1, 1, 2),
        (19, 1, 4, 2, 2, 1), (20, 4, 2, 1, 1, 2), (21, 4, 2, 2, 1, 1),
        (22, 3, 1, 2, 1, 2), (23, 3, 2, 1, 2, 2), (24, 3, 1, 1, 2, 3),
        (25, 5, 1, 2, 3, 2), (26, 4, 2, 1, 3, 2), (27, 2, 4, 1, 2, 1),
        (28, 3, 3, 2, 4, 2), (29, 3, 2, 4, 3, 4), (30, 2, 3, 1, 1, 2)
    ]
    
    return {
        'rater1': rater1_data,
        'rater2_additional': rater2_additional,
        'rater3': rater3_data,
        'rater4': rater4_data
    }

def create_combined_dataset(similarity_data, human_ratings):
    """Combine similarity and human rating data"""
    combined_data = []
    
    # Process each rater's data
    for rater_name, ratings in human_ratings.items():
        for rating_row in ratings:
            test_id = rating_row[0]
            if test_id in similarity_data:
                ranks = ['rank_1', 'rank_10', 'rank_100', 'rank_1000', 'rank_3000']
                for i, rank in enumerate(ranks):
                    combined_data.append({
                        'rater': rater_name,
                        'test_sample': test_id,
                        'rank': rank,
                        'similarity_score': similarity_data[test_id][rank],
                        'human_rating': rating_row[i+1]
                    })
    
    return pd.DataFrame(combined_data)

def create_human_rating_bins(df):
    """Create human rating bins"""
    def assign_bin(rating):
        return f"Rating {int(rating)}"
    
    df['rating_bin'] = df['human_rating'].apply(assign_bin)
    return df

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis - x and y swapped"""
    # Now x is human_rating, y is similarity_score
    x = df['human_rating'].values
    y = df['similarity_score'].values
    
    # Basic correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    # Linear regression
    lr = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    lr.fit(x_reshaped, y)
    y_pred = lr.predict(x_reshaped)
    r2 = r2_score(y, y_pred)
    
    # Slope and intercept
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # F-test for overall model significance
    n = len(x)
    k = 1  # number of predictors
    mse_res = np.sum((y - y_pred) ** 2) / (n - k - 1)
    mse_reg = np.sum((y_pred - np.mean(y)) ** 2) / k
    f_stat = mse_reg / mse_res
    f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    
    # 95% confidence interval for regression line
    x_sorted = np.sort(x)
    x_sorted_reshaped = x_sorted.reshape(-1, 1)
    y_pred_sorted = lr.predict(x_sorted_reshaped)
    
    # Standard error calculation for confidence intervals
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = np.mean(x)
    se_pred = np.sqrt(mse * (1/n + (x_sorted - x_mean)**2 / np.sum((x - x_mean)**2)))
    t_val = stats.t.ppf(0.975, n - 2)
    ci_lower = y_pred_sorted - t_val * se_pred
    ci_upper = y_pred_sorted + t_val * se_pred
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r2': r2,
        'slope': slope,
        'intercept': intercept,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'n_samples': n,
        'x_sorted': x_sorted,
        'y_pred_sorted': y_pred_sorted,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def create_scatter_plot(df, stats_results, output_dir):
    """Create scatter plot with regression analysis - axes swapped"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot - x is now human_rating, y is similarity_score
    plt.scatter(df['human_rating'], df['similarity_score'], 
                alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Regression line
    plt.plot(stats_results['x_sorted'], stats_results['y_pred_sorted'], 
            'r-', linewidth=2, label=f'Regression line (R² = {stats_results["r2"]:.3f})')
    
    # Confidence interval
    plt.fill_between(stats_results['x_sorted'], stats_results['ci_lower'], stats_results['ci_upper'],
                    alpha=0.2, color='red', label='95% Confidence Interval')
    
    # Customize axes
    plt.xlim(0.5, 5.5)
    plt.ylim(0.6, 1.0)
    
    # Labels and title
    plt.xlabel('Human Rating', fontsize=14, fontweight='bold')
    plt.ylabel('Music Similarity Score', fontsize=14, fontweight='bold')
    plt.title('Relationship between Human Rating and Music Similarity Score', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Statistical annotations
    stats_text = f'''Statistical Analysis:
Pearson r = {stats_results["pearson_r"]:.3f} (p = {stats_results["pearson_p"]:.3e})
Spearman ρ = {stats_results["spearman_r"]:.3f} (p = {stats_results["spearman_p"]:.3e})
R² = {stats_results["r2"]:.3f}
F-statistic = {stats_results["f_stat"]:.2f} (p = {stats_results["f_pvalue"]:.3e})
Slope = {stats_results["slope"]:.3f}
N = {stats_results["n_samples"]} observations'''
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Significance indicator
    if stats_results["pearson_p"] < 0.001:
        sig_text = "*** p < 0.001"
    elif stats_results["pearson_p"] < 0.01:
        sig_text = "** p < 0.01"
    elif stats_results["pearson_p"] < 0.05:
        sig_text = "* p < 0.05"
    else:
        sig_text = "n.s."
    
    plt.text(0.98, 0.02, sig_text, transform=plt.gca().transAxes, fontsize=14, fontweight='bold',
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if sig_text != "n.s." else 'lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plot_rating_vs_similarity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_boxplot(df, output_dir):
    """Create boxplot by human rating bins with mean values"""
    # Create bins for human ratings
    df_binned = create_human_rating_bins(df.copy())
    
    # Sort bins for proper ordering (Rating 1, Rating 2, etc.)
    bin_order = sorted(df_binned['rating_bin'].unique())
    
    # Create boxplot data
    boxplot_data = []
    boxplot_labels = []
    mean_values = []
    
    for bin_name in bin_order:
        bin_data = df_binned[df_binned['rating_bin'] == bin_name]['similarity_score']
        if len(bin_data) > 0:
            boxplot_data.append(bin_data)
            boxplot_labels.append(bin_name)
            mean_values.append(bin_data.mean())
    
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    bp = plt.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    
    # Customize boxplot colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean values as red diamonds
    for i, mean_val in enumerate(mean_values):
        plt.scatter(i+1, mean_val, marker='D', color='red', s=80, zorder=10, 
                   label='Mean' if i == 0 else "")
    
    # Add mean value annotations
    for i, (bin_name, data, mean_val) in enumerate(zip(boxplot_labels, boxplot_data, mean_values)):
        # Sample size annotation at top
        plt.text(i+1, plt.ylim()[1]*0.95, f'n={len(data)}', 
                ha='center', va='top', fontsize=11, fontweight='bold')
        
        # Mean value annotation next to the diamond
        plt.text(i+1.15, mean_val, f'{mean_val:.3f}', 
                ha='left', va='center', fontsize=10, fontweight='bold', color='red')
    
    plt.xlabel('Human Rating', fontsize=14, fontweight='bold')
    plt.ylabel('Music Similarity Score', fontsize=14, fontweight='bold')
    plt.title('Music Similarity Score Distribution by Human Rating', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Add statistics summary
    stats_summary = f'''Rating Statistics:
Total observations: {len(df_binned)}
Number of rating levels: {len(boxplot_labels)}
Overall mean similarity: {df_binned["similarity_score"].mean():.3f}
Overall std similarity: {df_binned["similarity_score"].std():.3f}'''
    
    plt.text(0.02, 0.98, stats_summary, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_rating_bins.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(stats_results, df):
    """Print detailed statistical results"""
    print("=" * 80)
    print("DETAILED STATISTICAL ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\nSample Size: {stats_results['n_samples']} observations")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson correlation coefficient: r = {stats_results['pearson_r']:.4f}")
    print(f"  Pearson p-value: p = {stats_results['pearson_p']:.6e}")
    print(f"  Spearman rank correlation: ρ = {stats_results['spearman_r']:.4f}")
    print(f"  Spearman p-value: p = {stats_results['spearman_p']:.6e}")
    
    print(f"\nLinear Regression Analysis:")
    print(f"  R-squared: R² = {stats_results['r2']:.4f}")
    print(f"  Slope: β = {stats_results['slope']:.4f}")
    print(f"  Intercept: α = {stats_results['intercept']:.4f}")
    print(f"  Regression equation: Similarity Score = {stats_results['intercept']:.3f} + {stats_results['slope']:.3f} × Human Rating")
    
    print(f"\nModel Significance Test:")
    print(f"  F-statistic: F = {stats_results['f_stat']:.3f}")
    print(f"  F-test p-value: p = {stats_results['f_pvalue']:.6e}")
    
    # Print binning summary
    df_binned = create_human_rating_bins(df.copy())
    print(f"\nHuman Rating Binning Summary:")
    bin_summary = df_binned.groupby('rating_bin').agg({
        'similarity_score': ['count', 'mean', 'std']
    }).round(4)
    print(bin_summary)
    
    print(f"\nInterpretation:")
    if stats_results['pearson_p'] < 0.05:
        direction = "positive" if stats_results['pearson_r'] > 0 else "negative"
        strength = ""
        if abs(stats_results['pearson_r']) < 0.3:
            strength = "weak"
        elif abs(stats_results['pearson_r']) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        print(f"  There is a statistically significant {strength} {direction} correlation")
        print(f"  between human ratings and music similarity scores.")
        print(f"  {stats_results['r2']*100:.1f}% of the variance in similarity scores is explained by human ratings.")
    else:
        print(f"  No statistically significant correlation was found between")
        print(f"  human ratings and music similarity scores.")

def main():
    # Set paths
    csv_path = "/home/xiruij/anticipation/extracted_music_samples/metadata.csv"
    output_dir = "/home/xiruij/anticipation/analysis_results"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print("Loading similarity data...")
    similarity_data = load_similarity_data(csv_path)
    
    print("Loading human rating data...")
    human_ratings = get_human_ratings()
    
    print("Combining datasets...")
    combined_df = create_combined_dataset(similarity_data, human_ratings)
    
    print(f"Total data points: {len(combined_df)}")
    print(f"Test samples covered: {combined_df['test_sample'].nunique()}")
    print(f"Number of raters: {combined_df['rater'].nunique()}")
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    stats_results = perform_statistical_analysis(combined_df)
    
    # Print detailed results
    print_detailed_results(stats_results, combined_df)
    
    # Create visualizations as separate plots
    print("\nCreating scatter plot...")
    create_scatter_plot(combined_df, stats_results, output_dir)
    
    print("\nCreating boxplot...")
    create_boxplot(combined_df, output_dir)
    
    # Save combined data with bins
    combined_df_with_bins = create_human_rating_bins(combined_df.copy())
    combined_df_with_bins.to_csv(f'{output_dir}/combined_rating_similarity_data.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - Scatter plot: scatter_plot_rating_vs_similarity.png")
    print(f"  - Boxplot: boxplot_rating_bins.png")

if __name__ == "__main__":
    main()
    
    # Create visualizations as separate plots
    print("\nCreating scatter plot...")
    create_scatter_plot(combined_df, stats_results, output_dir)
    
    print("\nCreating boxplot...")
    create_boxplot(combined_df, output_dir)
    
    # Save combined data with bins
    combined_df_with_bins = create_similarity_bins(combined_df.copy())
    combined_df_with_bins.to_csv(f'{output_dir}/combined_similarity_rating_data.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - Scatter plot: scatter_plot_similarity_vs_rating.png")
    print(f"  - Boxplot: boxplot_similarity_bins.png")

if __name__ == "__main__":
    main()
