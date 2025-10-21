import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 加载相似度矩阵
similarity_path = "/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers_gen.pt"
print(f"加载相似度矩阵: {similarity_path}")
similarity_matrix = torch.load(similarity_path, map_location=torch.device('cpu'))

print(f"原始矩阵形状: {similarity_matrix.shape}")

# 提取最后一层的 mean 和 max
similarity_mean = similarity_matrix[-1, 0, :, :]  # 最后一层，mean pooling
similarity_max = similarity_matrix[-1, 1, :, :]   # 最后一层，max pooling

print(f"Mean pooling 形状: {similarity_mean.shape}")
print(f"Max pooling 形状: {similarity_max.shape}")

# 将矩阵展平为一维数组
mean_values = similarity_mean.flatten().numpy()
max_values = similarity_max.flatten().numpy()

print(f"总共有 {len(mean_values)} 个数据点")

# 创建图表（2个子图）
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# ========== 左图：原始值散点图 + 回归线 ==========
ax1 = axes[0]
# 散点图
ax1.scatter(mean_values, max_values, alpha=0.3, s=1, color='blue')

# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(mean_values, max_values)
line_x = np.array([mean_values.min(), mean_values.max()])
line_y = slope * line_x + intercept

# 绘制回归线
ax1.plot(line_x, line_y, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.4f}')

ax1.set_xlabel('Mean Pooling Similarity', fontsize=14)
ax1.set_ylabel('Max Pooling Similarity', fontsize=14)
ax1.set_title('Mean vs Max Pooling (Original Values)', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# ========== 右图：标准化后的散点图 ==========
ax2 = axes[1]
# 标准化（Z-score）
mean_normalized = (mean_values - mean_values.mean()) / mean_values.std()
max_normalized = (max_values - max_values.mean()) / max_values.std()

# 散点图
ax2.scatter(mean_normalized, max_normalized, alpha=0.3, s=1, color='green')

# 标准化后的线性回归
slope_norm, intercept_norm, r_norm, _, _ = stats.linregress(mean_normalized, max_normalized)
line_x_norm = np.array([mean_normalized.min(), mean_normalized.max()])
line_y_norm = slope_norm * line_x_norm + intercept_norm

# 绘制回归线
ax2.plot(line_x_norm, line_y_norm, 'r-', linewidth=2, label=f'y = {slope_norm:.3f}x + {intercept_norm:.3f}\nR² = {r_norm**2:.4f}')

# 添加对角线参考
lim = max(abs(mean_normalized.min()), abs(mean_normalized.max()), 
          abs(max_normalized.min()), abs(max_normalized.max()))
ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1, label='y=x')

ax2.set_xlabel('Mean Pooling (Standardized)', fontsize=14)
ax2.set_ylabel('Max Pooling (Standardized)', fontsize=14)
ax2.set_title('Mean vs Max Pooling (Standardized)', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.axis('equal')

plt.tight_layout()

output_path = "/home/xiruij/anticipation/debug_mean_vs_max.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")

# 打印统计信息
print(f"\nMean Pooling 统计:")
print(f"  最小值: {mean_values.min():.4f}")
print(f"  最大值: {mean_values.max():.4f}")
print(f"  平均值: {mean_values.mean():.4f}")
print(f"  标准差: {mean_values.std():.4f}")

print(f"\nMax Pooling 统计:")
print(f"  最小值: {max_values.min():.4f}")
print(f"  最大值: {max_values.max():.4f}")
print(f"  平均值: {max_values.mean():.4f}")
print(f"  标准差: {max_values.std():.4f}")

print(f"\n线性关系:")
print(f"  Pearson 相关系数: {r_value:.4f}")
print(f"  R² (决定系数): {r_value**2:.4f}")
print(f"  回归方程: max = {slope:.4f} * mean + {intercept:.4f}")

plt.close()

