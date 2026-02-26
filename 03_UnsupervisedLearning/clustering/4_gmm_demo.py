"""
高斯混合模型（GMM - Gaussian Mixture Model）演示
GMM是一种基于概率的聚类算法，假设数据由多个高斯分布混合而成
相比KMeans，GMM可以：
1. 识别椭圆形簇（而不仅是球形）
2. 提供概率分配（软聚类）而不是硬分类
3. 使用EM算法优化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 1.5, 0.5], 
                       random_state=42)

print("数据形状:", X.shape)

# 2. 使用GMM进行聚类
# n_components: 高斯分布的数量（簇数）
# covariance_type: 协方差类型
#   - 'full': 每个分量有独立的协方差矩阵（默认）
#   - 'tied': 所有分量共享相同的协方差矩阵
#   - 'diag': 对角协方差矩阵
#   - 'spherical': 球形协方差矩阵（类似KMeans）
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 3. 预测标签（硬聚类）
labels = gmm.predict(X)
print(f"\n聚类标签: {np.unique(labels)}")

# 4. 预测概率（软聚类）
probs = gmm.predict_proba(X)
print(f"概率形状: {probs.shape}")
print(f"第一个样本的概率分布: {probs[0]}")

# 5. 获取模型参数
means = gmm.means_  # 各高斯分布的均值（聚类中心）
covariances = gmm.covariances_  # 协方差矩阵
weights = gmm.weights_  # 各分量的权重

print(f"\n聚类中心:\n{means}")
print(f"\n权重: {weights}")
print(f"对数似然: {gmm.score(X):.2f}")

# 6. 可视化函数：绘制高斯分布椭圆
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """根据均值和协方差矩阵绘制椭圆"""
    ax = ax or plt.gca()
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # 宽度和高度是特征值的2倍标准差
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    
    # 绘制椭圆
    ellipse = Ellipse(position, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# 7. 可视化GMM结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 绘制真实标签
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis', alpha=0.6)
axes[0].set_title('真实标签', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 绘制GMM聚类结果及高斯分布椭圆
axes[1].scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
axes[1].scatter(means[:, 0], means[:, 1], c='red', s=200, alpha=0.8,
               marker='X', edgecolors='black', linewidths=2, label='聚类中心')

# 绘制每个高斯分布的椭圆（1σ, 2σ, 3σ）
colors = ['red', 'green', 'blue']
for i in range(gmm.n_components):
    for n_std in [1, 2, 3]:
        draw_ellipse(means[i], covariances[i], ax=axes[1],
                    alpha=0.2/(n_std), facecolor=colors[i], 
                    edgecolor=colors[i], linewidth=2)

axes[1].set_title('GMM聚类结果及高斯分布椭圆', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'gmm_result.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 8. 比较不同协方差类型
print("\n" + "="*50)
print("比较不同协方差类型")
print("="*50)

covariance_types = ['full', 'tied', 'diag', 'spherical']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, cov_type in enumerate(covariance_types):
    gmm_temp = GaussianMixture(n_components=3, covariance_type=cov_type, 
                               random_state=42)
    gmm_temp.fit(X)
    labels_temp = gmm_temp.predict(X)
    
    # 可视化
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels_temp, s=50, 
                     cmap='viridis', alpha=0.6)
    axes[idx].scatter(gmm_temp.means_[:, 0], gmm_temp.means_[:, 1], 
                     c='red', s=200, alpha=0.8, marker='X', 
                     edgecolors='black', linewidths=2)
    
    # 绘制椭圆
    for i in range(gmm_temp.n_components):
        if cov_type == 'tied':
            cov = gmm_temp.covariances_
        else:
            cov = gmm_temp.covariances_[i]
        
        draw_ellipse(gmm_temp.means_[i], cov, ax=axes[idx],
                    alpha=0.3, facecolor='none', 
                    edgecolor='red', linewidth=2)
    
    axes[idx].set_title(f'协方差类型: {cov_type}\nAIC={gmm_temp.aic(X):.2f}, BIC={gmm_temp.bic(X):.2f}', 
                       fontsize=12)
    axes[idx].set_xlabel('特征1')
    axes[idx].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
cov_comparison_path = os.path.join(current_dir, 'gmm_covariance_comparison.png')
plt.savefig(cov_comparison_path, dpi=300, bbox_inches='tight')
print(f"协方差比较图已保存至: {cov_comparison_path}")

plt.show()

# 9. 使用AIC和BIC选择最佳组件数量
print("\n" + "="*50)
print("使用AIC/BIC选择最佳簇数量")
print("="*50)

n_components_range = range(1, 11)
aic_scores = []
bic_scores = []

for n in n_components_range:
    gmm_temp = GaussianMixture(n_components=n, covariance_type='full', 
                               random_state=42)
    gmm_temp.fit(X)
    aic_scores.append(gmm_temp.aic(X))
    bic_scores.append(gmm_temp.bic(X))

# 绘制AIC和BIC曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_components_range, aic_scores, 'bo-', linewidth=2, markersize=8, label='AIC')
ax.plot(n_components_range, bic_scores, 'rs-', linewidth=2, markersize=8, label='BIC')
ax.set_xlabel('组件数量', fontsize=12)
ax.set_ylabel('信息准则分数', fontsize=12)
ax.set_title('AIC/BIC vs 组件数量 (值越小越好)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 标记最佳值
best_aic = n_components_range[np.argmin(aic_scores)]
best_bic = n_components_range[np.argmin(bic_scores)]
ax.axvline(best_aic, color='blue', linestyle='--', alpha=0.5, label=f'最佳AIC={best_aic}')
ax.axvline(best_bic, color='red', linestyle='--', alpha=0.5, label=f'最佳BIC={best_bic}')

plt.tight_layout()

# 保存图像
aic_bic_path = os.path.join(current_dir, 'gmm_aic_bic.png')
plt.savefig(aic_bic_path, dpi=300, bbox_inches='tight')
print(f"AIC/BIC曲线已保存至: {aic_bic_path}")
print(f"最佳AIC组件数: {best_aic}")
print(f"最佳BIC组件数: {best_bic}")

plt.show()

# 10. 生成新样本
print("\n" + "="*50)
print("从GMM生成新样本")
print("="*50)

# 从拟合的GMM中采样新数据
X_new, y_new = gmm.sample(n_samples=100)
print(f"生成的新样本形状: {X_new.shape}")

# 可视化原始数据和生成的数据
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', 
               alpha=0.6, label='原始数据')
axes[0].set_title('原始数据', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].legend()

axes[1].scatter(X_new[:, 0], X_new[:, 1], c=y_new, s=50, cmap='viridis', 
               alpha=0.6, label='生成数据')
axes[1].set_title('从GMM生成的新样本', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()

plt.tight_layout()

# 保存图像
generated_path = os.path.join(current_dir, 'gmm_generated_samples.png')
plt.savefig(generated_path, dpi=300, bbox_inches='tight')
print(f"生成样本图已保存至: {generated_path}")

plt.show()

# 11. 算法总结
print("\n" + "="*50)
print("GMM算法总结")
print("="*50)
print("优点:")
print("  1. 软聚类，提供概率分配")
print("  2. 可以识别椭圆形簇")
print("  3. 可以生成新样本")
print("  4. 基于概率模型，有理论基础")
print("\n缺点:")
print("  1. 需要预先指定簇数量")
print("  2. 对初始化敏感")
print("  3. 可能陷入局部最优")
print("  4. 计算复杂度较高")
print("\nGMM vs KMeans:")
print("  - KMeans: 硬聚类，球形簇，计算快")
print("  - GMM: 软聚类，椭圆形簇，更灵活但计算慢")
