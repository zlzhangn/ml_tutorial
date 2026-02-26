"""
层次聚类算法演示
Hierarchical Clustering包括：
- 凝聚型（Agglomerative）：自底向上，从每个点作为一个簇开始，逐步合并
- 分裂型（Divisive）：自顶向下，从所有点在一个簇开始，逐步分裂
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)

print("数据形状:", X.shape)

# 2. 使用凝聚型层次聚类
# n_clusters: 聚类数量
# linkage: 链接方法
#   - 'ward': 最小化簇内方差（默认，效果通常最好）
#   - 'complete': 最大距离
#   - 'average': 平均距离
#   - 'single': 最小距离
linkage_methods = ['ward', 'complete', 'average', 'single']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # 执行层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hierarchical.fit_predict(X)
    
    # 可视化结果
    scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=labels, s=50, 
                               cmap='viridis', alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[idx].set_title(f'层次聚类 (linkage={method})', fontsize=12)
    axes[idx].set_xlabel('特征1')
    axes[idx].set_ylabel('特征2')
    
    # 添加聚类数量信息
    n_clusters_found = len(np.unique(labels))
    axes[idx].text(0.02, 0.98, f'簇数量: {n_clusters_found}', 
                  transform=axes[idx].transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', 
                  facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'hierarchical_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"比较图已保存至: {output_path}")

plt.show()

# 3. 绘制树状图（Dendrogram）
print("\n" + "="*50)
print("绘制层次聚类树状图")
print("="*50)

# 使用scipy计算链接矩阵
# method参数: 'ward', 'single', 'complete', 'average'
linkage_matrix = linkage(X, method='ward')

# 创建树状图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    linkage_matrix = linkage(X, method=method)
    
    # 绘制树状图
    dendrogram(linkage_matrix, ax=axes[idx], no_labels=True,
              color_threshold=0.3*max(linkage_matrix[:,2]))
    
    axes[idx].set_title(f'树状图 (linkage={method})', fontsize=12)
    axes[idx].set_xlabel('样本索引', fontsize=10)
    axes[idx].set_ylabel('距离', fontsize=10)
    axes[idx].axhline(y=0.3*max(linkage_matrix[:,2]), c='red', 
                     linestyle='--', linewidth=2, label='切割线')
    axes[idx].legend()

plt.tight_layout()

# 保存树状图
dendrogram_path = os.path.join(current_dir, 'hierarchical_dendrogram.png')
plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
print(f"树状图已保存至: {dendrogram_path}")

plt.show()

# 4. 使用距离阈值而非固定簇数量
print("\n" + "="*50)
print("使用距离阈值自动确定簇数量")
print("="*50)

distance_thresholds = [2, 3, 4, 5]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, threshold in enumerate(distance_thresholds):
    # distance_threshold: 距离阈值，超过此距离不合并
    # n_clusters: 设为None时使用distance_threshold
    hierarchical = AgglomerativeClustering(n_clusters=None, 
                                          distance_threshold=threshold,
                                          linkage='ward')
    labels = hierarchical.fit_predict(X)
    n_clusters = hierarchical.n_clusters_
    
    # 可视化
    scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=labels, s=50, 
                               cmap='tab20', alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[idx].set_title(f'距离阈值={threshold}, 簇数量={n_clusters}', fontsize=12)
    axes[idx].set_xlabel('特征1')
    axes[idx].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
threshold_path = os.path.join(current_dir, 'hierarchical_threshold.png')
plt.savefig(threshold_path, dpi=300, bbox_inches='tight')
print(f"阈值测试图已保存至: {threshold_path}")

plt.show()

# 5. 计算连接性约束的层次聚类
print("\n" + "="*50)
print("连接性约束层次聚类")
print("="*50)

from sklearn.neighbors import kneighbors_graph

# 创建连接性矩阵（只允许相邻点合并）
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 无连接性约束
hierarchical_no_conn = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_no_conn = hierarchical_no_conn.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=labels_no_conn, s=50, 
               cmap='viridis', alpha=0.6, edgecolors='black', linewidths=0.5)
axes[0].set_title('无连接性约束', fontsize=12)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 有连接性约束
hierarchical_conn = AgglomerativeClustering(n_clusters=3, linkage='ward',
                                           connectivity=connectivity)
labels_conn = hierarchical_conn.fit_predict(X)

axes[1].scatter(X[:, 0], X[:, 1], c=labels_conn, s=50, 
               cmap='viridis', alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1].set_title('有连接性约束', fontsize=12)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
connectivity_path = os.path.join(current_dir, 'hierarchical_connectivity.png')
plt.savefig(connectivity_path, dpi=300, bbox_inches='tight')
print(f"连接性约束图已保存至: {connectivity_path}")

plt.show()

# 6. 算法总结
print("\n" + "="*50)
print("层次聚类算法总结")
print("="*50)
print("优点:")
print("  1. 不需要预先指定簇的数量")
print("  2. 可以通过树状图直观理解数据结构")
print("  3. 对距离度量不敏感")
print("\n缺点:")
print("  1. 时间复杂度高 O(n²) 或 O(n³)")
print("  2. 对噪声和异常值敏感")
print("  3. 一旦合并/分裂，无法撤销")
print("\n链接方法选择:")
print("  - ward: 适用于大多数情况，倾向于产生大小相近的簇")
print("  - complete: 对异常值敏感，产生紧凑的簇")
print("  - average: 折中方案")
print("  - single: 可能产生链状簇，对噪声敏感")
