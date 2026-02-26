"""
Mean Shift均值漂移算法演示
Mean Shift是一种基于密度的聚类算法，特点：
1. 不需要预先指定簇的数量
2. 可以发现任意形状的簇
3. 对异常值具有一定的鲁棒性
4. 自动发现簇中心
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from itertools import cycle
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print("数据形状:", X.shape)
print("真实簇数量:", len(np.unique(y_true)))

# 2. 估计带宽（bandwidth）参数
# bandwidth是Mean Shift算法中最重要的参数，决定了搜索窗口的大小
# quantile: 用于估计带宽的分位数，范围[0,1]
# n_samples: 用于估计的样本数量
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(f"\n估计的带宽: {bandwidth:.3f}")

# 3. 使用Mean Shift进行聚类
# bandwidth: 搜索窗口的半径
# bin_seeding: 是否使用binning来加速
# cluster_all: 是否将所有点分配到簇（False时会有孤立点）
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
labels = ms.fit_predict(X)

# 4. 获取聚类结果
cluster_centers = ms.cluster_centers_  # 聚类中心
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

print(f"\n发现的簇数量: {n_clusters}")
print(f"聚类中心:\n{cluster_centers}")

# 5. 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 绘制真实标签
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis', alpha=0.6)
axes[0].set_title('真实标签', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 绘制Mean Shift聚类结果
colors = cycle('bgrcmyk')
for k, col in zip(range(n_clusters), colors):
    cluster_members = labels == k
    cluster_center = cluster_centers[k]
    
    # 绘制簇成员
    axes[1].scatter(X[cluster_members, 0], X[cluster_members, 1], 
                   c=col, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # 绘制簇中心
    axes[1].scatter(cluster_center[0], cluster_center[1], 
                   c=col, s=300, alpha=1, marker='*', 
                   edgecolors='black', linewidths=2)

axes[1].set_title(f'Mean Shift聚类结果 (bandwidth={bandwidth:.2f})', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'meanshift_result.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 6. 测试不同的带宽参数
print("\n" + "="*50)
print("测试不同的带宽参数")
print("="*50)

# 使用不同的quantile值估计带宽
quantiles = [0.1, 0.2, 0.3, 0.4]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, q in enumerate(quantiles):
    bw = estimate_bandwidth(X, quantile=q, n_samples=500)
    
    ms_temp = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True)
    labels_temp = ms_temp.fit_predict(X)
    centers_temp = ms_temp.cluster_centers_
    n_clusters_temp = len(np.unique(labels_temp))
    
    # 可视化
    colors = cycle('bgrcmykw')
    for k, col in zip(range(n_clusters_temp), colors):
        cluster_members = labels_temp == k
        axes[idx].scatter(X[cluster_members, 0], X[cluster_members, 1], 
                         c=col, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[idx].scatter(centers_temp[k, 0], centers_temp[k, 1], 
                         c=col, s=300, alpha=1, marker='*', 
                         edgecolors='black', linewidths=2)
    
    axes[idx].set_title(f'quantile={q}, bandwidth={bw:.2f}, 簇数={n_clusters_temp}', 
                       fontsize=11)
    axes[idx].set_xlabel('特征1')
    axes[idx].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
bandwidth_comparison_path = os.path.join(current_dir, 'meanshift_bandwidth_comparison.png')
plt.savefig(bandwidth_comparison_path, dpi=300, bbox_inches='tight')
print(f"带宽比较图已保存至: {bandwidth_comparison_path}")

plt.show()

# 7. 在不同形状的数据上测试Mean Shift
print("\n" + "="*50)
print("在不同形状数据上测试")
print("="*50)

from sklearn.datasets import make_moons, make_circles

datasets = [
    ('Blobs', X),
    ('Moons', make_moons(n_samples=300, noise=0.05, random_state=42)[0]),
    ('Circles', make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)[0])
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, data) in enumerate(datasets):
    # 估计带宽
    bw = estimate_bandwidth(data, quantile=0.2, n_samples=300)
    
    # 执行Mean Shift
    ms_temp = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True)
    labels_temp = ms_temp.fit_predict(data)
    centers_temp = ms_temp.cluster_centers_
    n_clusters_temp = len(np.unique(labels_temp))
    
    # 可视化
    scatter = axes[idx].scatter(data[:, 0], data[:, 1], c=labels_temp, 
                               s=50, cmap='viridis', alpha=0.6, 
                               edgecolors='black', linewidths=0.5)
    axes[idx].scatter(centers_temp[:, 0], centers_temp[:, 1], 
                     c='red', s=300, alpha=1, marker='*', 
                     edgecolors='black', linewidths=2, label='中心')
    
    axes[idx].set_title(f'{name}\n簇数={n_clusters_temp}, bandwidth={bw:.2f}', 
                       fontsize=12)
    axes[idx].set_xlabel('特征1')
    axes[idx].set_ylabel('特征2')
    axes[idx].legend()

plt.tight_layout()

# 保存图像
shapes_path = os.path.join(current_dir, 'meanshift_different_shapes.png')
plt.savefig(shapes_path, dpi=300, bbox_inches='tight')
print(f"不同形状数据测试图已保存至: {shapes_path}")

plt.show()

# 8. 添加噪声点测试鲁棒性
print("\n" + "="*50)
print("测试对噪声的鲁棒性")
print("="*50)

# 在原数据中添加噪声点
np.random.seed(42)
noise = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(50, 2))
X_with_noise = np.vstack([X, noise])

# 使用Mean Shift聚类
bandwidth_noise = estimate_bandwidth(X_with_noise, quantile=0.2, n_samples=500)
ms_noise = MeanShift(bandwidth=bandwidth_noise, bin_seeding=True, cluster_all=True)
labels_noise = ms_noise.fit_predict(X_with_noise)
centers_noise = ms_noise.cluster_centers_
n_clusters_noise = len(np.unique(labels_noise))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.6, label='原始数据')
axes[0].scatter(noise[:, 0], noise[:, 1], c='red', s=50, alpha=0.6, 
               marker='x', label='噪声点')
axes[0].set_title('添加噪声的数据', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].legend()

# Mean Shift结果
scatter = axes[1].scatter(X_with_noise[:, 0], X_with_noise[:, 1], 
                         c=labels_noise, s=50, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1].scatter(centers_noise[:, 0], centers_noise[:, 1], 
               c='red', s=300, alpha=1, marker='*', 
               edgecolors='black', linewidths=2)
axes[1].set_title(f'Mean Shift结果 (簇数={n_clusters_noise})', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
noise_path = os.path.join(current_dir, 'meanshift_noise_robustness.png')
plt.savefig(noise_path, dpi=300, bbox_inches='tight')
print(f"噪声鲁棒性测试图已保存至: {noise_path}")

plt.show()

# 9. 算法总结
print("\n" + "="*50)
print("Mean Shift算法总结")
print("="*50)
print("优点:")
print("  1. 不需要预先指定簇数量")
print("  2. 可以发现任意形状的簇")
print("  3. 只有一个参数（bandwidth）")
print("  4. 对异常值相对鲁棒")
print("\n缺点:")
print("  1. 计算复杂度高，时间复杂度O(n²)")
print("  2. bandwidth参数选择困难")
print("  3. 在高维空间表现不佳")
print("  4. 对密度变化大的数据表现不稳定")
print("\n参数调优:")
print("  - bandwidth太小: 产生过多小簇")
print("  - bandwidth太大: 簇被过度合并")
print("  - 建议: 使用estimate_bandwidth函数，调整quantile参数(0.1-0.3)")
