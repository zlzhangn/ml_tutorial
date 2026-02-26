"""
DBSCAN聚类算法演示
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
基于密度的聚类算法，可以发现任意形状的聚类，并能识别噪声点
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成月牙形数据（非凸形状）
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. 数据标准化（重要：DBSCAN对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_moons)

print("数据形状:", X_scaled.shape)

# 3. 使用DBSCAN进行聚类
# eps: 邻域半径，决定两个样本是否为邻居
# min_samples: 核心点的最小邻居数量
# metric: 距离度量方式
dbscan = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X_scaled)

# 4. 分析聚类结果
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # -1表示噪声点
n_noise = list(labels).count(-1)  # 噪声点数量

print(f"\n聚类数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")
print(f"所有标签: {unique_labels}")

# 5. 识别核心样本
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
print(f"核心样本数量: {np.sum(core_samples_mask)}")

# 6. 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 绘制原始数据
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], s=50, alpha=0.6)
axes[0].set_title('原始数据（月牙形）', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 绘制DBSCAN聚类结果
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for i, label in enumerate(unique_labels):
    if label == -1:
        # 噪声点用黑色表示
        col = 'black'
        marker = 'x'
        label_text = '噪声点'
    else:
        col = colors[label % len(colors)]
        marker = 'o'
        label_text = f'簇 {label}'
    
    class_member_mask = (labels == label)
    
    # 绘制核心样本（大圆圈）
    xy = X_scaled[class_member_mask & core_samples_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=col, marker=marker, s=100, 
                   alpha=0.8, edgecolors='black', linewidths=1.5,
                   label=label_text if i < 3 else "")
    
    # 绘制边界样本（小圆圈）
    xy = X_scaled[class_member_mask & ~core_samples_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=col, marker=marker, s=30, 
                   alpha=0.5, edgecolors='black', linewidths=0.5)

axes[1].set_title(f'DBSCAN聚类结果 (eps={dbscan.eps}, min_samples={dbscan.min_samples})', 
                 fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'dbscan_moons_result.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 7. 在圆形数据上测试DBSCAN
print("\n" + "="*50)
print("测试圆形数据")
print("="*50)

X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
X_circles_scaled = scaler.fit_transform(X_circles)

# 调整参数以适应圆形数据
dbscan_circles = DBSCAN(eps=0.25, min_samples=5)
labels_circles = dbscan_circles.fit_predict(X_circles_scaled)

n_clusters_circles = len(set(labels_circles)) - (1 if -1 in labels_circles else 0)
n_noise_circles = list(labels_circles).count(-1)

print(f"聚类数量: {n_clusters_circles}")
print(f"噪声点数量: {n_noise_circles}")

# 可视化圆形数据结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_circles_scaled[:, 0], X_circles_scaled[:, 1], s=50, alpha=0.6)
axes[0].set_title('原始数据（双圆形）', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 绘制聚类结果
unique_labels_circles = set(labels_circles)
for label in unique_labels_circles:
    if label == -1:
        col = 'black'
        marker = 'x'
    else:
        col = colors[label % len(colors)]
        marker = 'o'
    
    class_member_mask = (labels_circles == label)
    xy = X_circles_scaled[class_member_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=col, marker=marker, s=50, 
                   alpha=0.6, edgecolors='black', linewidths=0.5)

axes[1].set_title(f'DBSCAN聚类结果 (eps={dbscan_circles.eps})', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')

plt.tight_layout()

# 保存图像
circles_path = os.path.join(current_dir, 'dbscan_circles_result.png')
plt.savefig(circles_path, dpi=300, bbox_inches='tight')
print(f"图像已保存至: {circles_path}")

plt.show()

# 8. 参数调优建议
print("\n" + "="*50)
print("DBSCAN参数调优建议")
print("="*50)
print("1. eps (邻域半径):")
print("   - 太小: 大多数点会被标记为噪声")
print("   - 太大: 多个簇会合并成一个")
print("   - 建议: 使用K-distance图确定")
print("\n2. min_samples (最小样本数):")
print("   - 数据维度越高，该值应越大")
print("   - 建议: 至少为数据维度的2倍")
print("   - 常用值: 5-10")
