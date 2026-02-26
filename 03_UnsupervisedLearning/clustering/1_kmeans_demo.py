"""
KMeans聚类算法演示
KMeans是最常用的聚类算法，通过迭代将数据点分配到K个簇中
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
# make_blobs生成具有高斯分布的聚类数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

print("数据形状:", X.shape)
print("真实标签:", np.unique(y_true))

# 2. 使用KMeans进行聚类
# n_clusters: 聚类数量
# random_state: 随机种子，保证结果可复现
# n_init: 初始化次数，选择最佳结果
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 3. 获取聚类中心和标签
centers = kmeans.cluster_centers_  # 聚类中心
labels = kmeans.labels_  # 每个样本的聚类标签
inertia = kmeans.inertia_  # 样本到最近聚类中心的距离平方和

print(f"\n聚类中心:\n{centers}")
print(f"惯性(Inertia): {inertia:.2f}")

# 4. 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制真实标签
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis', alpha=0.6)
axes[0].set_title('真实标签', fontsize=14)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')

# 绘制KMeans聚类结果
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', alpha=0.6)
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, 
                marker='X', edgecolors='black', linewidths=2, label='聚类中心')
axes[1].set_title('KMeans聚类结果', fontsize=14)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()

plt.tight_layout()

# 保存图像（使用绝对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'kmeans_result.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 5. 肘部法则（Elbow Method）- 确定最佳K值
print("\n使用肘部法则确定最佳K值...")
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)

# 绘制肘部曲线
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('聚类数量 K', fontsize=12)
plt.ylabel('惯性(Inertia)', fontsize=12)
plt.title('肘部法则 - 确定最佳K值', fontsize=14)
plt.grid(True, alpha=0.3)

# 保存肘部曲线图
elbow_path = os.path.join(current_dir, 'kmeans_elbow.png')
plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
print(f"肘部曲线已保存至: {elbow_path}")

plt.show()

# 6. 预测新数据点
new_points = np.array([[0, 0], [4, 4]])
predictions = kmeans.predict(new_points)
print(f"\n新数据点 {new_points[0]} 属于簇: {predictions[0]}")
print(f"新数据点 {new_points[1]} 属于簇: {predictions[1]}")
