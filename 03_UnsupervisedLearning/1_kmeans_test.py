import os
os.environ['OMP_NUM_THREADS'] = '2'

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans    # k均值聚类
from sklearn.datasets import make_blobs    # 生成聚集分布的一组点

plt.rcParams['font.sans-serif'] = ['SimHei']    # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
# 300个数据点，3个簇（每个簇中心点自动生成），标准差越大数据越分散，y_ture是标签对应数据所属簇
# 默认是二维数据，可以通过n_features修改
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2, random_state=42)

print(X.shape)

# 先画出散点图
fig, ax =  plt.subplots(2, figsize=(10, 10))
ax[0].scatter(X[:, 0], X[:, 1], c="gray", s=50, label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()

# 2. 创建模型并训练
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 获取簇中心点
centers = kmeans.cluster_centers_

# 3. 预测：每个样本点的 簇标签
y_cluster = kmeans.predict(X)

# print(y_cluster)
# print(centers)

# 4. 画出聚类的散点图
ax[1].scatter(X[:, 0], X[:, 1], s=50, c=y_cluster) # s是点大小，c是颜色，alpha是透明的，marker是点的形状
ax[1].scatter(centers[:, 0], centers[:, 1], c="red", s=100, marker='o', label="簇中心")

ax[1].set_title("K-means聚类结果（K=3）")
ax[1].legend()
plt.show()