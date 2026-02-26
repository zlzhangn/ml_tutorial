import os
os.environ['OMP_NUM_THREADS'] = '2'

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans    # k均值聚类
from sklearn.datasets import make_blobs    # 生成聚集分布的一组点

from sklearn.metrics import silhouette_score, calinski_harabasz_score    # 轮廓系数，CH指数

plt.rcParams['font.sans-serif'] = ['SimHei']    # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2, random_state=42)

print(X.shape)

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
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_cluster)
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100, marker='o', label="簇中心")

plt.legend()
plt.show()

# 5. 打印评价指标
print("簇内平方和：", kmeans.inertia_)
print("轮廓系数: ", silhouette_score(X, y_cluster))
print("CH指数：", calinski_harabasz_score(X, y_cluster))