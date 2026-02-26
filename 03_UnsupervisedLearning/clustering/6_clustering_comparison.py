"""
聚类算法综合比较
比较不同聚类算法在各种数据集上的表现
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, 
                             MeanShift, estimate_bandwidth)
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备不同类型的数据集
print("准备数据集...")

datasets = [
    ('Blobs\n(球形簇)', 
     make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)[0]),
    
    ('Moons\n(非凸形)', 
     make_moons(n_samples=300, noise=0.05, random_state=42)[0]),
    
    ('Circles\n(环形)', 
     make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)[0]),
    
    ('Blobs\n(不同密度)', 
     make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 1.0, 1.5], 
                random_state=42)[0]),
    
    ('Anisotropic\n(各向异性)', 
     np.dot(make_blobs(n_samples=300, centers=3, cluster_std=0.6, 
                       random_state=42)[0], 
            [[0.6, -0.6], [-0.4, 0.8]]))
]

# 2. 配置聚类算法
def get_clustering_algorithms(X):
    """为给定数据集配置所有聚类算法"""
    # 数据标准化（对某些算法很重要）
    X_scaled = StandardScaler().fit_transform(X)
    
    # 估计bandwidth（用于Mean Shift）
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=300)
    if bandwidth <= 0:
        bandwidth = 0.5  # 默认值
    
    algorithms = [
        ('KMeans', 
         KMeans(n_clusters=3, random_state=42, n_init=10)),
        
        ('GMM', 
         GaussianMixture(n_components=3, covariance_type='full', random_state=42)),
        
        ('DBSCAN', 
         DBSCAN(eps=0.3, min_samples=5)),
        
        ('层次聚类\n(Ward)', 
         AgglomerativeClustering(n_clusters=3, linkage='ward')),
        
        ('Mean Shift', 
         MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True))
    ]
    
    return algorithms, X_scaled

# 3. 执行聚类并可视化
n_datasets = len(datasets)
n_algorithms = 5

fig, axes = plt.subplots(n_datasets, n_algorithms + 1, 
                        figsize=(18, 3.5 * n_datasets))

# 确保axes是二维数组
if n_datasets == 1:
    axes = axes.reshape(1, -1)

print("\n" + "="*70)
print("开始聚类比较")
print("="*70)

for i, (dataset_name, X) in enumerate(datasets):
    print(f"\n处理数据集: {dataset_name}")
    
    # 获取算法配置
    algorithms, X_scaled = get_clustering_algorithms(X)
    
    # 绘制原始数据
    axes[i, 0].scatter(X[:, 0], X[:, 1], s=30, alpha=0.6, color='gray')
    axes[i, 0].set_title(dataset_name, fontsize=11, fontweight='bold')
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    
    # 对每个算法进行聚类
    for j, (algo_name, algorithm) in enumerate(algorithms):
        # 计时
        start_time = time.time()
        
        try:
            # 执行聚类
            if algo_name in ['DBSCAN', 'Mean Shift']:
                labels = algorithm.fit_predict(X_scaled)
            else:
                labels = algorithm.fit_predict(X_scaled)
            
            elapsed_time = time.time() - start_time
            
            # 计算簇数量（排除噪声点-1）
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # 可视化
            scatter = axes[i, j + 1].scatter(X[:, 0], X[:, 1], c=labels, 
                                            s=30, cmap='viridis', alpha=0.6,
                                            edgecolors='black', linewidths=0.3)
            
            # 添加标题（包含性能信息）
            title = f'{algo_name}\n簇:{n_clusters}'
            if n_noise > 0:
                title += f', 噪声:{n_noise}'
            title += f'\n{elapsed_time:.3f}s'
            
            axes[i, j + 1].set_title(title, fontsize=10)
            axes[i, j + 1].set_xticks([])
            axes[i, j + 1].set_yticks([])
            
            print(f"  {algo_name:15s}: 簇数={n_clusters:2d}, "
                  f"噪声={n_noise:3d}, 用时={elapsed_time:.3f}s")
            
        except Exception as e:
            # 处理算法失败的情况
            axes[i, j + 1].text(0.5, 0.5, f'失败\n{str(e)[:20]}', 
                               ha='center', va='center',
                               transform=axes[i, j + 1].transAxes,
                               fontsize=9, color='red')
            axes[i, j + 1].set_title(algo_name, fontsize=10)
            axes[i, j + 1].set_xticks([])
            axes[i, j + 1].set_yticks([])
            print(f"  {algo_name:15s}: 失败 - {str(e)}")

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'clustering_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n综合比较图已保存至: {output_path}")

plt.show()

# 4. 算法特点总结表
print("\n" + "="*70)
print("聚类算法特点总结")
print("="*70)

summary_data = [
    ["算法", "簇形状", "需预设簇数", "噪声处理", "复杂度", "适用场景"],
    ["-" * 10] * 6,
    ["KMeans", "球形", "是", "一般", "O(n)", "大数据集，球形簇"],
    ["GMM", "椭圆形", "是", "一般", "O(n)", "概率模型，软聚类"],
    ["DBSCAN", "任意形状", "否", "优秀", "O(n²)", "非凸簇，含噪声"],
    ["层次聚类", "任意形状", "否*", "一般", "O(n²)", "小数据集，理解结构"],
    ["Mean Shift", "任意形状", "否", "良好", "O(n²)", "未知簇数，密度聚类"],
]

for row in summary_data:
    print(f"{row[0]:12s} | {row[1]:10s} | {row[2]:10s} | "
          f"{row[3]:10s} | {row[4]:8s} | {row[5]:20s}")

print("\n注: 层次聚类需要设置簇数或距离阈值")

# 5. 参数调优建议
print("\n" + "="*70)
print("参数调优建议")
print("="*70)

tuning_tips = {
    "KMeans": [
        "n_clusters: 使用肘部法则或轮廓系数确定",
        "n_init: 增加以获得更稳定的结果(默认10)",
        "max_iter: 数据复杂时可增加(默认300)"
    ],
    "GMM": [
        "n_components: 使用AIC/BIC选择",
        "covariance_type: 'full'灵活但慢，'spherical'快但限制多",
        "n_init: 增加以避免局部最优(默认1)"
    ],
    "DBSCAN": [
        "eps: 使用K-distance图确定，或从数据密度估计",
        "min_samples: 维度的2倍以上，常用5-10",
        "metric: 根据数据特性选择距离度量"
    ],
    "层次聚类": [
        "n_clusters: 从树状图确定",
        "linkage: ward通常最好，complete对异常值敏感",
        "distance_threshold: 可替代n_clusters，更灵活"
    ],
    "Mean Shift": [
        "bandwidth: 使用estimate_bandwidth，调整quantile(0.1-0.3)",
        "bin_seeding: 设为True加速",
        "cluster_all: 是否将所有点分配到簇"
    ]
}

for algo, tips in tuning_tips.items():
    print(f"\n{algo}:")
    for tip in tips:
        print(f"  • {tip}")
