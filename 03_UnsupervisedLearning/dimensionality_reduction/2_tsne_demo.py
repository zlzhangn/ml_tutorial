"""
t-SNE (t-分布随机邻域嵌入) 演示
t-Distributed Stochastic Neighbor Embedding
一种非线性降维方法，特别适合高维数据的可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("t-SNE降维演示")
print("="*70)

# 1. 加载手写数字数据集
digits = load_digits()
X = digits.data  # 64个特征
y = digits.target  # 10个类别

print(f"\n原始数据形状: {X.shape}")
print(f"类别数量: {len(np.unique(y))}")

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用t-SNE降维到2维
print("\n" + "="*70)
print("执行t-SNE降维...")
print("="*70)

# perplexity: 困惑度，平衡局部和全局结构（5-50之间）
# n_iter: 最大迭代次数
# learning_rate: 学习率（10-1000）
# random_state: 随机种子
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, 
           learning_rate=200, random_state=42, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)
elapsed_time = time.time() - start_time

print(f"\nt-SNE降维完成!")
print(f"降维后数据形状: {X_tsne.shape}")
print(f"用时: {elapsed_time:.2f}秒")
print(f"KL散度: {tsne.kl_divergence_:.4f}")

# 4. 可视化t-SNE结果
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 使用不同的颜色方案
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# 左图：t-SNE结果
for digit in range(10):
    mask = y == digit
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[digit]], label=str(digit), alpha=0.7, s=30)

axes[0].set_xlabel('t-SNE维度1', fontsize=12)
axes[0].set_ylabel('t-SNE维度2', fontsize=12)
axes[0].set_title(f't-SNE降维结果\n(perplexity={tsne.perplexity}, KL散度={tsne.kl_divergence_:.2f})',
                 fontsize=13)
axes[0].legend(title='数字', ncol=2, loc='best')
axes[0].grid(True, alpha=0.3)

# 右图：PCA结果（对比）
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

for digit in range(10):
    mask = y == digit
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[digit]], label=str(digit), alpha=0.7, s=30)

axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[1].set_title(f'PCA降维结果（对比）', fontsize=13)
axes[1].legend(title='数字', ncol=2, loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'tsne_vs_pca.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 5. 测试不同的perplexity参数
print("\n" + "="*70)
print("测试不同的perplexity参数")
print("="*70)

perplexity_values = [5, 15, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, perp in enumerate(perplexity_values):
    print(f"\n执行t-SNE (perplexity={perp})...")
    start = time.time()
    
    tsne_temp = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                    learning_rate=200, random_state=42)
    X_temp = tsne_temp.fit_transform(X_scaled)
    
    elapsed = time.time() - start
    print(f"  用时: {elapsed:.2f}秒, KL散度: {tsne_temp.kl_divergence_:.4f}")
    
    # 可视化
    for digit in range(10):
        mask = y == digit
        axes[idx].scatter(X_temp[mask, 0], X_temp[mask, 1],
                         c=[colors[digit]], alpha=0.7, s=25)
    
    axes[idx].set_title(f'perplexity={perp}\nKL散度={tsne_temp.kl_divergence_:.2f}',
                       fontsize=11)
    axes[idx].set_xlabel('t-SNE维度1')
    axes[idx].set_ylabel('t-SNE维度2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('不同perplexity参数对t-SNE结果的影响', fontsize=14)
plt.tight_layout()

# 保存图像
perplexity_path = os.path.join(current_dir, 'tsne_perplexity_comparison.png')
plt.savefig(perplexity_path, dpi=300, bbox_inches='tight')
print(f"\nperplexity比较图已保存至: {perplexity_path}")

plt.show()

# 6. 测试不同的学习率
print("\n" + "="*70)
print("测试不同的学习率")
print("="*70)

learning_rates = [10, 50, 200, 1000]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    print(f"\n执行t-SNE (learning_rate={lr})...")
    start = time.time()
    
    tsne_temp = TSNE(n_components=2, perplexity=30, n_iter=1000,
                    learning_rate=lr, random_state=42)
    X_temp = tsne_temp.fit_transform(X_scaled)
    
    elapsed = time.time() - start
    print(f"  用时: {elapsed:.2f}秒, KL散度: {tsne_temp.kl_divergence_:.4f}")
    
    # 可视化
    for digit in range(10):
        mask = y == digit
        axes[idx].scatter(X_temp[mask, 0], X_temp[mask, 1],
                         c=[colors[digit]], alpha=0.7, s=25)
    
    axes[idx].set_title(f'learning_rate={lr}\nKL散度={tsne_temp.kl_divergence_:.2f}',
                       fontsize=11)
    axes[idx].set_xlabel('t-SNE维度1')
    axes[idx].set_ylabel('t-SNE维度2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('不同学习率对t-SNE结果的影响', fontsize=14)
plt.tight_layout()

# 保存图像
lr_path = os.path.join(current_dir, 'tsne_learning_rate_comparison.png')
plt.savefig(lr_path, dpi=300, bbox_inches='tight')
print(f"\n学习率比较图已保存至: {lr_path}")

plt.show()

# 7. PCA预处理 + t-SNE（加速）
print("\n" + "="*70)
print("PCA预处理 + t-SNE（加速策略）")
print("="*70)

# 先用PCA降至50维
print("步骤1: 使用PCA降维到50维...")
pca_pre = PCA(n_components=50, random_state=42)
X_pca_pre = pca_pre.fit_transform(X_scaled)
print(f"PCA后数据形状: {X_pca_pre.shape}")
print(f"保留的方差: {pca_pre.explained_variance_ratio_.sum()*100:.2f}%")

# 再用t-SNE降至2维
print("\n步骤2: 使用t-SNE降维到2维...")
start_time = time.time()
tsne_with_pca = TSNE(n_components=2, perplexity=30, n_iter=1000,
                     learning_rate=200, random_state=42)
X_tsne_with_pca = tsne_with_pca.fit_transform(X_pca_pre)
elapsed_with_pca = time.time() - start_time

print(f"PCA+t-SNE完成! 用时: {elapsed_with_pca:.2f}秒")

# 对比直接t-SNE
print("\n对比: 直接t-SNE...")
start_time = time.time()
tsne_direct = TSNE(n_components=2, perplexity=30, n_iter=1000,
                   learning_rate=200, random_state=42)
X_tsne_direct = tsne_direct.fit_transform(X_scaled)
elapsed_direct = time.time() - start_time

print(f"直接t-SNE完成! 用时: {elapsed_direct:.2f}秒")
print(f"\n加速比: {elapsed_direct/elapsed_with_pca:.2f}x")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA + t-SNE
for digit in range(10):
    mask = y == digit
    axes[0].scatter(X_tsne_with_pca[mask, 0], X_tsne_with_pca[mask, 1],
                   c=[colors[digit]], label=str(digit), alpha=0.7, s=30)

axes[0].set_title(f'PCA(50维) + t-SNE\n用时: {elapsed_with_pca:.2f}秒', fontsize=13)
axes[0].set_xlabel('t-SNE维度1', fontsize=12)
axes[0].set_ylabel('t-SNE维度2', fontsize=12)
axes[0].legend(title='数字', ncol=2)
axes[0].grid(True, alpha=0.3)

# 直接t-SNE
for digit in range(10):
    mask = y == digit
    axes[1].scatter(X_tsne_direct[mask, 0], X_tsne_direct[mask, 1],
                   c=[colors[digit]], label=str(digit), alpha=0.7, s=30)

axes[1].set_title(f'直接t-SNE\n用时: {elapsed_direct:.2f}秒', fontsize=13)
axes[1].set_xlabel('t-SNE维度1', fontsize=12)
axes[1].set_ylabel('t-SNE维度2', fontsize=12)
axes[1].legend(title='数字', ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
pca_tsne_path = os.path.join(current_dir, 'tsne_with_pca_preprocessing.png')
plt.savefig(pca_tsne_path, dpi=300, bbox_inches='tight')
print(f"\nPCA预处理对比图已保存至: {pca_tsne_path}")

plt.show()

# 8. 在鸢尾花数据集上演示
print("\n" + "="*70)
print("鸢尾花数据集t-SNE降维")
print("="*70)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names

X_iris_scaled = StandardScaler().fit_transform(X_iris)

# t-SNE降维
tsne_iris = TSNE(n_components=2, perplexity=30, n_iter=1000,
                learning_rate=200, random_state=42)
X_iris_tsne = tsne_iris.fit_transform(X_iris_scaled)

# PCA降维（对比）
pca_iris = PCA(n_components=2, random_state=42)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-SNE
colors_iris = ['red', 'green', 'blue']
for i, (target_name, color) in enumerate(zip(target_names, colors_iris)):
    mask = y_iris == i
    axes[0].scatter(X_iris_tsne[mask, 0], X_iris_tsne[mask, 1],
                   c=color, label=target_name, alpha=0.7, s=50)

axes[0].set_title('t-SNE降维', fontsize=13)
axes[0].set_xlabel('t-SNE维度1', fontsize=11)
axes[0].set_ylabel('t-SNE维度2', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PCA
for i, (target_name, color) in enumerate(zip(target_names, colors_iris)):
    mask = y_iris == i
    axes[1].scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1],
                   c=color, label=target_name, alpha=0.7, s=50)

axes[1].set_title('PCA降维', fontsize=13)
axes[1].set_xlabel(f'PC1 ({pca_iris.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca_iris.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('鸢尾花数据集: t-SNE vs PCA', fontsize=14)
plt.tight_layout()

# 保存图像
iris_path = os.path.join(current_dir, 'tsne_iris.png')
plt.savefig(iris_path, dpi=300, bbox_inches='tight')
print(f"鸢尾花数据集图已保存至: {iris_path}")

plt.show()

# 9. t-SNE总结
print("\n" + "="*70)
print("t-SNE算法总结")
print("="*70)
print("优点:")
print("  1. 能保留数据的局部结构")
print("  2. 可视化效果优秀，簇分离清晰")
print("  3. 适合探索高维数据的结构")
print("  4. 可以发现非线性关系")
print("\n缺点:")
print("  1. 计算复杂度高，不适合大数据集")
print("  2. 每次运行结果可能不同（随机性）")
print("  3. 不能用于新数据（没有transform方法）")
print("  4. 全局结构可能被扭曲")
print("  5. 对参数敏感，需要调优")
print("\n使用场景:")
print("  • 高维数据的2D/3D可视化")
print("  • 探索数据的聚类结构")
print("  • 特征学习的中间步骤")
print("  • 数据质量检查")
print("\n参数调优:")
print("  • perplexity: 5-50，数据量大用大值")
print("  • learning_rate: 10-1000，通常200效果好")
print("  • n_iter: 至少1000，复杂数据用更多")
print("  • 建议: 先用PCA降到50维以下，再用t-SNE")
print("\n注意事项:")
print("  • t-SNE只用于可视化，不用于特征工程")
print("  • 距离和簇大小在t-SNE图中没有意义")
print("  • 多运行几次，选择KL散度最小的结果")
