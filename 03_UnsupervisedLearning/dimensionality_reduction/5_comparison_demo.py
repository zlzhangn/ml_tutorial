"""
降维方法综合比较
在多个数据集上比较不同降维方法的效果和性能
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.datasets import (load_digits, load_iris, make_classification,
                              make_moons, make_circles, make_swiss_roll)
from sklearn.preprocessing import StandardScaler
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("降维方法综合比较")
print("="*70)

current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 准备多个数据集
# ============================================================================
print("\n准备数据集...")

datasets = []

# 1. 鸢尾花数据集
iris = load_iris()
datasets.append(('鸢尾花\n(4D→2D)', iris.data, iris.target))

# 2. 手写数字数据集（子集）
digits = load_digits()
# 使用前500个样本加速
datasets.append(('手写数字\n(64D→2D)', digits.data[:500], digits.target[:500]))

# 3. 高维分类数据
X_class, y_class = make_classification(n_samples=300, n_features=20,
                                      n_informative=15, n_redundant=5,
                                      n_classes=3, n_clusters_per_class=1,
                                      random_state=42)
datasets.append(('高维分类\n(20D→2D)', X_class, y_class))

# 4. 月牙数据
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
datasets.append(('月牙形状\n(2D→2D)', X_moons, y_moons))

# 5. 圆环数据
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                   factor=0.5, random_state=42)
datasets.append(('圆环形状\n(2D→2D)', X_circles, y_circles))

print(f"共准备 {len(datasets)} 个数据集")

# ============================================================================
# 配置降维方法
# ============================================================================
def get_methods(n_classes):
    """获取所有降维方法"""
    methods = [
        ('PCA', PCA(n_components=2, random_state=42)),
        ('LDA', LinearDiscriminantAnalysis(n_components=min(2, n_classes-1))),
        ('t-SNE', TSNE(n_components=2, perplexity=30, n_iter=300, 
                      learning_rate=200, random_state=42, verbose=0)),
        ('Isomap', Isomap(n_components=2, n_neighbors=10))
    ]
    return methods

# ============================================================================
# 执行比较
# ============================================================================
print("\n" + "="*70)
print("执行降维并比较...")
print("="*70)

n_datasets = len(datasets)
n_methods = 4

fig, axes = plt.subplots(n_datasets, n_methods + 1, 
                        figsize=(18, 3.5 * n_datasets))

# 确保axes是二维数组
if n_datasets == 1:
    axes = axes.reshape(1, -1)

for row, (dataset_name, X, y) in enumerate(datasets):
    print(f"\n处理数据集: {dataset_name.replace(chr(10), ' ')}")
    
    # 标准化数据
    X_scaled = StandardScaler().fit_transform(X)
    n_classes = len(np.unique(y))
    
    # 绘制原始数据（前两个特征）
    if X.shape[1] >= 2:
        scatter = axes[row, 0].scatter(X[:, 0], X[:, 1], c=y, 
                                      cmap='viridis', s=20, alpha=0.6)
        axes[row, 0].set_title(f'{dataset_name}\n原始数据', fontsize=11)
    else:
        axes[row, 0].text(0.5, 0.5, '数据维度<2', ha='center', va='center',
                         transform=axes[row, 0].transAxes)
        axes[row, 0].set_title(dataset_name, fontsize=11)
    
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    
    # 应用各种降维方法
    methods = get_methods(n_classes)
    
    for col, (method_name, method) in enumerate(methods, start=1):
        try:
            # 计时
            start_time = time.time()
            
            if method_name == 'LDA':
                X_transformed = method.fit_transform(X_scaled, y)
            else:
                X_transformed = method.fit_transform(X_scaled)
            
            elapsed = time.time() - start_time
            
            # 可视化
            scatter = axes[row, col].scatter(X_transformed[:, 0], 
                                           X_transformed[:, 1],
                                           c=y, cmap='viridis', 
                                           s=20, alpha=0.6)
            
            axes[row, col].set_title(f'{method_name}\n{elapsed:.3f}秒', 
                                    fontsize=11)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
            print(f"  {method_name:10s}: {elapsed:.3f}秒")
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'失败\n{str(e)[:30]}', 
                              ha='center', va='center',
                              transform=axes[row, col].transAxes,
                              fontsize=9, color='red')
            axes[row, col].set_title(method_name, fontsize=11)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            print(f"  {method_name:10s}: 失败")

plt.suptitle('降维方法在不同数据集上的表现', fontsize=16, y=0.998)
plt.tight_layout()

# 保存图像
output_path = os.path.join(current_dir, 'dimensionality_reduction_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n综合比较图已保存至: {output_path}")

plt.show()

# ============================================================================
# 性能评估
# ============================================================================
print("\n" + "="*70)
print("性能评估 - 计算时间比较")
print("="*70)

# 在手写数字数据集上比较计算时间
X_perf = digits.data
y_perf = digits.target
X_perf_scaled = StandardScaler().fit_transform(X_perf)

methods_perf = [
    ('PCA', PCA(n_components=2, random_state=42)),
    ('LDA', LinearDiscriminantAnalysis(n_components=2)),
    ('t-SNE\n(n_iter=300)', TSNE(n_components=2, perplexity=30, n_iter=300,
                                  learning_rate=200, random_state=42, verbose=0)),
    ('Isomap', Isomap(n_components=2, n_neighbors=10))
]

times = []
method_names = []

for name, method in methods_perf:
    print(f"\n测试 {name}...")
    start = time.time()
    
    if 'LDA' in name:
        method.fit_transform(X_perf_scaled, y_perf)
    else:
        method.fit_transform(X_perf_scaled)
    
    elapsed = time.time() - start
    times.append(elapsed)
    method_names.append(name)
    print(f"  用时: {elapsed:.3f}秒")

# 可视化性能对比
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(method_names, times, color=['steelblue', 'orange', 'red', 'green'])
ax.set_xlabel('时间（秒）', fontsize=12)
ax.set_title(f'降维方法计算时间对比\n（手写数字数据集，{X_perf.shape[0]}样本×{X_perf.shape[1]}特征）',
            fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# 在每个条形上标注时间
for i, (bar, t) in enumerate(zip(bars, times)):
    ax.text(t + 0.1, i, f'{t:.3f}s', va='center', fontsize=11)

plt.tight_layout()

# 保存图像
perf_path = os.path.join(current_dir, 'dimensionality_reduction_performance.png')
plt.savefig(perf_path, dpi=300, bbox_inches='tight')
print(f"\n性能对比图已保存至: {perf_path}")

plt.show()

# ============================================================================
# 降维效果评估 - 保留信息量
# ============================================================================
print("\n" + "="*70)
print("降维效果评估 - 信息保留")
print("="*70)

# 使用PCA评估不同主成分数量的信息保留
X_eval = digits.data
X_eval_scaled = StandardScaler().fit_transform(X_eval)

# 计算所有主成分
pca_full = PCA(random_state=42)
pca_full.fit(X_eval_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 个体解释方差
axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, 'bo-', linewidth=2, markersize=6)
axes[0].set_xlabel('主成分编号', fontsize=12)
axes[0].set_ylabel('解释方差比', fontsize=12)
axes[0].set_title('各主成分解释方差', fontsize=13)
axes[0].grid(True, alpha=0.3)

# 累积解释方差
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
            'ro-', linewidth=2, markersize=6)
axes[1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95%')
axes[1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90%')
axes[1].axhline(y=0.80, color='blue', linestyle='--', linewidth=2, label='80%')
axes[1].set_xlabel('主成分数量', fontsize=12)
axes[1].set_ylabel('累积解释方差比', fontsize=12)
axes[1].set_title('累积解释方差', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 标注关键点
for threshold in [0.8, 0.9, 0.95]:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    axes[1].annotate(f'{n_comp}个\n({threshold*100:.0f}%)',
                    xy=(n_comp, threshold),
                    xytext=(n_comp + 5, threshold - 0.05),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='black'))

plt.suptitle('PCA信息保留分析（手写数字数据集）', fontsize=14)
plt.tight_layout()

# 保存图像
variance_path = os.path.join(current_dir, 'dimensionality_reduction_variance.png')
plt.savefig(variance_path, dpi=300, bbox_inches='tight')
print(f"信息保留分析图已保存至: {variance_path}")

plt.show()

# 打印关键信息
print("\n信息保留分析:")
for threshold in [0.8, 0.9, 0.95, 0.99]:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  达到{threshold*100:.0f}%信息需要 {n_comp} 个主成分 "
          f"(从{X_eval.shape[1]}维降至{n_comp}维)")

# ============================================================================
# 方法选择决策树
# ============================================================================
print("\n" + "="*70)
print("降维方法选择指南")
print("="*70)

guide = """
┌─────────────────────────────────────────────────────────────┐
│                    降维方法选择决策树                         │
└─────────────────────────────────────────────────────────────┘

1. 有标签数据？
   ├─ 是 → 使用 LDA (线性判别分析)
   │       • 最大化类别分离
   │       • 适合分类前的降维
   │
   └─ 否 → 继续...

2. 数据是否线性？
   ├─ 是 → 使用 PCA
   │       • 快速高效
   │       • 保留最大方差
   │
   └─ 否 → 继续...

3. 主要目的是什么？
   ├─ 可视化 → 使用 t-SNE
   │           • 效果最好
   │           • 但计算慢
   │
   ├─ 保留距离 → 使用 Isomap 或 MDS
   │             • 保持测地距离
   │             • 适合流形数据
   │
   ├─ 稀疏数据 → 使用 Truncated SVD
   │             • 不需要中心化
   │             • 适合文本数据
   │
   ├─ 非负数据 → 使用 NMF
   │             • 结果可解释
   │             • 适合主题建模
   │
   └─ 信号分离 → 使用 ICA
               • 统计独立
               • 适合混合信号

4. 数据量大小？
   ├─ 大数据 (>10000) → PCA, Truncated SVD
   ├─ 中等 (1000-10000) → PCA, LDA, Isomap
   └─ 小数据 (<1000) → 任何方法

5. 特殊考虑：
   • 需要逆变换？ → PCA
   • 需要新数据降维？ → PCA, LDA (避免t-SNE)
   • 计算资源有限？ → PCA, Truncated SVD
   • 追求最佳可视化？ → t-SNE
"""

print(guide)

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("总结")
print("="*70)

summary = """
降维方法对比总结:

1. PCA (主成分分析)
   优势: 快速、稳定、可解释
   劣势: 仅线性、对尺度敏感
   推荐: 大多数场景的首选

2. LDA (线性判别分析)
   优势: 监督学习、类别分离好
   劣势: 需要标签、假设高斯分布
   推荐: 分类问题的特征提取

3. t-SNE
   优势: 可视化效果最好、非线性
   劣势: 计算慢、不能变换新数据
   推荐: 数据探索和可视化

4. Isomap
   优势: 保持测地距离、非线性
   劣势: 对噪声敏感、计算较慢
   推荐: 流形数据的降维

实践建议:
• 先尝试PCA，如果效果不好再考虑其他方法
• 可视化用t-SNE，特征工程用PCA/LDA
• 大数据集避免使用t-SNE和Isomap
• 有标签数据优先考虑LDA
• 多尝试几种方法，比较效果
"""

print(summary)
