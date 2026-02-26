"""
其他降维方法演示
包括：SVD, NMF, Truncated SVD, Kernel PCA, Isomap, MDS等
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import (TruncatedSVD, NMF, KernelPCA, 
                                   FactorAnalysis, FastICA)
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target
X_scaled = StandardScaler().fit_transform(X)

print("="*70)
print("其他降维方法演示")
print("="*70)
print(f"\n数据形状: {X.shape}")
print(f"类别数量: {len(np.unique(y))}")

current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. Truncated SVD (截断奇异值分解)
# ============================================================================
print("\n" + "="*70)
print("1. Truncated SVD (截断奇异值分解)")
print("="*70)
print("适用于稀疏矩阵，常用于文本数据(TF-IDF)")

start_time = time.time()
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"解释方差比: {svd.explained_variance_ratio_.sum()*100:.2f}%")

# ============================================================================
# 2. NMF (非负矩阵分解)
# ============================================================================
print("\n" + "="*70)
print("2. NMF (非负矩阵分解)")
print("="*70)
print("要求数据非负，常用于主题建模和图像处理")

# NMF需要非负数据
X_nonneg = MinMaxScaler().fit_transform(X)

start_time = time.time()
nmf = NMF(n_components=2, init='nndsvd', random_state=42, max_iter=500)
X_nmf = nmf.fit_transform(X_nonneg)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"重构误差: {nmf.reconstruction_err_:.2f}")

# ============================================================================
# 3. Kernel PCA (核主成分分析)
# ============================================================================
print("\n" + "="*70)
print("3. Kernel PCA (核主成分分析)")
print("="*70)
print("使用核技巧实现非线性降维")

start_time = time.time()
# kernel可选: 'linear', 'poly', 'rbf', 'sigmoid'
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01, random_state=42)
X_kpca = kpca.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"核函数: {kpca.kernel}")

# ============================================================================
# 4. Isomap (等距映射)
# ============================================================================
print("\n" + "="*70)
print("4. Isomap (等距映射)")
print("="*70)
print("保持测地距离的非线性降维方法")

start_time = time.time()
# n_neighbors: 邻居数量，影响局部结构
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"重构误差: {isomap.reconstruction_error():.4f}")

# ============================================================================
# 5. MDS (多维缩放)
# ============================================================================
print("\n" + "="*70)
print("5. MDS (多维缩放)")
print("="*70)
print("保持样本间距离的降维方法")

start_time = time.time()
# metric: True使用度量MDS，False使用非度量MDS
mds = MDS(n_components=2, random_state=42, max_iter=300, n_init=1)
X_mds = mds.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"压力值(Stress): {mds.stress_:.2f}")

# ============================================================================
# 6. LLE (局部线性嵌入)
# ============================================================================
print("\n" + "="*70)
print("6. LLE (局部线性嵌入)")
print("="*70)
print("假设数据局部线性的流形学习方法")

start_time = time.time()
# method: 'standard', 'modified', 'hessian', 'ltsa'
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, 
                             method='standard', random_state=42)
X_lle = lle.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")
print(f"重构误差: {lle.reconstruction_error_:.4f}")

# ============================================================================
# 7. Factor Analysis (因子分析)
# ============================================================================
print("\n" + "="*70)
print("7. Factor Analysis (因子分析)")
print("="*70)
print("假设观测变量由少数潜在因子生成")

start_time = time.time()
fa = FactorAnalysis(n_components=2, random_state=42, max_iter=500)
X_fa = fa.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")

# ============================================================================
# 8. ICA (独立成分分析)
# ============================================================================
print("\n" + "="*70)
print("8. ICA (独立成分分析)")
print("="*70)
print("寻找统计独立的成分，常用于信号分离")

start_time = time.time()
ica = FastICA(n_components=2, random_state=42, max_iter=500)
X_ica = ica.fit_transform(X_scaled)
elapsed = time.time() - start_time

print(f"用时: {elapsed:.3f}秒")

# ============================================================================
# 可视化所有方法
# ============================================================================
print("\n" + "="*70)
print("可视化所有降维方法")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.ravel()

methods = [
    ('Truncated SVD', X_svd),
    ('NMF', X_nmf),
    ('Kernel PCA', X_kpca),
    ('Isomap', X_isomap),
    ('MDS', X_mds),
    ('LLE', X_lle),
    ('Factor Analysis', X_fa),
    ('ICA', X_ica)
]

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, (method_name, X_transformed) in enumerate(methods):
    for digit in range(10):
        mask = y == digit
        axes[idx].scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                         c=[colors[digit]], alpha=0.6, s=20)
    
    axes[idx].set_title(method_name, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('维度1', fontsize=10)
    axes[idx].set_ylabel('维度2', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

# 隐藏最后一个子图
axes[8].axis('off')

plt.suptitle('各种降维方法对比（手写数字数据集）', fontsize=15, y=0.995)
plt.tight_layout()

# 保存图像
output_path = os.path.join(current_dir, 'dimensionality_reduction_methods_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n对比图已保存至: {output_path}")

plt.show()

# ============================================================================
# 在瑞士卷数据上演示流形学习
# ============================================================================
print("\n" + "="*70)
print("流形学习方法在瑞士卷数据上的表现")
print("="*70)

# 生成瑞士卷数据
X_swiss, color_swiss = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

print(f"瑞士卷数据形状: {X_swiss.shape}")

# 3D可视化原始数据
fig = plt.figure(figsize=(16, 12))

# 原始3D数据
ax = fig.add_subplot(3, 3, 1, projection='3d')
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], 
          c=color_swiss, cmap='viridis', s=10)
ax.set_title('原始3D数据（瑞士卷）', fontsize=12, fontweight='bold')
ax.view_init(elev=10, azim=60)

# 应用各种降维方法
manifold_methods = [
    ('Isomap', Isomap(n_components=2, n_neighbors=10)),
    ('LLE', LocallyLinearEmbedding(n_components=2, n_neighbors=12, method='standard', random_state=42)),
    ('MDS', MDS(n_components=2, random_state=42, max_iter=100, n_init=1)),
    ('Kernel PCA (RBF)', KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)),
    ('Standard PCA', None)  # 使用sklearn的PCA
]

for idx, (name, method) in enumerate(manifold_methods, start=2):
    ax = fig.add_subplot(3, 3, idx)
    
    if name == 'Standard PCA':
        from sklearn.decomposition import PCA
        method = PCA(n_components=2, random_state=42)
    
    start = time.time()
    X_transformed = method.fit_transform(X_swiss)
    elapsed = time.time() - start
    
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                        c=color_swiss, cmap='viridis', s=10)
    ax.set_title(f'{name}\n(用时: {elapsed:.2f}秒)', fontsize=11)
    ax.set_xlabel('维度1', fontsize=9)
    ax.set_ylabel('维度2', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('流形学习方法对比（瑞士卷数据）', fontsize=15)
plt.tight_layout()

# 保存图像
swiss_path = os.path.join(current_dir, 'manifold_learning_swiss_roll.png')
plt.savefig(swiss_path, dpi=300, bbox_inches='tight')
print(f"瑞士卷对比图已保存至: {swiss_path}")

plt.show()

# ============================================================================
# 方法总结
# ============================================================================
print("\n" + "="*70)
print("各降维方法特点总结")
print("="*70)

summary = """
1. Truncated SVD
   • 特点: 不需要中心化，适合稀疏矩阵
   • 适用: 文本数据(TF-IDF)、推荐系统
   • 复杂度: O(n·k²)

2. NMF (非负矩阵分解)
   • 特点: 要求数据非负，结果可解释性强
   • 适用: 主题建模、图像处理、推荐系统
   • 复杂度: O(n·m·k·iter)

3. Kernel PCA
   • 特点: 非线性降维，使用核技巧
   • 适用: 非线性数据、复杂模式识别
   • 复杂度: O(n³)，计算慢

4. Isomap
   • 特点: 保持测地距离，适合流形学习
   • 适用: 非线性流形数据
   • 复杂度: O(n²log(n))

5. MDS (多维缩放)
   • 特点: 保持样本间距离
   • 适用: 相似度矩阵可视化
   • 复杂度: O(n³)

6. LLE (局部线性嵌入)
   • 特点: 保持局部线性结构
   • 适用: 流形数据、图像数据
   • 复杂度: O(d·n·k²)

7. Factor Analysis
   • 特点: 假设潜在因子模型
   • 适用: 心理学、社会科学
   • 复杂度: O(n·m²·k)

8. ICA (独立成分分析)
   • 特点: 寻找统计独立成分
   • 适用: 信号分离、脑电图分析
   • 复杂度: O(n·m²·iter)

选择建议:
• 线性+快速: PCA, Truncated SVD
• 非线性+可视化: t-SNE, Isomap, LLE
• 有监督: LDA
• 稀疏数据: Truncated SVD
• 非负数据: NMF
• 流形数据: Isomap, LLE
• 信号分离: ICA
"""

print(summary)

# 性能对比
print("\n" + "="*70)
print("计算复杂度对比（从快到慢）")
print("="*70)
print("1. PCA, Truncated SVD         - O(n·m·k)")
print("2. ICA, Factor Analysis       - O(n·m²·iter)")
print("3. LLE                        - O(d·n·k²)")
print("4. Isomap                     - O(n²log(n))")
print("5. MDS, Kernel PCA            - O(n³)")
print("6. t-SNE                      - O(n²) 但常数大")
print("\n注: n=样本数, m=特征数, k=降维后维度, d=原始维度")
