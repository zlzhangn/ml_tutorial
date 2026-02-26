"""
LDA (线性判别分析) 演示
Linear Discriminant Analysis是一种有监督的降维方法
目标是最大化类别间的分离度，同时最小化类别内的方差
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("LDA降维演示 - 鸢尾花数据集")
print("="*70)

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 4个特征
y = iris.target  # 3个类别
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\n原始数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"类别数量: {len(target_names)}")

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用LDA降维
# n_components: 降维后的维度数，最大为min(n_features, n_classes-1)
# LDA最多能降到n_classes-1维
max_components = len(target_names) - 1
print(f"\nLDA最多可降至: {max_components} 维")

# 降至2维
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

print(f"\nLDA降维后数据形状: {X_lda.shape}")

# 4. 分析LDA结果
print("\n" + "="*70)
print("LDA分析结果")
print("="*70)

# 解释方差比
explained_variance_ratio = lda.explained_variance_ratio_
print(f"\n各判别成分解释方差比:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  LD{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\n累积解释方差比: {explained_variance_ratio.sum():.4f} ({explained_variance_ratio.sum()*100:.2f}%)")

# 判别系数
print(f"\nLDA系数形状: {lda.coef_.shape}")
print(f"\n第一判别成分系数:")
for name, coef in zip(feature_names, lda.coef_[0]):
    print(f"  {name:20s}: {coef:7.4f}")

# 5. 可视化LDA vs PCA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据（前两个特征）
colors = ['red', 'green', 'blue']
for i, (target_name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    axes[0].scatter(X[mask, 0], X[mask, 1], c=color, label=target_name, 
                   alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

axes[0].set_xlabel(feature_names[0], fontsize=11)
axes[0].set_ylabel(feature_names[1], fontsize=11)
axes[0].set_title('原始数据（前两个特征）', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PCA降维
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

for i, (target_name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=target_name,
                   alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1].set_title('PCA降维（无监督）', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# LDA降维
for i, (target_name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    axes[2].scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=target_name,
                   alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

axes[2].set_xlabel(f'LD1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=11)
axes[2].set_ylabel(f'LD2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=11)
axes[2].set_title('LDA降维（有监督）', fontsize=13)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'lda_vs_pca_iris.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 6. LDA作为分类器
print("\n" + "="*70)
print("LDA作为分类器")
print("="*70)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 训练LDA分类器
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)

# 预测
y_pred = lda_clf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n分类准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. 葡萄酒数据集演示
print("\n" + "="*70)
print("葡萄酒数据集LDA降维")
print("="*70)

# 加载葡萄酒数据集
wine = load_wine()
X_wine = wine.data  # 13个特征
y_wine = wine.target  # 3个类别
wine_target_names = wine.target_names

print(f"\n原始数据形状: {X_wine.shape}")
print(f"类别: {wine_target_names}")

# 标准化
X_wine_scaled = StandardScaler().fit_transform(X_wine)

# LDA降维到2维
lda_wine = LinearDiscriminantAnalysis(n_components=2)
X_wine_lda = lda_wine.fit_transform(X_wine_scaled, y_wine)

# PCA降维到2维（对比）
pca_wine = PCA(n_components=2, random_state=42)
X_wine_pca = pca_wine.fit_transform(X_wine_scaled)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

wine_colors = ['darkred', 'gold', 'purple']

# PCA
for i, (name, color) in enumerate(zip(wine_target_names, wine_colors)):
    mask = y_wine == i
    axes[0].scatter(X_wine_pca[mask, 0], X_wine_pca[mask, 1],
                   c=color, label=name, alpha=0.7, s=50,
                   edgecolors='black', linewidths=0.5)

axes[0].set_xlabel(f'PC1 ({pca_wine.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca_wine.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0].set_title('PCA降维（无监督）', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LDA
for i, (name, color) in enumerate(zip(wine_target_names, wine_colors)):
    mask = y_wine == i
    axes[1].scatter(X_wine_lda[mask, 0], X_wine_lda[mask, 1],
                   c=color, label=name, alpha=0.7, s=50,
                   edgecolors='black', linewidths=0.5)

axes[1].set_xlabel(f'LD1 ({lda_wine.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'LD2 ({lda_wine.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1].set_title('LDA降维（有监督）', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('葡萄酒数据集: PCA vs LDA', fontsize=14)
plt.tight_layout()

# 保存图像
wine_path = os.path.join(current_dir, 'lda_wine.png')
plt.savefig(wine_path, dpi=300, bbox_inches='tight')
print(f"葡萄酒数据集图已保存至: {wine_path}")

plt.show()

# 8. 决策边界可视化
print("\n" + "="*70)
print("LDA决策边界可视化")
print("="*70)

# 使用LDA降维后的2维数据训练分类器
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_lda, y)

# 创建网格
x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 预测网格点
Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=2)

# 绘制数据点
for i, (target_name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=target_name,
               alpha=0.8, s=60, edgecolors='black', linewidths=1)

plt.xlabel(f'LD1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'LD2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=12)
plt.title('LDA决策边界', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
boundary_path = os.path.join(current_dir, 'lda_decision_boundary.png')
plt.savefig(boundary_path, dpi=300, bbox_inches='tight')
print(f"决策边界图已保存至: {boundary_path}")

plt.show()

# 9. LDA降维到1维
print("\n" + "="*70)
print("LDA降维到1维")
print("="*70)

# 降至1维
lda_1d = LinearDiscriminantAnalysis(n_components=1)
X_lda_1d = lda_1d.fit_transform(X_scaled, y)

print(f"降维后数据形状: {X_lda_1d.shape}")
print(f"解释方差比: {lda_1d.explained_variance_ratio_[0]:.4f}")

# 可视化1维投影
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i, (target_name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    
    # 在第i个子图中绘制
    axes[i].scatter(X_lda_1d[mask], np.zeros_like(X_lda_1d[mask]),
                   c=color, label=target_name, alpha=0.7, s=60,
                   edgecolors='black', linewidths=0.5)
    axes[i].set_ylabel(target_name, fontsize=11)
    axes[i].set_ylim(-0.5, 0.5)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_yticks([])
    
    if i < 2:
        axes[i].set_xticks([])

axes[2].set_xlabel('LD1', fontsize=12)
plt.suptitle('LDA降至1维的投影', fontsize=14)
plt.tight_layout()

# 保存图像
lda_1d_path = os.path.join(current_dir, 'lda_1d_projection.png')
plt.savefig(lda_1d_path, dpi=300, bbox_inches='tight')
print(f"1维投影图已保存至: {lda_1d_path}")

plt.show()

# 10. 多类别数据演示
print("\n" + "="*70)
print("多类别数据LDA降维")
print("="*70)

# 生成5类数据
X_multi, y_multi = make_classification(n_samples=500, n_features=10,
                                       n_informative=8, n_redundant=2,
                                       n_classes=5, n_clusters_per_class=1,
                                       random_state=42)

X_multi_scaled = StandardScaler().fit_transform(X_multi)

print(f"\n数据形状: {X_multi.shape}")
print(f"类别数量: {len(np.unique(y_multi))}")

# LDA降维（最多降到4维，因为5个类别）
lda_multi = LinearDiscriminantAnalysis(n_components=4)
X_multi_lda = lda_multi.fit_transform(X_multi_scaled, y_multi)

print(f"LDA降维后形状: {X_multi_lda.shape}")
print(f"\n各判别成分解释方差比:")
for i, ratio in enumerate(lda_multi.explained_variance_ratio_):
    print(f"  LD{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# 可视化前两个判别成分
plt.figure(figsize=(10, 7))

colors_multi = plt.cm.Set3(np.linspace(0, 1, 5))
for i in range(5):
    mask = y_multi == i
    plt.scatter(X_multi_lda[mask, 0], X_multi_lda[mask, 1],
               c=[colors_multi[i]], label=f'类别{i}', alpha=0.7, s=40,
               edgecolors='black', linewidths=0.3)

plt.xlabel(f'LD1 ({lda_multi.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'LD2 ({lda_multi.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('5类数据的LDA降维', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
multi_path = os.path.join(current_dir, 'lda_multiclass.png')
plt.savefig(multi_path, dpi=300, bbox_inches='tight')
print(f"\n多类别数据图已保存至: {multi_path}")

plt.show()

# 11. LDA总结
print("\n" + "="*70)
print("LDA算法总结")
print("="*70)
print("优点:")
print("  1. 有监督降维，充分利用标签信息")
print("  2. 最大化类别分离度")
print("  3. 计算高效，适合大数据集")
print("  4. 可同时用于降维和分类")
print("  5. 结果可解释性强")
print("\n缺点:")
print("  1. 假设数据服从高斯分布")
print("  2. 假设各类别协方差矩阵相同")
print("  3. 最多降至n_classes-1维")
print("  4. 只能捕获线性关系")
print("  5. 对异常值敏感")
print("\n使用场景:")
print("  • 有标签的降维任务")
print("  • 分类前的特征提取")
print("  • 类别分离可视化")
print("  • 多类分类问题")
print("\nLDA vs PCA:")
print("  PCA: 无监督，最大化方差，保留数据信息")
print("  LDA: 有监督，最大化类别分离，保留判别信息")
print("\n参数选择:")
print("  • n_components: 最大为min(n_features, n_classes-1)")
print("  • solver: 'svd'稳定但慢，'lsqr'快但可能不稳定")
print("  • shrinkage: 当样本量小时使用，防止过拟合")
