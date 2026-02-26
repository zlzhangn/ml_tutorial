"""
PCA (主成分分析) 演示
Principal Component Analysis是最常用的线性降维方法
通过找到数据方差最大的方向，将数据投影到低维空间
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("PCA降维演示 - 鸢尾花数据集")
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

# 2. 数据标准化（PCA对数据尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"标准化后数据均值: {X_scaled.mean(axis=0)}")
print(f"标准化后数据标准差: {X_scaled.std(axis=0)}")

# 3. 使用PCA降维到2维
# n_components: 保留的主成分数量
# whiten: 是否白化（使每个成分方差为1）
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA降维后数据形状: {X_pca.shape}")

# 4. 分析PCA结果
print("\n" + "="*70)
print("PCA分析结果")
print("="*70)

# 主成分（特征向量）
components = pca.components_
print(f"\n主成分形状: {components.shape}")
print(f"\n第一主成分系数:")
for name, coef in zip(feature_names, components[0]):
    print(f"  {name:20s}: {coef:7.4f}")

print(f"\n第二主成分系数:")
for name, coef in zip(feature_names, components[1]):
    print(f"  {name:20s}: {coef:7.4f}")

# 解释方差比
explained_variance_ratio = pca.explained_variance_ratio_
print(f"\n各主成分解释方差比:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\n累积解释方差比: {explained_variance_ratio.sum():.4f} ({explained_variance_ratio.sum()*100:.2f}%)")

# 5. 可视化降维结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 绘制原始数据（选择前两个特征）
for i, target_name in enumerate(target_names):
    mask = y == i
    axes[0].scatter(X[mask, 0], X[mask, 1], label=target_name, alpha=0.6, s=50)

axes[0].set_xlabel(feature_names[0], fontsize=11)
axes[0].set_ylabel(feature_names[1], fontsize=11)
axes[0].set_title('原始数据（前两个特征）', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 绘制PCA降维后的数据
for i, target_name in enumerate(target_names):
    mask = y == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=target_name, alpha=0.6, s=50)

axes[1].set_xlabel(f'第一主成分 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'第二主成分 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=11)
axes[1].set_title('PCA降维后数据', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'pca_iris_2d.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图像已保存至: {output_path}")

plt.show()

# 6. 确定最佳主成分数量
print("\n" + "="*70)
print("确定最佳主成分数量")
print("="*70)

# 计算所有主成分
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# 累积解释方差比
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 绘制解释方差图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 碎石图（Scree Plot）
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
           pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, 'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('主成分编号', fontsize=11)
axes[0].set_ylabel('解释方差比', fontsize=11)
axes[0].set_title('碎石图 (Scree Plot)', fontsize=13)
axes[0].grid(True, alpha=0.3)

# 累积解释方差图
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
            'bo-', linewidth=2, markersize=8)
axes[1].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95%阈值')
axes[1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90%阈值')
axes[1].set_xlabel('主成分数量', fontsize=11)
axes[1].set_ylabel('累积解释方差比', fontsize=11)
axes[1].set_title('累积解释方差', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
variance_path = os.path.join(current_dir, 'pca_variance_explained.png')
plt.savefig(variance_path, dpi=300, bbox_inches='tight')
print(f"解释方差图已保存至: {variance_path}")

plt.show()

# 找到达到95%方差的主成分数量
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\n达到95%解释方差需要 {n_components_95} 个主成分")

# 7. 手写数字数据集降维可视化
print("\n" + "="*70)
print("手写数字数据集降维")
print("="*70)

# 加载手写数字数据集
digits = load_digits()
X_digits = digits.data  # 64个特征 (8x8像素)
y_digits = digits.target  # 10个类别 (0-9)

print(f"\n原始数据形状: {X_digits.shape}")
print(f"类别数量: {len(np.unique(y_digits))}")

# 标准化
X_digits_scaled = StandardScaler().fit_transform(X_digits)

# PCA降维到2维
pca_digits = PCA(n_components=2, random_state=42)
X_digits_pca = pca_digits.fit_transform(X_digits_scaled)

print(f"降维后数据形状: {X_digits_pca.shape}")
print(f"前两个主成分解释方差: {pca_digits.explained_variance_ratio_.sum()*100:.2f}%")

# 可视化
plt.figure(figsize=(10, 8))

colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = y_digits == digit
    plt.scatter(X_digits_pca[mask, 0], X_digits_pca[mask, 1],
               c=[colors[digit]], label=str(digit), alpha=0.6, s=30)

plt.xlabel(f'第一主成分 ({pca_digits.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'第二主成分 ({pca_digits.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('手写数字数据集PCA降维到2维', fontsize=14)
plt.legend(title='数字', ncol=2, loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
digits_path = os.path.join(current_dir, 'pca_digits_2d.png')
plt.savefig(digits_path, dpi=300, bbox_inches='tight')
print(f"手写数字降维图已保存至: {digits_path}")

plt.show()

# 8. 可视化主成分方向
print("\n" + "="*70)
print("可视化前几个主成分（手写数字）")
print("="*70)

# 降维到多个维度查看
pca_multi = PCA(n_components=16, random_state=42)
pca_multi.fit(X_digits_scaled)

# 可视化前16个主成分
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

for i in range(16):
    # 将主成分重塑为8x8图像
    component_image = pca_multi.components_[i].reshape(8, 8)
    
    axes[i].imshow(component_image, cmap='RdBu_r')
    axes[i].set_title(f'PC{i+1} ({pca_multi.explained_variance_ratio_[i]*100:.1f}%)',
                     fontsize=10)
    axes[i].axis('off')

plt.suptitle('手写数字数据集的前16个主成分', fontsize=14, y=0.995)
plt.tight_layout()

# 保存图像
components_path = os.path.join(current_dir, 'pca_components_visualization.png')
plt.savefig(components_path, dpi=300, bbox_inches='tight')
print(f"主成分可视化已保存至: {components_path}")

plt.show()

# 9. PCA逆变换（重构数据）
print("\n" + "="*70)
print("PCA数据重构")
print("="*70)

# 使用不同数量的主成分重构数据
n_components_list = [1, 5, 10, 20, 64]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# 显示原始图像
sample_idx = 0
axes[0, 0].imshow(X_digits[sample_idx].reshape(8, 8), cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=10)
axes[0, 0].axis('off')

# 使用不同数量的主成分重构
for idx, n_comp in enumerate(n_components_list[:-1]):
    pca_temp = PCA(n_components=n_comp, random_state=42)
    X_transformed = pca_temp.fit_transform(X_digits_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_transformed)
    
    # 逆标准化
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    axes[0, idx+1].imshow(X_reconstructed[sample_idx].reshape(8, 8), cmap='gray')
    axes[0, idx+1].set_title(f'{n_comp}个主成分\n({pca_temp.explained_variance_ratio_.sum()*100:.1f}%)',
                            fontsize=9)
    axes[0, idx+1].axis('off')

# 第二行：另一个样本
sample_idx2 = 10
axes[1, 0].imshow(X_digits[sample_idx2].reshape(8, 8), cmap='gray')
axes[1, 0].set_title('原始图像', fontsize=10)
axes[1, 0].axis('off')

for idx, n_comp in enumerate(n_components_list[:-1]):
    pca_temp = PCA(n_components=n_comp, random_state=42)
    X_transformed = pca_temp.fit_transform(X_digits_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_transformed)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    axes[1, idx+1].imshow(X_reconstructed[sample_idx2].reshape(8, 8), cmap='gray')
    axes[1, idx+1].set_title(f'{n_comp}个主成分', fontsize=9)
    axes[1, idx+1].axis('off')

plt.suptitle('使用不同数量主成分重构手写数字', fontsize=13)
plt.tight_layout()

# 保存图像
reconstruction_path = os.path.join(current_dir, 'pca_reconstruction.png')
plt.savefig(reconstruction_path, dpi=300, bbox_inches='tight')
print(f"数据重构图已保存至: {reconstruction_path}")

plt.show()

# 10. PCA总结
print("\n" + "="*70)
print("PCA算法总结")
print("="*70)
print("优点:")
print("  1. 简单高效，计算复杂度低")
print("  2. 可解释性强，主成分有明确含义")
print("  3. 去除相关性，生成正交特征")
print("  4. 可用于数据压缩和去噪")
print("\n缺点:")
print("  1. 只能捕获线性关系")
print("  2. 对数据尺度敏感，需要标准化")
print("  3. 可能丢失重要但方差小的信息")
print("  4. 主成分可能难以解释（特别是高维数据）")
print("\n使用场景:")
print("  • 特征降维和特征提取")
print("  • 数据可视化（降至2D/3D）")
print("  • 去除噪声和冗余信息")
print("  • 加速机器学习算法")
print("\n参数选择:")
print("  • n_components: 使用累积解释方差选择（通常90-95%）")
print("  • whiten: 需要后续算法对特征尺度敏感时设为True")
