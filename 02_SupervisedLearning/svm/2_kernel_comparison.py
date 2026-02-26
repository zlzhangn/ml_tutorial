"""
SVM核函数对比
演示不同核函数在相同数据集上的表现
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=" * 60)
print("SVM核函数对比实验")
print("=" * 60)

# 1. 生成非线性可分的数据集
# 使用make_circles生成同心圆数据，这是一个典型的非线性问题
print("\n生成非线性数据集...")
X, y = datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

print(f"数据集大小: {X.shape}")
print(f"正类样本数: {np.sum(y == 1)}")
print(f"负类样本数: {np.sum(y == 0)}")

# 2. 数据标准化
print("\n数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 4. 定义不同的核函数和参数
kernels = {
    'linear': {'kernel': 'linear', 'C': 1.0},
    'poly-2': {'kernel': 'poly', 'degree': 2, 'C': 1.0},
    'poly-3': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
    'rbf': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'sigmoid': {'kernel': 'sigmoid', 'C': 1.0}
}

# 5. 训练和评估不同核函数的模型
print("\n" + "=" * 60)
print("训练不同核函数的SVM模型...")
print("=" * 60)

models = {}
results = {}

for name, params in kernels.items():
    print(f"\n训练 {name} 核函数...")
    
    # 创建并训练模型
    model = SVC(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 交叉验证（5折）
    # 交叉验证可以更全面地评估模型性能
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 保存模型和结果
    models[name] = model
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'support_vectors': model.n_support_
    }
    
    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  支持向量数量: {model.n_support_}")

# 6. 结果对比
print("\n" + "=" * 60)
print("核函数性能对比汇总:")
print("=" * 60)
print(f"{'核函数':<15} {'测试准确率':<12} {'交叉验证准确率':<20} {'支持向量数':<12}")
print("-" * 60)
for name, result in results.items():
    print(f"{name:<15} {result['accuracy']:<12.4f} "
          f"{result['cv_mean']:.4f}(±{result['cv_std']:.4f})       "
          f"{result['support_vectors']}")

# 7. 可视化决策边界
print("\n生成决策边界可视化...")

def plot_decision_boundary_2d(model, X, y, title):
    """
    绘制2D数据的SVM决策边界
    
    参数:
        model: 训练好的SVM模型
        X: 特征数据（2维）
        y: 标签数据
        title: 图表标题
    """
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', s=30)
    
    # 标记支持向量
    plt.scatter(model.support_vectors_[:, 0], 
               model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', 
               edgecolors='green', label='支持向量')
    
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'{title}\n准确率: {results[title.split()[0]]["accuracy"]:.4f}')
    plt.legend()

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('不同核函数的SVM决策边界对比', fontsize=16)

# 为每个模型绘制决策边界
for idx, (name, model) in enumerate(models.items()):
    row = idx // 3
    col = idx % 3
    plt.subplot(2, 3, idx + 1)
    plot_decision_boundary_2d(model, X_scaled, y, name)

# 删除多余的子图
if len(models) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('c:/MyWorkSpace/ML/06 ML/ml_tutorial/02_SupervisedLearning/svm/kernel_comparison.png', 
            dpi=300, bbox_inches='tight')
print("核函数对比图已保存！")
plt.show()

# 8. 混淆矩阵可视化（选择最佳模型）
print("\n生成最佳模型的混淆矩阵...")
best_kernel = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = models[best_kernel]
y_pred_best = best_model.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['类别 0', '类别 1'],
            yticklabels=['类别 0', '类别 1'])
plt.title(f'混淆矩阵 - {best_kernel} 核函数（最佳模型）')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('c:/MyWorkSpace/ML/06 ML/ml_tutorial/02_SupervisedLearning/svm/confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
print("混淆矩阵已保存！")
plt.show()

print("\n" + "=" * 60)
print(f"最佳核函数: {best_kernel}")
print(f"最佳准确率: {results[best_kernel]['accuracy']:.4f}")
print("=" * 60)
print("\n程序执行完成！")
