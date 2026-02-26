"""
支持向量机（SVM）基础分类示例
演示如何使用SVM进行简单的二分类任务
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 1. 加载数据集
# 使用sklearn自带的鸢尾花数据集，只取前两个类别进行二分类
print("=" * 50)
print("加载鸢尾花数据集...")
iris = datasets.load_iris()
# 只选择前两个类别（setosa和versicolor）
X = iris.data[iris.target != 2]  # 特征数据
y = iris.target[iris.target != 2]  # 标签数据

print(f"数据集大小: {X.shape}")
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 2. 数据预处理
# SVM对特征的尺度敏感，需要进行标准化
print("\n" + "=" * 50)
print("数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
# test_size=0.3表示30%的数据用于测试，70%用于训练
# random_state保证每次划分结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 4. 创建并训练SVM模型
print("\n" + "=" * 50)
print("训练SVM模型...")

# 创建线性核SVM分类器
# C: 正则化参数，C越大，对误分类的惩罚越大
# kernel: 核函数类型，'linear'表示线性核
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

# 创建RBF核（径向基函数）SVM分类器
# C: 正则化参数，C越大，对误分类的惩罚越大
# gamma: RBF核的系数，gamma越大，支持向量的影响范围越小
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

print("模型训练完成！")

# 5. 模型预测
print("\n" + "=" * 50)
print("模型预测...")
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

# 6. 模型评估
print("\n" + "=" * 50)
print("线性核SVM模型评估:")
print(f"准确率: {accuracy_score(y_test, y_pred_linear):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_linear, 
                          target_names=['setosa', 'versicolor']))

print("\n" + "=" * 50)
print("RBF核SVM模型评估:")
print(f"准确率: {accuracy_score(y_test, y_pred_rbf):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_rbf, 
                          target_names=['setosa', 'versicolor']))

# 7. 可视化决策边界（使用前两个特征）
print("\n" + "=" * 50)
print("生成决策边界可视化...")

def plot_decision_boundary(model, X, y, title):
    """
    绘制SVM决策边界
    
    参数:
        model: 训练好的SVM模型
        X: 特征数据（只使用前两个特征）
        y: 标签数据
        title: 图表标题
    """
    # 只使用前两个特征进行可视化
    X_2d = X[:, :2]
    
    # 创建网格点
    h = 0.02  # 步长
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 对网格点进行预测
    # 需要将网格点扩展到4个特征（用均值填充）
    X_mean = X.mean(axis=0)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_full = np.column_stack([
        grid_points,
        np.full((grid_points.shape[0], 2), X_mean[2:])
    ])
    
    Z = model.predict(grid_points_full)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                edgecolors='black', s=50)
    plt.xlabel('特征1 (Sepal length)')
    plt.ylabel('特征2 (Sepal width)')
    plt.title(title)

# 重新训练模型（使用所有数据以便可视化）
svm_linear_vis = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear_vis.fit(X_scaled, y)

svm_rbf_vis = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf_vis.fit(X_scaled, y)

# 创建子图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(svm_linear_vis, X_scaled, y, 
                      'SVM决策边界 (线性核)')

plt.subplot(1, 2, 2)
plot_decision_boundary(svm_rbf_vis, X_scaled, y, 
                      'SVM决策边界 (RBF核)')

plt.tight_layout()
plt.savefig('./svm/decision_boundary.png', dpi=300, bbox_inches='tight')
print("决策边界图已保存！")
plt.show()

print("\n" + "=" * 50)
print("程序执行完成！")
