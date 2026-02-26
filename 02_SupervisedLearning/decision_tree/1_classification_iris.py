"""
决策树分类器示例 - 鸢尾花分类
决策树是一种基于树结构的监督学习算法，通过学习特征的判断规则来进行分类
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=" * 60)
print("决策树分类器 - 鸢尾花分类示例")
print("=" * 60)

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target  # 标签：3种鸢尾花类别

print(f"\n数据集信息:")
print(f"样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")
print(f"类别分布: {np.bincount(y)}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 3. 创建决策树分类器
"""
主要参数说明:
- criterion: 分裂标准
  * 'gini': 基尼不纯度（默认）
  * 'entropy': 信息增益（熵）
- max_depth: 树的最大深度，用于防止过拟合
- min_samples_split: 内部节点再划分所需最小样本数
- min_samples_leaf: 叶子节点最少样本数
- max_features: 寻找最佳分割时考虑的特征数量
"""

# 使用基尼不纯度作为分裂标准
model_gini = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,  # 限制树的深度
    min_samples_split=2,  # 节点分裂所需的最小样本数
    min_samples_leaf=1,  # 叶子节点最小样本数
    random_state=42
)

# 4. 训练模型
print("\n训练决策树模型（基尼不纯度）...")
model_gini.fit(X_train, y_train)
print("训练完成！")

# 5. 模型信息
print(f"\n决策树深度: {model_gini.get_depth()}")
print(f"叶子节点数量: {model_gini.get_n_leaves()}")

# 6. 预测
y_pred = model_gini.predict(X_test)

# 7. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 8. 特征重要性
print("\n" + "=" * 60)
print("特征重要性分析")
print("=" * 60)

feature_importance = model_gini.feature_importances_
print("\n各特征重要性:")
for i, importance in enumerate(feature_importance):
    print(f"{iris.feature_names[i]}: {importance:.4f}")

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xticks(range(len(feature_importance)), iris.feature_names, rotation=45, ha='right')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('决策树特征重要性')
plt.tight_layout()
plt.savefig('./feature_importance.png', dpi=300, bbox_inches='tight')
print("\n特征重要性图已保存到 ./feature_importance.png")

# 9. 可视化决策树
print("\n" + "=" * 60)
print("决策树可视化")
print("=" * 60)

plt.figure(figsize=(20, 10))
plot_tree(
    model_gini,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,  # 填充颜色
    rounded=True,  # 圆角
    fontsize=10
)
plt.title('决策树结构（基尼不纯度）', fontsize=16)
plt.tight_layout()
plt.savefig('./decision_tree_gini.png', dpi=300, bbox_inches='tight')
print("决策树可视化图已保存到 ./decision_tree_gini.png")

# 10. 使用信息熵作为分裂标准进行对比
print("\n" + "=" * 60)
print("对比：使用信息熵作为分裂标准")
print("=" * 60)

model_entropy = DecisionTreeClassifier(
    criterion='entropy',  # 使用信息熵
    max_depth=3,
    random_state=42
)

model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

print(f"\n信息熵模型准确率: {accuracy_entropy:.4f}")
print(f"基尼不纯度模型准确率: {accuracy:.4f}")

# 11. 预测示例
print("\n" + "=" * 60)
print("预测示例")
print("=" * 60)

# 创建一些测试样本
test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 类似 setosa
    [6.5, 3.0, 5.5, 1.8],  # 类似 virginica
    [5.7, 2.8, 4.1, 1.3],  # 类似 versicolor
])

predictions = model_gini.predict(test_samples)
probabilities = model_gini.predict_proba(test_samples)

print("\n测试样本预测结果:")
for i, sample in enumerate(test_samples):
    print(f"\n样本 {i+1}: {sample}")
    print(f"预测类别: {iris.target_names[predictions[i]]}")
    print(f"各类别概率:")
    for j, prob in enumerate(probabilities[i]):
        print(f"  {iris.target_names[j]}: {prob:.4f}")

# 12. 决策路径
print("\n" + "=" * 60)
print("决策路径示例")
print("=" * 60)

# 获取第一个测试样本的决策路径
sample = test_samples[0].reshape(1, -1)
decision_path = model_gini.decision_path(sample)

print(f"\n样本 {test_samples[0]} 的决策路径:")
print(f"经过的节点: {decision_path.toarray()[0].nonzero()[0]}")

# 13. 决策树原理说明
print("\n" + "=" * 60)
print("决策树原理说明")
print("=" * 60)

print("""
决策树是一种树形结构的分类器:

1. 节点类型:
   - 根节点: 包含全部训练样本
   - 内部节点: 表示一个特征上的判断
   - 叶子节点: 表示最终的分类结果

2. 分裂标准:
   
   a) 基尼不纯度 (Gini):
      Gini = 1 - Σ(p_i)²
      其中 p_i 是类别 i 的概率
      基尼不纯度越小，纯度越高
   
   b) 信息熵 (Entropy):
      Entropy = -Σ(p_i × log2(p_i))
      熵越小，纯度越高
   
   c) 信息增益 (Information Gain):
      IG = 父节点熵 - 子节点加权平均熵

3. 优点:
   ✓ 易于理解和解释（白盒模型）
   ✓ 可以可视化
   ✓ 需要的数据预处理较少
   ✓ 可以处理数值和类别数据
   ✓ 可以处理多输出问题

4. 缺点:
   ✗ 容易过拟合（需要剪枝）
   ✗ 对数据变化敏感
   ✗ 可能创建有偏的树
   ✗ 对于不平衡数据集效果不佳

5. 防止过拟合的方法:
   - 限制树的深度 (max_depth)
   - 设置叶子节点最小样本数 (min_samples_leaf)
   - 设置分裂所需最小样本数 (min_samples_split)
   - 剪枝 (pruning)
""")

plt.show()
