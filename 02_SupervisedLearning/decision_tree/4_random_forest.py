"""
随机森林 - 决策树的集成方法
随机森林通过构建多棵决策树并集成它们的预测来提高性能和减少过拟合
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("随机森林 - 集成学习方法")
print("=" * 60)

# 1. 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"\n数据集信息:")
print(f"样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")
print(f"类别: {iris.target_names}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 3. 单棵决策树（基准模型）
print("\n" + "=" * 60)
print("单棵决策树（基准模型）")
print("=" * 60)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n准确率: {accuracy_dt:.4f}")
print(f"树的深度: {dt_model.get_depth()}")

# 交叉验证
cv_scores_dt = cross_val_score(dt_model, X_train, y_train, cv=5)
print(f"5折交叉验证准确率: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std() * 2:.4f})")

# 4. 随机森林
print("\n" + "=" * 60)
print("随机森林")
print("=" * 60)

"""
随机森林主要参数:
- n_estimators: 树的数量（默认100）
- max_depth: 每棵树的最大深度
- min_samples_split: 内部节点分裂所需的最小样本数
- min_samples_leaf: 叶子节点最小样本数
- max_features: 寻找最佳分割时考虑的特征数
  * 'sqrt': sqrt(n_features)
  * 'log2': log2(n_features)
  * None: n_features
- bootstrap: 是否使用自助采样
- oob_score: 是否使用袋外样本来估计泛化准确率
"""

rf_model = RandomForestClassifier(
    n_estimators=100,  # 100棵树
    max_depth=None,  # 不限制深度
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # 每次分裂考虑sqrt(n_features)个特征
    bootstrap=True,  # 使用自助采样
    oob_score=True,  # 计算袋外分数
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心
)

print("\n训练随机森林...")
rf_model.fit(X_train, y_train)
print("训练完成！")

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n准确率: {accuracy_rf:.4f}")
print(f"袋外分数 (OOB Score): {rf_model.oob_score_:.4f}")
print(f"树的数量: {rf_model.n_estimators}")

# 交叉验证
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"5折交叉验证准确率: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

print("\n分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_rf))

# 5. 特征重要性对比
print("\n" + "=" * 60)
print("特征重要性对比")
print("=" * 60)

dt_importance = dt_model.feature_importances_
rf_importance = rf_model.feature_importances_

print("\n单棵决策树:")
for i, importance in enumerate(dt_importance):
    print(f"{iris.feature_names[i]}: {importance:.4f}")

print("\n随机森林:")
for i, importance in enumerate(rf_importance):
    print(f"{iris.feature_names[i]}: {importance:.4f}")

# 绘制特征重要性对比
x = np.arange(len(iris.feature_names))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, dt_importance, width, label='单棵决策树')
plt.bar(x + width/2, rf_importance, width, label='随机森林')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('特征重要性对比')
plt.xticks(x, iris.feature_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('./rf_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("\n特征重要性对比图已保存")

# 6. 不同树数量的性能
print("\n" + "=" * 60)
print("不同树数量对性能的影响")
print("=" * 60)

n_estimators_range = [1, 5, 10, 20, 50, 100, 200, 300]
train_scores = []
test_scores = []
oob_scores = []

print("\n测试不同的树数量...")
for n in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_features='sqrt',
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    oob_score = rf.oob_score_
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    oob_scores.append(oob_score)
    
    print(f"  n_estimators={n:3d}: 训练={train_score:.4f}, 测试={test_score:.4f}, OOB={oob_score:.4f}")

# 绘制性能曲线
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='训练集准确率', linewidth=2)
plt.plot(n_estimators_range, test_scores, 's-', label='测试集准确率', linewidth=2)
plt.plot(n_estimators_range, oob_scores, '^-', label='OOB准确率', linewidth=2)
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('随机森林：树数量对性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('decision_tree/rf_n_estimators.png', dpi=300, bbox_inches='tight')
print("\n树数量影响图已保存")

# 7. 单棵树 vs 随机森林对比
print("\n" + "=" * 60)
print("单棵决策树 vs 随机森林 对比")
print("=" * 60)

comparison = pd.DataFrame({
    '指标': ['准确率', '交叉验证均值', '交叉验证标准差'],
    '单棵决策树': [
        f"{accuracy_dt:.4f}",
        f"{cv_scores_dt.mean():.4f}",
        f"{cv_scores_dt.std():.4f}"
    ],
    '随机森林': [
        f"{accuracy_rf:.4f}",
        f"{cv_scores_rf.mean():.4f}",
        f"{cv_scores_rf.std():.4f}"
    ]
})

print("\n", comparison.to_string(index=False))

# 8. 随机森林中各棵树的深度分布
print("\n" + "=" * 60)
print("随机森林中各棵树的深度分布")
print("=" * 60)

tree_depths = [tree.get_depth() for tree in rf_model.estimators_]
tree_leaves = [tree.get_n_leaves() for tree in rf_model.estimators_]

print(f"\n树的深度统计:")
print(f"  最小深度: {min(tree_depths)}")
print(f"  最大深度: {max(tree_depths)}")
print(f"  平均深度: {np.mean(tree_depths):.2f}")

print(f"\n叶子节点统计:")
print(f"  最小叶子数: {min(tree_leaves)}")
print(f"  最大叶子数: {max(tree_leaves)}")
print(f"  平均叶子数: {np.mean(tree_leaves):.2f}")

# 绘制深度分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(tree_depths, bins=20, edgecolor='black')
plt.xlabel('树的深度')
plt.ylabel('数量')
plt.title('随机森林中树的深度分布')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(tree_leaves, bins=20, edgecolor='black')
plt.xlabel('叶子节点数')
plt.ylabel('数量')
plt.title('随机森林中叶子节点数分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./rf_tree_distribution.png', dpi=300, bbox_inches='tight')
print("\n树深度分布图已保存")

# 9. 预测示例
print("\n" + "=" * 60)
print("预测示例")
print("=" * 60)

test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.5, 3.0, 5.5, 1.8],
    [5.7, 2.8, 4.1, 1.3],
])

# 单棵树预测
dt_predictions = dt_model.predict(test_samples)
dt_probabilities = dt_model.predict_proba(test_samples)

# 随机森林预测
rf_predictions = rf_model.predict(test_samples)
rf_probabilities = rf_model.predict_proba(test_samples)

print("\n预测对比:")
for i, sample in enumerate(test_samples):
    print(f"\n样本 {i+1}: {sample}")
    print(f"\n  单棵决策树:")
    print(f"    预测类别: {iris.target_names[dt_predictions[i]]}")
    print(f"    类别概率: {dt_probabilities[i]}")
    print(f"\n  随机森林:")
    print(f"    预测类别: {iris.target_names[rf_predictions[i]]}")
    print(f"    类别概率: {rf_probabilities[i]}")

# 10. 随机森林原理说明
print("\n" + "=" * 60)
print("随机森林原理说明")
print("=" * 60)

print("""
随机森林 (Random Forest) 原理:

1. 基本思想:
   - 集成学习方法（Bagging的变种）
   - 构建多棵决策树，每棵树独立训练
   - 最终预测通过投票（分类）或平均（回归）得出

2. 随机性来源:
   
   a) 样本随机（Bootstrap采样）:
      - 从原始训练集中有放回地随机抽取n个样本
      - 每棵树使用不同的训练子集
      - 约63.2%的样本会被选中（袋内样本）
      - 约36.8%的样本不会被选中（袋外样本，用于验证）
   
   b) 特征随机:
      - 每次分裂时只考虑随机选择的部分特征
      - 通常使用sqrt(n_features)个特征（分类）
      - 或n_features/3个特征（回归）

3. 优点:
   ✓ 准确率高，泛化能力强
   ✓ 对过拟合有很强的抵抗力
   ✓ 能处理高维数据
   ✓ 能评估特征重要性
   ✓ 对缺失值不敏感
   ✓ 可以并行训练（速度快）
   ✓ 无需特征缩放

4. 缺点:
   ✗ 模型较大，占用内存多
   ✗ 解释性不如单棵决策树
   ✗ 训练时间较长（但可并行）
   ✗ 对噪声较多的分类/回归问题可能过拟合

5. OOB (Out-of-Bag) 评估:
   - 每棵树约有36.8%的样本未被使用（袋外样本）
   - 可以用这些样本来评估模型性能
   - 无需额外的验证集
   - OOB分数通常接近交叉验证分数

6. 与单棵决策树的对比:
   决策树: 简单、易解释，但容易过拟合
   随机森林: 准确、稳定，但复杂、难解释

7. 适用场景:
   ✓ 需要高准确率的分类/回归任务
   ✓ 特征维度高、样本量大
   ✓ 需要特征重要性分析
   ✓ 数据有噪声或缺失值
""")

print(f"\n当前随机森林模型总结:")
print(f"树的数量: {rf_model.n_estimators}")
print(f"平均树深度: {np.mean(tree_depths):.2f}")
print(f"测试集准确率: {accuracy_rf:.4f}")
print(f"OOB准确率: {rf_model.oob_score_:.4f}")
print(f"最重要的特征: {iris.feature_names[np.argmax(rf_importance)]}")

plt.show()
