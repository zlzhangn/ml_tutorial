"""
决策树分类 - 心脏病预测
使用决策树对心脏病数据进行分类，并与其他模型对比
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("决策树分类 - 心脏病预测")
print("=" * 60)

# 1. 加载数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/heart_disease.csv")

try:
    data = pd.read_csv(data_path)
    print(f"\n成功加载数据集，共 {len(data)} 条记录")
except FileNotFoundError:
    print("\n错误：找不到 heart_disease.csv 文件")
    exit(1)

# 数据清洗
data.dropna(inplace=True)
print(f"清洗后剩余 {len(data)} 条记录")

# 2. 数据预处理
X = data.drop("是否患有心脏病", axis=1)
y = data["是否患有心脏病"]

print(f"\n特征数量: {X.shape[1]}")
print(f"类别分布:\n{y.value_counts()}")

# 对类别特征进行标签编码
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"\n已对 {len(categorical_columns)} 个类别特征进行编码")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化（决策树不需要，但用于与其他模型对比）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 创建基础决策树模型
print("\n" + "=" * 60)
print("基础决策树模型")
print("=" * 60)

dt_basic = DecisionTreeClassifier(random_state=42)
dt_basic.fit(X_train, y_train)

y_pred_basic = dt_basic.predict(X_test)
y_prob_basic = dt_basic.predict_proba(X_test)[:, 1]

accuracy_basic = accuracy_score(y_test, y_pred_basic)
roc_auc_basic = roc_auc_score(y_test, y_prob_basic)

print(f"\n基础模型（无限制深度）:")
print(f"准确率: {accuracy_basic:.4f}")
print(f"ROC AUC: {roc_auc_basic:.4f}")
print(f"树的深度: {dt_basic.get_depth()}")
print(f"叶子节点数: {dt_basic.get_n_leaves()}")

# 5. 使用网格搜索调优参数
print("\n" + "=" * 60)
print("参数调优（网格搜索）")
print("=" * 60)

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

print("\n开始网格搜索...")
print(f"参数组合总数: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion'])}")

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n最佳参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型
dt_tuned = grid_search.best_estimator_
y_pred_tuned = dt_tuned.predict(X_test)
y_prob_tuned = dt_tuned.predict_proba(X_test)[:, 1]

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print(f"\n调优后模型:")
print(f"准确率: {accuracy_tuned:.4f}")
print(f"ROC AUC: {roc_auc_tuned:.4f}")
print(f"树的深度: {dt_tuned.get_depth()}")
print(f"叶子节点数: {dt_tuned.get_n_leaves()}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_tuned, target_names=['无心脏病', '有心脏病']))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_tuned))

# 6. 特征重要性分析
print("\n" + "=" * 60)
print("特征重要性分析")
print("=" * 60)

feature_importance = dt_tuned.feature_importances_
feature_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': feature_importance
}).sort_values('重要性', ascending=False)

print("\n特征重要性排序:")
print(feature_df.to_string(index=False))

# 绘制特征重要性
plt.figure(figsize=(12, 6))
plt.barh(feature_df['特征'], feature_df['重要性'])
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('决策树特征重要性分析')
plt.tight_layout()
plt.savefig('decision_tree/heart_disease_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n特征重要性图已保存")

# 7. 与其他模型对比
print("\n" + "=" * 60)
print("模型对比")
print("=" * 60)

# 逻辑回归
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# 朴素贝叶斯
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
y_prob_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

# 性能对比
comparison = pd.DataFrame({
    '模型': ['决策树（基础）', '决策树（调优）', '逻辑回归', '朴素贝叶斯'],
    '准确率': [
        accuracy_score(y_test, y_pred_basic),
        accuracy_score(y_test, y_pred_tuned),
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb)
    ],
    'ROC AUC': [
        roc_auc_score(y_test, y_prob_basic),
        roc_auc_score(y_test, y_prob_tuned),
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_nb)
    ]
})

print("\n", comparison.to_string(index=False))

# 8. ROC曲线对比
plt.figure(figsize=(10, 8))

# 计算ROC曲线
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_tuned)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)

plt.plot(fpr_dt, tpr_dt, label=f'决策树 (AUC={roc_auc_tuned:.3f})', linewidth=2)
plt.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC={roc_auc_score(y_test, y_prob_lr):.3f})', linewidth=2)
plt.plot(fpr_nb, tpr_nb, label=f'朴素贝叶斯 (AUC={roc_auc_score(y_test, y_prob_nb):.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')

plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('decision_tree/roc_comparison.png', dpi=300, bbox_inches='tight')
print("\nROC曲线对比图已保存")

# 9. 可视化决策树
print("\n" + "=" * 60)
print("决策树可视化")
print("=" * 60)

plt.figure(figsize=(25, 15))
plot_tree(
    dt_tuned,
    feature_names=X.columns.tolist(),
    class_names=['无心脏病', '有心脏病'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('心脏病预测决策树', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree/heart_disease_tree.png', dpi=300, bbox_inches='tight')
print("决策树结构图已保存")

# 10. 决策规则提取
print("\n" + "=" * 60)
print("决策规则示例")
print("=" * 60)

from sklearn.tree import _tree

def tree_to_rules(tree, feature_names):
    """将决策树转换为规则"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # 左子树
            left_rule = rule + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], depth + 1, left_rule)
            
            # 右子树
            right_rule = rule + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            # 叶子节点
            class_idx = np.argmax(tree_.value[node])
            class_name = '有心脏病' if class_idx == 1 else '无心脏病'
            rules.append({
                'rule': ' AND '.join(rule),
                'prediction': class_name,
                'samples': int(tree_.n_node_samples[node])
            })
    
    recurse(0, 1, [])
    return rules

rules = tree_to_rules(dt_tuned, X.columns.tolist())

# 只显示前5条规则
print("\n前5条决策规则:")
for i, rule in enumerate(rules[:5], 1):
    print(f"\n规则 {i}:")
    print(f"  条件: {rule['rule']}")
    print(f"  预测: {rule['prediction']}")
    print(f"  样本数: {rule['samples']}")

print(f"\n总共提取了 {len(rules)} 条规则")

# 11. 决策树优缺点总结
print("\n" + "=" * 60)
print("决策树方法总结")
print("=" * 60)

print("""
决策树的优势:
✓ 易于理解和解释（白盒模型）
✓ 可视化直观
✓ 需要的数据预处理少（不需要归一化）
✓ 可以处理数值和类别数据
✓ 可以处理缺失值
✓ 可以处理多输出问题
✓ 能够自动学习特征交互

决策树的劣势:
✗ 容易过拟合（特别是深度过大时）
✗ 对训练数据的微小变化敏感
✗ 可能创建过于复杂的树
✗ 对不平衡数据集效果不佳
✗ 预测边界呈轴对齐的矩形

改进方法:
→ 剪枝（pre-pruning和post-pruning）
→ 集成方法（随机森林、梯度提升树）
→ 使用交叉验证选择参数
→ 限制树的复杂度

与其他模型的对比:
- vs 逻辑回归: 决策树能捕捉非线性关系，但可能过拟合
- vs 朴素贝叶斯: 决策树不需要特征独立假设
- vs KNN: 决策树训练慢但预测快，且模型可解释
""")

plt.show()
