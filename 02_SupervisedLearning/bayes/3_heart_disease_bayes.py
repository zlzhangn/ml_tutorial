"""
使用高斯朴素贝叶斯分类器预测心脏病
对比朴素贝叶斯与逻辑回归的性能
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

print("=" * 50)
print("朴素贝叶斯 vs 逻辑回归 - 心脏病预测")
print("=" * 50)

# 1. 加载数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/heart_disease.csv")

try:
    heart_data = pd.read_csv(data_path)
    print(f"\n成功加载数据集，共 {len(heart_data)} 条记录")
except FileNotFoundError:
    print("\n错误：找不到 heart_disease.csv 文件")
    print("请确保数据文件存在于 data/ 目录下")
    exit(1)

# 数据清洗
heart_data.dropna(inplace=True)
print(f"清洗后剩余 {len(heart_data)} 条记录")

# 显示数据集信息
print("\n数据集前5行:")
print(heart_data.head())

# 2. 数据预处理
# 分离特征和目标变量
X = heart_data.drop("是否患有心脏病", axis=1)
y = heart_data["是否患有心脏病"]

print(f"\n特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
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
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 特征标准化
# 对于高斯朴素贝叶斯，标准化不是必须的，但可以提高某些情况下的性能
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\n训练集大小: {x_train.shape}")
print(f"测试集大小: {x_test.shape}")

# 5. 创建并训练朴素贝叶斯模型
print("\n" + "=" * 50)
print("朴素贝叶斯模型")
print("=" * 50)

nb_model = GaussianNB()
nb_model.fit(x_train_scaled, y_train)

# 预测
y_pred_nb = nb_model.predict(x_test_scaled)
y_prob_nb = nb_model.predict_proba(x_test_scaled)[:, 1]

# 评估
accuracy_nb = accuracy_score(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)

print(f"\n准确率: {accuracy_nb:.4f}")
print(f"ROC AUC: {roc_auc_nb:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_nb, target_names=['无心脏病', '有心脏病']))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_nb))

# 交叉验证
cv_scores_nb = cross_val_score(nb_model, x_train_scaled, y_train, cv=5)
print(f"\n5折交叉验证平均准确率: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std() * 2:.4f})")

# 6. 创建并训练逻辑回归模型（用于对比）
print("\n" + "=" * 50)
print("逻辑回归模型")
print("=" * 50)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(x_train_scaled, y_train)

# 预测
y_pred_lr = lr_model.predict(x_test_scaled)
y_prob_lr = lr_model.predict_proba(x_test_scaled)[:, 1]

# 评估
accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"\n准确率: {accuracy_lr:.4f}")
print(f"ROC AUC: {roc_auc_lr:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_lr, target_names=['无心脏病', '有心脏病']))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_lr))

# 交叉验证
cv_scores_lr = cross_val_score(lr_model, x_train_scaled, y_train, cv=5)
print(f"\n5折交叉验证平均准确率: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# 7. 模型对比
print("\n" + "=" * 50)
print("模型性能对比")
print("=" * 50)

comparison = pd.DataFrame({
    '模型': ['朴素贝叶斯', '逻辑回归'],
    '准确率': [accuracy_nb, accuracy_lr],
    'ROC AUC': [roc_auc_nb, roc_auc_lr],
    '交叉验证均值': [cv_scores_nb.mean(), cv_scores_lr.mean()],
    '交叉验证标准差': [cv_scores_nb.std(), cv_scores_lr.std()]
})

print("\n", comparison.to_string(index=False))

# 8. 贝叶斯定理解释
print("\n" + "=" * 50)
print("贝叶斯定理解释")
print("=" * 50)

print("""
贝叶斯定理: P(疾病|症状) = P(症状|疾病) * P(疾病) / P(症状)

其中:
- P(疾病|症状): 后验概率 - 在观察到症状后患病的概率
- P(症状|疾病): 似然 - 患病情况下出现这些症状的概率
- P(疾病): 先验概率 - 患病的基础概率
- P(症状): 边缘概率 - 出现这些症状的总概率

朴素贝叶斯的"朴素"假设:
假设所有特征(症状)在给定类别(疾病状态)的条件下相互独立。
虽然这个假设在现实中往往不成立，但朴素贝叶斯在很多场景下仍然表现良好。

优点:
1. 训练速度快，计算效率高
2. 对小规模数据表现良好
3. 对缺失数据不敏感
4. 可以处理多分类问题

缺点:
1. 特征独立性假设在实际中很难满足
2. 对于连续特征，需要假设其分布（如高斯分布）
3. 对输入数据的准备方式比较敏感
""")

print(f"\n训练数据中的先验概率:")
print(f"无心脏病: {(y_train == 0).sum() / len(y_train):.4f}")
print(f"有心脏病: {(y_train == 1).sum() / len(y_train):.4f}")

print(f"\n模型学到的类别先验概率:")
print(f"无心脏病: {nb_model.class_prior_[0]:.4f}")
print(f"有心脏病: {nb_model.class_prior_[1]:.4f}")
