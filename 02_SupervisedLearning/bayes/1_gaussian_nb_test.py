"""
高斯朴素贝叶斯分类器示例
适用于特征服从正态分布的数据
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 1. 加载数据集 - 使用经典的鸢尾花数据集
print("=" * 50)
print("高斯朴素贝叶斯分类器 - 鸢尾花分类示例")
print("=" * 50)

iris = load_iris()
X = iris.data  # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target  # 标签：3种鸢尾花类别

print(f"\n数据集大小: {X.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"目标类别: {iris.target_names}")

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify保持类别比例
)

print(f"\n训练集大小: {x_train.shape}")
print(f"测试集大小: {x_test.shape}")

# 3. 创建高斯朴素贝叶斯模型
# 高斯朴素贝叶斯假设每个特征都服从正态分布
# 它会计算每个类别下每个特征的均值和方差
model = GaussianNB()

# 4. 训练模型
# 贝叶斯方法基于贝叶斯定理: P(类别|特征) = P(特征|类别) * P(类别) / P(特征)
# 朴素贝叶斯假设特征之间相互独立
model.fit(x_train, y_train)

print("\n模型训练完成！")
print(f"每个类别的先验概率: {model.class_prior_}")

# 5. 预测
y_pred = model.predict(x_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 7. 预测概率
# 返回样本属于每个类别的概率
y_prob = model.predict_proba(x_test[:5])
print("\n前5个测试样本的预测概率:")
for i, prob in enumerate(y_prob):
    print(f"样本 {i+1}: {prob}")
    print(f"  预测类别: {iris.target_names[y_pred[i]]}")
    print(f"  真实类别: {iris.target_names[y_test[i]]}")
