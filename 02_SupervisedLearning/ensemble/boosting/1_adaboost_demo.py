# -*- coding: utf-8 -*-
"""
AdaBoost 演示脚本

AdaBoost (Adaptive Boosting) 是一种基本的 Boosting 算法，它通过迭代地训练弱学习器，
并根据错误分类的样本调整权重来改进模型性能。

特点：
- 重点关注被错误分类的样本
- 逐步降低强分类器的训练误差
- 对异常值较为敏感
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录（如果不存在）
image_dir = Path(__file__).parent / 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def demo_adaboost_breast_cancer():
    """
    演示 AdaBoost 在乳腺癌数据集上的分类效果
    """
    print("\n" + "="*60)
    print("AdaBoost 演示：乳腺癌数据集分类")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建 AdaBoost 分类器（使用决策树作为基础分类器）
    # n_estimators: 弱学习器的数量
    # learning_rate: 学习率，控制每个弱学习器对最终结果的贡献
    # random_state: 随机种子，保证可重现性
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),  # 使用决策树桩（深度为1）作为弱学习器
        n_estimators=50,  # 训练 50 个弱学习器
        learning_rate=1.0,  # 学习率
        random_state=42
    )
    
    # 训练模型
    print("\n正在训练 AdaBoost 模型...")
    ada_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = ada_clf.predict(X_train)
    y_test_pred = ada_clf.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n模型性能：")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 打印详细的分类报告
    print(f"\n测试集分类报告：")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title('AdaBoost 混淆矩阵 - 乳腺癌数据集')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_adaboost_confusion_matrix.png'), dpi=150)
    print(f"\n混淆矩阵已保存至: ./images/1_adaboost_confusion_matrix.png")
    plt.close()
    
    # 绘制特征重要性
    feature_importance = ada_clf.feature_importances_
    indices = np.argsort(feature_importance)[-10:]  # 取重要性最高的 10 个特征
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importance[indices], color='steelblue')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('AdaBoost 特征重要性排名（Top 10）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_adaboost_feature_importance.png'), dpi=150)
    print(f"特征重要性已保存至: ./images/1_adaboost_feature_importance.png")
    plt.close()


def compare_n_estimators():
    """
    比较不同数量弱学习器对 AdaBoost 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同弱学习器数量对模型性能的影响")
    print("="*60)
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 仅使用二分类问题（类别 0 和 1）
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试不同的 n_estimators
    n_estimators_list = np.arange(1, 51, 5)
    train_scores = []
    test_scores = []
    
    print(f"\n正在测试不同数量的弱学习器 (1 到 50)...")
    for n_est in n_estimators_list:
        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_est,
            learning_rate=1.0,
            random_state=42
        )
        ada_clf.fit(X_train, y_train)
        
        train_score = ada_clf.score(X_train, y_train)
        test_score = ada_clf.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=6)
    plt.xlabel('弱学习器数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('AdaBoost：不同弱学习器数量的性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_adaboost_estimators_comparison.png'), dpi=150)
    print(f"学习曲线已保存至: ./images/1_adaboost_estimators_comparison.png")
    plt.close()


def compare_learning_rates():
    """
    比较不同学习率对 AdaBoost 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同学习率对模型性能的影响")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试不同的学习率
    learning_rates = [0.1, 0.5, 1.0, 1.5, 2.0]
    train_scores = []
    test_scores = []
    
    print(f"\n正在测试不同的学习率...")
    for lr in learning_rates:
        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=lr,
            random_state=42
        )
        ada_clf.fit(X_train, y_train)
        
        train_score = ada_clf.score(X_train, y_train)
        test_score = ada_clf.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"学习率 {lr}: 训练准确率={train_score:.4f}, 测试准确率={test_score:.4f}")
    
    # 绘制学习率对比
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=8)
    plt.plot(learning_rates, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=8)
    plt.xlabel('学习率 (learning_rate)')
    plt.ylabel('准确率')
    plt.title('AdaBoost：不同学习率的性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_adaboost_learning_rate_comparison.png'), dpi=150)
    print(f"学习率对比已保存至: ./images/1_adaboost_learning_rate_comparison.png")
    plt.close()


if __name__ == '__main__':
    # 运行所有演示
    demo_adaboost_breast_cancer()
    compare_n_estimators()
    compare_learning_rates()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)
