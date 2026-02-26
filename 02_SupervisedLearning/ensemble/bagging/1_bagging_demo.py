# -*- coding: utf-8 -*-
"""
Bagging 基础演示脚本

Bagging (Bootstrap Aggregating) 是一种通过重采样来降低模型方差的集成学习方法。

核心思想：
1. 从原始数据集中随机有放回地抽样，生成多个子数据集
2. 在每个子数据集上训练相同的基础学习器
3. 对多个学习器的预测结果进行聚合（分类：投票，回归：平均）
4. 通过减少方差来改进模型性能

特点：
- 并行训练，速度快
- 减少方差，不减少偏差
- 不易过拟合
- 适合高方差的弱学习器
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def demo_bagging_classification():
    """
    演示 Bagging 在分类任务中的应用
    """
    print("\n" + "="*60)
    print("Bagging 演示：分类任务 - 乳腺癌数据集")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建单个决策树分类器（基础学习器）
    dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # 创建 Bagging 分类器
    # n_estimators: 基础学习器的数量
    # max_samples: 每次采样的样本数
    # max_features: 每次采样的特征数
    # bootstrap: 是否使用有放回抽样
    # n_jobs: 并行化处理的核心数
    bagging_clf = BaggingClassifier(
        estimator=dt_clf,
        n_estimators=10,  # 使用 10 个决策树
        max_samples=1.0,  # 使用 100% 的样本
        max_features=1.0,  # 使用 100% 的特征
        bootstrap=True,  # 有放回抽样
        random_state=42,
        n_jobs=-1  # 使用所有 CPU 核心
    )
    
    # 训练模型
    print("\n正在训练 Bagging 分类器...")
    bagging_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = bagging_clf.predict(X_train)
    y_test_pred = bagging_clf.predict(X_test)
    
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
    plt.title('Bagging 混淆矩阵 - 乳腺癌数据集')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_bagging_confusion_matrix.png'), dpi=150)
    print(f"\n混淆矩阵已保存至: ./images/1_bagging_confusion_matrix.png")
    plt.close()
    
    # 绘制特征重要性
    feature_importance = bagging_clf.estimators_features_
    # 计算各特征在 Bagging 中被使用的频率
    feature_usage = np.zeros(X.shape[1])
    for features in feature_importance:
        feature_usage[features] += 1
    feature_usage = feature_usage / len(bagging_clf.estimators_)
    
    indices = np.argsort(feature_usage)[-10:]  # 取使用频率最高的 10 个特征
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_usage[indices], color='steelblue')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征使用频率')
    plt.title('Bagging 特征使用频率 (Top 10)')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_bagging_feature_usage.png'), dpi=150)
    print(f"特征使用频率已保存至: ./images/1_bagging_feature_usage.png")
    plt.close()


def compare_n_estimators():
    """
    比较不同基础学习器数量对 Bagging 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同基础学习器数量对模型性能的影响")
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
    
    print(f"\n正在测试不同数量的基础学习器 (1 到 50)...")
    for n_est in n_estimators_list:
        bagging_clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1
        )
        bagging_clf.fit(X_train, y_train)
        
        train_score = bagging_clf.score(X_train, y_train)
        test_score = bagging_clf.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=6)
    plt.xlabel('基础学习器数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('Bagging：不同基础学习器数量的性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_bagging_estimators_comparison.png'), dpi=150)
    print(f"学习曲线已保存至: ./images/1_bagging_estimators_comparison.png")
    plt.close()


def demo_bagging_regression():
    """
    演示 Bagging 在回归任务中的应用
    """
    print("\n" + "="*60)
    print("演示：Bagging 回归任务")
    print("="*60)
    
    # 生成合成回归数据集
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建 Bagging 回归器
    bagging_reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10, random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    print("\n正在训练 Bagging 回归器...")
    bagging_reg.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = bagging_reg.predict(X_train)
    y_test_pred = bagging_reg.predict(X_test)
    
    # 计算性能指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n模型性能：")
    print(f"训练集 MSE: {train_mse:.4f}, R² 得分: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R² 得分: {test_r2:.4f}")
    
    # 绘制预测结果与实际值的对比
    plt.figure(figsize=(12, 5))
    
    # 子图1：训练集
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, s=30, color='steelblue')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'训练集预测结果 (R²={train_r2:.4f})')
    
    # 子图2：测试集
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='forestgreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'测试集预测结果 (R²={test_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_bagging_regression_predictions.png'), dpi=150)
    print(f"\n预测结果已保存至: ./images/1_bagging_regression_predictions.png")
    plt.close()


def compare_base_estimators():
    """
    比较不同基础学习器对 Bagging 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同基础学习器对 Bagging 性能的影响")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 定义不同的基础学习器
    base_estimators = {
        '决策树 (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        '决策树 (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
        '决策树 (depth=20)': DecisionTreeClassifier(max_depth=20, random_state=42),
    }
    
    print(f"\n正在测试不同的基础学习器...")
    results = {}
    
    for name, estimator in base_estimators.items():
        bagging_clf = BaggingClassifier(
            estimator=estimator,
            n_estimators=20,
            random_state=42,
            n_jobs=-1
        )
        bagging_clf.fit(X_train, y_train)
        
        train_score = bagging_clf.score(X_train, y_train)
        test_score = bagging_clf.score(X_test, y_test)
        
        results[name] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        print(f"  {name}: 训练准确率={train_score:.4f}, 测试准确率={test_score:.4f}")
    
    # 绘制对比结果
    names = list(results.keys())
    train_scores = [results[n]['train_accuracy'] for n in names]
    test_scores = [results[n]['test_accuracy'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_scores, width, label='训练集准确率', color='steelblue')
    plt.bar(x + width/2, test_scores, width, label='测试集准确率', color='coral')
    
    plt.xlabel('基础学习器类型')
    plt.ylabel('准确率')
    plt.title('Bagging：不同基础学习器的性能对比')
    plt.xticks(x, names, rotation=15, ha='right')
    plt.legend()
    plt.ylim([0.85, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + 0.005, f'{train:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, test + 0.005, f'{test:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_bagging_base_estimators_comparison.png'), dpi=150)
    print(f"\n基础学习器对比已保存至: ./images/1_bagging_base_estimators_comparison.png")
    plt.close()


if __name__ == '__main__':
    # 运行所有演示
    demo_bagging_classification()
    compare_n_estimators()
    demo_bagging_regression()
    compare_base_estimators()
    
    print("\n" + "="*60)
    print("所有 Bagging 演示完成！")
    print("="*60)
