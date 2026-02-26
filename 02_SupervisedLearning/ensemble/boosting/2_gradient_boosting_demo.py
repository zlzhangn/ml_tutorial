# -*- coding: utf-8 -*-
"""
Gradient Boosting 演示脚本

Gradient Boosting 是一种强大的集成学习方法，它通过逐步训练决策树来拟合残差，
从而不断改进模型的预测能力。

特点：
- 基于损失函数的梯度方向进行优化
- 对各类数据集都有很好的泛化性能
- 相比 AdaBoost，对异常值不太敏感
- 通常具有更强的预测能力
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = Path(__file__).parent / 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def demo_gb_classification():
    """
    演示 Gradient Boosting 在分类任务中的应用
    """
    print("\n" + "="*60)
    print("Gradient Boosting 演示：分类任务 - 乳腺癌数据集")
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
    
    # 创建 Gradient Boosting 分类器
    # n_estimators: 梯度提升阶段数（基础学习器数量）
    # learning_rate: 学习率，控制每个弱学习器的贡献程度
    # max_depth: 每个决策树的最大深度
    # subsample: 用于拟合每个基础学习器的样本比例
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,  # 训练 100 个决策树
        learning_rate=0.1,  # 学习率
        max_depth=5,  # 每个树的最大深度
        subsample=0.8,  # 使用 80% 的样本进行训练
        random_state=42
    )
    
    # 训练模型
    print("\n正在训练 Gradient Boosting 分类器...")
    gb_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = gb_clf.predict(X_train)
    y_test_pred = gb_clf.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n模型性能：")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 打印详细的分类报告
    print(f"\n测试集分类报告：")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))
    
    # 绘制特征重要性（Top 15）
    feature_importance = gb_clf.feature_importances_
    indices = np.argsort(feature_importance)[-15:]  # 取重要性最高的 15 个特征
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='forestgreen')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('Gradient Boosting 特征重要性排名（Top 15）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_gb_classification_feature_importance.png'), dpi=150)
    print(f"\n特征重要性已保存至: ./images/2_gb_classification_feature_importance.png")
    plt.close()


def demo_gb_regression():
    """
    演示 Gradient Boosting 在回归任务中的应用
    """
    print("\n" + "="*60)
    print("Gradient Boosting 演示：回归任务")
    print("="*60)
    
    # 生成合成回归数据集
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建 Gradient Boosting 回归器
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    # 训练模型
    print("\n正在训练 Gradient Boosting 回归器...")
    gb_reg.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = gb_reg.predict(X_train)
    y_test_pred = gb_reg.predict(X_test)
    
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
    plt.savefig(os.path.join(image_dir, '2_gb_regression_predictions.png'), dpi=150)
    print(f"\n预测结果已保存至: ./images/2_gb_regression_predictions.png")
    plt.close()


def compare_gb_parameters():
    """
    比较不同超参数对 Gradient Boosting 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同超参数对 Gradient Boosting 的影响")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. 比较不同的 n_estimators
    print(f"\n(1) 测试不同的 n_estimators 参数...")
    n_estimators_list = np.arange(10, 151, 10)
    train_scores_n = []
    test_scores_n = []
    
    for n_est in n_estimators_list:
        gb_clf = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        train_scores_n.append(gb_clf.score(X_train, y_train))
        test_scores_n.append(gb_clf.score(X_test, y_test))
    
    # 2. 比较不同的 learning_rate
    print(f"\n(2) 测试不同的 learning_rate 参数...")
    learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    train_scores_lr = []
    test_scores_lr = []
    
    for lr in learning_rates:
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=5,
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        train_scores_lr.append(gb_clf.score(X_train, y_train))
        test_scores_lr.append(gb_clf.score(X_test, y_test))
    
    # 3. 比较不同的 max_depth
    print(f"\n(3) 测试不同的 max_depth 参数...")
    max_depths = [1, 2, 3, 4, 5, 7, 10]
    train_scores_md = []
    test_scores_md = []
    
    for md in max_depths:
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=md,
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        train_scores_md.append(gb_clf.score(X_train, y_train))
        test_scores_md.append(gb_clf.score(X_test, y_test))
    
    # 绘制对比结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # n_estimators 对比
    axes[0].plot(n_estimators_list, train_scores_n, 'o-', label='训练集', linewidth=2)
    axes[0].plot(n_estimators_list, test_scores_n, 's-', label='测试集', linewidth=2)
    axes[0].set_xlabel('n_estimators')
    axes[0].set_ylabel('准确率')
    axes[0].set_title('n_estimators 的影响')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # learning_rate 对比
    axes[1].plot(learning_rates, train_scores_lr, 'o-', label='训练集', linewidth=2)
    axes[1].plot(learning_rates, test_scores_lr, 's-', label='测试集', linewidth=2)
    axes[1].set_xlabel('learning_rate')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('learning_rate 的影响')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # max_depth 对比
    axes[2].plot(max_depths, train_scores_md, 'o-', label='训练集', linewidth=2)
    axes[2].plot(max_depths, test_scores_md, 's-', label='测试集', linewidth=2)
    axes[2].set_xlabel('max_depth')
    axes[2].set_ylabel('准确率')
    axes[2].set_title('max_depth 的影响')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_gb_hyperparameters_comparison.png'), dpi=150)
    print(f"\n超参数对比已保存至: ./images/2_gb_hyperparameters_comparison.png")
    plt.close()


if __name__ == '__main__':
    # 运行所有演示
    demo_gb_classification()
    demo_gb_regression()
    compare_gb_parameters()
    
    print("\n" + "="*60)
    print("所有 Gradient Boosting 演示完成！")
    print("="*60)
