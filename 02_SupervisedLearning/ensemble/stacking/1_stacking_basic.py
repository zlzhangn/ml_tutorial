#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║                  Stacking 基础演示 - 基础用法                  ║
║                   Stacking Basic Demonstration                ║
╚════════════════════════════════════════════════════════════════╝

功能说明：
本脚本演示 Stacking 集成学习的基础概念和用法
- 演示基础 Stacking 分类器
- 演示 Stacking 回归器
- 演示不同基础学习器的组合
- 分析 Stacking 的性能优势

Stacking 的原理：
1. 用多个基础学习器进行训练 (Base Learners)
2. 基础学习器的预测结果作为元特征 (Meta Features)
3. 用元学习器学习这些特征 (Meta Learner)
4. 最终用元学习器进行预测

依赖包：
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# sklearn 相关库
from sklearn.datasets import load_breast_cancer, load_iris, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 基础学习器
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR

# Stacking - sklearn 的 StackingClassifier/Regressor
from sklearn.ensemble import StackingClassifier, StackingRegressor

# 模型评估
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def ensure_images_dir():
    """
    确保 images 目录存在
    """
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    return images_dir


# ─────────────────────────────────────────────────────────────
# 演示 1: Stacking 分类器基础
# ─────────────────────────────────────────────────────────────

def demo_stacking_classification():
    """
    演示 Stacking 分类器的基本使用
    
    步骤：
    1. 定义多个基础分类器 (Base Learners)
       - 决策树
       - 随机森林
       - KNN
       - SVM
    
    2. 定义元学习器 (Meta Learner)
       - 逻辑回归 (简单高效)
    
    3. 创建 Stacking 分类器
    4. 训练和评估
    """
    print("\n" + "="*60)
    print("演示 1: Stacking 分类器 - 基础使用")
    print("="*60)
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化 (Stacking 中很重要)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ─── 定义基础学习器 ───
    print("\n【定义基础学习器】")
    
    # 基础学习器列表
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    for name, clf in base_learners:
        print(f"  ✓ {name.upper()} - {clf.__class__.__name__}")
    
    # ─── 定义元学习器 ───
    print("\n【定义元学习器】")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    print(f"  ✓ Logistic Regression - 学习基础分类器的预测")
    
    # ─── 创建 Stacking 分类器 ───
    print("\n【创建 Stacking 分类器】")
    stacking_clf = StackingClassifier(
        estimators=base_learners,      # 基础学习器
        final_estimator=meta_learner,  # 元学习器
        cv=5                           # 5 折交叉验证生成元特征
    )
    print("  ✓ Stacking 分类器已创建")
    
    # ─── 训练和评估 ───
    print("\n【训练模型】")
    stacking_clf.fit(X_train, y_train)
    print("  ✓ Stacking 分类器训练完成")
    
    # 获取预测
    y_pred = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)
    
    # ─── 性能评估 ───
    print("\n【性能评估】")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n【分类报告】")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # ─── 可视化混淆矩阵 ───
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names,
                cbar_kws={'label': '样本数'})
    plt.title(f'Stacking 分类器 - 混淆矩阵\n准确率: {accuracy:.4f}', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 混淆矩阵已保存: ./images/1_confusion_matrix.png")
    
    # ─── 比较各基础学习器的性能 ───
    print("\n【各基础学习器的性能对比】")
    base_scores = {}
    for name, clf in base_learners:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        base_scores[name.upper()] = score
        print(f"  {name.upper():4s}: {score:.4f}")
    
    print(f"\n  Stacking: {accuracy:.4f} (综合基础学习器)")
    
    return {
        'accuracy': accuracy,
        'base_scores': base_scores,
        'stacking_clf': stacking_clf
    }


# ─────────────────────────────────────────────────────────────
# 演示 2: Stacking 回归器
# ─────────────────────────────────────────────────────────────

def demo_stacking_regression():
    """
    演示 Stacking 回归器的基本使用
    
    与分类类似，但用于回归问题
    基础学习器：决策树、随机森林、KNN、SVR
    元学习器：Ridge 回归 (正则化线性回归)
    """
    print("\n" + "="*60)
    print("演示 2: Stacking 回归器")
    print("="*60)
    
    # 生成回归数据集
    X, y = make_regression(
        n_samples=300, n_features=20, n_informative=15,
        noise=10, random_state=42
    )
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ─── 定义基础回归器 ───
    print("\n【定义基础学习器】")
    base_learners = [
        ('dt', DecisionTreeRegressor(max_depth=15, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
        ('svr', SVR(kernel='rbf'))
    ]
    
    for name, reg in base_learners:
        print(f"  ✓ {name.upper()} - {reg.__class__.__name__}")
    
    # ─── 定义元学习器 ───
    print("\n【定义元学习器】")
    meta_learner = Ridge(alpha=1.0)
    print("  ✓ Ridge 回归 - 学习基础回归器的预测")
    
    # ─── 创建 Stacking 回归器 ───
    print("\n【创建 Stacking 回归器】")
    stacking_reg = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
    print("  ✓ Stacking 回归器已创建")
    
    # ─── 训练和评估 ───
    print("\n【训练模型】")
    stacking_reg.fit(X_train, y_train)
    print("  ✓ Stacking 回归器训练完成")
    
    # 获取预测
    y_pred = stacking_reg.predict(X_test)
    
    # ─── 性能评估 ───
    print("\n【性能评估】")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  均方误差 (MSE):   {mse:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  R² 分数:         {r2:.4f}")
    
    # ─── 比较各基础学习器的性能 ───
    print("\n【各基础学习器的性能对比 (R² 分数)】")
    base_scores = {}
    for name, reg in base_learners:
        reg.fit(X_train, y_train)
        score = reg.score(X_test, y_test)
        base_scores[name.upper()] = score
        print(f"  {name.upper():4s}: {score:.4f}")
    
    print(f"\n  Stacking: {r2:.4f} (综合基础学习器)")
    
    # ─── 可视化预测结果 ───
    indices = np.argsort(y_test)[:50]  # 取前 50 个样本
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(indices)), y_test[indices], label='真实值', alpha=0.6, s=50)
    plt.scatter(range(len(indices)), y_pred[indices], label='预测值', alpha=0.6, s=50)
    plt.xlabel('样本索引')
    plt.ylabel('目标值')
    plt.title(f'Stacking 回归器 - 预测结果 (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '2_regression_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 预测结果已保存: ./images/2_regression_predictions.png")
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'base_scores': base_scores
    }


# ─────────────────────────────────────────────────────────────
# 演示 3: 对比不同的基础学习器组合
# ─────────────────────────────────────────────────────────────

def compare_base_learner_combinations():
    """
    演示不同的基础学习器组合对 Stacking 性能的影响
    
    测试不同的组合：
    1. 简单组合 (2 个学习器)
    2. 标准组合 (4 个学习器)
    3. 复杂组合 (6 个学习器)
    """
    print("\n" + "="*60)
    print("演示 3: 不同基础学习器组合对性能的影响")
    print("="*60)
    
    # 加载 Iris 数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 二分类问题 (为了简化)
    mask = y != 2
    X, y = X[mask], y[mask]
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 定义不同的组合
    combinations = {
        '简单组合\n(2个学习器)': [
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
        ],
        '标准组合\n(4个学习器)': [
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ],
        '复杂组合\n(6个学习器)': [
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('knn3', KNeighborsClassifier(n_neighbors=3)),
            ('knn5', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]
    }
    
    # 评估不同组合
    results = {}
    print("\n【评估不同的基础学习器组合】")
    
    for combo_name, base_learners in combinations.items():
        # 创建 Stacking 分类器
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5
        )
        
        # 训练和评估
        stacking_clf.fit(X_train, y_train)
        score = stacking_clf.score(X_test, y_test)
        results[combo_name] = score
        print(f"  {combo_name}: {score:.4f}")
    
    # ─── 可视化对比 ───
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(names, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('准确率', fontsize=11, fontweight='bold')
    ax.set_title('不同基础学习器组合对 Stacking 性能的影响', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '3_base_learners_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 对比图已保存: ./images/3_base_learners_comparison.png")
    
    return results


# ─────────────────────────────────────────────────────────────
# 演示 4: Stacking vs 其他集成方法的对比
# ─────────────────────────────────────────────────────────────

def compare_with_other_methods():
    """
    对比 Stacking 与其他集成方法的性能
    
    对比方法：
    1. Stacking (多个基础学习器 + 元学习器)
    2. Voting 分类器 (简单投票)
    3. 单个最好的基础学习器 (Random Forest)
    4. 简单平均 (多个学习器的简单平均)
    """
    print("\n" + "="*60)
    print("演示 4: Stacking vs 其他集成方法")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 导入 Voting 分类器
    from sklearn.ensemble import VotingClassifier
    
    # ─── 定义基础学习器 ───
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    # ─── 方法 1: Stacking ───
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    # ─── 方法 2: Voting (Hard) ───
    voting_hard = VotingClassifier(estimators=base_learners, voting='hard')
    
    # ─── 方法 3: Voting (Soft) ───
    voting_soft = VotingClassifier(estimators=base_learners, voting='soft')
    
    # ─── 方法 4: 单个 Random Forest ───
    rf_single = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # ─── 训练和评估 ───
    print("\n【评估不同方法】")
    methods = {
        'Stacking': stacking_clf,
        'Voting Hard': voting_hard,
        'Voting Soft': voting_soft,
        'Random Forest': rf_single
    }
    
    results = {}
    for method_name, method_clf in methods.items():
        method_clf.fit(X_train, y_train)
        score = method_clf.score(X_test, y_test)
        results[method_name] = score
        print(f"  {method_name:15s}: {score:.4f}")
    
    # ─── 可视化对比 ───
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = ax.bar(names, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('准确率', fontsize=11, fontweight='bold')
    ax.set_title('Stacking vs 其他集成方法', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '4_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 对比图已保存: ./images/4_methods_comparison.png")
    
    return results


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    """
    运行所有演示
    """
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "Stacking 集成学习 - 基础演示" + " "*20 + "║")
    print("╚" + "="*58 + "╝")
    
    # 演示 1: Stacking 分类器
    result1 = demo_stacking_classification()
    
    # 演示 2: Stacking 回归器
    result2 = demo_stacking_regression()
    
    # 演示 3: 不同基础学习器组合
    result3 = compare_base_learner_combinations()
    
    # 演示 4: Stacking vs 其他方法
    result4 = compare_with_other_methods()
    
    # 总结
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n✓ 所有图表已保存到 ./images/ 目录")
    print("\n关键要点：")
    print("1. Stacking 通过组合多个基础学习器的预测来改进性能")
    print("2. 基础学习器应该具有多样性，这样元学习器才能有效学习")
    print("3. 元学习器通常使用简单的模型（如逻辑回归、Ridge）")
    print("4. Stacking 可用于分类和回归问题")
    print("5. Stacking 的性能通常好于单个集成方法")


if __name__ == '__main__':
    main()
