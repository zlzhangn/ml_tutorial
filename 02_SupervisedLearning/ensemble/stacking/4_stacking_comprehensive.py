#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║             Stacking 综合演示 - 全面对比分析                  ║
║           Comprehensive Stacking Analysis and Comparison     ║
╚════════════════════════════════════════════════════════════════╝

功能说明：
本脚本进行全面的 Stacking 综合分析，包括：
1. 不同元学习器的选择对 Stacking 性能的影响
2. 基础学习器多样性的重要性
3. Stacking 与其他集成方法的全面对比（Bagging, Boosting, Voting）
4. Stacking 的参数敏感性分析

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
import time

# sklearn 相关库
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 基础学习器
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 模型评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def ensure_images_dir():
    """确保 images 目录存在"""
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    return images_dir


# ─────────────────────────────────────────────────────────────
# 演示 1: 不同元学习器的影响
# ─────────────────────────────────────────────────────────────

def demo_meta_learner_impact():
    """
    演示不同的元学习器对 Stacking 性能的影响
    
    测试的元学习器：
    1. Logistic Regression（线性）
    2. Random Forest（非线性）
    3. SVM（非线性）
    4. Gradient Boosting（非线性）
    """
    print("\n" + "="*60)
    print("演示 1: 不同元学习器对 Stacking 性能的影响")
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
    
    # 定义基础学习器（保持一致）
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    # 定义不同的元学习器
    meta_learners = {
        '逻辑回归\n(线性)': LogisticRegression(random_state=42, max_iter=1000),
        '随机森林\n(非线性)': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM\n(非线性)': SVC(kernel='rbf', probability=True, random_state=42),
        '梯度提升\n(非线性)': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    print("\n【测试不同的元学习器】")
    results = {}
    
    for meta_name, meta_clf in meta_learners.items():
        # 创建 Stacking 分类器
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_clf,
            cv=5
        )
        
        # 训练
        start = time.time()
        stacking_clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        # 评估
        score = stacking_clf.score(X_test, y_test)
        y_pred = stacking_clf.predict(X_test)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[meta_name] = {
            'accuracy': score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'time': train_time
        }
        
        print(f"  {meta_name:15s}: 准确率={score:.4f}, F1={f1:.4f}, 时间={train_time:.4f}s")
    
    # ─── 可视化对比 ───
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    f1_scores = [results[name]['f1'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='准确率', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 分数', alpha=0.8, color='#e74c3c')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('元学习器类型', fontsize=11, fontweight='bold')
    ax.set_ylabel('性能指标', fontsize=11, fontweight='bold')
    ax.set_title('不同元学习器对 Stacking 性能的影响', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '4_meta_learner_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 图表已保存: ./images/4_meta_learner_impact.png")
    
    return results


# ─────────────────────────────────────────────────────────────
# 演示 2: 基础学习器多样性的重要性
# ─────────────────────────────────────────────────────────────

def demo_base_learner_diversity():
    """
    演示基础学习器的多样性对 Stacking 性能的影响
    
    对比：
    1. 所有基础学习器相同 (低多样性)
    2. 混合不同类型的学习器 (高多样性)
    """
    print("\n" + "="*60)
    print("演示 2: 基础学习器多样性的重要性")
    print("="*60)
    
    # 加载数据
    data = load_wine()
    X, y = data.data, data.target
    
    # 二分类
    mask = y != 2
    X, y = X[mask], y[mask]
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n【对比不同的基础学习器组合】")
    
    # 方案 1: 低多样性（都是基于树的模型）
    print("\n  方案 1: 低多样性 (都是基于树的模型)")
    low_diversity_learners = [
        ('dt1', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('dt2', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf1', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    
    stacking_low_div = StackingClassifier(
        estimators=low_diversity_learners,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    stacking_low_div.fit(X_train, y_train)
    score_low_div = stacking_low_div.score(X_test, y_test)
    print(f"    准确率: {score_low_div:.4f}")
    
    # 方案 2: 高多样性（混合不同类型的模型）
    print("\n  方案 2: 高多样性 (混合不同类型的模型)")
    high_diversity_learners = [
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('nb', GaussianNB()),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    stacking_high_div = StackingClassifier(
        estimators=high_diversity_learners,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    stacking_high_div.fit(X_train, y_train)
    score_high_div = stacking_high_div.score(X_test, y_test)
    print(f"    准确率: {score_high_div:.4f}")
    
    # 对比单个最好的模型
    print("\n  参考: 单个最好的模型（Gradient Boosting）")
    gb_single = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_single.fit(X_train, y_train)
    score_single = gb_single.score(X_test, y_test)
    print(f"    准确率: {score_single:.4f}")
    
    # ─── 可视化对比 ───
    methods = ['低多样性\n(4个树模型)', '高多样性\n(6个混合模型)', '单个模型\n(梯度提升)']
    scores = [score_low_div, score_high_div, score_single]
    colors = ['#e74c3c', '#2ecc71', '#95a5a6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('准确率', fontsize=11, fontweight='bold')
    plt.title('基础学习器多样性对 Stacking 性能的影响', fontsize=12, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '4_diversity_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 图表已保存: ./images/4_diversity_importance.png")
    
    improvement = score_high_div - score_low_div
    if improvement > 0:
        print(f"\n【结论】高多样性的基础学习器提升了 {improvement:.4f} 的准确率")
    else:
        print(f"\n【结论】多样性在此数据集上的改进不显著")


# ─────────────────────────────────────────────────────────────
# 演示 3: Stacking vs 其他集成方法的全面对比
# ─────────────────────────────────────────────────────────────

def comprehensive_ensemble_comparison():
    """
    全面对比 Stacking 与其他集成方法
    
    对比方法：
    1. Bagging (Bootstrap Aggregating)
    2. Boosting (AdaBoost, Gradient Boosting)
    3. Voting (Hard & Soft)
    4. Stacking
    """
    print("\n" + "="*60)
    print("演示 3: Stacking vs 其他集成方法的全面对比")
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
    
    # 定义基础学习器（所有方法使用相同的基础学习器）
    base_learners_simple = [
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    print("\n【对比不同的集成方法】")
    
    methods = {}
    
    # ─── 方法 1: Bagging ───
    print("\n  方法 1: Bagging (随机森林)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    start = time.time()
    rf.fit(X_train, y_train)
    methods['Bagging'] = {
        'score': rf.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['Bagging']['score']:.4f}, 时间: {methods['Bagging']['time']:.4f}s")
    
    # ─── 方法 2: Boosting (AdaBoost) ───
    print("\n  方法 2: Boosting (AdaBoost)")
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    start = time.time()
    adaboost.fit(X_train, y_train)
    methods['AdaBoost'] = {
        'score': adaboost.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['AdaBoost']['score']:.4f}, 时间: {methods['AdaBoost']['time']:.4f}s")
    
    # ─── 方法 3: Boosting (Gradient Boosting) ───
    print("\n  方法 3: Boosting (梯度提升)")
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    start = time.time()
    gb.fit(X_train, y_train)
    methods['Gradient Boosting'] = {
        'score': gb.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['Gradient Boosting']['score']:.4f}, 时间: {methods['Gradient Boosting']['time']:.4f}s")
    
    # ─── 方法 4: Voting (Hard) ───
    print("\n  方法 4: Voting (硬投票)")
    voting_hard = VotingClassifier(estimators=base_learners_simple, voting='hard')
    start = time.time()
    voting_hard.fit(X_train, y_train)
    methods['Voting Hard'] = {
        'score': voting_hard.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['Voting Hard']['score']:.4f}, 时间: {methods['Voting Hard']['time']:.4f}s")
    
    # ─── 方法 5: Voting (Soft) ───
    print("\n  方法 5: Voting (软投票)")
    voting_soft = VotingClassifier(estimators=base_learners_simple, voting='soft')
    start = time.time()
    voting_soft.fit(X_train, y_train)
    methods['Voting Soft'] = {
        'score': voting_soft.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['Voting Soft']['score']:.4f}, 时间: {methods['Voting Soft']['time']:.4f}s")
    
    # ─── 方法 6: Stacking ───
    print("\n  方法 6: Stacking")
    stacking = StackingClassifier(
        estimators=base_learners_simple,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    start = time.time()
    stacking.fit(X_train, y_train)
    methods['Stacking'] = {
        'score': stacking.score(X_test, y_test),
        'time': time.time() - start
    }
    print(f"    准确率: {methods['Stacking']['score']:.4f}, 时间: {methods['Stacking']['time']:.4f}s")
    
    # ─── 可视化对比 ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(methods.keys())
    scores = [methods[name]['score'] for name in names]
    times = [methods[name]['time'] for name in names]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']
    
    # 准确率对比
    ax = axes[0]
    bars = ax.bar(names, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('准确率', fontsize=11, fontweight='bold')
    ax.set_title('准确率对比', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 速度对比
    ax = axes[1]
    bars = ax.bar(names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('训练时间 (秒)', fontsize=11, fontweight='bold')
    ax.set_title('速度对比', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('6 种集成方法的全面对比', fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '4_ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 对比图已保存: ./images/4_ensemble_comparison.png")
    
    # ─── 性能排名 ───
    print("\n【性能排名】")
    sorted_methods = sorted(methods.items(), key=lambda x: x[1]['score'], reverse=True)
    for i, (method, metrics) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method:20s}: 准确率 {metrics['score']:.4f}")
    
    return methods


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    """运行所有演示"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "Stacking 综合演示 - 全面对比分析" + " "*14 + "║")
    print("╚" + "="*58 + "╝")
    
    # 演示 1: 元学习器的影响
    result1 = demo_meta_learner_impact()
    
    # 演示 2: 基础学习器多样性
    demo_base_learner_diversity()
    
    # 演示 3: 全面对比
    result3 = comprehensive_ensemble_comparison()
    
    # 总结
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n✓ 所有图表已保存到 ./images/ 目录")
    print("\n关键要点：")
    print("1. 元学习器的选择会影响 Stacking 的性能")
    print("2. 基础学习器的多样性对 Stacking 至关重要")
    print("3. Stacking 通常性能最好，但计算成本较高")
    print("4. 应根据性能和速度权衡选择合适的集成方法")
    print("5. 不同的数据集可能需要不同的集成方法")


if __name__ == '__main__':
    main()
