#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║             多层 Stacking 演示 - 深层集成方法                  ║
║      Multi-Level Stacking Demonstration - Deep Ensembling   ║
╚════════════════════════════════════════════════════════════════╝

功能说明：
本脚本演示多层 Stacking （也称为 Multi-level Stacking）
多层 Stacking 的思想是在第一层 Stacking 的基础上，继续进行 Stacking

多层 Stacking 的结构：
┌─────────────────────────────────────────┐
│   第 0 层：原始数据                      │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
  ┌────▼────┐    ┌────▼────┐
  │第 1 层   │    │第 1 层   │
  │基础学习器│    │基础学习器│
  └────┬────┘    └────┬────┘
       │              │
       └────┬─────────┘
            │
       ┌────▼─────┐
       │第 2 层    │
       │Meta 学习器│
       └──────────┘

多层 Stacking 的优势：
1. 可以学习更高层次的特征组合
2. 通常性能更好
3. 但计算量也更大

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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 基础学习器
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Stacking
from sklearn.ensemble import StackingClassifier

# 模型评估
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

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
# 演示 1: 二层 Stacking (基础)
# ─────────────────────────────────────────────────────────────

def demo_two_level_stacking():
    """
    演示两层 Stacking 的实现
    
    第一层：基础学习器
    第二层：元学习器
    """
    print("\n" + "="*60)
    print("演示 1: 两层 Stacking (标准 Stacking)")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n【两层 Stacking 结构】")
    print("  第 0 层: 原始特征 (30 维)")
    
    # ─── 第一层：基础学习器 ───
    print("\n  第 1 层: 基础学习器")
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    for name, _ in base_learners:
        print(f"    - {name.upper()}")
    
    # ─── 第二层：元学习器 ───
    print("\n  第 2 层: 元学习器")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    print(f"    - Logistic Regression (学习 4 个基础学习器的预测)")
    
    # ─── 创建和训练 Stacking 分类器 ───
    print("\n【训练两层 Stacking】")
    print("  正在构建模型...")
    
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
    
    print("  正在训练...")
    start = time.time()
    stacking_clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"  ✓ 训练完成，耗时: {train_time:.4f}s")
    
    # ─── 评估 ───
    print("\n【评估性能】")
    score = stacking_clf.score(X_test, y_test)
    print(f"  准确率: {score:.4f}")
    
    y_pred = stacking_clf.predict(X_test)
    print("\n【分类报告】")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # ─── 可视化混淆矩阵 ───
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names,
                cbar_kws={'label': '样本数'})
    plt.title(f'两层 Stacking - 混淆矩阵\n准确率: {score:.4f}', 
              fontsize=12, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '3_two_level_stacking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 混淆矩阵已保存: ./images/3_two_level_stacking.png")
    
    return {'score': score, 'train_time': train_time}


# ─────────────────────────────────────────────────────────────
# 演示 2: 三层 Stacking (多层 Stacking)
# ─────────────────────────────────────────────────────────────

def demo_three_level_stacking():
    """
    演示三层 Stacking 的实现
    
    第一层：基础学习器 A（生成元特征）
    第二层：基础学习器 B（在元特征基础上学习）
    第三层：最终元学习器（综合第二层的预测）
    """
    print("\n" + "="*60)
    print("演示 2: 三层 Stacking (多层集成)")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n【三层 Stacking 结构】")
    print("  第 0 层: 原始特征 (30 维)")
    
    # ─── 第一层：基础学习器 ───
    print("\n  第 1 层: 基础学习器（生成元特征）")
    level1_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    for name, _ in level1_learners:
        print(f"    - {name.upper()}")
    print("  生成元特征: 3 维 (每个学习器的概率预测)")
    
    # ─── 使用第一层构建第一个 Stacking ───
    print("\n  第 2 层: 学习元特征的基础学习器")
    
    level2_base_learners = [
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    for name, _ in level2_base_learners:
        print(f"    - {name.upper()}")
    
    # ─── 第三层：最终元学习器 ───
    print("\n  第 3 层: 最终元学习器")
    final_meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    print("    - Logistic Regression (综合第 2 层的预测)")
    
    # ─── 创建第一层 Stacking ───
    print("\n【构建和训练三层 Stacking】")
    print("  正在构建第一层 Stacking...")
    
    level1_stacking = StackingClassifier(
        estimators=level1_learners,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    print("  正在训练第一层...")
    start = time.time()
    level1_stacking.fit(X_train, y_train)
    level1_score = level1_stacking.score(X_test, y_test)
    print(f"  ✓ 第一层 Stacking 完成，准确率: {level1_score:.4f}")
    
    # 使用第一层的元特征构建第二层
    print("\n  正在构建第二层 Stacking...")
    
    level2_stacking = StackingClassifier(
        estimators=level2_base_learners,
        final_estimator=final_meta_learner,
        cv=5
    )
    
    # 获取第一层的元特征
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict
    
    # 手动构建三层 Stacking（更直观）
    # 第一层生成元特征
    X_meta_level1_train = np.zeros((X_train.shape[0], len(level1_learners)))
    X_meta_level1_test = np.zeros((X_test.shape[0], len(level1_learners)))
    
    for i, (name, learner) in enumerate(level1_learners):
        # 使用交叉验证获取训练集的元特征
        X_meta_level1_train[:, i] = cross_val_predict(
            learner, X_train, y_train, cv=5, method='predict_proba'
        )[:, 1]
        
        # 用完整训练集训练，预测测试集
        learner.fit(X_train, y_train)
        X_meta_level1_test[:, i] = learner.predict_proba(X_test)[:, 1]
    
    print(f"  ✓ 第一层元特征形状: {X_meta_level1_train.shape}")
    
    # 第二层：在元特征基础上再做 Stacking
    print("  正在训练第二层...")
    
    X_meta_level2_train = np.zeros((X_meta_level1_train.shape[0], len(level2_base_learners)))
    X_meta_level2_test = np.zeros((X_meta_level1_test.shape[0], len(level2_base_learners)))
    
    for i, (name, learner) in enumerate(level2_base_learners):
        X_meta_level2_train[:, i] = cross_val_predict(
            learner, X_meta_level1_train, y_train, cv=5, method='predict_proba'
        )[:, 1]
        
        learner.fit(X_meta_level1_train, y_train)
        X_meta_level2_test[:, i] = learner.predict_proba(X_meta_level1_test)[:, 1]
    
    print(f"  ✓ 第二层元特征形状: {X_meta_level2_train.shape}")
    
    # 第三层：最终元学习器
    print("  正在训练第三层...")
    final_meta_learner.fit(X_meta_level2_train, y_train)
    
    # 最终预测
    y_pred_three_level = final_meta_learner.predict(X_meta_level2_test)
    
    total_train_time = time.time() - start
    three_level_score = accuracy_score(y_test, y_pred_three_level)
    
    print(f"  ✓ 三层 Stacking 完成，耗时: {total_train_time:.4f}s")
    
    # ─── 评估 ───
    print("\n【性能评估】")
    print(f"  第一层准确率:  {level1_score:.4f}")
    print(f"  三层 Stacking 准确率: {three_level_score:.4f}")
    
    improvement = three_level_score - level1_score
    if improvement > 0:
        print(f"  性能提升: +{improvement:.4f}")
    else:
        print(f"  性能变化: {improvement:.4f}")
    
    # ─── 可视化对比 ───
    cm = confusion_matrix(y_test, y_pred_three_level)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names,
                cbar_kws={'label': '样本数'})
    plt.title(f'三层 Stacking - 混淆矩阵\n准确率: {three_level_score:.4f}', 
              fontsize=12, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '3_three_level_stacking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 混淆矩阵已保存: ./images/3_three_level_stacking.png")
    
    return {
        'two_level_score': level1_score,
        'three_level_score': three_level_score,
        'train_time': total_train_time
    }


# ─────────────────────────────────────────────────────────────
# 演示 3: 多层 Stacking 的性能对比
# ─────────────────────────────────────────────────────────────

def compare_stacking_layers():
    """
    对比不同层数 Stacking 的性能和复杂度
    """
    print("\n" + "="*60)
    print("演示 3: 多层 Stacking 的层数对性能的影响")
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
    
    print("\n【对比不同层数的 Stacking】")
    
    # ─── 单层模型（Random Forest）───
    print("\n  1 层（基础模型 - Random Forest）:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    start = time.time()
    rf.fit(X_train, y_train)
    time_1layer = time.time() - start
    score_1layer = rf.score(X_test, y_test)
    print(f"    准确率: {score_1layer:.4f}, 时间: {time_1layer:.4f}s")
    
    # ─── 两层 Stacking ───
    print("\n  2 层（Stacking）:")
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    stacking_2layer = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    start = time.time()
    stacking_2layer.fit(X_train, y_train)
    time_2layer = time.time() - start
    score_2layer = stacking_2layer.score(X_test, y_test)
    print(f"    准确率: {score_2layer:.4f}, 时间: {time_2layer:.4f}s")
    
    # ─── 三层 Stacking ───
    print("\n  3 层（多层 Stacking）:")
    # 简化的三层实现
    from sklearn.model_selection import cross_val_predict
    
    level1_learners = [
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ]
    
    start = time.time()
    
    # 第一层元特征
    X_meta_l1 = np.zeros((X_train.shape[0], len(level1_learners)))
    X_meta_l1_test = np.zeros((X_test.shape[0], len(level1_learners)))
    
    for i, (name, learner) in enumerate(level1_learners):
        X_meta_l1[:, i] = cross_val_predict(
            learner, X_train, y_train, cv=5, method='predict_proba'
        )[:, 1]
        learner.fit(X_train, y_train)
        X_meta_l1_test[:, i] = learner.predict_proba(X_test)[:, 1]
    
    # 第二层
    level2_learner = SVC(kernel='rbf', probability=True, random_state=42)
    X_meta_l2 = cross_val_predict(
        level2_learner, X_meta_l1, y_train, cv=5, method='predict_proba'
    )[:, 1]
    
    level2_learner.fit(X_meta_l1, y_train)
    X_meta_l2_test = level2_learner.predict_proba(X_meta_l1_test)[:, 1].reshape(-1, 1)
    
    # 第三层
    level3_learner = LogisticRegression(random_state=42, max_iter=1000)
    level3_learner.fit(X_meta_l2.reshape(-1, 1), y_train)
    y_pred_3layer = level3_learner.predict(X_meta_l2_test)
    
    time_3layer = time.time() - start
    score_3layer = accuracy_score(y_test, y_pred_3layer)
    print(f"    准确率: {score_3layer:.4f}, 时间: {time_3layer:.4f}s")
    
    # ─── 可视化对比 ───
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 准确率对比
    ax = axes[0]
    layers = ['1 层\n(Random Forest)', '2 层\n(Stacking)', '3 层\n(多层Stacking)']
    scores = [score_1layer, score_2layer, score_3layer]
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(layers, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('准确率', fontsize=11, fontweight='bold')
    ax.set_title('准确率对比', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 时间对比
    ax = axes[1]
    times = [time_1layer, time_2layer, time_3layer]
    bars = ax.bar(layers, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('训练时间 (秒)', fontsize=11, fontweight='bold')
    ax.set_title('时间对比 (复杂度)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('不同层数 Stacking 的对比', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '3_layers_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 对比图已保存: ./images/3_layers_comparison.png")
    
    return {
        '1_layer': {'score': score_1layer, 'time': time_1layer},
        '2_layers': {'score': score_2layer, 'time': time_2layer},
        '3_layers': {'score': score_3layer, 'time': time_3layer}
    }


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    """运行所有演示"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*12 + "多层 Stacking 集成学习演示" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    # 演示 1: 两层 Stacking
    result1 = demo_two_level_stacking()
    
    # 演示 2: 三层 Stacking
    result2 = demo_three_level_stacking()
    
    # 演示 3: 多层 Stacking 对比
    result3 = compare_stacking_layers()
    
    # 总结
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n✓ 所有图表已保存到 ./images/ 目录")
    print("\n关键要点：")
    print("1. 多层 Stacking 通过多层元学习器实现更复杂的特征组合")
    print("2. 每增加一层，复杂度会显著增加")
    print("3. 性能改进通常在 2-3 层后开始饱和")
    print("4. 应根据数据规模和计算资源选择合适的层数")
    print("5. 常见的是 2-3 层 Stacking，很少使用超过 4 层")


if __name__ == '__main__':
    main()
