#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║                    Blending 演示 - 简化的 Stacking             ║
║              Blending Demonstration - Simplified Stacking     ║
╚════════════════════════════════════════════════════════════════╝

功能说明：
本脚本演示 Blending 集成学习方法
Blending 是 Stacking 的简化版本

Blending vs Stacking 的区别：
1. Blending 不使用交叉验证生成元特征，而是用验证集
2. Blending 计算量更小，适合大数据集
3. Blending 的性能通常略低于 Stacking，但速度更快

Blending 的步骤：
1. 将训练集分为两部分：训练集和验证集
2. 用训练集训练基础学习器
3. 用基础学习器在验证集上进行预测（生成元特征）
4. 用元特征训练元学习器
5. 用测试集进行最终预测

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
from sklearn.datasets import load_breast_cancer, load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 基础学习器
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR

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
    """确保 images 目录存在"""
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    return images_dir


# ─────────────────────────────────────────────────────────────
# 演示 1: Blending 分类器 (手动实现)
# ─────────────────────────────────────────────────────────────

def demo_blending_classification():
    """
    演示如何手动实现 Blending 分类器
    
    步骤：
    1. 将训练集分为训练集和验证集
    2. 用训练集训练基础学习器
    3. 用基础学习器在验证集上预测（生成元特征）
    4. 用验证集的元特征和标签训练元学习器
    5. 用测试集进行预测
    """
    print("\n" + "="*60)
    print("演示 1: Blending 分类器 (手动实现)")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # ─── 第一步：划分数据集 ───
    print("\n【数据集划分】")
    
    # 先分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 再从 temp 中分出训练集和验证集
    # Blending 通常用 50:50 的比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  训练集大小:  {X_train.shape[0]} 样本")
    print(f"  验证集大小:  {X_val.shape[0]} 样本 (用于生成元特征)")
    print(f"  测试集大小:  {X_test.shape[0]} 样本")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # ─── 第二步：定义基础学习器 ───
    print("\n【定义基础学习器】")
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    for name, _ in base_learners:
        print(f"  ✓ {name.upper()}")
    
    # ─── 第三步：在训练集上训练基础学习器 ───
    print("\n【在训练集上训练基础学习器】")
    fitted_learners = []
    for name, clf in base_learners:
        clf.fit(X_train, y_train)
        fitted_learners.append((name, clf))
        print(f"  ✓ {name.upper()} 已训练")
    
    # ─── 第四步：在验证集上生成元特征 ───
    print("\n【在验证集上生成元特征】")
    
    # 用基础学习器在验证集上预测
    meta_features_val = []
    for name, clf in fitted_learners:
        # 获取概率预测作为元特征
        pred = clf.predict_proba(X_val)
        meta_features_val.append(pred)
        print(f"  ✓ {name.upper()} 预测形状: {pred.shape}")
    
    # 合并所有元特征
    X_meta_val = np.hstack(meta_features_val)
    print(f"  合并后的元特征形状: {X_meta_val.shape}")
    
    # ─── 第五步：训练元学习器 ───
    print("\n【训练元学习器】")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(X_meta_val, y_val)
    print("  ✓ 元学习器 (Logistic Regression) 已训练")
    
    # ─── 第六步：在测试集上生成元特征 ───
    print("\n【在测试集上进行预测】")
    meta_features_test = []
    for name, clf in fitted_learners:
        pred = clf.predict_proba(X_test)
        meta_features_test.append(pred)
    
    X_meta_test = np.hstack(meta_features_test)
    print(f"  测试集元特征形状: {X_meta_test.shape}")
    
    # ─── 第七步：用元学习器在测试集上预测 ───
    y_pred = meta_learner.predict(X_meta_test)
    y_pred_proba = meta_learner.predict_proba(X_meta_test)
    
    # ─── 评估性能 ───
    print("\n【性能评估】")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  准确率: {accuracy:.4f}")
    
    print("\n【分类报告】")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # ─── 可视化混淆矩阵 ───
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names,
                cbar_kws={'label': '样本数'})
    plt.title(f'Blending 分类器 - 混淆矩阵\n准确率: {accuracy:.4f}', 
              fontsize=12, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '2_blending_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 混淆矩阵已保存: ./images/2_blending_confusion_matrix.png")
    
    # ─── 对比各基础学习器的性能 ───
    print("\n【各基础学习器的性能对比】")
    for name, clf in fitted_learners:
        score = clf.score(X_test, y_test)
        print(f"  {name.upper():4s}: {score:.4f}")
    
    print(f"\n  Blending: {accuracy:.4f} (通过元学习器综合)")
    
    return {'accuracy': accuracy}


# ─────────────────────────────────────────────────────────────
# 演示 2: Blending 回归器
# ─────────────────────────────────────────────────────────────

def demo_blending_regression():
    """
    演示 Blending 回归器的实现
    """
    print("\n" + "="*60)
    print("演示 2: Blending 回归器")
    print("="*60)
    
    # 生成回归数据
    X, y = make_regression(
        n_samples=300, n_features=20, n_informative=15,
        noise=10, random_state=42
    )
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # ─── 定义基础回归器 ───
    print("\n【定义基础学习器】")
    base_learners = [
        ('dt', DecisionTreeRegressor(max_depth=15, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
        ('svr', SVR(kernel='rbf'))
    ]
    
    for name, _ in base_learners:
        print(f"  ✓ {name.upper()}")
    
    # ─── 训练基础学习器 ───
    print("\n【在训练集上训练基础学习器】")
    fitted_learners = []
    for name, reg in base_learners:
        reg.fit(X_train, y_train)
        fitted_learners.append((name, reg))
        print(f"  ✓ {name.upper()} 已训练")
    
    # ─── 生成验证集的元特征 ───
    print("\n【生成元特征】")
    meta_features_val = []
    for name, reg in fitted_learners:
        pred = reg.predict(X_val).reshape(-1, 1)
        meta_features_val.append(pred)
    
    X_meta_val = np.hstack(meta_features_val)
    print(f"  验证集元特征形状: {X_meta_val.shape}")
    
    # ─── 训练元学习器 ───
    print("\n【训练元学习器】")
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(X_meta_val, y_val)
    print("  ✓ 元学习器 (Ridge) 已训练")
    
    # ─── 生成测试集的元特征并预测 ───
    print("\n【测试集预测】")
    meta_features_test = []
    for name, reg in fitted_learners:
        pred = reg.predict(X_test).reshape(-1, 1)
        meta_features_test.append(pred)
    
    X_meta_test = np.hstack(meta_features_test)
    y_pred = meta_learner.predict(X_meta_test)
    
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
    
    # ─── 可视化预测结果 ───
    indices = np.argsort(y_test)[:50]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(indices)), y_test[indices], label='真实值', alpha=0.6, s=50)
    plt.scatter(range(len(indices)), y_pred[indices], label='预测值', alpha=0.6, s=50)
    plt.xlabel('样本索引')
    plt.ylabel('目标值')
    plt.title(f'Blending 回归器 - 预测结果 (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '2_blending_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 预测结果已保存: ./images/2_blending_regression.png")
    
    return {'r2_score': r2, 'rmse': rmse}


# ─────────────────────────────────────────────────────────────
# 演示 3: Blending vs Stacking 性能和速度对比
# ─────────────────────────────────────────────────────────────

def compare_blending_vs_stacking():
    """
    对比 Blending 和 Stacking 的性能和速度
    
    Blending 优势：
    - 速度更快（无需交叉验证）
    - 内存占用少
    
    Stacking 优势：
    - 充分利用训练数据（交叉验证）
    - 通常性能更好
    """
    print("\n" + "="*60)
    print("演示 3: Blending vs Stacking 对比")
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
    
    # ─── 导入 StackingClassifier ───
    from sklearn.ensemble import StackingClassifier
    
    # 定义基础学习器
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    
    # ─── Stacking ───
    print("\n【Stacking】")
    print("  正在训练... (使用 5 折交叉验证)")
    start = time.time()
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    stacking_time = time.time() - start
    stacking_score = stacking_clf.score(X_test, y_test)
    print(f"  训练时间: {stacking_time:.4f}s")
    print(f"  准确率: {stacking_score:.4f}")
    
    # ─── Blending ───
    print("\n【Blending】")
    print("  正在训练... (使用验证集)")
    start = time.time()
    
    # 手动实现 Blending
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42
    )
    
    # 训练基础学习器
    fitted_learners = []
    for name, clf in base_learners:
        clf.fit(X_train_split, y_train_split)
        fitted_learners.append((name, clf))
    
    # 生成元特征
    meta_features_val = []
    for name, clf in fitted_learners:
        pred = clf.predict_proba(X_val)
        meta_features_val.append(pred)
    X_meta_val = np.hstack(meta_features_val)
    
    # 训练元学习器
    meta_learner_blending = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner_blending.fit(X_meta_val, y_val)
    
    blending_time = time.time() - start
    
    # 测试集预测
    meta_features_test = []
    for name, clf in fitted_learners:
        pred = clf.predict_proba(X_test)
        meta_features_test.append(pred)
    X_meta_test = np.hstack(meta_features_test)
    blending_score = meta_learner_blending.score(X_meta_test, y_test)
    
    print(f"  训练时间: {blending_time:.4f}s")
    print(f"  准确率: {blending_score:.4f}")
    
    # ─── 对比 ───
    print("\n【对比总结】")
    speedup = stacking_time / blending_time
    print(f"  速度提升: {speedup:.2f}x (Blending 更快)")
    print(f"  准确率差异: {abs(stacking_score - blending_score):.4f}")
    
    if stacking_score > blending_score:
        print(f"  Stacking 准确率更高")
    else:
        print(f"  Blending 准确率更高")
    
    # ─── 可视化对比 ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确率对比
    ax = axes[0]
    methods = ['Stacking', 'Blending']
    scores = [stacking_score, blending_score]
    bars = ax.bar(methods, scores, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('准确率', fontsize=11, fontweight='bold')
    ax.set_title('准确率对比', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 速度对比
    ax = axes[1]
    times = [stacking_time, blending_time]
    bars = ax.bar(methods, times, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('训练时间 (秒)', fontsize=11, fontweight='bold')
    ax.set_title('速度对比', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Blending vs Stacking 对比', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    images_dir = ensure_images_dir()
    plt.savefig(images_dir / '3_blending_vs_stacking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 对比图已保存: ./images/3_blending_vs_stacking.png")
    
    return {
        'stacking': {'score': stacking_score, 'time': stacking_time},
        'blending': {'score': blending_score, 'time': blending_time}
    }


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    """运行所有演示"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "Blending 集成学习演示" + " "*21 + "║")
    print("╚" + "="*58 + "╝")
    
    # 演示 1: Blending 分类器
    result1 = demo_blending_classification()
    
    # 演示 2: Blending 回归器
    result2 = demo_blending_regression()
    
    # 演示 3: Blending vs Stacking
    result3 = compare_blending_vs_stacking()
    
    # 总结
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n✓ 所有图表已保存到 ./images/ 目录")
    print("\n关键要点：")
    print("1. Blending 是 Stacking 的简化版本，不使用交叉验证")
    print("2. Blending 使用验证集来生成元特征")
    print("3. Blending 比 Stacking 更快，但性能可能略低")
    print("4. Blending 适合大数据集和对速度有要求的场景")
    print("5. Blending 的核心思想与 Stacking 相同：用基础学习器的预测作为元特征")


if __name__ == '__main__':
    main()
