# -*- coding: utf-8 -*-
"""
XGBoost 演示脚本

XGBoost (Extreme Gradient Boosting) 是 Gradient Boosting 的一个高度优化的开源实现，
具有更快的训练速度、更好的性能和更强的可扩展性。

特点：
- 支持正则化（L1 和 L2）来防止过拟合
- 支持缺失值处理
- 支持并行化处理
- 提供特征重要性评估
- 在 Kaggle 竞赛中广泛使用
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = Path(__file__).parent / 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 尝试导入 xgboost，如果未安装则显示提示信息
try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("警告: XGBoost 未安装，请运行: pip install xgboost")


def demo_xgboost_classification():
    """
    演示 XGBoost 在分类任务中的应用
    """
    if not XGB_INSTALLED:
        print("XGBoost 未安装，跳过 XGBoost 分类演示")
        return
    
    print("\n" + "="*60)
    print("XGBoost 演示：分类任务 - 乳腺癌数据集")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 标准化特征（XGBoost 对特征缩放的敏感性较低，但仍建议进行标准化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建 XGBoost 分类器
    # n_estimators: 树的数量
    # max_depth: 树的最大深度
    # learning_rate: 学习率（eta），控制步长
    # subsample: 每次迭代时使用的样本比例
    # colsample_bytree: 每次分割时使用的特征比例
    # reg_alpha: L1 正则化参数
    # reg_lambda: L2 正则化参数
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,  # 树的数量
        max_depth=6,  # 每个树的最大深度
        learning_rate=0.1,  # 学习率
        subsample=0.8,  # 样本比例
        colsample_bytree=0.8,  # 特征比例
        reg_alpha=0.1,  # L1 正则化
        reg_lambda=1.0,  # L2 正则化
        random_state=42,
        n_jobs=-1,  # 使用所有 CPU 核心进行并行化
        eval_metric='logloss'  # 评估指标
    )
    
    # 训练模型
    print("\n正在训练 XGBoost 分类器...")
    xgb_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = xgb_clf.predict(X_train)
    y_test_pred = xgb_clf.predict(X_test)
    
    # 获取预测概率（用于 ROC 曲线）
    y_test_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"\n模型性能：")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"ROC AUC 得分: {roc_auc:.4f}")
    
    # 打印详细的分类报告
    print(f"\n测试集分类报告：")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))
    
    # 绘制特征重要性（Top 15）
    feature_importance = xgb_clf.feature_importances_
    indices = np.argsort(feature_importance)[-15:]  # 取重要性最高的 15 个特征
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='coral')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('XGBoost 特征重要性排名（Top 15）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_xgboost_feature_importance.png'), dpi=150)
    print(f"\n特征重要性已保存至: ./images/3_xgboost_feature_importance.png")
    plt.close()
    
    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='随机分类器')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('XGBoost ROC 曲线')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_xgboost_roc_curve.png'), dpi=150)
    print(f"ROC 曲线已保存至: ./images/3_xgboost_roc_curve.png")
    plt.close()


def compare_boosting_methods():
    """
    比较 XGBoost 与其他 Boosting 方法的性能
    （需要 AdaBoost 和 Gradient Boosting）
    """
    if not XGB_INSTALLED:
        print("XGBoost 未安装，跳过方法对比")
        return
    
    print("\n" + "="*60)
    print("演示：不同 Boosting 方法的性能对比")
    print("="*60)
    
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建不同的 Boosting 分类器
    classifiers = {
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    print(f"\n正在训练和评估所有模型...")
    for name, clf in classifiers.items():
        print(f"  - {name}...", end=' ')
        
        # 训练
        clf.fit(X_train, y_train)
        
        # 评估
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        
        results[name] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        print(f"完成 (训练: {train_score:.4f}, 测试: {test_score:.4f})")
    
    # 绘制对比结果
    methods = list(results.keys())
    train_scores = [results[m]['train_accuracy'] for m in methods]
    test_scores = [results[m]['test_accuracy'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_scores, width, label='训练集准确率', color='steelblue')
    plt.bar(x + width/2, test_scores, width, label='测试集准确率', color='coral')
    
    plt.xlabel('Boosting 方法')
    plt.ylabel('准确率')
    plt.title('不同 Boosting 方法的性能对比')
    plt.xticks(x, methods)
    plt.legend()
    plt.ylim([0.85, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + 0.005, f'{train:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, test + 0.005, f'{test:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_boosting_methods_comparison.png'), dpi=150)
    print(f"\n方法对比已保存至: ./images/3_boosting_methods_comparison.png")
    plt.close()


def demo_xgboost_cv():
    """
    演示 XGBoost 的交叉验证
    """
    if not XGB_INSTALLED:
        return
    
    print("\n" + "="*60)
    print("演示：XGBoost 交叉验证")
    print("="*60)
    
    # 加载数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 只使用二分类（类别 0 和 1）
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    # 创建 XGBoost 分类器
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # 进行 5 折交叉验证
    print(f"\n正在进行 5 折交叉验证...")
    cv_scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')
    
    print(f"交叉验证得分: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 绘制交叉验证结果
    plt.figure(figsize=(10, 6))
    folds = np.arange(1, len(cv_scores) + 1)
    plt.bar(folds, cv_scores, color='steelblue', alpha=0.7, label='各折准确率')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2, label=f'平均值: {cv_scores.mean():.4f}')
    plt.fill_between(folds, cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(), 
                     alpha=0.2, color='r', label=f'标准差范围')
    
    plt.xlabel('折数')
    plt.ylabel('准确率')
    plt.title('XGBoost 5 折交叉验证结果')
    plt.xticks(folds)
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_xgboost_cv_results.png'), dpi=150)
    print(f"\n交叉验证结果已保存至: ./images/3_xgboost_cv_results.png")
    plt.close()


if __name__ == '__main__':
    if not XGB_INSTALLED:
        print("\n" + "="*60)
        print("XGBoost 未安装！")
        print("="*60)
        print("\n请运行以下命令安装 XGBoost：")
        print("  pip install xgboost")
        print("\n然后再运行此脚本。")
        print("\n"*60)
    else:
        # 运行所有演示
        demo_xgboost_classification()
        compare_boosting_methods()
        demo_xgboost_cv()
        
        print("\n" + "="*60)
        print("所有 XGBoost 演示完成！")
        print("="*60)
