# -*- coding: utf-8 -*-
"""
Boosting 方法综合对比脚本

本脚本展示了四种主要的 Boosting 方法的综合对比，
包括性能、训练时间、特征重要性等多个方面。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import time
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = Path(__file__).parent / 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 尝试导入 xgboost 和 lightgbm
try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("提示: XGBoost 未安装，对比中将不包含 XGBoost")

try:
    import lightgbm as lgb
    LGB_INSTALLED = True
except ImportError:
    LGB_INSTALLED = False
    print("提示: LightGBM 未安装，对比中将不包含 LightGBM")


def comprehensive_comparison():
    """
    进行四种 Boosting 方法的综合对比
    """
    print("\n" + "="*70)
    print("Boosting 方法综合对比")
    print("="*70)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集：乳腺癌数据集")
    print(f"训练集大小：{X_train.shape}")
    print(f"测试集大小：{X_test.shape}")
    
    # 定义 Boosting 分类器
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
            subsample=0.8,
            random_state=42
        )
    }
    
    # 添加 XGBoost（如果已安装）
    if XGB_INSTALLED:
        classifiers['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # 添加 LightGBM（如果已安装）
    if LGB_INSTALLED:
        classifiers['LightGBM'] = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            verbose=-1,
            random_state=42,
            n_jobs=-1
        )
    
    # 训练和评估所有模型
    results = {}
    
    print(f"\n{'方法':<20} {'训练时间':<12} {'测试准确率':<12} {'ROC AUC':<12} {'精确率':<12} {'召回率':<12} {'F1 分数':<12}")
    print("-" * 100)
    
    for name, clf in classifiers.items():
        # 记录训练时间
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 进行预测
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # 计算性能指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        results[name] = {
            'train_time': train_time,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_importances': clf.feature_importances_,
            'clf': clf
        }
        
        print(f"{name:<20} {train_time:<12.4f} {test_accuracy:<12.4f} {roc_auc:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    return results, classifiers, X_train, X_test, y_train, y_test, data


def plot_comprehensive_comparison(results, data):
    """
    绘制综合对比图表
    """
    print("\n正在生成对比图表...")
    
    methods = list(results.keys())
    
    # 1. 性能指标对比
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = {
        '测试准确率': 'test_accuracy',
        'ROC AUC': 'roc_auc',
        '精确率': 'precision',
        '召回率': 'recall',
        'F1 分数': 'f1',
        '训练时间': 'train_time'
    }
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
    
    for idx, (metric_name, metric_key) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        
        if metric_key == 'train_time':
            values = [results[m][metric_key] for m in methods]
            bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.7)
            ax.set_ylabel('时间（秒）')
        else:
            values = [results[m][metric_key] for m in methods]
            bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.7)
            ax.set_ylabel('得分')
            ax.set_ylim([0, 1.1])
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Boosting 方法性能对比', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '5_comprehensive_comparison_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"性能对比图已保存至: ./images/5_comprehensive_comparison_metrics.png")
    plt.close()
    
    # 2. 特征重要性对比（Top 10）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        if idx < 4:  # 最多显示 4 个方法
            ax = axes[idx]
            
            feature_importance = results[method]['feature_importances']
            indices = np.argsort(feature_importance)[-10:]  # 取 Top 10
            
            ax.barh(range(len(indices)), feature_importance[indices], color=colors[idx], alpha=0.7)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(np.array(data.feature_names)[indices], fontsize=9)
            ax.set_xlabel('特征重要性')
            ax.set_title(f'{method} - 特征重要性 (Top 10)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    # 隐藏未使用的子图
    for idx in range(len(methods), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '5_feature_importance_comparison.png'), dpi=150)
    print(f"特征重要性对比图已保存至: ./images/5_feature_importance_comparison.png")
    plt.close()
    
    # 3. 雷达图对比
    try:
        from math import pi
        
        categories = ['准确率', 'ROC AUC', '精确率', '召回率', 'F1分数']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors_radar = ['steelblue', 'coral', 'mediumseagreen', 'gold']
        
        for idx, method in enumerate(methods):
            values = [
                results[method]['test_accuracy'],
                results[method]['roc_auc'],
                results[method]['precision'],
                results[method]['recall'],
                results[method]['f1']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors_radar[idx], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('Boosting 方法多维度对比雷达图', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, '5_radar_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"雷达图对比已保存至: ./images/5_radar_comparison.png")
        plt.close()
    except Exception as e:
        print(f"雷达图生成失败: {e}")


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("Boosting 集成学习方法综合演示")
    print("="*70)
    
    # 检查依赖
    print(f"\n依赖检查：")
    print(f"  - scikit-learn: ✓")
    print(f"  - XGBoost: {'✓' if XGB_INSTALLED else '✗'}")
    print(f"  - LightGBM: {'✓' if LGB_INSTALLED else '✗'}")
    
    # 进行综合对比
    results, classifiers, X_train, X_test, y_train, y_test, data = comprehensive_comparison()
    
    # 生成对比图表
    plot_comprehensive_comparison(results, data)
    
    print("\n" + "="*70)
    print("综合对比完成！所有图表已保存到 images/ 目录")
    print("="*70)
    
    # 总结
    print("\n总结与建议：")
    print("-" * 70)
    
    best_accuracy = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_speed = min(results.items(), key=lambda x: x[1]['train_time'])
    best_auc = max(results.items(), key=lambda x: x[1]['roc_auc'])
    
    print(f"✓ 最高准确率: {best_accuracy[0]} ({best_accuracy[1]['test_accuracy']:.4f})")
    print(f"✓ 最快训练速度: {best_speed[0]} ({best_speed[1]['train_time']:.4f}s)")
    print(f"✓ 最高 ROC AUC: {best_auc[0]} ({best_auc[1]['roc_auc']:.4f})")
    
    print("\n建议使用场景：")
    print("  • AdaBoost: 简单演示、教学、小规模数据")
    print("  • Gradient Boosting: 中等规模数据、通用场景")
    if XGB_INSTALLED:
        print("  • XGBoost: 竞赛、大规模数据、需要高精度")
    if LGB_INSTALLED:
        print("  • LightGBM: 大规模数据、实时预测、内存限制")
    
    print("="*70)


if __name__ == '__main__':
    main()
