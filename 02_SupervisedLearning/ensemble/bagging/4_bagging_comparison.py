# -*- coding: utf-8 -*-
"""
Bagging 方法综合对比脚本

本脚本展示了四种主要的 Bagging 方法的全面对比：
1. Bagging （基础 Bagging）
2. Random Forest （随机森林）
3. Extra Trees （极端随机树）
4. Pasting （不使用 Bootstrap 的 Bagging）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def comprehensive_comparison():
    """
    进行四种 Bagging 方法的综合对比
    """
    print("\n" + "="*70)
    print("Bagging 方法综合对比")
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
    
    # 定义四种 Bagging 方法
    classifiers = {
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
            n_estimators=50,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=50,
            max_depth=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Pasting': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
            n_estimators=50,
            bootstrap=False,  # Pasting：不使用 Bootstrap
            random_state=42,
            n_jobs=-1
        )
    }
    
    # 训练和评估所有模型
    results = {}
    
    print(f"\n{'方法':<20} {'训练时间':<12} {'测试准确率':<12} {'精确率':<12} {'召回率':<12} {'F1 分数':<12}")
    print("-" * 100)
    
    for name, clf in classifiers.items():
        # 记录训练时间
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 进行预测
        y_test_pred = clf.predict(X_test)
        
        # 计算性能指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        results[name] = {
            'train_time': train_time,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"{name:<20} {train_time:<12.4f} {test_accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    return results


def plot_comprehensive_comparison(results):
    """
    绘制综合对比图表
    """
    print("\n正在生成对比图表...")
    
    methods = list(results.keys())
    
    # 1. 性能指标对比
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = {
        '测试准确率': 'test_accuracy',
        '精确率': 'precision',
        '召回率': 'recall',
        'F1 分数': 'f1',
        '训练时间': 'train_time',
    }
    
    colors = ['steelblue', 'mediumseagreen', 'darkorange', 'coral']
    
    for idx, (metric_name, metric_key) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        
        if metric_key == 'train_time':
            values = [results[m][metric_key] for m in methods]
            bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('时间（秒）')
        else:
            values = [results[m][metric_key] for m in methods]
            bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('得分')
            ax.set_ylim([0.9, 1.0])
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 隐藏最后一个子图
    axes[1, 2].set_visible(False)
    
    plt.suptitle('Bagging 方法性能对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_bagging_methods_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"性能对比图已保存至: ./images/4_bagging_methods_comparison.png")
    plt.close()
    
    # 2. 方法特性对比表（用柱状图表示）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精确率 vs 召回率
    precision_values = [results[m]['precision'] for m in methods]
    recall_values = [results[m]['recall'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0].bar(x - width/2, precision_values, width, label='精确率', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, recall_values, width, label='召回率', color='coral', alpha=0.8)
    
    axes[0].set_xlabel('Bagging 方法')
    axes[0].set_ylabel('得分')
    axes[0].set_title('精确率 vs 召回率对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim([0.9, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 准确率 vs 训练时间
    accuracy_values = [results[m]['test_accuracy'] for m in methods]
    time_values = [results[m]['train_time'] for m in methods]
    
    axes[1].scatter(time_values, accuracy_values, s=200, alpha=0.6, c=colors, edgecolors='black', linewidth=2)
    
    for i, method in enumerate(methods):
        axes[1].annotate(method, (time_values[i], accuracy_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[1].set_xlabel('训练时间（秒）')
    axes[1].set_ylabel('测试准确率')
    axes[1].set_title('准确率 vs 训练时间')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_bagging_accuracy_vs_time.png'), dpi=150)
    print(f"准确率 vs 训练时间已保存至: ./images/4_bagging_accuracy_vs_time.png")
    plt.close()
    
    # 3. 雷达图对比
    try:
        from math import pi
        
        categories = ['准确率', '精确率', '召回率', 'F1分数']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors_radar = ['steelblue', 'mediumseagreen', 'darkorange', 'coral']
        
        for idx, method in enumerate(methods):
            values = [
                results[method]['test_accuracy'],
                results[method]['precision'],
                results[method]['recall'],
                results[method]['f1']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors_radar[idx], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('Bagging 方法多维度对比雷达图', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, '4_bagging_radar_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"雷达图对比已保存至: ./images/4_bagging_radar_comparison.png")
        plt.close()
    except Exception as e:
        print(f"雷达图生成失败: {e}")


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("Bagging 集成学习方法综合演示")
    print("="*70)
    
    # 进行综合对比
    results = comprehensive_comparison()
    
    # 生成对比图表
    plot_comprehensive_comparison(results)
    
    print("\n" + "="*70)
    print("综合对比完成！所有图表已保存到 images/ 目录")
    print("="*70)
    
    # 总结
    print("\n总结与建议：")
    print("-" * 70)
    
    best_accuracy = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_speed = min(results.items(), key=lambda x: x[1]['train_time'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    
    print(f"✓ 最高准确率: {best_accuracy[0]} ({best_accuracy[1]['test_accuracy']:.4f})")
    print(f"✓ 最快训练速度: {best_speed[0]} ({best_speed[1]['train_time']:.4f}s)")
    print(f"✓ 最高 F1 分数: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    
    print("\n方法选择建议：")
    print("  • Bagging：灵活，可使用任何基础分类器")
    print("  • Random Forest：平衡性好，特征随机增加多样性")
    print("  • Extra Trees：训练快，适合大数据集")
    print("  • Pasting：内存占用少，适合内存有限的场景")
    
    print("="*70)


if __name__ == '__main__':
    main()
