# -*- coding: utf-8 -*-
"""
Extra Trees (极端随机树) 演示脚本

Extra Trees (Extremely Randomized Trees) 是随机森林的一个变种。

主要区别：
1. 随机森林：在所有特征上寻找最优分割点
2. Extra Trees：随机选择分割点，减少计算复杂度

特点：
- 训练速度更快（随机分割，不需要搜索最优点）
- 方差较大（随机性更强）
- 偏差较小（集成多个不同的树）
- 适合大规模数据集
- 计算复杂度：O(log n)，而随机森林是 O(n log n)

适用场景：
- 大规模数据集
- 需要快速训练
- 特征维度高
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
import os
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def demo_extra_trees_classification():
    """
    演示 Extra Trees 在分类任务中的应用
    """
    print("\n" + "="*60)
    print("Extra Trees 演示：分类任务 - 乳腺癌数据集")
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
    
    # 创建 Extra Trees 分类器
    # n_estimators: 森林中树的数量
    # max_depth: 树的最大深度
    # max_features: 考虑的特征数（'sqrt' 表示 sqrt(n_features)）
    # bootstrap: 是否使用 Bootstrap 样本
    # n_jobs: 并行核心数
    et_clf = ExtraTreesClassifier(
        n_estimators=100,  # 100 个树
        max_depth=20,  # 最大深度
        max_features='sqrt',  # 每次分割考虑 sqrt(n_features) 个特征
        bootstrap=True,  # 使用 Bootstrap
        random_state=42,
        n_jobs=-1  # 并行化
    )
    
    # 训练模型
    print("\n正在训练 Extra Trees 分类器...")
    start_time = time.time()
    et_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"模型训练完成！耗时: {train_time:.4f} 秒")
    
    # 进行预测
    y_train_pred = et_clf.predict(X_train)
    y_test_pred = et_clf.predict(X_test)
    
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title('Extra Trees 混淆矩阵 - 乳腺癌数据集')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_extra_trees_confusion_matrix.png'), dpi=150)
    print(f"\n混淆矩阵已保存至: ./images/3_extra_trees_confusion_matrix.png")
    plt.close()
    
    # 绘制特征重要性
    feature_importance = et_clf.feature_importances_
    indices = np.argsort(feature_importance)[-15:]  # 取重要性最高的 15 个特征
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='darkorange')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('Extra Trees 特征重要性排名（Top 15）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_extra_trees_feature_importance.png'), dpi=150)
    print(f"特征重要性已保存至: ./images/3_extra_trees_feature_importance.png")
    plt.close()


def compare_split_strategies():
    """
    比较随机森林和 Extra Trees 在分割策略上的差异
    """
    print("\n" + "="*60)
    print("演示：随机森林 vs Extra Trees 的速度对比")
    print("="*60)
    
    # 生成大规模数据集
    print("\n生成大规模数据集...")
    X, y = make_classification(
        n_samples=10000,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"数据集大小：{X_train.shape[0]} 训练样本，{X_train.shape[1]} 个特征")
    
    # 导入随机森林进行对比
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=50,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    print(f"\n正在训练和比较...")
    for name, model in models.items():
        # 记录训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 评估
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        results[name] = {
            'train_time': train_time,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        print(f"{name:15} - 训练时间: {train_time:.4f}s, 准确率: {test_score:.4f}")
    
    # 绘制对比结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    names = list(results.keys())
    
    # 训练时间
    train_times = [results[n]['train_time'] for n in names]
    axes[0].bar(names, train_times, color=['steelblue', 'coral'], alpha=0.7)
    axes[0].set_ylabel('训练时间（秒）')
    axes[0].set_title('训练时间对比')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (name, time_val) in enumerate(zip(names, train_times)):
        axes[0].text(i, time_val + 0.01, f'{time_val:.4f}s', ha='center', va='bottom')
    
    # 训练准确率
    train_accs = [results[n]['train_accuracy'] for n in names]
    axes[1].bar(names, train_accs, color=['steelblue', 'coral'], alpha=0.7)
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练集准确率对比')
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (name, acc) in enumerate(zip(names, train_accs)):
        axes[1].text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom')
    
    # 测试准确率
    test_accs = [results[n]['test_accuracy'] for n in names]
    axes[2].bar(names, test_accs, color=['steelblue', 'coral'], alpha=0.7)
    axes[2].set_ylabel('准确率')
    axes[2].set_title('测试集准确率对比')
    axes[2].set_ylim([0.9, 1.0])
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (name, acc) in enumerate(zip(names, test_accs)):
        axes[2].text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_random_forest_vs_extra_trees.png'), dpi=150)
    print(f"\n对比结果已保存至: ./images/3_random_forest_vs_extra_trees.png")
    plt.close()


def demo_extra_trees_regression():
    """
    演示 Extra Trees 在回归任务中的应用
    """
    print("\n" + "="*60)
    print("演示：Extra Trees 回归任务")
    print("="*60)
    
    # 生成合成回归数据集
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        noise=15,
        random_state=42
    )
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建 Extra Trees 回归器
    et_reg = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    print("\n正在训练 Extra Trees 回归器...")
    et_reg.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = et_reg.predict(X_train)
    y_test_pred = et_reg.predict(X_test)
    
    # 计算性能指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n模型性能：")
    print(f"训练集 MSE: {train_mse:.4f}, R² 得分: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R² 得分: {test_r2:.4f}")
    
    # 绘制预测结果
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
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='darkorange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'测试集预测结果 (R²={test_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_extra_trees_regression.png'), dpi=150)
    print(f"\n预测结果已保存至: ./images/3_extra_trees_regression.png")
    plt.close()


def compare_n_estimators():
    """
    比较不同树的数量对 Extra Trees 性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同树的数量对 Extra Trees 性能的影响")
    print("="*60)
    
    # 加载数据集
    data = load_digits()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试不同的 n_estimators
    n_estimators_list = np.arange(10, 201, 20)
    train_scores = []
    test_scores = []
    
    print(f"\n正在测试不同数量的树 (10 到 200)...")
    for n_est in n_estimators_list:
        et_clf = ExtraTreesClassifier(
            n_estimators=n_est,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        et_clf.fit(X_train, y_train)
        
        train_score = et_clf.score(X_train, y_train)
        test_score = et_clf.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=6)
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('Extra Trees：不同树数量的性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_extra_trees_n_estimators.png'), dpi=150)
    print(f"学习曲线已保存至: ./images/3_extra_trees_n_estimators.png")
    plt.close()


if __name__ == '__main__':
    # 运行所有演示
    demo_extra_trees_classification()
    compare_split_strategies()
    demo_extra_trees_regression()
    compare_n_estimators()
    
    print("\n" + "="*60)
    print("所有 Extra Trees 演示完成！")
    print("="*60)
