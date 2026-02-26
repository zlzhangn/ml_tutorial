# -*- coding: utf-8 -*-
"""
随机森林 (Random Forest) 演示脚本

随机森林是 Bagging 的一个改进版本，同时在特征和样本两个维度进行随机采样。

核心创新：
1. 样本随机：Bootstrap 有放回抽样，不同的样本子集
2. 特征随机：每个分割点只考虑特征的随机子集，增加树之间的多样性
3. 结果聚合：多个树的投票/平均

主要优点：
- 比 Bagging 更快（特征采样减少计算量）
- 特征随机增加模型多样性
- 自动提供特征重要性
- 处理高维数据更好
- 天然支持并行化

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 images 目录
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def demo_random_forest_classification():
    """
    演示随机森林在分类任务中的应用
    """
    print("\n" + "="*60)
    print("随机森林演示：分类任务 - 乳腺癌数据集")
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
    
    # 创建随机森林分类器
    # n_estimators: 森林中树的数量
    # max_depth: 每个树的最大深度
    # min_samples_split: 分割内部节点所需的最小样本数
    # min_samples_leaf: 叶节点所需的最小样本数
    # max_features: 寻找最佳分割时要考虑的特征数量
    # bootstrap: 是否使用 Bootstrap 样本
    rf_clf = RandomForestClassifier(
        n_estimators=100,  # 100 个决策树
        max_depth=20,  # 每个树的最大深度
        min_samples_split=5,  # 分割所需最小样本数
        min_samples_leaf=2,  # 叶节点所需最小样本数
        max_features='sqrt',  # 考虑 sqrt(n_features) 个特征
        random_state=42,
        n_jobs=-1  # 并行化
    )
    
    # 训练模型
    print("\n正在训练随机森林分类器...")
    rf_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = rf_clf.predict(X_train)
    y_test_pred = rf_clf.predict(X_test)
    
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title('随机森林混淆矩阵 - 乳腺癌数据集')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_confusion_matrix.png'), dpi=150)
    print(f"\n混淆矩阵已保存至: ./images/2_random_forest_confusion_matrix.png")
    plt.close()
    
    # 绘制特征重要性
    feature_importance = rf_clf.feature_importances_
    indices = np.argsort(feature_importance)[-15:]  # 取重要性最高的 15 个特征
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='mediumseagreen')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('随机森林特征重要性排名（Top 15）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_feature_importance.png'), dpi=150)
    print(f"特征重要性已保存至: ./images/2_random_forest_feature_importance.png")
    plt.close()


def compare_n_estimators():
    """
    比较不同树的数量对随机森林性能的影响
    """
    print("\n" + "="*60)
    print("演示：不同树的数量对随机森林性能的影响")
    print("="*60)
    
    # 加载数据集
    data = load_wine()
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
        rf_clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        rf_clf.fit(X_train, y_train)
        
        train_score = rf_clf.score(X_train, y_train)
        test_score = rf_clf.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=6)
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('随机森林：不同树数量的性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_n_estimators.png'), dpi=150)
    print(f"学习曲线已保存至: ./images/2_random_forest_n_estimators.png")
    plt.close()


def demo_oob_score():
    """
    演示随机森林的 OOB (Out-of-Bag) 评分
    """
    print("\n" + "="*60)
    print("演示：随机森林的 OOB (Out-of-Bag) 评分")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建使用 OOB 评分的随机森林
    # oob_score=True 会自动计算 OOB 评分
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        oob_score=True,  # 启用 OOB 评分
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    print("\n正在训练随机森林（启用 OOB 评分）...")
    rf_clf.fit(X_train, y_train)
    print("模型训练完成！")
    
    # OOB 评分是对模型泛化性能的无偏估计
    oob_score = rf_clf.oob_score_
    train_score = rf_clf.score(X_train, y_train)
    test_score = rf_clf.score(X_test, y_test)
    
    print(f"\n模型评分：")
    print(f"OOB 评分（无偏估计）: {oob_score:.4f}")
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 绘制三个评分的对比
    scores = [train_score, oob_score, test_score]
    names = ['训练集准确率', 'OOB 评分', '测试集准确率']
    
    plt.figure(figsize=(10, 6))
    colors = ['steelblue', 'coral', 'mediumseagreen']
    bars = plt.bar(names, scores, color=colors, alpha=0.7, edgecolor='black')
    
    # 在柱子上显示数值
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('准确率')
    plt.title('随机森林：OOB 评分与测试集准确率对比')
    plt.ylim([0.9, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_oob_score.png'), dpi=150)
    print(f"\nOOB 评分对比已保存至: ./images/2_random_forest_oob_score.png")
    plt.close()


def demo_max_features():
    """
    比较不同的 max_features 参数对随机森林性能的影响
    """
    print("\n" + "="*60)
    print("演示：max_features 参数对性能的影响")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 定义不同的 max_features
    max_features_options = [
        'sqrt',      # sqrt(n_features)
        'log2',      # log2(n_features)
        None,        # 所有特征
    ]
    
    print(f"\n正在测试不同的 max_features...")
    results = {}
    
    for max_feat in max_features_options:
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1
        )
        rf_clf.fit(X_train, y_train)
        
        train_score = rf_clf.score(X_train, y_train)
        test_score = rf_clf.score(X_test, y_test)
        
        feat_name = max_feat if max_feat is not None else '全部特征'
        results[feat_name] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        print(f"  {feat_name}: 训练={train_score:.4f}, 测试={test_score:.4f}")
    
    # 绘制对比结果
    names = list(results.keys())
    train_scores = [results[n]['train_accuracy'] for n in names]
    test_scores = [results[n]['test_accuracy'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_scores, width, label='训练集准确率', color='steelblue')
    plt.bar(x + width/2, test_scores, width, label='测试集准确率', color='coral')
    
    plt.xlabel('max_features 参数')
    plt.ylabel('准确率')
    plt.title('随机森林：max_features 参数的影响')
    plt.xticks(x, names)
    plt.legend()
    plt.ylim([0.9, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + 0.002, f'{train:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, test + 0.002, f'{test:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_max_features.png'), dpi=150)
    print(f"\nmax_features 对比已保存至: ./images/2_random_forest_max_features.png")
    plt.close()


def demo_random_forest_regression():
    """
    演示随机森林在回归任务中的应用
    """
    print("\n" + "="*60)
    print("演示：随机森林回归任务")
    print("="*60)
    
    # 生成合成回归数据集
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=300, n_features=20, noise=15, random_state=42)
    
    print(f"\n数据集信息：")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建随机森林回归器
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    print("\n正在训练随机森林回归器...")
    rf_reg.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 进行预测
    y_train_pred = rf_reg.predict(X_train)
    y_test_pred = rf_reg.predict(X_test)
    
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
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='mediumseagreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'测试集预测结果 (R²={test_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_random_forest_regression.png'), dpi=150)
    print(f"\n预测结果已保存至: ./images/2_random_forest_regression.png")
    plt.close()


if __name__ == '__main__':
    # 运行所有演示
    demo_random_forest_classification()
    compare_n_estimators()
    demo_oob_score()
    demo_max_features()
    demo_random_forest_regression()
    
    print("\n" + "="*60)
    print("所有随机森林演示完成！")
    print("="*60)
