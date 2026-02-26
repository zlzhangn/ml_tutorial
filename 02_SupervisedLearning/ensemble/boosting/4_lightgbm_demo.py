# -*- coding: utf-8 -*-
"""
LightGBM 演示脚本

LightGBM (Light Gradient Boosting Machine) 是微软开发的一个快速、分布式的梯度提升框架，
采用叶子生长策略，速度快，内存占用少。

特点：
- 训练速度快（相比 XGBoost）
- 内存占用小
- 支持分类特征处理
- 支持并行和 GPU 学习
- 更好的大型数据集处理能力
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
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

# 尝试导入 lightgbm，如果未安装则显示提示信息
try:
    import lightgbm as lgb
    LGB_INSTALLED = True
except ImportError:
    LGB_INSTALLED = False
    print("警告: LightGBM 未安装，请运行: pip install lightgbm")


def demo_lightgbm_classification():
    """
    演示 LightGBM 在分类任务中的应用
    """
    if not LGB_INSTALLED:
        print("LightGBM 未安装，跳过 LightGBM 分类演示")
        return
    
    print("\n" + "="*60)
    print("LightGBM 演示：分类任务 - 乳腺癌数据集")
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
    
    # 创建 LightGBM 分类器
    # num_leaves: 最大叶子数（LightGBM 使用叶子数而不是深度）
    # learning_rate: 学习率
    # n_estimators: 树的数量
    # feature_fraction: 每次分割时使用的特征比例
    # bagging_fraction: 每次迭代时使用的样本比例
    # bagging_freq: 每 N 次迭代进行一次 bagging
    lgb_clf = lgb.LGBMClassifier(
        num_leaves=31,  # 最大叶子数
        learning_rate=0.05,  # 学习率
        n_estimators=100,  # 树的数量
        feature_fraction=0.8,  # 特征比例
        bagging_fraction=0.8,  # 样本比例
        bagging_freq=5,  # 每 5 次迭代进行 bagging
        verbose=-1,  # 不输出详细日志
        random_state=42,
        n_jobs=-1  # 并行化
    )
    
    # 训练模型
    print("\n正在训练 LightGBM 分类器...")
    start_time = time.time()
    lgb_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"模型训练完成！耗时: {train_time:.4f} 秒")
    
    # 进行预测
    y_train_pred = lgb_clf.predict(X_train)
    y_test_pred = lgb_clf.predict(X_test)
    
    # 获取预测概率
    y_test_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
    
    # 计算性能指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"\n模型性能：")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"ROC AUC 得分: {roc_auc:.4f}")
    print(f"训练时间: {train_time:.4f} 秒")
    
    # 打印详细的分类报告
    print(f"\n测试集分类报告：")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))
    
    # 绘制特征重要性（Top 15）
    feature_importance = lgb_clf.feature_importances_
    indices = np.argsort(feature_importance)[-15:]  # 取重要性最高的 15 个特征
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='mediumseagreen')
    plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
    plt.xlabel('特征重要性')
    plt.title('LightGBM 特征重要性排名（Top 15）')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_lightgbm_feature_importance.png'), dpi=150)
    print(f"\n特征重要性已保存至: ./images/4_lightgbm_feature_importance.png")
    plt.close()


def demo_lightgbm_large_dataset():
    """
    演示 LightGBM 在大规模数据集上的性能优势
    """
    if not LGB_INSTALLED:
        return
    
    print("\n" + "="*60)
    print("演示：LightGBM 在大规模数据集上的性能")
    print("="*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
    
    # 生成大规模数据集
    print("\n正在生成大规模数据集...")
    X_large, y_large = make_classification(
        n_samples=10000,  # 10000 个样本
        n_features=100,  # 100 个特征
        n_informative=50,
        n_redundant=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_large, y_large, test_size=0.2, random_state=42
    )
    
    print(f"数据集大小: {X_train.shape[0]} 训练样本，{X_test.shape[0]} 测试样本，{X_train.shape[1]} 个特征")
    
    # 定义模型
    models = {
        'LightGBM': lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            verbose=-1,
            n_jobs=-1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    # 如果 XGBoost 可用，添加到对比中
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    results = {}
    
    print(f"\n正在训练和评估所有模型...")
    for name, model in models.items():
        print(f"  - {name}...", end=' ')
        
        # 记录训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 评估
        test_score = model.score(X_test, y_test)
        
        results[name] = {
            'train_time': train_time,
            'test_accuracy': test_score
        }
        
        print(f"完成 (耗时: {train_time:.4f}秒, 准确率: {test_score:.4f})")
    
    # 绘制结果对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练时间对比
    names = list(results.keys())
    train_times = [results[n]['train_time'] for n in names]
    
    axes[0].bar(names, train_times, color=['mediumseagreen', 'steelblue', 'coral'][:len(names)])
    axes[0].set_ylabel('训练时间（秒）')
    axes[0].set_title('不同模型的训练时间对比')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (name, time_val) in enumerate(zip(names, train_times)):
        axes[0].text(i, time_val + 0.1, f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # 准确率对比
    accuracies = [results[n]['test_accuracy'] for n in names]
    
    axes[1].bar(names, accuracies, color=['mediumseagreen', 'steelblue', 'coral'][:len(names)])
    axes[1].set_ylabel('测试集准确率')
    axes[1].set_title('不同模型的准确率对比')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (name, acc) in enumerate(zip(names, accuracies)):
        axes[1].text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_lightgbm_large_dataset_comparison.png'), dpi=150)
    print(f"\n大规模数据集对比已保存至: ./images/4_lightgbm_large_dataset_comparison.png")
    plt.close()


def demo_lightgbm_early_stopping():
    """
    演示 LightGBM 的早停（Early Stopping）功能
    """
    if not LGB_INSTALLED:
        return
    
    print("\n" + "="*60)
    print("演示：LightGBM 早停功能")
    print("="*60)
    
    # 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 进一步分割训练集为训练和验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 创建 LightGBM 分类器（带早停）
    print("\n正在训练 LightGBM 模型（启用早停）...")
    lgb_clf = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,  # 设置很大的迭代次数
        verbose=-1,
        random_state=42
    )
    
    # 使用早停训练
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],  # 验证集
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # 如果验证集性能 50 轮没有改进，则停止
            lgb.log_evaluation(period=0)  # 不输出日志
        ]
    )
    
    # 获取迭代次数
    best_iteration = lgb_clf.best_iteration_
    
    # 评估
    test_accuracy = lgb_clf.score(X_test, y_test)
    
    print(f"最佳迭代次数: {best_iteration}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 绘制训练过程
    # 注意：需要重新训练来获取详细的训练历史
    train_scores = []
    val_scores = []
    
    for n_est in range(10, best_iteration + 1, 10):
        lgb_temp = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=n_est,
            verbose=-1,
            random_state=42
        )
        lgb_temp.fit(X_tr, y_tr)
        train_scores.append(lgb_temp.score(X_tr, y_tr))
        val_scores.append(lgb_temp.score(X_val, y_val))
    
    plt.figure(figsize=(10, 6))
    iterations = range(10, best_iteration + 1, 10)
    plt.plot(iterations, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
    plt.plot(iterations, val_scores, 's-', label='验证集准确率', linewidth=2, markersize=6)
    plt.axvline(x=best_iteration, color='r', linestyle='--', label=f'最佳迭代: {best_iteration}')
    
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.title('LightGBM 早停演示')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_lightgbm_early_stopping.png'), dpi=150)
    print(f"\n早停过程已保存至: ./images/4_lightgbm_early_stopping.png")
    plt.close()


if __name__ == '__main__':
    if not LGB_INSTALLED:
        print("\n" + "="*60)
        print("LightGBM 未安装！")
        print("="*60)
        print("\n请运行以下命令安装 LightGBM：")
        print("  pip install lightgbm")
        print("\n然后再运行此脚本。")
        print("="*60)
    else:
        # 运行所有演示
        demo_lightgbm_classification()
        demo_lightgbm_large_dataset()
        demo_lightgbm_early_stopping()
        
        print("\n" + "="*60)
        print("所有 LightGBM 演示完成！")
        print("="*60)
