"""
数据平衡处理 - 生产环境数据预处理技术
Imbalanced Data Handling for Production Environment

本文件演示了处理不平衡数据的各种方法，适用于生产环境
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, f1_score,
    balanced_accuracy_score
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, 
    NearMiss
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_imbalanced_dataset(imbalance_ratio=0.1):
    """创建不平衡数据集"""
    print("="*60)
    print("创建不平衡数据集")
    print("="*60)
    
    # 创建不平衡数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[1-imbalance_ratio, imbalance_ratio],
        flip_y=0.01,
        random_state=42
    )
    
    # 统计类别分布
    class_counts = Counter(y)
    
    print(f"\n类别分布:")
    print(f"  类别 0 (多数类): {class_counts[0]} 样本 ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"  类别 1 (少数类): {class_counts[1]} 样本 ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"  不平衡比率: {class_counts[0]/class_counts[1]:.2f}:1")
    
    # 可视化类别分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 柱状图
    axes[0].bar(['类别 0 (多数)', '类别 1 (少数)'], 
               [class_counts[0], class_counts[1]], 
               color=['blue', 'red'], alpha=0.7)
    axes[0].set_ylabel('样本数量')
    axes[0].set_title('类别分布')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 饼图
    axes[1].pie([class_counts[0], class_counts[1]], 
               labels=['类别 0 (多数)', '类别 1 (少数)'],
               autopct='%1.1f%%', colors=['blue', 'red'], alpha=0.7)
    axes[1].set_title('类别比例')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'imbalanced_data.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: imbalanced_data.png")
    plt.close()
    
    return X, y


# ============================================================
# 1. 基线模型（不处理不平衡）
# ============================================================

def baseline_model(X, y):
    """不处理不平衡数据的基线模型"""
    print("\n" + "="*60)
    print("1. 基线模型（不处理不平衡）")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n评估指标:")
    print(f"  准确率 (Accuracy): {model.score(X_test, y_test):.4f}")
    print(f"  F1分数 (少数类): {f1_score(y_test, y_pred):.4f}")
    print(f"  平衡准确率: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n⚠️  问题: 模型可能倾向于预测多数类，少数类召回率低")
    
    return model, X_train, X_test, y_train, y_test


# ============================================================
# 2. 过采样方法 - Random OverSampling
# ============================================================

def random_oversampling_demo(X, y):
    """随机过采样"""
    print("\n" + "="*60)
    print("2. Random OverSampling (随机过采样)")
    print("="*60)
    print("原理: 随机复制少数类样本")
    print("优点: 简单、快速")
    print("缺点: 可能导致过拟合")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 3. 过采样方法 - SMOTE
# ============================================================

def smote_demo(X, y):
    """SMOTE (Synthetic Minority Over-sampling Technique)"""
    print("\n" + "="*60)
    print("3. SMOTE (合成少数类过采样)")
    print("="*60)
    print("原理: 在少数类样本之间插值生成新样本")
    print("优点: 生成新样本，降低过拟合风险")
    print("缺点: 可能生成噪声样本")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    print("\n评估指标:")
    print(f"  F1分数: {f1_score(y_test, y_pred):.4f}")
    print(f"  平衡准确率: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, X_train_res, y_train_res


# ============================================================
# 4. 过采样方法 - BorderlineSMOTE
# ============================================================

def borderline_smote_demo(X, y):
    """BorderlineSMOTE: 只对边界样本应用SMOTE"""
    print("\n" + "="*60)
    print("4. BorderlineSMOTE (边界SMOTE)")
    print("="*60)
    print("原理: 只对靠近决策边界的少数类样本应用SMOTE")
    print("优点: 聚焦于难分样本，效果常优于标准SMOTE")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用BorderlineSMOTE
    bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_train_res, y_train_res = bsmote.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 5. 过采样方法 - ADASYN
# ============================================================

def adasyn_demo(X, y):
    """ADASYN (Adaptive Synthetic Sampling)"""
    print("\n" + "="*60)
    print("5. ADASYN (自适应合成采样)")
    print("="*60)
    print("原理: 根据学习难度自适应生成样本")
    print("优点: 为难分样本生成更多合成样本")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用ADASYN
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 6. 欠采样方法 - Random UnderSampling
# ============================================================

def random_undersampling_demo(X, y):
    """随机欠采样"""
    print("\n" + "="*60)
    print("6. Random UnderSampling (随机欠采样)")
    print("="*60)
    print("原理: 随机删除多数类样本")
    print("优点: 快速、减少数据量")
    print("缺点: 可能丢失重要信息")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 7. 欠采样方法 - Tomek Links
# ============================================================

def tomek_links_demo(X, y):
    """Tomek Links: 移除边界模糊样本"""
    print("\n" + "="*60)
    print("7. Tomek Links (移除Tomek链接)")
    print("="*60)
    print("原理: 移除不同类的最近邻对中的多数类样本")
    print("优点: 清理决策边界")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用TomekLinks
    tomek = TomekLinks()
    X_train_res, y_train_res = tomek.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    print(f"移除样本数: {len(y_train) - len(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 8. 混合方法 - SMOTEENN
# ============================================================

def smoteenn_demo(X, y):
    """SMOTEENN: SMOTE + ENN (Edited Nearest Neighbours)"""
    print("\n" + "="*60)
    print("8. SMOTEENN (SMOTE + ENN)")
    print("="*60)
    print("原理: 先用SMOTE过采样，再用ENN清理噪声")
    print("优点: 结合过采样和欠采样的优点")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 9. 混合方法 - SMOTETomek
# ============================================================

def smotetomek_demo(X, y):
    """SMOTETomek: SMOTE + Tomek Links"""
    print("\n" + "="*60)
    print("9. SMOTETomek (SMOTE + Tomek Links)")
    print("="*60)
    print("原理: 先用SMOTE过采样，再用Tomek清理边界")
    print("优点: 平衡数据并清理决策边界")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用SMOTETomek
    smotetomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smotetomek.fit_resample(X_train, y_train)
    
    print(f"\n重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_train_res)}")
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    return model, X_train_res, y_train_res


# ============================================================
# 10. 集成方法 - BalancedRandomForest
# ============================================================

def balanced_random_forest_demo(X, y):
    """BalancedRandomForest: 平衡随机森林"""
    print("\n" + "="*60)
    print("10. BalancedRandomForest (平衡随机森林)")
    print("="*60)
    print("原理: 每棵树训练时自动平衡类别")
    print("优点: 无需额外采样，集成学习")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练BalancedRandomForest
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        sampling_strategy='all'
    )
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    print("\n评估指标:")
    print(f"  F1分数: {f1_score(y_test, y_pred):.4f}")
    print(f"  平衡准确率: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model


# ============================================================
# 11. 类别权重调整
# ============================================================

def class_weight_demo(X, y):
    """调整类别权重"""
    print("\n" + "="*60)
    print("11. Class Weight (类别权重调整)")
    print("="*60)
    print("原理: 给少数类更高的权重")
    print("优点: 不改变数据集大小，适用于大多数模型")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 计算类别权重
    class_counts = Counter(y_train)
    total = len(y_train)
    class_weights = {0: total / (2 * class_counts[0]), 
                    1: total / (2 * class_counts[1])}
    
    print(f"\n类别权重: {class_weights}")
    
    # 使用class_weight='balanced'（自动计算）
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # 自动计算平衡权重
    )
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    print("\n评估指标:")
    print(f"  F1分数: {f1_score(y_test, y_pred):.4f}")
    print(f"  平衡准确率: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model


# ============================================================
# 12. 不同方法的比较
# ============================================================

def compare_methods(X, y):
    """比较所有方法"""
    print("\n" + "="*60)
    print("12. 不同方法的性能比较")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    methods = [
        ('无处理', None),
        ('RandomOverSampler', RandomOverSampler(random_state=42)),
        ('SMOTE', SMOTE(random_state=42)),
        ('BorderlineSMOTE', BorderlineSMOTE(random_state=42)),
        ('ADASYN', ADASYN(random_state=42)),
        ('RandomUnderSampler', RandomUnderSampler(random_state=42)),
        ('SMOTETomek', SMOTETomek(random_state=42)),
        ('SMOTEENN', SMOTEENN(random_state=42))
    ]
    
    results = []
    
    for name, sampler in methods:
        if sampler is None:
            X_train_res, y_train_res = X_train, y_train
        else:
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        
        # 训练模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_res, y_train_res)
        
        # 评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results.append({
            '方法': name,
            '训练样本数': len(y_train_res),
            'F1分数': f1_score(y_test, y_pred),
            '平衡准确率': balanced_accuracy_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
        })
    
    # 添加类别权重方法
    model_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_weighted.fit(X_train, y_train)
    y_pred = model_weighted.predict(X_test)
    y_pred_proba = model_weighted.predict_proba(X_test)[:, 1]
    
    results.append({
        '方法': 'Class Weight',
        '训练样本数': len(y_train),
        'F1分数': f1_score(y_test, y_pred),
        '平衡准确率': balanced_accuracy_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
    })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1分数', ascending=False)
    
    print("\n性能比较:")
    print(results_df.to_string(index=False))
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['F1分数', '平衡准确率', 'AUC-ROC']
    colors = ['steelblue', 'coral', 'green']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        sorted_df = results_df.sort_values(metric, ascending=True)
        axes[idx].barh(sorted_df['方法'], sorted_df[metric], color=color, alpha=0.7)
        axes[idx].set_xlabel(metric)
        axes[idx].set_title(f'{metric}比较')
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'imbalance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("\n比较结果已保存: imbalance_comparison.png")
    plt.close()
    
    return results_df


# ============================================================
# 13. 生产环境最佳实践
# ============================================================

def production_best_practices(X, y):
    """生产环境不平衡数据处理最佳实践"""
    print("\n" + "="*60)
    print("13. 生产环境最佳实践")
    print("="*60)
    
    # 步骤1: 评估不平衡程度
    print("\n步骤1: 评估不平衡程度")
    class_counts = Counter(y)
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"不平衡比率: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 2:
        print("✓ 轻度不平衡 (<2:1) - 可能不需要特殊处理")
    elif imbalance_ratio < 10:
        print("⚠️  中度不平衡 (2-10:1) - 建议使用采样或权重")
    else:
        print("🚨 严重不平衡 (>10:1) - 必须使用专门技术")
    
    # 步骤2: 选择合适的方法
    print("\n步骤2: 根据情况选择方法")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 组合策略：SMOTE + class_weight
    print("\n推荐策略: SMOTE + Random Forest with class_weight")
    
    # 应用SMOTE（适度过采样，不完全平衡）
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 少数类达到多数类50%
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"采样前: {Counter(y_train)}")
    print(f"采样后: {Counter(y_train_res)}")
    
    # 训练平衡随机森林
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)
    
    # 评估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n模型性能:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    # 步骤3: 阈值调整
    print("\n步骤3: 根据业务需求调整决策阈值")
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # 可视化PR曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, color='blue')
    ax.set_xlabel('召回率 (Recall)')
    ax.set_ylabel('精确率 (Precision)')
    ax.set_title('Precision-Recall 曲线')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'pr_curve.png'), 
                dpi=300, bbox_inches='tight')
    print("PR曲线已保存: pr_curve.png")
    plt.close()
    
    # 步骤4: 保存处理器
    print("\n步骤4: 保存采样器和模型")
    import joblib
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(smote, os.path.join(model_dir, 'smote_sampler.joblib'))
    joblib.dump(model, os.path.join(model_dir, 'balanced_model.joblib'))
    print("采样器和模型已保存")
    
    # 步骤5: 评估指标选择
    print("\n步骤5: 使用适当的评估指标")
    print("⚠️  不平衡数据不要只看准确率！")
    print("\n推荐指标:")
    print("  - F1分数 (平衡精确率和召回率)")
    print("  - 平衡准确率 (各类别准确率的平均)")
    print("  - AUC-ROC (整体分类能力)")
    print("  - Precision-Recall曲线 (不同阈值的权衡)")
    print("  - 混淆矩阵 (详细的分类结果)")
    
    return model, smote


# ============================================================
# 14. 方法选择指南
# ============================================================

def selection_guide():
    """提供方法选择指南"""
    print("\n" + "="*60)
    print("14. 不平衡数据处理方法选择指南")
    print("="*60)
    
    guide = """
    ┌─────────────────────────────────────────────────────────────┐
    │            不平衡数据处理方法选择指南                        │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. 评估不平衡程度                                            │
    │    - 轻度 (<2:1): 可能不需要特殊处理                         │
    │    - 中度 (2-10:1): 使用采样或权重                           │
    │    - 严重 (>10:1): 组合多种技术                              │
    ├─────────────────────────────────────────────────────────────┤
    │ 2. 数据量考虑                                                │
    │    - 数据量小 (<1000): 使用过采样(SMOTE)                     │
    │    - 数据量中等: 使用SMOTE或class_weight                     │
    │    - 数据量大 (>100k): 使用欠采样或class_weight              │
    ├─────────────────────────────────────────────────────────────┤
    │ 3. 方法推荐                                                  │
    │                                                              │
    │    过采样方法:                                               │
    │      ✓ RandomOverSampler: 简单快速，baseline               │
    │      ✓ SMOTE: 最常用，效果好                                │
    │      ✓ BorderlineSMOTE: 聚焦难分样本                        │
    │      ✓ ADASYN: 自适应，适合复杂分布                         │
    │                                                              │
    │    欠采样方法:                                               │
    │      ✓ RandomUnderSampler: 快速，数据量大时使用            │
    │      ✓ TomekLinks: 清理边界噪声                             │
    │      ✓ NearMiss: 保留有代表性的多数类样本                   │
    │                                                              │
    │    混合方法:                                                 │
    │      ✓ SMOTETomek: 平衡 + 清理边界                          │
    │      ✓ SMOTEENN: 平衡 + 去噪                                │
    │                                                              │
    │    算法级方法:                                               │
    │      ✓ class_weight: 简单有效，不改变数据                   │
    │      ✓ BalancedRandomForest: 集成学习 + 自动平衡           │
    │      ✓ EasyEnsemble: 多个平衡子集的集成                     │
    ├─────────────────────────────────────────────────────────────┤
    │ 4. 生产环境推荐策略                                          │
    │                                                              │
    │    策略1 (通用):                                             │
    │      SMOTE (适度过采样) + class_weight                      │
    │                                                              │
    │    策略2 (大数据):                                           │
    │      RandomUnderSampler + class_weight                      │
    │                                                              │
    │    策略3 (高质量):                                           │
    │      SMOTETomek + BalancedRandomForest                      │
    │                                                              │
    │    策略4 (简单):                                             │
    │      仅使用 class_weight='balanced'                         │
    ├─────────────────────────────────────────────────────────────┤
    │ 5. 关键注意事项                                              │
    │                                                              │
    │    ⚠️  只在训练集上应用采样！                                │
    │    ⚠️  测试集保持原始分布！                                  │
    │    ⚠️  使用分层划分(stratify)！                              │
    │    ⚠️  使用合适的评估指标(F1, AUC)！                         │
    │    ⚠️  根据业务需求调整决策阈值！                            │
    │    ⚠️  交叉验证评估模型性能！                                │
    └─────────────────────────────────────────────────────────────┘
    
    💡 快速决策树:
    
    是否严重不平衡(>10:1)?
      ├─ 是 → 组合使用SMOTE + class_weight + 集成方法
      └─ 否 ↓
    
    数据量是否很大(>100k)?
      ├─ 是 → RandomUnderSampler + class_weight
      └─ 否 ↓
    
    是否有足够计算资源?
      ├─ 是 → SMOTETomek + BalancedRandomForest
      └─ 否 → class_weight='balanced'
    """
    
    print(guide)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("数据平衡处理 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建不平衡数据集
    X, y = create_imbalanced_dataset(imbalance_ratio=0.1)
    
    # 1. 基线模型
    baseline, X_train, X_test, y_train, y_test = baseline_model(X, y)
    
    # 2-11. 各种方法演示
    ros_model, X_ros, y_ros = random_oversampling_demo(X, y)
    smote_model, X_smote, y_smote = smote_demo(X, y)
    bsmote_model, X_bsmote, y_bsmote = borderline_smote_demo(X, y)
    adasyn_model, X_adasyn, y_adasyn = adasyn_demo(X, y)
    rus_model, X_rus, y_rus = random_undersampling_demo(X, y)
    tomek_model, X_tomek, y_tomek = tomek_links_demo(X, y)
    smoteenn_model, X_smoteenn, y_smoteenn = smoteenn_demo(X, y)
    smotetomek_model, X_smotetomek, y_smotetomek = smotetomek_demo(X, y)
    brf_model = balanced_random_forest_demo(X, y)
    cw_model = class_weight_demo(X, y)
    
    # 12. 方法比较
    comparison_results = compare_methods(X, y)
    
    # 13. 生产环境最佳实践
    prod_model, prod_sampler = production_best_practices(X, y)
    
    # 14. 选择指南
    selection_guide()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
