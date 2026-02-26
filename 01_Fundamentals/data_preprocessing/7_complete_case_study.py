"""
完整综合案例 - 生产环境数据预处理
Complete Case Study for Production Environment

本文件演示了从原始数据到模型部署的完整流程，整合所有预处理技术
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 步骤1: 数据加载和探索
# ============================================================

def load_and_explore_data():
    """加载数据并进行探索性分析"""
    print("="*70)
    print("步骤1: 数据加载和探索性分析 (EDA)")
    print("="*70)
    
    # 创建模拟真实场景的数据集
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟客户流失预测数据集
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'monthly_income': np.random.normal(5000, 2000, n_samples),
        'account_balance': np.random.normal(10000, 8000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'num_products': np.random.randint(1, 5, n_samples),
        'tenure_months': np.random.randint(0, 120, n_samples),
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], n_samples),
        'has_credit_card': np.random.choice([0, 1], n_samples),
        'is_active_member': np.random.choice([0, 1], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 不平衡数据
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些现实中的数据质量问题
    # 1. 缺失值
    df.loc[np.random.choice(df.index, 80), 'monthly_income'] = np.nan
    df.loc[np.random.choice(df.index, 50), 'credit_score'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'city'] = np.nan
    
    # 2. 异常值
    df.loc[np.random.choice(df.index, 10), 'account_balance'] = df['account_balance'] * 10  # 极端值
    df.loc[np.random.choice(df.index, 5), 'age'] = 150  # 不合理的年龄
    
    # 3. 重复记录
    df = pd.concat([df, df.sample(5)], ignore_index=True)
    
    print(f"\n数据集基本信息:")
    print(f"  样本数: {len(df)}")
    print(f"  特征数: {df.shape[1]}")
    print(f"  目标变量: churn (客户流失)")
    
    print(f"\n数据预览:")
    print(df.head())
    
    print(f"\n数据类型:")
    print(df.dtypes)
    
    print(f"\n基本统计:")
    print(df.describe())
    
    # 目标变量分布
    print(f"\n目标变量分布:")
    print(df['churn'].value_counts())
    print(f"  不流失: {(df['churn']==0).sum()} ({(df['churn']==0).sum()/len(df)*100:.1f}%)")
    print(f"  流失: {(df['churn']==1).sum()} ({(df['churn']==1).sum()/len(df)*100:.1f}%)")
    print(f"  不平衡比率: {(df['churn']==0).sum()/(df['churn']==1).sum():.2f}:1")
    
    # 缺失值分析
    print(f"\n缺失值统计:")
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100
    })
    print(missing_stats[missing_stats['缺失数量'] > 0])
    
    print(f"\n重复记录数: {df.duplicated().sum()}")
    
    return df


# ============================================================
# 步骤2: 数据清洗
# ============================================================

def data_cleaning(df):
    """数据清洗"""
    print("\n" + "="*70)
    print("步骤2: 数据清洗")
    print("="*70)
    
    df_clean = df.copy()
    
    # 1. 删除customer_id（不是特征）
    print("\n1. 删除ID列")
    df_clean = df_clean.drop('customer_id', axis=1)
    print(f"   ✓ 删除 customer_id")
    
    # 2. 处理重复记录
    print("\n2. 处理重复记录")
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    print(f"   ✓ 删除 {before - after} 条重复记录")
    
    # 3. 处理明显的异常值（基于业务规则）
    print("\n3. 处理业务规则异常值")
    
    # 年龄应该在18-100之间
    invalid_age = ((df_clean['age'] < 18) | (df_clean['age'] > 100)).sum()
    df_clean['age'] = df_clean['age'].clip(18, 100)
    print(f"   ✓ 修正 {invalid_age} 个异常年龄值")
    
    # 信用分数应该在300-850之间
    df_clean.loc[df_clean['credit_score'] < 300, 'credit_score'] = 300
    df_clean.loc[df_clean['credit_score'] > 850, 'credit_score'] = 850
    print(f"   ✓ 修正信用分数范围")
    
    # 账户余额不应为负
    negative_balance = (df_clean['account_balance'] < 0).sum()
    df_clean['account_balance'] = df_clean['account_balance'].clip(0, None)
    print(f"   ✓ 修正 {negative_balance} 个负余额")
    
    print(f"\n清洗后数据集大小: {df_clean.shape}")
    
    return df_clean


# ============================================================
# 步骤3: 特征工程
# ============================================================

def feature_engineering(df):
    """特征工程"""
    print("\n" + "="*70)
    print("步骤3: 特征工程")
    print("="*70)
    
    df_fe = df.copy()
    
    # 1. 创建新特征
    print("\n1. 创建新特征")
    
    # 余额收入比
    df_fe['balance_income_ratio'] = df_fe['account_balance'] / (df_fe['monthly_income'] + 1)
    print("   ✓ balance_income_ratio: 账户余额/月收入")
    
    # 平均产品持有时间
    df_fe['avg_product_tenure'] = df_fe['tenure_months'] / (df_fe['num_products'] + 1)
    print("   ✓ avg_product_tenure: 平均产品持有时间")
    
    # 是否高净值客户
    df_fe['is_high_value'] = (df_fe['account_balance'] > df_fe['account_balance'].quantile(0.75)).astype(int)
    print("   ✓ is_high_value: 是否高净值客户")
    
    # 年龄段
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=[0, 30, 45, 60, 100], 
                                 labels=['Young', 'Middle', 'Senior', 'Elderly'])
    print("   ✓ age_group: 年龄段分组")
    
    print(f"\n新增特征: {['balance_income_ratio', 'avg_product_tenure', 'is_high_value', 'age_group']}")
    print(f"当前特征数: {df_fe.shape[1]}")
    
    return df_fe


# ============================================================
# 步骤4: 分离特征和目标变量，划分数据集
# ============================================================

def split_data(df):
    """划分数据集"""
    print("\n" + "="*70)
    print("步骤4: 划分训练集和测试集")
    print("="*70)
    
    # 分离特征和目标
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # 分层划分（保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    print(f"\n训练集类别分布:")
    print(f"  不流失: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
    print(f"  流失: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    
    print(f"\n测试集类别分布:")
    print(f"  不流失: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
    print(f"  流失: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


# ============================================================
# 步骤5: 构建预处理Pipeline
# ============================================================

def build_preprocessing_pipeline():
    """构建预处理Pipeline"""
    print("\n" + "="*70)
    print("步骤5: 构建预处理Pipeline")
    print("="*70)
    
    # 定义特征类型
    numeric_features = [
        'age', 'monthly_income', 'account_balance', 'credit_score',
        'num_products', 'tenure_months', 'balance_income_ratio', 'avg_product_tenure'
    ]
    
    binary_features = ['has_credit_card', 'is_active_member', 'is_high_value']
    
    ordinal_features = ['age_group']
    ordinal_categories = [['Young', 'Middle', 'Senior', 'Elderly']]
    
    nominal_features = ['city', 'education', 'employment_status']
    
    print(f"\n特征分类:")
    print(f"  数值特征 ({len(numeric_features)}个): {numeric_features[:3]}...")
    print(f"  二值特征 ({len(binary_features)}个): {binary_features}")
    print(f"  有序类别 ({len(ordinal_features)}个): {ordinal_features}")
    print(f"  无序类别 ({len(nominal_features)}个): {nominal_features}")
    
    # 数值特征处理
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 二值特征处理（只需填充）
    binary_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # 有序类别处理
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, 
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])
    
    # 无序类别处理
    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 组合所有转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features)
        ],
        remainder='drop'
    )
    
    print("\nPipeline结构:")
    print("┌─ 数值特征")
    print("│   ├─ SimpleImputer(median)")
    print("│   └─ StandardScaler()")
    print("├─ 二值特征")
    print("│   └─ SimpleImputer(most_frequent)")
    print("├─ 有序类别")
    print("│   ├─ SimpleImputer(Unknown)")
    print("│   └─ OrdinalEncoder()")
    print("└─ 无序类别")
    print("    ├─ SimpleImputer(Unknown)")
    print("    └─ OneHotEncoder()")
    
    return preprocessor


# ============================================================
# 步骤6: 构建完整的ML Pipeline（包含SMOTE）
# ============================================================

def build_complete_pipeline(preprocessor):
    """构建完整的ML Pipeline"""
    print("\n" + "="*70)
    print("步骤6: 构建完整的ML Pipeline")
    print("="*70)
    
    # 使用imbalanced-learn的Pipeline（支持SMOTE）
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),  # 少数类达到多数类80%
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # 额外的类别权重
        ))
    ])
    
    print("\n完整Pipeline:")
    print("1. 数据预处理 (ColumnTransformer)")
    print("2. SMOTE过采样 (sampling_strategy=0.8)")
    print("3. RandomForestClassifier (class_weight='balanced')")
    
    return pipeline


# ============================================================
# 步骤7: 模型训练和交叉验证
# ============================================================

def train_and_validate(pipeline, X_train, y_train):
    """训练模型并进行交叉验证"""
    print("\n" + "="*70)
    print("步骤7: 模型训练和交叉验证")
    print("="*70)
    
    print("\n进行5折交叉验证...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    print(f"\n交叉验证 AUC-ROC 分数:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"  平均: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\n在完整训练集上训练最终模型...")
    pipeline.fit(X_train, y_train)
    print("✓ 训练完成")
    
    return pipeline


# ============================================================
# 步骤8: 模型评估
# ============================================================

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """全面评估模型"""
    print("\n" + "="*70)
    print("步骤8: 模型评估")
    print("="*70)
    
    # 预测
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 基本指标
    print("\n训练集分类报告:")
    print(classification_report(y_train, y_train_pred, 
                               target_names=['不流失', '流失']))
    
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['不流失', '流失']))
    
    # AUC-ROC
    train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nAUC-ROC:")
    print(f"  训练集: {train_auc:.4f}")
    print(f"  测试集: {test_auc:.4f}")
    
    # 混淆矩阵
    print("\n混淆矩阵（测试集）:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # 可视化
    visualize_results(y_test, y_test_pred, y_test_proba, cm)
    
    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'confusion_matrix': cm
    }


def visualize_results(y_test, y_pred, y_proba, cm):
    """可视化评估结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('混淆矩阵')
    axes[0, 0].set_xlabel('预测类别')
    axes[0, 0].set_ylabel('真实类别')
    axes[0, 0].set_xticklabels(['不流失', '流失'])
    axes[0, 0].set_yticklabels(['不流失', '流失'])
    
    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('假正例率(FPR)')
    axes[0, 1].set_ylabel('真正例率(TPR)')
    axes[0, 1].set_title('ROC曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 0].plot(recall, precision, linewidth=2)
    axes[1, 0].set_xlabel('召回率(Recall)')
    axes[1, 0].set_ylabel('精确率(Precision)')
    axes[1, 0].set_title('Precision-Recall曲线')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 预测概率分布
    axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='不流失', color='blue')
    axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='流失', color='red')
    axes[1, 1].set_xlabel('预测概率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('预测概率分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'complete_case_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存: complete_case_evaluation.png")
    plt.close()


# ============================================================
# 步骤9: 特征重要性分析
# ============================================================

def feature_importance_analysis(pipeline, feature_names):
    """分析特征重要性"""
    print("\n" + "="*70)
    print("步骤9: 特征重要性分析")
    print("="*70)
    
    # 获取RandomForest的特征重要性
    clf = pipeline.named_steps['classifier']
    importances = clf.feature_importances_
    
    # 获取转换后的特征名
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names_transformed = preprocessor.get_feature_names_out()
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names_transformed,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 重要特征:")
    print(importance_df.head(15).to_string(index=False))
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 15
    top_features = importance_df.head(top_n)
    ax.barh(range(top_n), top_features['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('重要性')
    ax.set_title(f'Top {top_n} 特征重要性')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n特征重要性图已保存: feature_importance.png")
    plt.close()
    
    return importance_df


# ============================================================
# 步骤10: 模型保存和部署
# ============================================================

def save_model(pipeline, metrics):
    """保存模型和相关信息"""
    print("\n" + "="*70)
    print("步骤10: 模型保存和部署准备")
    print("="*70)
    
    # 创建保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存Pipeline
    pipeline_path = os.path.join(output_dir, f'churn_prediction_pipeline_{timestamp}.joblib')
    joblib.dump(pipeline, pipeline_path)
    print(f"\n✓ Pipeline已保存: {pipeline_path}")
    
    # 保存模型元数据
    metadata = {
        'timestamp': timestamp,
        'train_auc': metrics['train_auc'],
        'test_auc': metrics['test_auc'],
        'model_type': 'RandomForestClassifier',
        'preprocessing': 'StandardScaler + OneHotEncoder + SMOTE',
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    import json
    metadata_path = os.path.join(output_dir, f'model_metadata_{timestamp}.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"✓ 元数据已保存: {metadata_path}")
    
    print("\n模型部署就绪!")
    print("  - 加载: pipeline = joblib.load(pipeline_path)")
    print("  - 预测: predictions = pipeline.predict(new_data)")
    
    return pipeline_path


# ============================================================
# 步骤11: 生产环境使用示例
# ============================================================

def production_usage_example(pipeline_path):
    """演示生产环境使用"""
    print("\n" + "="*70)
    print("步骤11: 生产环境使用示例")
    print("="*70)
    
    # 加载模型
    print("\n加载训练好的Pipeline...")
    pipeline = joblib.load(pipeline_path)
    print("✓ Pipeline加载完成")
    
    # 模拟新客户数据
    new_customers = pd.DataFrame({
        'age': [35, 28, 55],
        'monthly_income': [6000, 4500, 8000],
        'account_balance': [15000, 5000, 30000],
        'credit_score': [720, 650, 800],
        'num_products': [2, 1, 3],
        'tenure_months': [24, 6, 60],
        'city': ['Beijing', 'Shanghai', 'Hangzhou'],
        'education': ['Master', 'Bachelor', 'PhD'],
        'employment_status': ['Employed', 'Self-employed', 'Employed'],
        'has_credit_card': [1, 0, 1],
        'is_active_member': [1, 1, 0],
        'balance_income_ratio': [2.5, 1.1, 3.75],
        'avg_product_tenure': [12, 6, 20],
        'is_high_value': [1, 0, 1],
        'age_group': ['Middle', 'Young', 'Senior']
    })
    
    print("\n新客户数据:")
    print(new_customers)
    
    # 预测
    print("\n进行预测...")
    predictions = pipeline.predict(new_customers)
    probabilities = pipeline.predict_proba(new_customers)
    
    print("\n预测结果:")
    for i in range(len(new_customers)):
        print(f"\n客户 {i+1}:")
        print(f"  预测: {'流失' if predictions[i] == 1 else '不流失'}")
        print(f"  流失概率: {probabilities[i][1]:.4f}")
        print(f"  风险评级: ", end='')
        if probabilities[i][1] < 0.3:
            print("低风险 🟢")
        elif probabilities[i][1] < 0.7:
            print("中风险 🟡")
        else:
            print("高风险 🔴")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：完整流程"""
    print("="*70)
    print("完整综合案例 - 客户流失预测")
    print("从原始数据到生产部署的完整流程")
    print("="*70)
    
    # 步骤1: 加载和探索数据
    df = load_and_explore_data()
    
    # 步骤2: 数据清洗
    df_clean = data_cleaning(df)
    
    # 步骤3: 特征工程
    df_fe = feature_engineering(df_clean)
    
    # 步骤4: 划分数据集
    X_train, X_test, y_train, y_test = split_data(df_fe)
    
    # 步骤5: 构建预处理Pipeline
    preprocessor = build_preprocessing_pipeline()
    
    # 步骤6: 构建完整Pipeline
    pipeline = build_complete_pipeline(preprocessor)
    
    # 步骤7: 训练和验证
    pipeline = train_and_validate(pipeline, X_train, y_train)
    
    # 步骤8: 评估模型
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    
    # 步骤9: 特征重要性分析
    importance_df = feature_importance_analysis(pipeline, X_train.columns.tolist())
    
    # 步骤10: 保存模型
    pipeline_path = save_model(pipeline, metrics)
    
    # 步骤11: 生产使用示例
    production_usage_example(pipeline_path)
    
    print("\n" + "="*70)
    print("完整流程演示完成！")
    print("="*70)
    
    print("\n项目交付物:")
    print("  ✓ 训练好的Pipeline模型")
    print("  ✓ 模型元数据和性能指标")
    print("  ✓ 可视化评估报告")
    print("  ✓ 特征重要性分析")
    print("  ✓ 生产环境使用示例")
    
    print("\n生产环境部署清单:")
    print("  □ 将Pipeline部署到服务器")
    print("  □ 创建API接口")
    print("  □ 设置监控和日志")
    print("  □ 准备数据验证和异常处理")
    print("  □ 建立模型更新机制")
    print("  □ 准备回滚策略")


if __name__ == "__main__":
    main()
