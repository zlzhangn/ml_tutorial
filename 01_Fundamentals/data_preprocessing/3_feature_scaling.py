"""
数据标准化与归一化 - 生产环境数据预处理技术
Feature Scaling for Production Environment

本文件演示了数据缩放的各种方法，适用于生产环境
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, QuantileTransformer, PowerTransformer
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_sample_data():
    """创建示例数据集，包含不同尺度的特征"""
    np.random.seed(42)
    
    n_samples = 200
    data = {
        'age': np.random.normal(35, 15, n_samples),  # 范围: 0-100
        'salary': np.random.normal(60000, 20000, n_samples),  # 范围: 数万
        'experience': np.random.normal(8, 5, n_samples),  # 范围: 0-30
        'score': np.random.normal(75, 15, n_samples),  # 范围: 0-100
        'purchases': np.random.poisson(5, n_samples),  # 计数数据
        'target': np.random.randint(0, 2, n_samples)  # 二分类标签
    }
    
    df = pd.DataFrame(data)
    # 确保数据在合理范围内
    df['age'] = df['age'].clip(18, 70)
    df['salary'] = df['salary'].clip(20000, 150000)
    df['experience'] = df['experience'].clip(0, 30)
    df['score'] = df['score'].clip(0, 100)
    
    print("原始数据统计:")
    print(df.describe())
    print("\n不同特征的尺度差异很大，需要进行特征缩放！")
    
    return df


# ============================================================
# 1. StandardScaler (Z-Score标准化)
# ============================================================

def standard_scaler_demo(df):
    """StandardScaler: 均值为0，标准差为1"""
    print("\n" + "="*60)
    print("1. StandardScaler (Z-Score标准化)")
    print("="*60)
    print("公式: X_scaled = (X - mean) / std")
    print("结果: 均值=0, 标准差=1")
    print("适用: 特征近似正态分布，对异常值敏感")
    
    # 提取特征
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 应用StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 转换为DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n标准化后的统计:")
    print(df_scaled.describe())
    
    # 查看转换参数
    print("\n各特征的均值 (mean_):")
    for col, mean in zip(feature_cols, scaler.mean_):
        print(f"  {col}: {mean:.2f}")
    
    print("\n各特征的标准差 (scale_):")
    for col, scale in zip(feature_cols, scaler.scale_):
        print(f"  {col}: {scale:.2f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for idx, col in enumerate(feature_cols):
        # 原始数据
        axes[0, idx].hist(df[col], bins=30, color='blue', alpha=0.7)
        axes[0, idx].set_title(f'{col}\n(原始)')
        axes[0, idx].set_xlabel('值')
        axes[0, idx].set_ylabel('频数')
        
        # 标准化后
        axes[1, idx].hist(df_scaled[col], bins=30, color='green', alpha=0.7)
        axes[1, idx].set_title(f'{col}\n(StandardScaler)')
        axes[1, idx].set_xlabel('值')
        axes[1, idx].set_ylabel('频数')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'standard_scaler.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: standard_scaler.png")
    plt.close()
    
    return scaler, df_scaled


# ============================================================
# 2. MinMaxScaler (最小-最大归一化)
# ============================================================

def minmax_scaler_demo(df, feature_range=(0, 1)):
    """MinMaxScaler: 缩放到指定范围"""
    print("\n" + "="*60)
    print("2. MinMaxScaler (最小-最大归一化)")
    print("="*60)
    print(f"公式: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min")
    print(f"结果: 数据缩放到 [{feature_range[0]}, {feature_range[1]}]")
    print("适用: 需要固定范围，对异常值敏感")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 应用MinMaxScaler
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n归一化后的统计:")
    print(df_scaled.describe())
    
    print("\n各特征的最小值 (data_min_):")
    for col, min_val in zip(feature_cols, scaler.data_min_):
        print(f"  {col}: {min_val:.2f}")
    
    print("\n各特征的最大值 (data_max_):")
    for col, max_val in zip(feature_cols, scaler.data_max_):
        print(f"  {col}: {max_val:.2f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df_scaled[col], bins=30, color='orange', alpha=0.7)
        axes[idx].set_title(f'{col}\n(MinMaxScaler)')
        axes[idx].set_xlabel('值')
        axes[idx].set_ylabel('频数')
        axes[idx].axvline(feature_range[0], color='red', linestyle='--', label='min')
        axes[idx].axvline(feature_range[1], color='red', linestyle='--', label='max')
        axes[idx].legend()
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'minmax_scaler.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: minmax_scaler.png")
    plt.close()
    
    return scaler, df_scaled


# ============================================================
# 3. RobustScaler (鲁棒缩放器)
# ============================================================

def robust_scaler_demo(df):
    """RobustScaler: 对异常值鲁棒的缩放"""
    print("\n" + "="*60)
    print("3. RobustScaler (鲁棒缩放器)")
    print("="*60)
    print("公式: X_scaled = (X - median) / IQR")
    print("结果: 使用中位数和四分位距，对异常值不敏感")
    print("适用: 数据包含异常值")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 应用RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n鲁棒缩放后的统计:")
    print(df_scaled.describe())
    
    print("\n各特征的中位数 (center_):")
    for col, center in zip(feature_cols, scaler.center_):
        print(f"  {col}: {center:.2f}")
    
    print("\n各特征的IQR (scale_):")
    for col, scale in zip(feature_cols, scaler.scale_):
        print(f"  {col}: {scale:.2f}")
    
    return scaler, df_scaled


# ============================================================
# 4. MaxAbsScaler (最大绝对值缩放)
# ============================================================

def maxabs_scaler_demo(df):
    """MaxAbsScaler: 按最大绝对值缩放到[-1, 1]"""
    print("\n" + "="*60)
    print("4. MaxAbsScaler (最大绝对值缩放)")
    print("="*60)
    print("公式: X_scaled = X / max(|X|)")
    print("结果: 数据缩放到 [-1, 1]，保持稀疏性")
    print("适用: 稀疏数据，已经中心化的数据")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 应用MaxAbsScaler
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n最大绝对值缩放后的统计:")
    print(df_scaled.describe())
    
    print("\n各特征的最大绝对值 (max_abs_):")
    for col, max_abs in zip(feature_cols, scaler.max_abs_):
        print(f"  {col}: {max_abs:.2f}")
    
    return scaler, df_scaled


# ============================================================
# 5. Normalizer (L2归一化)
# ============================================================

def normalizer_demo(df):
    """Normalizer: 将每个样本缩放到单位范数"""
    print("\n" + "="*60)
    print("5. Normalizer (样本归一化)")
    print("="*60)
    print("公式: X_scaled = X / ||X||")
    print("结果: 每个样本的L2范数为1")
    print("适用: 文本分类、聚类，关注方向而非幅度")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 应用Normalizer (L2范数)
    scaler = Normalizer(norm='l2')
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\nL2归一化后的统计:")
    print(df_scaled.describe())
    
    # 验证L2范数
    l2_norms = np.linalg.norm(X_scaled, axis=1)
    print(f"\n样本L2范数 (应该接近1): mean={l2_norms.mean():.6f}, std={l2_norms.std():.6f}")
    
    return scaler, df_scaled


# ============================================================
# 6. QuantileTransformer (分位数转换)
# ============================================================

def quantile_transformer_demo(df):
    """QuantileTransformer: 转换为均匀或正态分布"""
    print("\n" + "="*60)
    print("6. QuantileTransformer (分位数转换)")
    print("="*60)
    print("原理: 通过分位数映射，将特征转换为指定分布")
    print("适用: 非线性转换，处理偏态分布")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # 转换为均匀分布
    print("\n6.1 转换为均匀分布:")
    scaler_uniform = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_uniform = scaler_uniform.fit_transform(X)
    df_uniform = pd.DataFrame(X_uniform, columns=feature_cols)
    print(df_uniform.describe())
    
    # 转换为正态分布
    print("\n6.2 转换为正态分布:")
    scaler_normal = QuantileTransformer(output_distribution='normal', random_state=42)
    X_normal = scaler_normal.fit_transform(X)
    df_normal = pd.DataFrame(X_normal, columns=feature_cols)
    print(df_normal.describe())
    
    # 可视化对比
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for idx, col in enumerate(feature_cols):
        # 原始分布
        axes[0, idx].hist(df[col], bins=30, color='blue', alpha=0.7)
        axes[0, idx].set_title(f'{col}\n(原始)')
        
        # 均匀分布
        axes[1, idx].hist(df_uniform[col], bins=30, color='green', alpha=0.7)
        axes[1, idx].set_title(f'{col}\n(均匀分布)')
        
        # 正态分布
        axes[2, idx].hist(df_normal[col], bins=30, color='red', alpha=0.7)
        axes[2, idx].set_title(f'{col}\n(正态分布)')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'quantile_transformer.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: quantile_transformer.png")
    plt.close()
    
    return scaler_uniform, scaler_normal


# ============================================================
# 7. PowerTransformer (幂变换)
# ============================================================

def power_transformer_demo(df):
    """PowerTransformer: Box-Cox和Yeo-Johnson变换"""
    print("\n" + "="*60)
    print("7. PowerTransformer (幂变换)")
    print("="*60)
    print("原理: 使数据更接近高斯分布")
    print("Box-Cox: 仅适用于正数")
    print("Yeo-Johnson: 适用于任意数据")
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    
    # Yeo-Johnson变换（可处理负值）
    print("\n使用Yeo-Johnson变换:")
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n变换后的统计:")
    print(df_scaled.describe())
    
    print("\n各特征的λ参数 (lambdas_):")
    for col, lambda_val in zip(feature_cols, scaler.lambdas_):
        print(f"  {col}: {lambda_val:.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for idx, col in enumerate(feature_cols):
        # 原始分布
        axes[0, idx].hist(df[col], bins=30, color='blue', alpha=0.7)
        axes[0, idx].set_title(f'{col}\n(原始)')
        
        # 变换后
        axes[1, idx].hist(df_scaled[col], bins=30, color='purple', alpha=0.7)
        axes[1, idx].set_title(f'{col}\n(PowerTransformer)')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'power_transformer.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: power_transformer.png")
    plt.close()
    
    return scaler, df_scaled


# ============================================================
# 8. 不同缩放方法的比较
# ============================================================

def compare_scalers(df):
    """比较不同缩放方法的效果"""
    print("\n" + "="*60)
    print("8. 不同缩放方法比较")
    print("="*60)
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    y = df['target'].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 定义不同的缩放器
    scalers = {
        '无缩放': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'PowerTransformer': PowerTransformer()
    }
    
    results = []
    
    for name, scaler in scalers.items():
        if scaler is None:
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        # 训练逻辑回归模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 评估
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        
        results.append({
            '缩放方法': name,
            '训练准确率': train_acc,
            '测试准确率': test_acc
        })
        
        print(f"\n{name}:")
        print(f"  训练准确率: {train_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
    
    # 创建结果表
    results_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("综合比较:")
    print(results_df.to_string(index=False))
    
    # 可视化比较
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x_pos - width/2, results_df['训练准确率'], width, label='训练准确率', alpha=0.8)
    ax.bar(x_pos + width/2, results_df['测试准确率'], width, label='测试准确率', alpha=0.8)
    
    ax.set_xlabel('缩放方法')
    ax.set_ylabel('准确率')
    ax.set_title('不同缩放方法对模型性能的影响')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['缩放方法'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'scaler_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("\n比较结果已保存: scaler_comparison.png")
    plt.close()
    
    return results_df


# ============================================================
# 9. 生产环境最佳实践
# ============================================================

def production_best_practices(df):
    """生产环境特征缩放最佳实践"""
    print("\n" + "="*60)
    print("9. 生产环境最佳实践")
    print("="*60)
    
    feature_cols = ['age', 'salary', 'experience', 'score', 'purchases']
    X = df[feature_cols].values
    y = df['target'].values
    
    # 步骤1: 划分数据集（重要：先划分再缩放！）
    print("\n步骤1: 先划分数据集")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 步骤2: 在训练集上拟合缩放器
    print("\n步骤2: 在训练集上拟合缩放器")
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("缩放器已在训练集上拟合完成")
    
    # 步骤3: 转换训练集和测试集
    print("\n步骤3: 分别转换训练集和测试集")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("数据转换完成")
    
    # 步骤4: 保存缩放器
    print("\n步骤4: 保存缩放器用于生产环境")
    import joblib
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"缩放器已保存: {scaler_path}")
    
    # 步骤5: 加载并使用（模拟生产环境）
    print("\n步骤5: 加载缩放器处理新数据")
    loaded_scaler = joblib.load(scaler_path)
    
    # 模拟新数据
    new_data = np.array([[30, 55000, 5, 80, 3]])
    new_data_scaled = loaded_scaler.transform(new_data)
    
    print("\n新数据 (原始):")
    print(pd.DataFrame(new_data, columns=feature_cols))
    print("\n新数据 (缩放后):")
    print(pd.DataFrame(new_data_scaled, columns=feature_cols))
    
    # 步骤6: 注意事项
    print("\n" + "="*60)
    print("重要注意事项:")
    print("="*60)
    print("1. ⚠️ 必须先划分数据，再进行缩放！")
    print("2. ⚠️ 缩放器只在训练集上fit，然后transform训练集和测试集")
    print("3. ⚠️ 保存缩放器，确保生产环境使用相同的缩放参数")
    print("4. ⚠️ 不要在整个数据集上fit，会导致数据泄露")
    print("5. ⚠️ 新数据必须使用相同的缩放器进行transform")
    
    return scaler, X_train_scaled, X_test_scaled


# ============================================================
# 10. 缩放方法选择指南
# ============================================================

def scaling_selection_guide():
    """提供缩放方法选择指南"""
    print("\n" + "="*60)
    print("10. 缩放方法选择指南")
    print("="*60)
    
    guide = """
    ┌─────────────────────────────────────────────────────────────┐
    │               特征缩放方法选择指南                           │
    ├─────────────────────────────────────────────────────────────┤
    │ StandardScaler (Z-Score标准化)                              │
    │   适用: 特征近似正态分布                                     │
    │   优点: 保留异常值信息，适合大多数ML算法                     │
    │   缺点: 对异常值敏感                                         │
    │   推荐: 线性回归、逻辑回归、SVM、神经网络                    │
    ├─────────────────────────────────────────────────────────────┤
    │ MinMaxScaler (最小-最大归一化)                              │
    │   适用: 需要固定范围(如[0,1])，特征分布均匀                  │
    │   优点: 保持特征的原始分布形状                               │
    │   缺点: 对异常值非常敏感                                     │
    │   推荐: 神经网络、图像处理、需要固定范围的算法               │
    ├─────────────────────────────────────────────────────────────┤
    │ RobustScaler (鲁棒缩放器)                                   │
    │   适用: 数据包含较多异常值                                   │
    │   优点: 对异常值鲁棒，使用中位数和IQR                        │
    │   缺点: 不保证特定的输出范围                                 │
    │   推荐: 有异常值的数据集                                     │
    ├─────────────────────────────────────────────────────────────┤
    │ MaxAbsScaler (最大绝对值缩放)                               │
    │   适用: 稀疏数据、已中心化的数据                             │
    │   优点: 保持稀疏性，缩放到[-1,1]                             │
    │   缺点: 对异常值敏感                                         │
    │   推荐: 稀疏矩阵、文本数据(TF-IDF后)                         │
    ├─────────────────────────────────────────────────────────────┤
    │ Normalizer (L2归一化)                                       │
    │   适用: 文本分类、聚类，关注方向不关注幅度                   │
    │   优点: 每个样本独立缩放，保持方向                           │
    │   缺点: 改变特征间的相对关系                                 │
    │   推荐: 文本分类、余弦相似度、聚类                           │
    ├─────────────────────────────────────────────────────────────┤
    │ QuantileTransformer (分位数转换)                            │
    │   适用: 非线性关系、偏态分布                                 │
    │   优点: 对异常值鲁棒，可转换为任意分布                       │
    │   缺点: 改变特征间的线性关系                                 │
    │   推荐: 偏态分布严重、需要均匀或正态分布                     │
    ├─────────────────────────────────────────────────────────────┤
    │ PowerTransformer (幂变换)                                   │
    │   适用: 偏态分布，需要转换为正态分布                         │
    │   优点: 使数据更接近正态分布                                 │
    │   缺点: 仅适用于连续数据                                     │
    │   推荐: 线性模型、假设正态性的算法                           │
    └─────────────────────────────────────────────────────────────┘
    
    💡 快速选择建议:
    1. 默认选择: StandardScaler (适用于大多数情况)
    2. 有异常值: RobustScaler 或 QuantileTransformer
    3. 神经网络: MinMaxScaler 或 StandardScaler
    4. 树模型: 通常不需要缩放 (决策树、随机森林、XGBoost)
    5. 距离算法: 必须缩放 (KNN、K-Means、SVM)
    6. 偏态分布: PowerTransformer 或 QuantileTransformer
    """
    
    print(guide)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("数据标准化与归一化 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 1. StandardScaler
    standard_scaler, df_standard = standard_scaler_demo(df)
    
    # 2. MinMaxScaler
    minmax_scaler, df_minmax = minmax_scaler_demo(df)
    
    # 3. RobustScaler
    robust_scaler, df_robust = robust_scaler_demo(df)
    
    # 4. MaxAbsScaler
    maxabs_scaler, df_maxabs = maxabs_scaler_demo(df)
    
    # 5. Normalizer
    normalizer, df_normalized = normalizer_demo(df)
    
    # 6. QuantileTransformer
    qt_uniform, qt_normal = quantile_transformer_demo(df)
    
    # 7. PowerTransformer
    power_scaler, df_power = power_transformer_demo(df)
    
    # 8. 比较不同方法
    comparison_results = compare_scalers(df)
    
    # 9. 生产环境最佳实践
    prod_scaler, X_train_scaled, X_test_scaled = production_best_practices(df)
    
    # 10. 选择指南
    scaling_selection_guide()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
