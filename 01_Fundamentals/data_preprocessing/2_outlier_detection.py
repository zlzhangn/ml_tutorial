"""
异常值检测与处理 - 生产环境数据预处理技术
Outlier Detection and Handling for Production Environment

本文件演示了检测和处理异常值的各种方法，适用于生产环境
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_sample_data_with_outliers():
    """创建包含异常值的示例数据集"""
    np.random.seed(42)
    
    # 正常数据
    normal_data = {
        'age': np.random.normal(35, 10, 95).astype(int),
        'salary': np.random.normal(60000, 15000, 95),
        'experience': np.random.normal(8, 4, 95),
        'score': np.random.normal(80, 10, 95)
    }
    
    # 添加异常值
    outlier_data = {
        'age': [150, 200, -10, 5, 250],  # 异常年龄
        'salary': [500000, 1000, 1000000, 100, 800000],  # 异常薪资
        'experience': [50, -5, 100, 0.5, 60],  # 异常经验
        'score': [150, 10, 200, 5, 180]  # 异常分数
    }
    
    # 合并数据
    df_normal = pd.DataFrame(normal_data)
    df_outliers = pd.DataFrame(outlier_data)
    df = pd.concat([df_normal, df_outliers], ignore_index=True)
    
    print("数据集基本信息:")
    print(df.describe())
    
    return df


# ============================================================
# 1. 统计学方法 - Z-Score
# ============================================================

def zscore_detection(df, threshold=3):
    """使用Z-Score方法检测异常值"""
    print("\n" + "="*60)
    print("1. Z-Score 异常值检测")
    print("="*60)
    print(f"原理: 数据点距离均值超过{threshold}个标准差即为异常值")
    
    df_zscore = df.copy()
    
    # 计算每列的Z-Score
    z_scores = np.abs(stats.zscore(df))
    
    # 标记异常值（任意特征的Z-Score超过阈值）
    outlier_mask = (z_scores > threshold).any(axis=1)
    
    print(f"\n检测到的异常值数量: {outlier_mask.sum()}")
    print("\n异常值样本:")
    print(df[outlier_mask])
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, col in enumerate(df.columns):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(df.index[~outlier_mask], df.loc[~outlier_mask, col], 
                  c='blue', label='正常值', alpha=0.6)
        ax.scatter(df.index[outlier_mask], df.loc[outlier_mask, col], 
                  c='red', label='异常值', alpha=0.6)
        ax.set_xlabel('索引')
        ax.set_ylabel(col)
        ax.set_title(f'{col} - Z-Score检测')
        ax.legend()
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'zscore_detection.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存: zscore_detection.png")
    plt.close()
    
    return outlier_mask


# ============================================================
# 2. 统计学方法 - IQR (四分位距)
# ============================================================

def iqr_detection(df):
    """使用IQR方法检测异常值"""
    print("\n" + "="*60)
    print("2. IQR (四分位距) 异常值检测")
    print("="*60)
    print("原理: Q1 - 1.5*IQR < 正常值 < Q3 + 1.5*IQR")
    
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    outlier_details = {}
    
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 检测异常值
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers
        
        outlier_details[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': col_outliers.sum()
        }
        
        print(f"\n{col}:")
        print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
        print(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  异常值数量: {col_outliers.sum()}")
    
    print(f"\n总异常值数量: {outlier_mask.sum()}")
    print("\n异常值样本:")
    print(df[outlier_mask])
    
    # 箱线图可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, col in enumerate(df.columns):
        ax = axes[idx // 2, idx % 2]
        ax.boxplot(df[col], vert=True)
        ax.set_ylabel(col)
        ax.set_title(f'{col} - 箱线图')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'iqr_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    print("\n箱线图已保存: iqr_boxplot.png")
    plt.close()
    
    return outlier_mask, outlier_details


# ============================================================
# 3. 机器学习方法 - Isolation Forest
# ============================================================

def isolation_forest_detection(df, contamination=0.05):
    """使用Isolation Forest检测异常值"""
    print("\n" + "="*60)
    print("3. Isolation Forest 异常值检测")
    print("="*60)
    print(f"原理: 基于随机森林，异常值更容易被孤立")
    print(f"contamination参数: {contamination} (预期异常值比例)")
    
    # 训练Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(df)
    
    # -1 表示异常值，1 表示正常值
    outlier_mask = predictions == -1
    
    # 获取异常分数（越小越异常）
    scores = iso_forest.score_samples(df)
    
    print(f"\n检测到的异常值数量: {outlier_mask.sum()}")
    print("\n异常值样本及其异常分数:")
    outlier_df = df[outlier_mask].copy()
    outlier_df['anomaly_score'] = scores[outlier_mask]
    outlier_df = outlier_df.sort_values('anomaly_score')
    print(outlier_df)
    
    # 可视化异常分数分布
    plt.figure(figsize=(10, 6))
    plt.hist(scores[~outlier_mask], bins=50, alpha=0.6, label='正常值', color='blue')
    plt.hist(scores[outlier_mask], bins=20, alpha=0.6, label='异常值', color='red')
    plt.xlabel('异常分数 (越小越异常)')
    plt.ylabel('频数')
    plt.title('Isolation Forest - 异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'isolation_forest.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存: isolation_forest.png")
    plt.close()
    
    return outlier_mask, scores


# ============================================================
# 4. 机器学习方法 - Local Outlier Factor (LOF)
# ============================================================

def lof_detection(df, n_neighbors=20, contamination=0.05):
    """使用Local Outlier Factor检测异常值"""
    print("\n" + "="*60)
    print("4. Local Outlier Factor (LOF) 异常值检测")
    print("="*60)
    print(f"原理: 基于局部密度，异常值的局部密度明显低于邻居")
    print(f"n_neighbors: {n_neighbors}, contamination: {contamination}")
    
    # 训练LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(df)
    
    # -1 表示异常值，1 表示正常值
    outlier_mask = predictions == -1
    
    # 获取LOF分数（负值越大越异常）
    lof_scores = lof.negative_outlier_factor_
    
    print(f"\n检测到的异常值数量: {outlier_mask.sum()}")
    print("\n异常值样本及其LOF分数:")
    outlier_df = df[outlier_mask].copy()
    outlier_df['lof_score'] = lof_scores[outlier_mask]
    outlier_df = outlier_df.sort_values('lof_score')
    print(outlier_df)
    
    return outlier_mask, lof_scores


# ============================================================
# 5. 机器学习方法 - Elliptic Envelope (协方差估计)
# ============================================================

def elliptic_envelope_detection(df, contamination=0.05):
    """使用Elliptic Envelope检测异常值"""
    print("\n" + "="*60)
    print("5. Elliptic Envelope 异常值检测")
    print("="*60)
    print(f"原理: 假设数据服从高斯分布，使用协方差矩阵检测异常")
    print(f"contamination: {contamination}")
    
    # 训练Elliptic Envelope
    ee = EllipticEnvelope(contamination=contamination, random_state=42)
    predictions = ee.fit_predict(df)
    
    # -1 表示异常值，1 表示正常值
    outlier_mask = predictions == -1
    
    # 获取马氏距离
    mahalanobis_dist = ee.mahalanobis(df)
    
    print(f"\n检测到的异常值数量: {outlier_mask.sum()}")
    print("\n异常值样本及其马氏距离:")
    outlier_df = df[outlier_mask].copy()
    outlier_df['mahalanobis_distance'] = mahalanobis_dist[outlier_mask]
    outlier_df = outlier_df.sort_values('mahalanobis_distance', ascending=False)
    print(outlier_df)
    
    return outlier_mask, mahalanobis_dist


# ============================================================
# 6. 异常值处理方法
# ============================================================

def handle_outliers(df, outlier_mask):
    """演示多种异常值处理方法"""
    print("\n" + "="*60)
    print("6. 异常值处理方法")
    print("="*60)
    
    # 方法1: 删除异常值
    print("\n6.1 删除异常值:")
    df_removed = df[~outlier_mask].copy()
    print(f"原始数据量: {len(df)}, 删除后: {len(df_removed)}")
    print(df_removed.describe())
    
    # 方法2: 用边界值替换（Winsorization）
    print("\n6.2 边界值替换 (Winsorization):")
    df_winsorized = df.copy()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值替换为边界值
        df_winsorized[col] = df_winsorized[col].clip(lower_bound, upper_bound)
    
    print(df_winsorized.describe())
    
    # 方法3: 用均值/中位数替换
    print("\n6.3 用中位数替换异常值:")
    df_median_replaced = df.copy()
    for col in df.columns:
        median_val = df.loc[~outlier_mask, col].median()
        df_median_replaced.loc[outlier_mask, col] = median_val
    
    print(df_median_replaced[outlier_mask])
    
    # 方法4: 对数变换（适用于右偏数据）
    print("\n6.4 对数变换:")
    df_log = df.copy()
    for col in df.columns:
        # 确保所有值为正数
        min_val = df[col].min()
        if min_val <= 0:
            df_log[col] = np.log1p(df[col] - min_val + 1)
        else:
            df_log[col] = np.log1p(df[col])
    
    print(df_log.describe())
    
    # 方法5: 标记为缺失值，然后用填充方法处理
    print("\n6.5 标记为缺失值:")
    df_marked_missing = df.copy()
    df_marked_missing[outlier_mask] = np.nan
    print(f"标记为缺失值的数量: {df_marked_missing.isnull().sum().sum()}")
    
    return df_removed, df_winsorized, df_median_replaced, df_log, df_marked_missing


# ============================================================
# 7. 多种方法比较
# ============================================================

def compare_methods(df):
    """比较不同异常值检测方法的结果"""
    print("\n" + "="*60)
    print("7. 不同检测方法比较")
    print("="*60)
    
    # 使用不同方法检测
    zscore_mask = zscore_detection(df, threshold=3)
    iqr_mask, _ = iqr_detection(df)
    iso_mask, _ = isolation_forest_detection(df, contamination=0.05)
    lof_mask, _ = lof_detection(df, n_neighbors=20, contamination=0.05)
    ee_mask, _ = elliptic_envelope_detection(df, contamination=0.05)
    
    # 创建比较表
    comparison = pd.DataFrame({
        'Z-Score': zscore_mask.astype(int),
        'IQR': iqr_mask.astype(int),
        'Isolation Forest': iso_mask.astype(int),
        'LOF': lof_mask.astype(int),
        'Elliptic Envelope': ee_mask.astype(int)
    })
    
    # 计算每个样本被标记为异常的次数
    comparison['异常标记次数'] = comparison.sum(axis=1)
    
    print("\n各方法检测统计:")
    print(comparison.sum())
    
    print("\n被多个方法标记为异常的样本（强异常值）:")
    strong_outliers = comparison[comparison['异常标记次数'] >= 3]
    print(strong_outliers)
    
    # 可视化比较
    fig, ax = plt.subplots(figsize=(10, 6))
    method_counts = comparison.sum()[:-1]  # 排除最后一列
    method_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('检测方法')
    ax.set_ylabel('检测到的异常值数量')
    ax.set_title('不同异常值检测方法比较')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'method_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("\n比较结果已保存: method_comparison.png")
    plt.close()
    
    return comparison


# ============================================================
# 8. 生产环境最佳实践
# ============================================================

def production_best_practices(df):
    """生产环境异常值处理最佳实践"""
    print("\n" + "="*60)
    print("8. 生产环境最佳实践")
    print("="*60)
    
    # 步骤1: 先进行探索性数据分析
    print("\n步骤1: 数据分布分析")
    print(df.describe())
    
    # 步骤2: 使用多种方法检测，投票决定
    print("\n步骤2: 多方法投票检测")
    zscore_mask = (np.abs(stats.zscore(df)) > 3).any(axis=1)
    iqr_mask = pd.Series([False] * len(df))
    for col in df.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        iqr_mask |= (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_mask = iso_forest.fit_predict(df) == -1
    
    # 投票：至少2个方法认为是异常值
    vote_count = zscore_mask.astype(int) + iqr_mask.astype(int) + iso_mask.astype(int)
    final_outlier_mask = vote_count >= 2
    
    print(f"Z-Score检测: {zscore_mask.sum()}")
    print(f"IQR检测: {iqr_mask.sum()}")
    print(f"Isolation Forest检测: {iso_mask.sum()}")
    print(f"投票结果(≥2): {final_outlier_mask.sum()}")
    
    # 步骤3: 根据业务规则处理
    print("\n步骤3: 业务规则处理")
    df_processed = df.copy()
    
    # 对于明显不合理的值（如年龄<0或>120），直接设为边界值
    for col in df.columns:
        if col == 'age':
            df_processed[col] = df_processed[col].clip(0, 120)
        elif col == 'salary':
            df_processed[col] = df_processed[col].clip(0, 1000000)
        elif col == 'experience':
            df_processed[col] = df_processed[col].clip(0, 50)
        elif col == 'score':
            df_processed[col] = df_processed[col].clip(0, 100)
    
    # 步骤4: 对剩余异常值使用Winsorization
    print("\n步骤4: Winsorization处理")
    for col in df_processed.columns:
        Q1 = df_processed[col].quantile(0.05)
        Q3 = df_processed[col].quantile(0.95)
        df_processed[col] = df_processed[col].clip(Q1, Q3)
    
    print("\n处理后的数据统计:")
    print(df_processed.describe())
    
    # 步骤5: 保存异常检测器
    print("\n步骤5: 保存异常检测器")
    import joblib
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    detector_path = os.path.join(model_dir, 'outlier_detector.joblib')
    joblib.dump(iso_forest, detector_path)
    print(f"异常检测器已保存: {detector_path}")
    
    return df_processed, final_outlier_mask


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("异常值检测与处理 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data_with_outliers()
    
    # 1. Z-Score检测
    zscore_mask = zscore_detection(df)
    
    # 2. IQR检测
    iqr_mask, iqr_details = iqr_detection(df)
    
    # 3. Isolation Forest
    iso_mask, iso_scores = isolation_forest_detection(df)
    
    # 4. LOF
    lof_mask, lof_scores = lof_detection(df)
    
    # 5. Elliptic Envelope
    ee_mask, mahal_dist = elliptic_envelope_detection(df)
    
    # 6. 处理异常值
    df_removed, df_winsorized, df_median, df_log, df_missing = handle_outliers(df, iqr_mask)
    
    # 7. 方法比较
    comparison = compare_methods(df)
    
    # 8. 生产环境最佳实践
    df_processed, final_mask = production_best_practices(df)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n关键要点:")
    print("1. 统计方法(Z-Score, IQR): 简单快速，适用于单变量")
    print("2. ML方法(Isolation Forest, LOF): 适用于高维多变量")
    print("3. 使用多种方法投票，提高检测准确性")
    print("4. 结合业务规则，不能盲目删除异常值")
    print("5. Winsorization是生产环境常用的稳健处理方法")
    print("6. 保存检测器，确保训练集和测试集使用相同策略")


if __name__ == "__main__":
    main()
