"""
缺失值处理 - 生产环境数据预处理技术
Missing Values Handling for Production Environment

本文件演示了处理缺失值的各种方法，适用于生产环境
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # 需要显式启用
import warnings
warnings.filterwarnings('ignore')


def create_sample_data_with_missing():
    """创建包含缺失值的示例数据集"""
    data = {
        'age': [25, 30, np.nan, 45, 50, np.nan, 35, 40],
        'salary': [50000, 60000, 55000, np.nan, 75000, 65000, np.nan, 70000],
        'experience': [2, 5, 3, np.nan, 12, 8, 6, np.nan],
        'city': ['Beijing', 'Shanghai', np.nan, 'Beijing', 'Shanghai', 'Guangzhou', 'Beijing', np.nan],
        'score': [85, 90, 78, np.nan, 95, 88, 82, 91]
    }
    df = pd.DataFrame(data)
    print("原始数据（包含缺失值）:")
    print(df)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n缺失值百分比:")
    print(df.isnull().sum() / len(df) * 100)
    return df


# ============================================================
# 1. 简单填充方法 (Simple Imputation)
# ============================================================

def simple_imputation_demo(df):
    """演示简单填充方法"""
    print("\n" + "="*60)
    print("1. 简单填充方法 (Simple Imputation)")
    print("="*60)
    
    # 1.1 使用均值填充（适用于数值型特征）
    print("\n1.1 均值填充 (Mean Imputation):")
    imputer_mean = SimpleImputer(strategy='mean')
    numeric_cols = ['age', 'salary', 'experience', 'score']
    df_mean = df.copy()
    df_mean[numeric_cols] = imputer_mean.fit_transform(df[numeric_cols])
    print(df_mean[numeric_cols].head())
    
    # 1.2 使用中位数填充（对异常值更鲁棒）
    print("\n1.2 中位数填充 (Median Imputation):")
    imputer_median = SimpleImputer(strategy='median')
    df_median = df.copy()
    df_median[numeric_cols] = imputer_median.fit_transform(df[numeric_cols])
    print(df_median[numeric_cols].head())
    
    # 1.3 使用众数填充（适用于类别型特征）
    print("\n1.3 众数填充 (Most Frequent Imputation):")
    imputer_mode = SimpleImputer(strategy='most_frequent')
    df_mode = df.copy()
    df_mode['city'] = imputer_mode.fit_transform(df[['city']])
    print(df_mode['city'])
    
    # 1.4 使用常数填充
    print("\n1.4 常数填充 (Constant Imputation):")
    imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
    df_constant = df.copy()
    df_constant[numeric_cols] = imputer_constant.fit_transform(df[numeric_cols])
    print(df_constant[numeric_cols].head())
    
    return df_mean, df_median, df_mode, df_constant


# ============================================================
# 2. KNN填充 (KNN Imputation)
# ============================================================

def knn_imputation_demo(df):
    """演示KNN填充方法 - 基于相似样本填充"""
    print("\n" + "="*60)
    print("2. KNN填充 (KNN Imputation)")
    print("="*60)
    print("原理: 使用K个最近邻样本的均值来填充缺失值")
    
    # 只对数值型特征应用KNN填充
    numeric_cols = ['age', 'salary', 'experience', 'score']
    
    # n_neighbors=3: 使用3个最近邻
    # weights='distance': 使用距离加权
    imputer_knn = KNNImputer(n_neighbors=3, weights='distance')
    df_knn = df.copy()
    df_knn[numeric_cols] = imputer_knn.fit_transform(df[numeric_cols])
    
    print("\nKNN填充后的数据:")
    print(df_knn[numeric_cols])
    
    return df_knn


# ============================================================
# 3. 迭代填充 (Iterative/MICE Imputation)
# ============================================================

def iterative_imputation_demo(df):
    """演示迭代填充方法 - 多元特征插补"""
    print("\n" + "="*60)
    print("3. 迭代填充 (Iterative/MICE Imputation)")
    print("="*60)
    print("原理: 将每个缺失特征建模为其他特征的函数")
    
    numeric_cols = ['age', 'salary', 'experience', 'score']
    
    # max_iter: 最大迭代次数
    # random_state: 随机种子，保证可复现
    imputer_iterative = IterativeImputer(max_iter=10, random_state=42)
    df_iterative = df.copy()
    df_iterative[numeric_cols] = imputer_iterative.fit_transform(df[numeric_cols])
    
    print("\n迭代填充后的数据:")
    print(df_iterative[numeric_cols])
    
    return df_iterative


# ============================================================
# 4. 前向填充和后向填充 (Forward/Backward Fill)
# ============================================================

def forward_backward_fill_demo(df):
    """演示前向/后向填充 - 适用于时间序列数据"""
    print("\n" + "="*60)
    print("4. 前向/后向填充 (Forward/Backward Fill)")
    print("="*60)
    
    # 前向填充：用前一个值填充
    print("\n4.1 前向填充 (Forward Fill):")
    df_ffill = df.fillna(method='ffill')
    print(df_ffill)
    
    # 后向填充：用后一个值填充
    print("\n4.2 后向填充 (Backward Fill):")
    df_bfill = df.fillna(method='bfill')
    print(df_bfill)
    
    # 组合使用：先前向后后向
    print("\n4.3 组合填充 (Forward then Backward):")
    df_combined = df.fillna(method='ffill').fillna(method='bfill')
    print(df_combined)
    
    return df_ffill, df_bfill, df_combined


# ============================================================
# 5. 插值法 (Interpolation)
# ============================================================

def interpolation_demo(df):
    """演示插值法 - 适用于连续数据"""
    print("\n" + "="*60)
    print("5. 插值法 (Interpolation)")
    print("="*60)
    
    # 线性插值
    print("\n5.1 线性插值 (Linear Interpolation):")
    df_linear = df.interpolate(method='linear')
    print(df_linear[['age', 'salary', 'experience', 'score']])
    
    # 多项式插值
    print("\n5.2 多项式插值 (Polynomial Interpolation):")
    df_poly = df.interpolate(method='polynomial', order=2)
    print(df_poly[['age', 'salary', 'experience', 'score']])
    
    return df_linear, df_poly


# ============================================================
# 6. 删除缺失值 (Drop Missing Values)
# ============================================================

def drop_missing_demo(df):
    """演示删除缺失值的方法"""
    print("\n" + "="*60)
    print("6. 删除缺失值 (Drop Missing Values)")
    print("="*60)
    
    # 删除包含任何缺失值的行
    print("\n6.1 删除包含缺失值的行:")
    df_dropna_rows = df.dropna()
    print(f"原始行数: {len(df)}, 删除后行数: {len(df_dropna_rows)}")
    print(df_dropna_rows)
    
    # 删除包含缺失值的列（谨慎使用）
    print("\n6.2 删除包含缺失值的列:")
    df_dropna_cols = df.dropna(axis=1)
    print(f"原始列数: {df.shape[1]}, 删除后列数: {df_dropna_cols.shape[1]}")
    print(df_dropna_cols.columns.tolist())
    
    # 删除缺失值比例超过阈值的行
    print("\n6.3 删除缺失值超过阈值的行 (thresh=4, 至少4个非空值):")
    df_dropna_thresh = df.dropna(thresh=4)
    print(f"原始行数: {len(df)}, 删除后行数: {len(df_dropna_thresh)}")
    print(df_dropna_thresh)
    
    return df_dropna_rows, df_dropna_cols, df_dropna_thresh


# ============================================================
# 7. 生产环境最佳实践
# ============================================================

def production_best_practices(df):
    """演示生产环境中的最佳实践"""
    print("\n" + "="*60)
    print("7. 生产环境最佳实践")
    print("="*60)
    
    # 7.1 先分析缺失模式
    print("\n7.1 缺失值分析:")
    missing_info = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100,
        '数据类型': df.dtypes
    })
    print(missing_info)
    
    # 7.2 根据特征类型和缺失比例选择策略
    df_processed = df.copy()
    
    # 对于数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n7.2 数值型特征: {numeric_cols}")
    
    for col in numeric_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio < 0.05:  # 缺失率 < 5%，使用均值
            df_processed[col].fillna(df[col].mean(), inplace=True)
            print(f"  - {col}: 缺失率 {missing_ratio:.2%}, 使用均值填充")
        elif missing_ratio < 0.30:  # 缺失率 5-30%，使用中位数
            df_processed[col].fillna(df[col].median(), inplace=True)
            print(f"  - {col}: 缺失率 {missing_ratio:.2%}, 使用中位数填充")
        else:  # 缺失率 > 30%，考虑删除或使用KNN
            print(f"  - {col}: 缺失率 {missing_ratio:.2%}, 建议删除或使用高级方法")
    
    # 对于类别型特征
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n7.3 类别型特征: {categorical_cols}")
    
    for col in categorical_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio < 0.30:  # 缺失率 < 30%，使用众数
            df_processed[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  - {col}: 缺失率 {missing_ratio:.2%}, 使用众数填充")
        else:  # 缺失率 > 30%，使用特殊类别
            df_processed[col].fillna('Unknown', inplace=True)
            print(f"  - {col}: 缺失率 {missing_ratio:.2%}, 使用'Unknown'填充")
    
    print("\n7.4 处理后的数据:")
    print(df_processed)
    print("\n缺失值检查:")
    print(df_processed.isnull().sum())
    
    return df_processed


# ============================================================
# 8. 保存和加载填充器（用于生产环境）
# ============================================================

def save_load_imputer_demo(df):
    """演示如何保存和加载填充器，用于生产环境"""
    print("\n" + "="*60)
    print("8. 保存和加载填充器（生产环境）")
    print("="*60)
    
    import joblib
    import os
    
    # 创建模型保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练填充器
    numeric_cols = ['age', 'salary', 'experience', 'score']
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df[numeric_cols])
    
    # 保存填充器
    imputer_path = os.path.join(model_dir, 'imputer.joblib')
    joblib.dump(imputer, imputer_path)
    print(f"\n填充器已保存到: {imputer_path}")
    
    # 加载填充器（模拟生产环境）
    loaded_imputer = joblib.load(imputer_path)
    print("填充器已加载")
    
    # 使用加载的填充器处理新数据
    df_transformed = df.copy()
    df_transformed[numeric_cols] = loaded_imputer.transform(df[numeric_cols])
    print("\n使用加载的填充器处理后的数据:")
    print(df_transformed[numeric_cols])
    
    return loaded_imputer


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("缺失值处理 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data_with_missing()
    
    # 1. 简单填充
    df_mean, df_median, df_mode, df_constant = simple_imputation_demo(df)
    
    # 2. KNN填充
    df_knn = knn_imputation_demo(df)
    
    # 3. 迭代填充
    df_iterative = iterative_imputation_demo(df)
    
    # 4. 前向/后向填充
    df_ffill, df_bfill, df_combined = forward_backward_fill_demo(df)
    
    # 5. 插值法
    df_linear, df_poly = interpolation_demo(df)
    
    # 6. 删除缺失值
    df_dropna_rows, df_dropna_cols, df_dropna_thresh = drop_missing_demo(df)
    
    # 7. 生产环境最佳实践
    df_processed = production_best_practices(df)
    
    # 8. 保存和加载填充器
    loaded_imputer = save_load_imputer_demo(df)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n关键要点:")
    print("1. 根据缺失比例选择不同策略")
    print("2. 数值型特征：均值/中位数/KNN/迭代填充")
    print("3. 类别型特征：众数/常数填充")
    print("4. 时间序列：前向/后向填充/插值")
    print("5. 生产环境：保存填充器，确保训练集和测试集使用相同策略")


if __name__ == "__main__":
    main()
