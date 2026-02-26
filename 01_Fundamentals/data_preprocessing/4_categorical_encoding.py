"""
类别编码 - 生产环境数据预处理技术
Categorical Encoding for Production Environment

本文件演示了处理类别特征的各种编码方法，适用于生产环境
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce  # 需要安装: pip install category-encoders
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_sample_data():
    """创建包含类别特征的示例数据集"""
    np.random.seed(42)
    
    n_samples = 200
    
    # 类别特征
    cities = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou']
    education = ['High School', 'Bachelor', 'Master', 'PhD']
    job_types = ['IT', 'Finance', 'Education', 'Healthcare', 'Manufacturing', 
                 'Retail', 'Government', 'Other']
    
    data = {
        'city': np.random.choice(cities, n_samples),
        'education': np.random.choice(education, n_samples),
        'job_type': np.random.choice(job_types, n_samples),
        'age': np.random.randint(22, 60, n_samples),
        'salary': np.random.randint(30000, 150000, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    print("数据集基本信息:")
    print(df.head(10))
    print("\n数据类型:")
    print(df.dtypes)
    print("\n类别特征的唯一值数量:")
    for col in ['city', 'education', 'job_type']:
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    {df[col].unique()}")
    
    return df


# ============================================================
# 1. LabelEncoder (标签编码)
# ============================================================

def label_encoder_demo(df):
    """LabelEncoder: 将类别转换为整数"""
    print("\n" + "="*60)
    print("1. LabelEncoder (标签编码)")
    print("="*60)
    print("原理: 将类别映射为0到n-1的整数")
    print("适用: 有序类别特征、目标变量")
    print("注意: 会引入人工的顺序关系，不适合无序类别")
    
    df_encoded = df.copy()
    
    # 对每个类别特征应用LabelEncoder
    encoders = {}
    for col in ['city', 'education', 'job_type']:
        le = LabelEncoder()
        df_encoded[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        
        print(f"\n{col} 的编码映射:")
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        for original, encoded in mapping.items():
            print(f"  {original} -> {encoded}")
    
    print("\n编码后的数据示例:")
    print(df_encoded[['city', 'city_encoded', 'education', 'education_encoded', 
                      'job_type', 'job_type_encoded']].head(10))
    
    # 逆变换示例
    print("\n逆变换 (将编码转回原始类别):")
    sample_encoded = df_encoded['city_encoded'].head(5).values
    sample_decoded = encoders['city'].inverse_transform(sample_encoded)
    print(f"编码值: {sample_encoded}")
    print(f"原始类别: {sample_decoded}")
    
    return encoders, df_encoded


# ============================================================
# 2. OrdinalEncoder (序数编码)
# ============================================================

def ordinal_encoder_demo(df):
    """OrdinalEncoder: 适用于有序类别"""
    print("\n" + "="*60)
    print("2. OrdinalEncoder (序数编码)")
    print("="*60)
    print("原理: 按指定顺序将类别映射为整数")
    print("适用: 有明确顺序的类别特征（如教育程度、等级）")
    
    # 定义教育程度的顺序
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
    
    print(f"\n定义的教育顺序: {education_order}")
    
    # 应用OrdinalEncoder
    encoder = OrdinalEncoder(categories=[education_order], 
                            handle_unknown='use_encoded_value', 
                            unknown_value=-1)
    
    df_encoded = df.copy()
    df_encoded['education_ordinal'] = encoder.fit_transform(df[['education']])
    
    print("\n编码后的数据:")
    comparison = df_encoded[['education', 'education_ordinal']].drop_duplicates().sort_values('education_ordinal')
    print(comparison)
    
    # 可视化顺序
    edu_stats = df_encoded.groupby(['education', 'education_ordinal'])['salary'].mean().reset_index()
    edu_stats = edu_stats.sort_values('education_ordinal')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edu_stats['education'], edu_stats['salary'], color='steelblue', alpha=0.7)
    ax.set_xlabel('教育程度')
    ax.set_ylabel('平均薪资')
    ax.set_title('教育程度与薪资关系（体现顺序性）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'ordinal_encoder.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: ordinal_encoder.png")
    plt.close()
    
    return encoder, df_encoded


# ============================================================
# 3. OneHotEncoder (独热编码)
# ============================================================

def onehot_encoder_demo(df):
    """OneHotEncoder: 创建二进制列"""
    print("\n" + "="*60)
    print("3. OneHotEncoder (独热编码)")
    print("="*60)
    print("原理: 为每个类别创建一个二进制列")
    print("适用: 无序类别特征，类别数量不太多")
    print("注意: 高基数类别会导致维度爆炸")
    
    # 方法1: 使用sklearn的OneHotEncoder
    print("\n方法1: sklearn.preprocessing.OneHotEncoder")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    city_encoded = encoder.fit_transform(df[['city']])
    
    # 创建列名
    feature_names = encoder.get_feature_names_out(['city'])
    df_onehot = pd.DataFrame(city_encoded, columns=feature_names)
    
    print(f"\n原始特征: city ({df['city'].nunique()} 个类别)")
    print(f"编码后特征数: {df_onehot.shape[1]}")
    print("\n编码后的数据示例:")
    print(pd.concat([df[['city']].reset_index(drop=True), df_onehot], axis=1).head(10))
    
    # 方法2: 使用pandas的get_dummies
    print("\n方法2: pandas.get_dummies (更简单)")
    df_dummies = pd.get_dummies(df[['city', 'job_type']], prefix=['city', 'job'])
    print(f"\n编码后的特征:")
    print(df_dummies.columns.tolist())
    print("\n数据示例:")
    print(df_dummies.head())
    
    # 避免虚拟变量陷阱 (Dummy Variable Trap)
    print("\n方法3: drop_first=True (避免多重共线性)")
    df_dummies_drop = pd.get_dummies(df[['city', 'job_type']], 
                                     prefix=['city', 'job'], 
                                     drop_first=True)
    print(f"原始列数: {df_dummies.shape[1]}")
    print(f"drop_first后列数: {df_dummies_drop.shape[1]}")
    
    return encoder, df_onehot, df_dummies


# ============================================================
# 4. Binary Encoding (二进制编码)
# ============================================================

def binary_encoder_demo(df):
    """Binary Encoding: 更节省空间的编码方式"""
    print("\n" + "="*60)
    print("4. Binary Encoding (二进制编码)")
    print("="*60)
    print("原理: 先用整数编码，再转为二进制")
    print("适用: 高基数类别特征，比OneHot更节省空间")
    print("优势: n个类别只需log2(n)列，而OneHot需要n列")
    
    # 使用category_encoders库
    encoder = ce.BinaryEncoder(cols=['job_type'])
    df_encoded = encoder.fit_transform(df[['job_type']])
    
    print(f"\n原始特征 job_type: {df['job_type'].nunique()} 个类别")
    print(f"二进制编码后: {df_encoded.shape[1]} 列")
    print("\n编码后的数据示例:")
    print(pd.concat([df[['job_type']].reset_index(drop=True), df_encoded], axis=1).head(10))
    
    # 比较维度
    comparison_data = {
        '编码方法': ['OneHot', 'Binary'],
        '类别数量': [df['job_type'].nunique(), df['job_type'].nunique()],
        '生成列数': [df['job_type'].nunique(), df_encoded.shape[1]]
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\n维度比较:")
    print(comparison_df)
    
    return encoder, df_encoded


# ============================================================
# 5. Target Encoding (目标编码)
# ============================================================

def target_encoder_demo(df):
    """Target Encoding: 基于目标变量的编码"""
    print("\n" + "="*60)
    print("5. Target Encoding (目标编码)")
    print("="*60)
    print("原理: 用类别对应的目标均值替换类别")
    print("适用: 高基数类别特征")
    print("注意: 容易过拟合，需要使用交叉验证")
    
    # 使用category_encoders库的TargetEncoder
    encoder = ce.TargetEncoder(cols=['city', 'job_type'], min_samples_leaf=10, smoothing=1.0)
    
    # 重要：分离训练集和测试集
    X = df[['city', 'job_type', 'age', 'salary']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 在训练集上fit
    encoder.fit(X_train[['city', 'job_type']], y_train)
    
    # 分别transform训练集和测试集
    X_train_encoded = encoder.transform(X_train[['city', 'job_type']])
    X_test_encoded = encoder.transform(X_test[['city', 'job_type']])
    
    print("\n训练集编码示例 (city):")
    train_comparison = pd.DataFrame({
        'city': X_train['city'].values[:10],
        'city_encoded': X_train_encoded['city'].values[:10],
        'target': y_train.values[:10]
    })
    print(train_comparison)
    
    print("\n各城市的目标均值:")
    city_target_mean = pd.DataFrame({
        'city': df['city'].unique(),
        'target_mean': [df[df['city'] == city]['target'].mean() for city in df['city'].unique()]
    }).sort_values('target_mean', ascending=False)
    print(city_target_mean)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(city_target_mean['city'], city_target_mean['target_mean'], color='coral', alpha=0.7)
    ax.set_xlabel('目标均值')
    ax.set_ylabel('城市')
    ax.set_title('Target Encoding - 各城市的目标均值')
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'target_encoder.png'), 
                dpi=300, bbox_inches='tight')
    print("\n可视化已保存: target_encoder.png")
    plt.close()
    
    return encoder, X_train_encoded, X_test_encoded


# ============================================================
# 6. Frequency Encoding (频率编码)
# ============================================================

def frequency_encoder_demo(df):
    """Frequency Encoding: 用出现频率替换类别"""
    print("\n" + "="*60)
    print("6. Frequency Encoding (频率编码)")
    print("="*60)
    print("原理: 用类别的出现频率替换类别")
    print("适用: 类别的频率与目标相关时")
    
    df_encoded = df.copy()
    
    # 计算频率
    for col in ['city', 'job_type']:
        # 计算每个类别的出现次数
        freq_map = df[col].value_counts(normalize=True).to_dict()
        
        # 应用映射
        df_encoded[f'{col}_freq'] = df[col].map(freq_map)
        
        print(f"\n{col} 的频率编码:")
        freq_df = pd.DataFrame({
            'category': list(freq_map.keys()),
            'frequency': list(freq_map.values())
        }).sort_values('frequency', ascending=False)
        print(freq_df)
    
    print("\n编码后的数据示例:")
    print(df_encoded[['city', 'city_freq', 'job_type', 'job_type_freq']].head(10))
    
    return df_encoded


# ============================================================
# 7. Hash Encoding (哈希编码)
# ============================================================

def hash_encoder_demo(df):
    """Hash Encoding: 使用哈希函数编码"""
    print("\n" + "="*60)
    print("7. Hash Encoding (哈希编码)")
    print("="*60)
    print("原理: 使用哈希函数将类别映射到固定数量的桶")
    print("适用: 极高基数类别特征、内存受限")
    print("注意: 可能有哈希冲突")
    
    # 方法1: sklearn的FeatureHasher
    print("\n方法1: sklearn.feature_extraction.FeatureHasher")
    n_features = 5  # 哈希空间大小
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    
    # 准备数据（需要是字符串列表的列表）
    city_list = [[city] for city in df['city']]
    hashed_features = hasher.transform(city_list).toarray()
    
    df_hashed = pd.DataFrame(hashed_features, columns=[f'hash_{i}' for i in range(n_features)])
    
    print(f"\n原始类别数: {df['city'].nunique()}")
    print(f"哈希空间大小: {n_features}")
    print("\n哈希编码后的数据示例:")
    print(pd.concat([df[['city']].reset_index(drop=True), df_hashed], axis=1).head(10))
    
    # 方法2: category_encoders的HashingEncoder
    print("\n方法2: category_encoders.HashingEncoder")
    encoder = ce.HashingEncoder(cols=['job_type'], n_components=6)
    df_hash_encoded = encoder.fit_transform(df[['job_type']])
    
    print(f"\n原始类别数: {df['job_type'].nunique()}")
    print(f"哈希编码后列数: {df_hash_encoded.shape[1]}")
    print("\n编码后的数据示例:")
    print(df_hash_encoded.head(10))
    
    return hasher, encoder, df_hashed


# ============================================================
# 8. 不同编码方法的比较
# ============================================================

def compare_encoders(df):
    """比较不同编码方法对模型性能的影响"""
    print("\n" + "="*60)
    print("8. 不同编码方法比较")
    print("="*60)
    
    X = df[['city', 'education', 'job_type', 'age', 'salary']]
    y = df['target']
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = []
    
    # 1. Label Encoding
    print("\n测试 Label Encoding...")
    X_train_label = X_train.copy()
    X_test_label = X_test.copy()
    for col in ['city', 'education', 'job_type']:
        le = LabelEncoder()
        X_train_label[col] = le.fit_transform(X_train[col])
        X_test_label[col] = le.transform(X_test[col])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_label, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train_label))
    test_acc = accuracy_score(y_test, rf.predict(X_test_label))
    results.append({'方法': 'Label Encoding', '特征数': X_train_label.shape[1], 
                   '训练准确率': train_acc, '测试准确率': test_acc})
    
    # 2. OneHot Encoding
    print("测试 OneHot Encoding...")
    X_train_onehot = pd.get_dummies(X_train, columns=['city', 'education', 'job_type'])
    X_test_onehot = pd.get_dummies(X_test, columns=['city', 'education', 'job_type'])
    # 确保训练集和测试集有相同的列
    X_train_onehot, X_test_onehot = X_train_onehot.align(X_test_onehot, join='left', axis=1, fill_value=0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_onehot, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train_onehot))
    test_acc = accuracy_score(y_test, rf.predict(X_test_onehot))
    results.append({'方法': 'OneHot Encoding', '特征数': X_train_onehot.shape[1], 
                   '训练准确率': train_acc, '测试准确率': test_acc})
    
    # 3. Target Encoding
    print("测试 Target Encoding...")
    encoder_target = ce.TargetEncoder(cols=['city', 'education', 'job_type'])
    X_train_target = X_train.copy()
    X_test_target = X_test.copy()
    X_train_target[['city', 'education', 'job_type']] = encoder_target.fit_transform(
        X_train[['city', 'education', 'job_type']], y_train)
    X_test_target[['city', 'education', 'job_type']] = encoder_target.transform(
        X_test[['city', 'education', 'job_type']])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_target, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train_target))
    test_acc = accuracy_score(y_test, rf.predict(X_test_target))
    results.append({'方法': 'Target Encoding', '特征数': X_train_target.shape[1], 
                   '训练准确率': train_acc, '测试准确率': test_acc})
    
    # 4. Frequency Encoding
    print("测试 Frequency Encoding...")
    X_train_freq = X_train.copy()
    X_test_freq = X_test.copy()
    for col in ['city', 'education', 'job_type']:
        freq_map = X_train[col].value_counts(normalize=True).to_dict()
        X_train_freq[col] = X_train[col].map(freq_map)
        X_test_freq[col] = X_test[col].map(freq_map).fillna(0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_freq, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train_freq))
    test_acc = accuracy_score(y_test, rf.predict(X_test_freq))
    results.append({'方法': 'Frequency Encoding', '特征数': X_train_freq.shape[1], 
                   '训练准确率': train_acc, '测试准确率': test_acc})
    
    # 创建结果表
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("编码方法性能比较:")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 特征数量比较
    axes[0].bar(results_df['方法'], results_df['特征数'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('编码方法')
    axes[0].set_ylabel('特征数量')
    axes[0].set_title('不同编码方法的特征数量')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 准确率比较
    x = np.arange(len(results_df))
    width = 0.35
    axes[1].bar(x - width/2, results_df['训练准确率'], width, label='训练准确率', alpha=0.8)
    axes[1].bar(x + width/2, results_df['测试准确率'], width, label='测试准确率', alpha=0.8)
    axes[1].set_xlabel('编码方法')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('不同编码方法的模型性能')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['方法'], rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'encoder_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("\n比较结果已保存: encoder_comparison.png")
    plt.close()
    
    return results_df


# ============================================================
# 9. 生产环境最佳实践
# ============================================================

def production_best_practices(df):
    """生产环境类别编码最佳实践"""
    print("\n" + "="*60)
    print("9. 生产环境最佳实践")
    print("="*60)
    
    X = df[['city', 'education', 'job_type', 'age', 'salary']]
    y = df['target']
    
    # 步骤1: 划分数据集
    print("\n步骤1: 先划分数据集")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 步骤2: 选择合适的编码器
    print("\n步骤2: 为不同类型的特征选择编码器")
    print("  - education (有序): OrdinalEncoder")
    print("  - city (低基数无序): OneHotEncoder")
    print("  - job_type (中高基数): TargetEncoder")
    
    # 有序编码
    ordinal_encoder = OrdinalEncoder(
        categories=[['High School', 'Bachelor', 'Master', 'PhD']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # OneHot编码
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Target编码
    target_encoder = ce.TargetEncoder(cols=['job_type'], min_samples_leaf=10, smoothing=1.0)
    
    # 步骤3: 在训练集上fit
    print("\n步骤3: 在训练集上拟合所有编码器")
    ordinal_encoder.fit(X_train[['education']])
    onehot_encoder.fit(X_train[['city']])
    target_encoder.fit(X_train[['job_type']], y_train)
    
    # 步骤4: 转换数据
    print("\n步骤4: 转换训练集和测试集")
    
    # 转换训练集
    edu_train = ordinal_encoder.transform(X_train[['education']])
    city_train = onehot_encoder.transform(X_train[['city']])
    job_train = target_encoder.transform(X_train[['job_type']])
    
    X_train_processed = np.hstack([
        edu_train,
        city_train,
        job_train,
        X_train[['age', 'salary']].values
    ])
    
    # 转换测试集
    edu_test = ordinal_encoder.transform(X_test[['education']])
    city_test = onehot_encoder.transform(X_test[['city']])
    job_test = target_encoder.transform(X_test[['job_type']])
    
    X_test_processed = np.hstack([
        edu_test,
        city_test,
        job_test,
        X_test[['age', 'salary']].values
    ])
    
    print(f"训练集形状: {X_train_processed.shape}")
    print(f"测试集形状: {X_test_processed.shape}")
    
    # 步骤5: 保存编码器
    print("\n步骤5: 保存所有编码器")
    import joblib
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(ordinal_encoder, os.path.join(model_dir, 'ordinal_encoder.joblib'))
    joblib.dump(onehot_encoder, os.path.join(model_dir, 'onehot_encoder.joblib'))
    joblib.dump(target_encoder, os.path.join(model_dir, 'target_encoder.joblib'))
    
    print("所有编码器已保存")
    
    # 步骤6: 处理未知类别
    print("\n步骤6: 处理未知类别的策略")
    print("  - OrdinalEncoder: use_encoded_value=-1")
    print("  - OneHotEncoder: handle_unknown='ignore' (全0向量)")
    print("  - TargetEncoder: 使用全局均值")
    
    return ordinal_encoder, onehot_encoder, target_encoder, X_train_processed, X_test_processed


# ============================================================
# 10. 编码方法选择指南
# ============================================================

def encoding_selection_guide():
    """提供编码方法选择指南"""
    print("\n" + "="*60)
    print("10. 类别编码方法选择指南")
    print("="*60)
    
    guide = """
    ┌─────────────────────────────────────────────────────────────┐
    │               类别编码方法选择指南                           │
    ├─────────────────────────────────────────────────────────────┤
    │ Label Encoding (标签编码)                                   │
    │   适用: 有序类别、目标变量                                   │
    │   优点: 简单、节省内存                                       │
    │   缺点: 引入虚假的顺序关系                                   │
    │   推荐: 树模型可用，线性模型需谨慎                           │
    ├─────────────────────────────────────────────────────────────┤
    │ Ordinal Encoding (序数编码)                                 │
    │   适用: 有明确顺序的类别(教育、评级、等级)                   │
    │   优点: 保留有意义的顺序信息                                 │
    │   缺点: 需要人工定义顺序                                     │
    │   推荐: 所有模型类型                                         │
    ├─────────────────────────────────────────────────────────────┤
    │ OneHot Encoding (独热编码)                                  │
    │   适用: 低基数无序类别(< 10-15个类别)                        │
    │   优点: 不引入虚假关系，模型易理解                           │
    │   缺点: 高基数时维度爆炸                                     │
    │   推荐: 线性模型、神经网络                                   │
    ├─────────────────────────────────────────────────────────────┤
    │ Binary Encoding (二进制编码)                                │
    │   适用: 中高基数类别(10-100个)                               │
    │   优点: 比OneHot节省空间(log2(n)列)                          │
    │   缺点: 编码不够直观                                         │
    │   推荐: 高基数特征                                           │
    ├─────────────────────────────────────────────────────────────┤
    │ Target Encoding (目标编码)                                  │
    │   适用: 高基数类别(> 100个)                                  │
    │   优点: 维度低、可能提升性能                                 │
    │   缺点: 易过拟合、需要交叉验证                               │
    │   推荐: 树模型、高基数特征                                   │
    ├─────────────────────────────────────────────────────────────┤
    │ Frequency Encoding (频率编码)                               │
    │   适用: 频率与目标相关的类别                                 │
    │   优点: 简单、维度低                                         │
    │   缺点: 丢失类别本身的信息                                   │
    │   推荐: 补充特征                                             │
    ├─────────────────────────────────────────────────────────────┤
    │ Hash Encoding (哈希编码)                                    │
    │   适用: 极高基数(> 1000个)、内存受限                         │
    │   优点: 固定维度、处理未知类别                               │
    │   缺点: 哈希冲突、不可逆                                     │
    │   推荐: 超大规模数据、在线学习                               │
    └─────────────────────────────────────────────────────────────┘
    
    💡 决策树:
    
    类别特征
        │
        ├─ 有明确顺序？
        │   └─ 是 → Ordinal Encoding
        │   └─ 否 ↓
        │
        ├─ 类别数量 < 15？
        │   └─ 是 → OneHot Encoding
        │   └─ 否 ↓
        │
        ├─ 类别数量 < 100？
        │   └─ 是 → Binary Encoding 或 Target Encoding
        │   └─ 否 ↓
        │
        └─ 类别数量 > 100
            └─ Target Encoding 或 Hash Encoding
    
    ⚠️  重要注意事项:
    1. 先划分数据集，再进行编码
    2. 编码器只在训练集上fit
    3. 保存编码器用于生产环境
    4. 处理未知类别的策略要提前定义
    5. Target Encoding需要交叉验证避免过拟合
    6. 树模型对编码方式不敏感，线性模型很敏感
    """
    
    print(guide)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("类别编码 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 1. Label Encoding
    label_encoders, df_label = label_encoder_demo(df)
    
    # 2. Ordinal Encoding
    ordinal_encoder, df_ordinal = ordinal_encoder_demo(df)
    
    # 3. OneHot Encoding
    onehot_encoder, df_onehot_sklearn, df_onehot_pandas = onehot_encoder_demo(df)
    
    # 4. Binary Encoding
    binary_encoder, df_binary = binary_encoder_demo(df)
    
    # 5. Target Encoding
    target_encoder, X_train_target, X_test_target = target_encoder_demo(df)
    
    # 6. Frequency Encoding
    df_frequency = frequency_encoder_demo(df)
    
    # 7. Hash Encoding
    hash_hasher, hash_encoder, df_hash = hash_encoder_demo(df)
    
    # 8. 比较不同编码方法
    comparison_results = compare_encoders(df)
    
    # 9. 生产环境最佳实践
    ord_enc, oh_enc, tgt_enc, X_train_proc, X_test_proc = production_best_practices(df)
    
    # 10. 选择指南
    encoding_selection_guide()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
