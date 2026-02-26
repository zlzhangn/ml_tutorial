"""
Pipeline构建 - 生产环境数据预处理技术
ML Pipeline Construction for Production Environment

本文件演示了如何构建完整的机器学习Pipeline，适用于生产环境
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def create_sample_data():
    """创建示例数据集"""
    np.random.seed(42)
    
    n_samples = 500
    
    data = {
        # 数值特征
        'age': np.random.randint(20, 70, n_samples),
        'salary': np.random.randint(30000, 150000, n_samples),
        'experience': np.random.randint(0, 30, n_samples),
        'score': np.random.randint(60, 100, n_samples),
        
        # 类别特征
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'job_type': np.random.choice(['IT', 'Finance', 'Education', 'Healthcare', 'Other'], n_samples),
        
        # 目标变量
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值
    df.loc[np.random.choice(df.index, 30), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 25), 'salary'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'city'] = np.nan
    
    print("="*60)
    print("示例数据集")
    print("="*60)
    print(f"\n数据形状: {df.shape}")
    print(f"\n前5行:")
    print(df.head())
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    print(f"\n数据类型:")
    print(df.dtypes)
    
    return df


# ============================================================
# 1. 基础Pipeline
# ============================================================

def basic_pipeline_demo(df):
    """演示基础Pipeline"""
    print("\n" + "="*60)
    print("1. 基础Pipeline")
    print("="*60)
    print("原理: 将多个步骤串联成一个流程")
    
    # 准备数据（只使用数值特征简化演示）
    numeric_features = ['age', 'salary', 'experience', 'score']
    X = df[numeric_features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 构建Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # 步骤1: 填充缺失值
        ('scaler', StandardScaler()),                   # 步骤2: 标准化
        ('classifier', LogisticRegression(random_state=42))  # 步骤3: 分类器
    ])
    
    print("\nPipeline步骤:")
    for idx, (name, step) in enumerate(pipeline.steps, 1):
        print(f"  {idx}. {name}: {step.__class__.__name__}")
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 评估
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    print("\n优点:")
    print("  ✓ 代码简洁，避免重复")
    print("  ✓ 防止数据泄露（自动在训练集fit，测试集transform）")
    print("  ✓ 便于超参数调优")
    print("  ✓ 易于部署")
    
    return pipeline


# ============================================================
# 2. ColumnTransformer - 处理不同类型的特征
# ============================================================

def column_transformer_demo(df):
    """演示ColumnTransformer处理异构数据"""
    print("\n" + "="*60)
    print("2. ColumnTransformer - 处理不同类型特征")
    print("="*60)
    print("原理: 对不同列应用不同的转换")
    
    # 定义特征类型
    numeric_features = ['age', 'salary', 'experience', 'score']
    categorical_features = ['city', 'education', 'job_type']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 数值特征处理器
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理器
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 组合所有转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 完整Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("\nColumnTransformer配置:")
    print(f"  数值特征 ({len(numeric_features)}个): {numeric_features}")
    print(f"    → SimpleImputer(median) → StandardScaler")
    print(f"  类别特征 ({len(categorical_features)}个): {categorical_features}")
    print(f"    → SimpleImputer(constant) → OneHotEncoder")
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 评估
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 查看转换后的特征数量
    X_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    print(f"\n原始特征数: {X_train.shape[1]}")
    print(f"转换后特征数: {X_transformed.shape[1]}")
    
    return pipeline


# ============================================================
# 3. 自定义Transformer
# ============================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """自定义对数转换器"""
    
    def __init__(self, features=None):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if self.features is None:
            # 对所有数值列应用
            return np.log1p(X_copy)
        else:
            # 对指定列应用
            for feature in self.features:
                if feature in X_copy.columns:
                    X_copy[feature] = np.log1p(X_copy[feature])
            return X_copy


class OutlierClipper(BaseEstimator, TransformerMixin):
    """自定义异常值裁剪器"""
    
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.clip_values = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X_df.select_dtypes(include=[np.number]).columns:
            lower = X_df[col].quantile(self.lower_quantile)
            upper = X_df[col].quantile(self.upper_quantile)
            self.clip_values[col] = (lower, upper)
        return self
    
    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (lower, upper) in self.clip_values.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower, upper)
        return X_copy


def custom_transformer_demo(df):
    """演示自定义Transformer"""
    print("\n" + "="*60)
    print("3. 自定义Transformer")
    print("="*60)
    print("原理: 创建符合sklearn接口的自定义转换器")
    
    numeric_features = ['age', 'salary', 'experience', 'score']
    X = df[numeric_features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 构建包含自定义转换器的Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', OutlierClipper(lower_quantile=0.05, upper_quantile=0.95)),
        ('log_transform', LogTransformer(features=['salary'])),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("\nPipeline步骤 (包含自定义Transformer):")
    for idx, (name, step) in enumerate(pipeline.steps, 1):
        print(f"  {idx}. {name}: {step.__class__.__name__}")
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 评估
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    print("\n自定义Transformer的要点:")
    print("  1. 继承BaseEstimator和TransformerMixin")
    print("  2. 实现fit()和transform()方法")
    print("  3. __init__()中定义参数但不要访问数据")
    print("  4. fit()中学习参数，transform()中应用转换")
    
    return pipeline


# ============================================================
# 4. FeatureUnion - 并行特征工程
# ============================================================

def feature_union_demo(df):
    """演示FeatureUnion并行处理特征"""
    print("\n" + "="*60)
    print("4. FeatureUnion - 并行特征工程")
    print("="*60)
    print("原理: 并行应用多个转换器，然后合并结果")
    
    numeric_features = ['age', 'salary', 'experience', 'score']
    X = df[numeric_features].fillna(0)  # 简化演示
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建特征工程的并行流程
    feature_engineering = FeatureUnion([
        ('original', StandardScaler()),  # 原始特征标准化
        ('pca', PCA(n_components=2))     # PCA降维特征
    ])
    
    # 完整Pipeline
    pipeline = Pipeline([
        ('features', feature_engineering),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("\nFeatureUnion配置:")
    print("  并行分支1: StandardScaler (保留原始特征)")
    print("  并行分支2: PCA (提取主成分)")
    print("  → 合并所有特征")
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 评估
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 查看特征数量
    X_transformed = pipeline.named_steps['features'].transform(X_train)
    print(f"\n原始特征数: {X_train.shape[1]}")
    print(f"转换后特征数: {X_transformed.shape[1]} (4个标准化特征 + 2个PCA特征)")
    
    return pipeline


# ============================================================
# 5. 包含特征选择的Pipeline
# ============================================================

def feature_selection_pipeline_demo(df):
    """演示包含特征选择的Pipeline"""
    print("\n" + "="*60)
    print("5. 包含特征选择的Pipeline")
    print("="*60)
    print("原理: 在Pipeline中加入特征选择步骤")
    
    numeric_features = ['age', 'salary', 'experience', 'score']
    categorical_features = ['city', 'education', 'job_type']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 数值和类别特征预处理
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 完整Pipeline（包含特征选择）
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=10)),  # 选择最好的10个特征
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("\nPipeline配置:")
    print("  1. 数据预处理 (ColumnTransformer)")
    print("  2. 特征选择 (SelectKBest, k=10)")
    print("  3. 分类器 (RandomForestClassifier)")
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 评估
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 查看选择的特征
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X_train)
    selected_mask = pipeline.named_steps['feature_selection'].get_support()
    
    print(f"\n预处理后特征数: {X_preprocessed.shape[1]}")
    print(f"选择后特征数: {selected_mask.sum()}")
    
    return pipeline


# ============================================================
# 6. 超参数调优Pipeline
# ============================================================

def hyperparameter_tuning_demo(df):
    """演示Pipeline的超参数调优"""
    print("\n" + "="*60)
    print("6. Pipeline超参数调优")
    print("="*60)
    print("原理: 使用GridSearchCV同时调优预处理和模型参数")
    
    numeric_features = ['age', 'salary', 'experience', 'score']
    X = df[numeric_features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 构建Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 定义超参数网格（使用 步骤名__参数名 格式）
    param_grid = {
        'imputer__strategy': ['mean', 'median'],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
    
    print("\n超参数搜索空间:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n开始网格搜索...")
    grid_search.fit(X_train, y_train)
    
    # 最佳参数
    print("\n最佳参数:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # 最佳模型评估
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n最佳模型测试集准确率: {accuracy:.4f}")
    print(f"交叉验证最佳得分: {grid_search.best_score_:.4f}")
    
    return best_pipeline, grid_search


# ============================================================
# 7. 保存和加载Pipeline
# ============================================================

def save_load_pipeline_demo(pipeline):
    """演示Pipeline的保存和加载"""
    print("\n" + "="*60)
    print("7. 保存和加载Pipeline")
    print("="*60)
    print("原理: 使用joblib保存整个Pipeline（包括所有步骤）")
    
    # 保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    pipeline_path = os.path.join(model_dir, 'complete_pipeline.joblib')
    
    # 保存Pipeline
    joblib.dump(pipeline, pipeline_path)
    print(f"\nPipeline已保存到: {pipeline_path}")
    
    # 加载Pipeline
    loaded_pipeline = joblib.load(pipeline_path)
    print("Pipeline已加载")
    
    # 验证加载的Pipeline
    print("\n加载的Pipeline步骤:")
    for idx, (name, step) in enumerate(loaded_pipeline.steps, 1):
        print(f"  {idx}. {name}: {step.__class__.__name__}")
    
    print("\n优点:")
    print("  ✓ 一次性保存所有预处理步骤和模型")
    print("  ✓ 确保训练和部署使用相同的预处理")
    print("  ✓ 简化部署流程")
    print("  ✓ 版本管理方便")
    
    return loaded_pipeline


# ============================================================
# 8. 生产环境完整Pipeline
# ============================================================

def production_pipeline_demo(df):
    """演示生产环境的完整Pipeline"""
    print("\n" + "="*60)
    print("8. 生产环境完整Pipeline")
    print("="*60)
    
    # 定义特征
    numeric_features = ['age', 'salary', 'experience', 'score']
    categorical_features = ['city', 'education', 'job_type']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 数值特征Pipeline（包含异常值处理）
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征Pipeline
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # 删除未指定的列
    )
    
    # 完整Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\n生产环境Pipeline结构:")
    print("┌─ 数值特征处理")
    print("│   ├─ SimpleImputer (中位数填充)")
    print("│   ├─ OutlierClipper (异常值裁剪)")
    print("│   └─ StandardScaler (标准化)")
    print("├─ 类别特征处理")
    print("│   ├─ SimpleImputer (Unknown填充)")
    print("│   └─ OneHotEncoder (独热编码)")
    print("└─ RandomForestClassifier (分类器)")
    
    # 训练
    print("\n训练Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 交叉验证
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 测试集评估
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别 0', '类别 1']))
    
    # 保存Pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, 'production_pipeline.joblib')
    joblib.dump(pipeline, pipeline_path)
    print(f"\n生产Pipeline已保存: {pipeline_path}")
    
    return pipeline


# ============================================================
# 9. Pipeline使用示例（生产环境）
# ============================================================

def production_usage_demo():
    """演示生产环境中如何使用Pipeline"""
    print("\n" + "="*60)
    print("9. 生产环境Pipeline使用示例")
    print("="*60)
    
    # 加载保存的Pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    pipeline_path = os.path.join(model_dir, 'production_pipeline.joblib')
    
    if not os.path.exists(pipeline_path):
        print("⚠️  Pipeline文件不存在，请先运行production_pipeline_demo()")
        return
    
    pipeline = joblib.load(pipeline_path)
    print("✓ Pipeline已加载")
    
    # 模拟新数据（来自生产环境）
    new_data = pd.DataFrame({
        'age': [35, 28, 52],
        'salary': [75000, 55000, 120000],
        'experience': [10, 5, 20],
        'score': [85, 78, 92],
        'city': ['Beijing', 'Shanghai', 'Unknown_City'],  # 包含未见过的类别
        'education': ['Master', 'Bachelor', 'PhD'],
        'job_type': ['IT', 'Finance', 'Healthcare']
    })
    
    print("\n新数据（来自生产环境）:")
    print(new_data)
    
    # 预测（Pipeline自动应用所有预处理步骤）
    predictions = pipeline.predict(new_data)
    prediction_proba = pipeline.predict_proba(new_data)
    
    print("\n预测结果:")
    for idx, (pred, proba) in enumerate(zip(predictions, prediction_proba)):
        print(f"  样本 {idx+1}: 类别 {pred}, 概率 {proba}")
    
    print("\n优点:")
    print("  ✓ 一行代码完成所有预处理 + 预测")
    print("  ✓ 自动处理缺失值")
    print("  ✓ 自动处理未见过的类别")
    print("  ✓ 确保与训练时相同的预处理")
    print("  ✓ 不会出现训练/推理不一致的问题")


# ============================================================
# 10. Pipeline最佳实践指南
# ============================================================

def pipeline_best_practices():
    """Pipeline最佳实践指南"""
    print("\n" + "="*60)
    print("10. Pipeline最佳实践指南")
    print("="*60)
    
    guide = """
    ┌─────────────────────────────────────────────────────────────┐
    │              Pipeline最佳实践指南                            │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. 基本原则                                                  │
    │    ✓ 将所有预处理步骤都放入Pipeline                         │
    │    ✓ 确保Pipeline是可序列化的（可保存和加载）               │
    │    ✓ 避免在Pipeline外部进行数据转换                         │
    │    ✓ 使用ColumnTransformer处理不同类型的特征                │
    ├─────────────────────────────────────────────────────────────┤
    │ 2. 命名规范                                                  │
    │    ✓ 使用有意义的步骤名称                                   │
    │    ✓ 数值特征处理: 'numeric_transformer'                    │
    │    ✓ 类别特征处理: 'categorical_transformer'                │
    │    ✓ 预处理器: 'preprocessor'                               │
    │    ✓ 模型: 'classifier' 或 'regressor'                      │
    ├─────────────────────────────────────────────────────────────┤
    │ 3. 特征工程                                                  │
    │    ✓ 缺失值处理放在最前面                                   │
    │    ✓ 异常值处理在填充之后                                   │
    │    ✓ 特征缩放在编码之后                                     │
    │    ✓ 特征选择在所有预处理之后                               │
    ├─────────────────────────────────────────────────────────────┤
    │ 4. 数据泄露防护                                              │
    │    ⚠️  只在训练集上fit Pipeline                             │
    │    ⚠️  测试集只调用transform/predict                        │
    │    ⚠️  交叉验证要在完整Pipeline上进行                       │
    │    ⚠️  不要在Pipeline外提前处理数据                         │
    ├─────────────────────────────────────────────────────────────┤
    │ 5. 超参数调优                                                │
    │    ✓ 使用GridSearchCV或RandomizedSearchCV                  │
    │    ✓ 参数格式: '步骤名__参数名'                             │
    │    ✓ 同时调优预处理和模型参数                               │
    │    ✓ 使用交叉验证评估                                       │
    ├─────────────────────────────────────────────────────────────┤
    │ 6. 保存和版本管理                                            │
    │    ✓ 使用joblib保存Pipeline                                 │
    │    ✓ 保存时包含版本号和时间戳                               │
    │    ✓ 记录数据分布和性能指标                                 │
    │    ✓ 保存训练脚本和配置文件                                 │
    ├─────────────────────────────────────────────────────────────┤
    │ 7. 生产环境部署                                              │
    │    ✓ 测试Pipeline在新数据上的表现                           │
    │    ✓ 处理未见过的类别（handle_unknown='ignore'）            │
    │    ✓ 添加输入验证                                           │
    │    ✓ 记录预测日志                                           │
    │    ✓ 监控模型性能                                           │
    ├─────────────────────────────────────────────────────────────┤
    │ 8. 调试技巧                                                  │
    │    ✓ 使用pipeline.named_steps访问各步骤                     │
    │    ✓ 检查中间步骤的输出                                     │
    │    ✓ 使用verbose参数查看详细信息                            │
    │    ✓ 单独测试每个转换器                                     │
    ├─────────────────────────────────────────────────────────────┤
    │ 9. 性能优化                                                  │
    │    ✓ 使用memory参数缓存中间结果                             │
    │    ✓ 并行处理（n_jobs=-1）                                  │
    │    ✓ 避免不必要的复制                                       │
    │    ✓ 使用稀疏矩阵（sparse=True）                            │
    ├─────────────────────────────────────────────────────────────┤
    │ 10. 常见陷阱                                                 │
    │    ❌ 在整个数据集上fit再划分                               │
    │    ❌ 测试集上调用fit_transform                             │
    │    ❌ Pipeline外部预处理数据                                │
    │    ❌ 忘记处理未知类别                                      │
    │    ❌ 参数名称错误                                          │
    └─────────────────────────────────────────────────────────────┘
    
    💡 Pipeline模板示例:
    
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # 定义特征
    numeric_features = [...]
    categorical_features = [...]
    
    # 数值特征处理
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合预处理
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 完整Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', YourClassifier())
    ])
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 预测
    predictions = pipeline.predict(X_test)
    
    # 保存
    joblib.dump(pipeline, 'model_pipeline.joblib')
    """
    
    print(guide)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("Pipeline构建 - 生产环境数据预处理技术")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 1. 基础Pipeline
    basic_pipe = basic_pipeline_demo(df)
    
    # 2. ColumnTransformer
    column_pipe = column_transformer_demo(df)
    
    # 3. 自定义Transformer
    custom_pipe = custom_transformer_demo(df)
    
    # 4. FeatureUnion
    union_pipe = feature_union_demo(df)
    
    # 5. 特征选择Pipeline
    selection_pipe = feature_selection_pipeline_demo(df)
    
    # 6. 超参数调优
    tuned_pipe, grid_search = hyperparameter_tuning_demo(df)
    
    # 7. 保存和加载
    loaded_pipe = save_load_pipeline_demo(column_pipe)
    
    # 8. 生产环境完整Pipeline
    prod_pipe = production_pipeline_demo(df)
    
    # 9. 生产使用示例
    production_usage_demo()
    
    # 10. 最佳实践指南
    pipeline_best_practices()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n关键要点:")
    print("1. Pipeline将所有预处理和模型步骤串联")
    print("2. ColumnTransformer处理不同类型的特征")
    print("3. 自定义Transformer扩展Pipeline功能")
    print("4. 使用GridSearchCV调优整个Pipeline")
    print("5. 保存Pipeline确保训练和部署一致")
    print("6. 生产环境必须使用Pipeline!")


if __name__ == "__main__":
    main()
