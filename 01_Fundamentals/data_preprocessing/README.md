# 数据预处理技术 - 生产环境完整指南

本目录包含所有主流的、适用于生产环境的数据预处理技术的完整演示代码。所有代码都包含详细的中文注释，便于学习和参考。

## 📁 目录结构

```
data_preprocessing/
├── 1_missing_values.py          # 缺失值处理
├── 2_outlier_detection.py       # 异常值检测与处理
├── 3_feature_scaling.py         # 数据标准化与归一化
├── 4_categorical_encoding.py    # 类别编码
├── 5_imbalanced_data.py        # 不平衡数据处理
├── 6_pipeline_construction.py   # Pipeline构建
├── 7_complete_case_study.py    # 完整综合案例
├── models/                      # 保存的模型和转换器
└── README.md                    # 本文件
```

## 📚 各文件内容介绍

### 1. 缺失值处理 (`1_missing_values.py`)

**核心技术：**
- ✅ 简单填充（均值、中位数、众数、常数）
- ✅ KNN填充（基于相似样本）
- ✅ 迭代填充（MICE多元插补）
- ✅ 前向/后向填充（时间序列）
- ✅ 插值法（线性、多项式）
- ✅ 删除缺失值策略

**生产环境要点：**
- 根据缺失比例选择策略
- 数值型和类别型特征分别处理
- 保存填充器用于生产环境

### 2. 异常值检测与处理 (`2_outlier_detection.py`)

**核心技术：**
- ✅ 统计方法（Z-Score、IQR）
- ✅ 机器学习方法（Isolation Forest、LOF、Elliptic Envelope）
- ✅ 多方法投票检测
- ✅ 异常值处理（删除、Winsorization、替换、转换）

**生产环境要点：**
- 使用多种方法提高准确性
- 结合业务规则
- Winsorization是稳健的处理方法

### 3. 数据标准化与归一化 (`3_feature_scaling.py`)

**核心技术：**
- ✅ StandardScaler（Z-Score标准化）
- ✅ MinMaxScaler（最小-最大归一化）
- ✅ RobustScaler（鲁棒缩放）
- ✅ MaxAbsScaler（最大绝对值缩放）
- ✅ Normalizer（L2归一化）
- ✅ QuantileTransformer（分位数转换）
- ✅ PowerTransformer（幂变换）

**生产环境要点：**
- 先划分数据，再缩放
- 缩放器只在训练集上fit
- 根据特征分布和模型选择方法

### 4. 类别编码 (`4_categorical_encoding.py`)

**核心技术：**
- ✅ Label Encoding（标签编码）
- ✅ Ordinal Encoding（序数编码）
- ✅ OneHot Encoding（独热编码）
- ✅ Binary Encoding（二进制编码）
- ✅ Target Encoding（目标编码）
- ✅ Frequency Encoding（频率编码）
- ✅ Hash Encoding（哈希编码）

**生产环境要点：**
- 有序类别用Ordinal
- 低基数用OneHot
- 高基数用Target或Hash
- 处理未知类别

### 5. 不平衡数据处理 (`5_imbalanced_data.py`)

**核心技术：**
- ✅ 过采样（RandomOverSampler、SMOTE、BorderlineSMOTE、ADASYN）
- ✅ 欠采样（RandomUnderSampler、TomekLinks、NearMiss）
- ✅ 混合方法（SMOTETomek、SMOTEENN）
- ✅ 算法级方法（class_weight、BalancedRandomForest）
- ✅ 阈值调整

**生产环境要点：**
- 评估不平衡程度
- 小数据用过采样，大数据用欠采样
- 使用适当的评估指标（F1、AUC）

### 6. Pipeline构建 (`6_pipeline_construction.py`)

**核心技术：**
- ✅ 基础Pipeline
- ✅ ColumnTransformer（处理异构数据）
- ✅ 自定义Transformer
- ✅ FeatureUnion（并行特征工程）
- ✅ 特征选择Pipeline
- ✅ 超参数调优
- ✅ 保存和加载

**生产环境要点：**
- 所有预处理都放入Pipeline
- 防止数据泄露
- 确保训练和部署一致

### 7. 完整综合案例 (`7_complete_case_study.py`)

**内容：**
- ✅ 真实数据集的完整处理流程
- ✅ 从原始数据到部署的全流程
- ✅ 模型评估和优化
- ✅ 生产环境部署示例

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib scipy imbalanced-learn category-encoders joblib
```

### 运行示例

```python
# 运行任意一个文件
python 1_missing_values.py

# 或者在Python中导入
from data_preprocessing import missing_values
missing_values.main()
```

## 📊 数据预处理流程图

```
原始数据
    ↓
1. 数据探索
    ├─ 缺失值分析
    ├─ 异常值分析
    ├─ 数据分布分析
    └─ 类别特征分析
    ↓
2. 数据清洗
    ├─ 处理缺失值
    ├─ 处理异常值
    ├─ 处理重复值
    └─ 数据类型转换
    ↓
3. 特征工程
    ├─ 类别编码
    ├─ 特征缩放
    ├─ 特征选择
    └─ 特征创建
    ↓
4. 数据平衡
    ├─ 过采样/欠采样
    └─ 类别权重调整
    ↓
5. Pipeline构建
    ├─ 组合所有步骤
    ├─ 超参数调优
    └─ 模型训练
    ↓
6. 模型评估
    ├─ 交叉验证
    ├─ 测试集评估
    └─ 性能分析
    ↓
7. 模型部署
    ├─ 保存Pipeline
    ├─ API接口
    └─ 监控和维护
```

## 🎯 生产环境核心原则

### 1. 数据泄露防护
```python
❌ 错误做法：
X_scaled = scaler.fit_transform(X)  # 在整个数据集上fit
X_train, X_test = train_test_split(X_scaled)

✅ 正确做法：
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # 只在训练集上fit
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Pipeline必备
```python
✅ 使用Pipeline:
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)  # 一键训练
predictions = pipeline.predict(X_test)  # 一键预测
```

### 3. 保存和版本管理
```python
import joblib
from datetime import datetime

# 保存时添加时间戳
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(pipeline, f'model_pipeline_{timestamp}.joblib')
```

### 4. 处理未知类别
```python
# OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# OrdinalEncoder
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
```

## 📈 性能对比指南

### 缺失值填充方法选择

| 方法 | 速度 | 准确性 | 适用场景 |
|------|------|--------|----------|
| 简单填充 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 快速baseline |
| KNN填充 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 中小数据集 |
| 迭代填充 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 高质量需求 |

### 特征缩放方法选择

| 模型类型 | 推荐方法 | 原因 |
|----------|----------|------|
| 线性模型 | StandardScaler | 假设正态分布 |
| 神经网络 | MinMaxScaler / StandardScaler | 需要固定范围 |
| 树模型 | 不需要 | 对尺度不敏感 |
| KNN/SVM | StandardScaler | 基于距离 |

### 类别编码方法选择

| 类别数量 | 推荐方法 | 备注 |
|----------|----------|------|
| < 15 | OneHot | 维度可控 |
| 15-100 | Binary / Target | 节省空间 |
| > 100 | Target / Hash | 高基数特征 |
| 有序 | Ordinal | 保留顺序信息 |

## 🔧 常见问题解决

### 问题1：内存不足
```python
# 使用稀疏矩阵
encoder = OneHotEncoder(sparse_output=True)

# 分批处理
from sklearn.utils import gen_batches
for batch in gen_batches(n, batch_size):
    process_batch(X[batch])
```

### 问题2：训练太慢
```python
# 并行处理
model = RandomForestClassifier(n_jobs=-1)

# 减少数据量（欠采样）
sampler = RandomUnderSampler()
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

### 问题3：模型过拟合
```python
# 使用交叉验证
scores = cross_val_score(pipeline, X, y, cv=5)

# 正则化
model = LogisticRegression(C=0.1)  # 增加正则化

# 减少特征
selector = SelectKBest(k=10)
```

## 📖 学习路径建议

1. **初学者路径**
   - 1_missing_values.py → 3_feature_scaling.py → 4_categorical_encoding.py → 6_pipeline_construction.py

2. **进阶路径**
   - 2_outlier_detection.py → 5_imbalanced_data.py → 7_complete_case_study.py

3. **生产环境路径**
   - 6_pipeline_construction.py → 7_complete_case_study.py

## 🌟 最佳实践清单

- [ ] 先进行探索性数据分析（EDA）
- [ ] 先划分数据集，再进行预处理
- [ ] 所有预处理步骤都放入Pipeline
- [ ] 使用交叉验证评估模型
- [ ] 保存所有转换器和模型
- [ ] 处理未知类别和缺失值
- [ ] 使用适当的评估指标
- [ ] 记录数据分布和统计信息
- [ ] 版本管理代码和模型
- [ ] 监控生产环境性能

## 📚 参考资源

- [Scikit-learn官方文档](https://scikit-learn.org/)
- [Imbalanced-learn文档](https://imbalanced-learn.org/)
- [Category Encoders文档](https://contrib.scikit-learn.org/category_encoders/)

## 🤝 贡献

欢迎提出建议和改进！

## 📄 许可证

MIT License

---

**注意**：所有代码使用绝对路径保存文件到当前项目目录，确保路径正确。

**作者**：ML Tutorial
**更新日期**：2026-02-13
