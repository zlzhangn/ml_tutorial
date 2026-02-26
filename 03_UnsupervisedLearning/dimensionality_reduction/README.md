# 降维方法演示

本目录包含常见降维方法的演示代码和说明。

## 📁 文件说明

| 文件 | 说明 | 关键算法 |
|------|------|----------|
| `1_pca_demo.py` | PCA主成分分析演示 | PCA, 方差分析 |
| `2_tsne_demo.py` | t-SNE降维演示 | t-SNE, 参数调优 |
| `3_lda_demo.py` | LDA线性判别分析演示 | LDA, 监督降维 |
| `4_other_methods_demo.py` | 其他降维方法演示 | SVD, NMF, Isomap等 |
| `5_comparison_demo.py` | 降维方法综合比较 | 多方法对比 |

## 🚀 快速开始

### 运行单个演示

```bash
# 进入dimensionality_reduction目录
cd 03_UnsupervisedLearning/dimensionality_reduction

# 运行PCA演示
python 1_pca_demo.py

# 运行t-SNE演示
python 2_tsne_demo.py

# 运行LDA演示
python 3_lda_demo.py

# 运行其他方法演示
python 4_other_methods_demo.py

# 运行综合比较
python 5_comparison_demo.py
```

## 📊 算法比较

### 主要降维方法

| 方法 | 类型 | 线性/非线性 | 监督/无监督 | 复杂度 | 适用场景 |
|------|------|------------|------------|--------|---------|
| **PCA** | 线性投影 | 线性 | 无监督 | O(n·m·k) | 通用降维 |
| **t-SNE** | 流形学习 | 非线性 | 无监督 | O(n²) | 可视化 |
| **LDA** | 线性投影 | 线性 | 监督 | O(n·m·k) | 分类任务 |
| **Isomap** | 流形学习 | 非线性 | 无监督 | O(n²log n) | 流形数据 |
| **NMF** | 矩阵分解 | 线性 | 无监督 | O(n·m·k·iter) | 非负数据 |
| **ICA** | 信号分离 | 线性 | 无监督 | O(n·m²·iter) | 信号处理 |

## 🎯 方法选择指南

### 决策流程

```
开始
  ↓
是否有标签？
  ├─ 是 → 使用 LDA
  │       最大化类别分离
  │
  └─ 否 → 数据是否线性？
           ├─ 是 → 使用 PCA
           │       快速、稳定
           │
           └─ 否 → 目的是什么？
                    ├─ 可视化 → t-SNE
                    ├─ 保留距离 → Isomap
                    ├─ 稀疏数据 → SVD
                    └─ 非负数据 → NMF
```

### 详细对比

#### 1. PCA (主成分分析)
- **优点**: 
  - 计算快速，适合大数据
  - 结果稳定，可重复
  - 可解释性强
  - 能够逆变换
- **缺点**:
  - 仅捕获线性关系
  - 对数据尺度敏感
  - 可能丢失小方差但重要的信息
- **使用场景**:
  - 特征降维
  - 数据压缩
  - 噪声过滤
  - 加速机器学习

#### 2. t-SNE (t-分布随机邻域嵌入)
- **优点**:
  - 可视化效果极佳
  - 能保留局部结构
  - 发现非线性关系
  - 簇分离清晰
- **缺点**:
  - 计算非常慢
  - 无法变换新数据
  - 每次结果可能不同
  - 对参数敏感
- **使用场景**:
  - 高维数据可视化
  - 探索数据结构
  - 质量检查
  - 结果展示

#### 3. LDA (线性判别分析)
- **优点**:
  - 监督降维，利用标签
  - 最大化类别分离
  - 既能降维又能分类
  - 结果可解释
- **缺点**:
  - 需要标签数据
  - 假设高斯分布
  - 最多降至n_classes-1维
  - 对异常值敏感
- **使用场景**:
  - 分类前的特征提取
  - 多类分类问题
  - 类别可视化
  - 有监督降维

#### 4. 其他方法

**Isomap (等距映射)**
- 保持测地距离的流形学习
- 适合非线性流形数据
- 对噪声敏感

**NMF (非负矩阵分解)**
- 要求数据非负
- 结果可解释性强
- 适合主题建模、推荐系统

**ICA (独立成分分析)**
- 寻找统计独立成分
- 适合信号分离
- 常用于脑电图分析

## 📈 参数调优指南

### PCA
```python
# n_components: 保留的主成分数量
pca = PCA(n_components=0.95)  # 保留95%方差
# 或指定具体数量
pca = PCA(n_components=10)
```

**技巧**:
- 使用累积方差曲线选择n_components
- 通常保留90-95%方差
- 数据量大时可保留更少主成分

### t-SNE
```python
tsne = TSNE(
    n_components=2,       # 降至2维或3维
    perplexity=30,        # 5-50之间
    learning_rate=200,    # 10-1000
    n_iter=1000,          # 至少1000
    random_state=42
)
```

**技巧**:
- perplexity: 数据量大用大值 (30-50)
- learning_rate: 通常200效果好
- 先用PCA降至50维以下再用t-SNE
- 多运行几次选择KL散度最小的

### LDA
```python
# n_components: 最大为min(n_features, n_classes-1)
lda = LinearDiscriminantAnalysis(
    n_components=2,
    solver='svd'  # 'svd'或'lsqr'
)
```

**技巧**:
- n_components通常设为n_classes-1
- solver='svd'更稳定但慢
- 样本量小时使用shrinkage参数

## 💻 代码示例

### 基本使用

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. 查看解释方差
print(pca.explained_variance_ratio_)
```

### t-SNE可视化

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 先用PCA预处理（加速）
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# 2. t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# 3. 可视化
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.show()
```

### LDA分类降维

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. LDA降维（需要标签）
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 2. 同时可用于分类
y_pred = lda.predict(X_test)
```

## 📦 依赖库

```python
numpy
matplotlib
scikit-learn
scipy
```

安装命令：
```bash
pip install numpy matplotlib scikit-learn scipy
```

## 📝 注意事项

### 数据预处理

1. **标准化**: 大多数方法对数据尺度敏感
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **缺失值处理**: 降维前必须处理
   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='mean')
   X_imputed = imputer.fit_transform(X)
   ```

3. **异常值**: 对PCA和LDA影响大，建议先处理

### 性能优化

1. **大数据集**:
   - 使用PCA或Truncated SVD
   - 避免t-SNE和Isomap
   - 考虑增量PCA

2. **t-SNE加速**:
   - 先用PCA降至50维
   - 减少n_iter
   - 使用样本子集

3. **内存优化**:
   - 使用稀疏矩阵（Truncated SVD）
   - 批处理大数据

### 常见错误

1. **未标准化**: 导致PCA效果差
2. **t-SNE用于特征工程**: t-SNE只用于可视化
3. **LDA维度过高**: 不能超过n_classes-1
4. **过度降维**: 丢失太多信息

## 🔍 评估降维效果

### 内部指标

1. **解释方差比** (PCA, LDA)
   ```python
   print(f"保留方差: {pca.explained_variance_ratio_.sum():.2%}")
   ```

2. **重构误差**
   ```python
   X_reconstructed = pca.inverse_transform(X_pca)
   error = np.mean((X - X_reconstructed) ** 2)
   ```

3. **KL散度** (t-SNE)
   ```python
   print(f"KL散度: {tsne.kl_divergence_:.4f}")
   ```

### 外部评估

1. **分类任务**: 比较降维前后的分类准确率
2. **聚类任务**: 比较降维前后的聚类效果
3. **可视化质量**: 人工检查类别分离度

## 📚 参考资料

- [Scikit-learn降维文档](https://scikit-learn.org/stable/modules/decomposition.html)
- [t-SNE原理](https://distill.pub/2016/misread-tsne/)
- [PCA详解](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [流形学习](https://scikit-learn.org/stable/modules/manifold.html)

## 🎓 学习路径

1. **基础** (`1_pca_demo.py`)
   - 理解PCA原理
   - 掌握方差解释
   - 学会参数选择

2. **进阶** (`2_tsne_demo.py`, `3_lda_demo.py`)
   - 非线性降维
   - 监督降维
   - 参数调优

3. **扩展** (`4_other_methods_demo.py`)
   - 特殊场景方法
   - 流形学习
   - 信号处理

4. **实践** (`5_comparison_demo.py`)
   - 方法对比
   - 场景选择
   - 性能评估

## ⚠️ 常见问题

**Q: 如何选择降维方法？**
A: 根据任务：有标签→LDA，可视化→t-SNE，通用→PCA，流形→Isomap

**Q: PCA保留多少主成分？**
A: 通常保留90-95%方差，或根据累积方差曲线的"肘部"选择

**Q: t-SNE为什么每次结果不同？**
A: t-SNE有随机性，设置random_state可固定结果

**Q: 降维后准确率下降怎么办？**
A: 1) 保留更多维度 2) 尝试其他方法 3) 检查是否丢失重要信息

**Q: 如何加速t-SNE？**
A: 1) 先用PCA降至50维 2) 减少迭代次数 3) 使用样本子集

**Q: LDA能降到几维？**
A: 最多n_classes-1维，例如3类数据最多降至2维

**Q: 降维会丢失信息吗？**
A: 会，但通过保留主要成分可最小化损失。使用解释方差评估损失程度

**Q: 稀疏数据用什么方法？**
A: Truncated SVD，不需要中心化，保持稀疏性

## 🌟 最佳实践

1. **数据探索阶段**
   - 先用PCA快速了解数据
   - 再用t-SNE详细可视化

2. **特征工程阶段**
   - 无监督: PCA
   - 有监督: LDA
   - 稀疏数据: Truncated SVD

3. **模型训练阶段**
   - 在验证集上选择最优维度
   - 避免在测试集上调参

4. **结果展示阶段**
   - 使用t-SNE做最终可视化
   - 配合主成分解释增强可解释性

5. **性能优化阶段**
   - 大数据优先PCA
   - 逐步尝试复杂方法
   - 权衡精度和速度
