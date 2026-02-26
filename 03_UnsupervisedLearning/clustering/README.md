# 聚类算法演示

本目录包含常见聚类算法的演示代码和说明。

## 📁 文件说明

| 文件 | 说明 | 关键算法 |
|------|------|----------|
| `1_kmeans_demo.py` | KMeans聚类演示 | KMeans, 肘部法则 |
| `2_dbscan_demo.py` | DBSCAN聚类演示 | DBSCAN, 密度聚类 |
| `3_hierarchical_demo.py` | 层次聚类演示 | Agglomerative, 树状图 |
| `4_gmm_demo.py` | 高斯混合模型演示 | GMM, 软聚类 |
| `5_meanshift_demo.py` | Mean Shift聚类演示 | Mean Shift, 带宽估计 |
| `6_clustering_comparison.py` | 聚类算法综合比较 | 多算法对比 |

## 🚀 快速开始

### 运行单个演示

```bash
# 进入clustering目录
cd 03_UnsupervisedLearning/clustering

# 运行KMeans演示
python 1_kmeans_demo.py

# 运行DBSCAN演示
python 2_dbscan_demo.py

# 运行层次聚类演示
python 3_hierarchical_demo.py

# 运行GMM演示
python 4_gmm_demo.py

# 运行Mean Shift演示
python 5_meanshift_demo.py

# 运行综合比较
python 6_clustering_comparison.py
```

## 📊 算法比较

### 1. KMeans
- **优点**: 简单快速，适合大数据集
- **缺点**: 需要预设簇数，只能识别球形簇
- **适用场景**: 簇大小相似、形状近似球形的数据

### 2. DBSCAN
- **优点**: 可识别任意形状簇，自动检测噪声
- **缺点**: 参数敏感，对密度变化大的数据效果差
- **适用场景**: 含噪声、非凸形状的数据

### 3. 层次聚类
- **优点**: 生成树状图，直观理解数据结构
- **缺点**: 计算复杂度高，内存消耗大
- **适用场景**: 小到中等规模数据，需要理解簇层次结构

### 4. GMM（高斯混合模型）
- **优点**: 软聚类，提供概率分配，可识别椭圆形簇
- **缺点**: 需要预设簇数，可能陷入局部最优
- **适用场景**: 需要概率输出、椭圆形簇的场景

### 5. Mean Shift
- **优点**: 不需要预设簇数，可识别任意形状簇
- **缺点**: 计算慢，参数选择困难
- **适用场景**: 簇数未知、计算资源充足的场景

## 📈 算法选择指南

```
开始
  ↓
是否知道簇的数量？
  ├─ 是 → 簇是否为球形？
  │        ├─ 是 → 使用 KMeans
  │        └─ 否 → 需要概率输出？
  │                 ├─ 是 → 使用 GMM
  │                 └─ 否 → 使用层次聚类
  │
  └─ 否 → 数据是否包含噪声？
           ├─ 是 → 使用 DBSCAN
           └─ 否 → 数据规模大？
                    ├─ 是 → 使用 Mean Shift
                    └─ 否 → 使用层次聚类
```

## 🎯 参数调优技巧

### KMeans
- 使用**肘部法则**或**轮廓系数**确定最佳K值
- 增加`n_init`获得更稳定的结果

### DBSCAN
- 使用**K-distance图**确定`eps`参数
- `min_samples`通常设为维度的2倍

### 层次聚类
- 通过**树状图**确定最佳簇数
- `ward`链接通常效果最好

### GMM
- 使用**AIC/BIC**选择最佳组件数
- `covariance_type='full'`最灵活但最慢

### Mean Shift
- 使用`estimate_bandwidth`函数自动估计
- 调整`quantile`参数(0.1-0.3)

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

1. **数据标准化**: DBSCAN和Mean Shift对数据尺度敏感，使用前需标准化
2. **参数敏感性**: 所有算法都对参数敏感，需要根据数据特点调整
3. **计算复杂度**: 层次聚类和Mean Shift不适合大数据集
4. **噪声处理**: 只有DBSCAN能有效识别和排除噪声点
5. **可解释性**: KMeans和层次聚类结果更容易解释

## 🔍 评估指标

聚类算法常用评估指标：

- **轮廓系数 (Silhouette Score)**: 衡量簇内紧密度和簇间分离度
- **Calinski-Harabasz指数**: 簇间方差与簇内方差的比值
- **Davies-Bouldin指数**: 簇内距离与簇间距离的比值（越小越好）
- **调整兰德指数 (ARI)**: 与真实标签比较（需要真实标签）

## 📚 参考资料

- [Scikit-learn聚类文档](https://scikit-learn.org/stable/modules/clustering.html)
- [聚类算法比较](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)

## 🎓 学习路径

1. 从`1_kmeans_demo.py`开始，理解基本的聚类概念
2. 学习`2_dbscan_demo.py`，了解密度聚类
3. 通过`3_hierarchical_demo.py`理解层次结构
4. 掌握`4_gmm_demo.py`的概率聚类
5. 学习`5_meanshift_demo.py`的自适应聚类
6. 最后运行`6_clustering_comparison.py`进行综合对比

## ⚠️ 常见问题

**Q: 如何选择聚类算法？**
A: 根据数据特点和需求：已知簇数→KMeans/GMM，未知簇数→DBSCAN/Mean Shift，需要层次关系→层次聚类

**Q: 聚类结果不理想怎么办？**
A: 1) 尝试数据标准化 2) 调整算法参数 3) 尝试其他算法 4) 检查数据质量

**Q: 如何评估聚类质量？**
A: 使用轮廓系数、Calinski-Harabasz指数等内部指标，或与真实标签比较（如果有）

**Q: 大数据集用什么算法？**
A: KMeans或Mini-Batch KMeans，避免使用层次聚类和Mean Shift
