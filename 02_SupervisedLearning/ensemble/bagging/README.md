# Bagging 集成学习方法演示

本文件夹包含 Bagging 集成学习的多个演示脚本，展示了不同 Bagging 方法的应用和特点。

## 文件说明

### 1. `1_bagging_demo.py` - Bagging 基础演示
**Bagging (Bootstrap Aggregating)** 是通过重采样来降低模型方差的集成学习方法。

**主要特点：**
- 有放回抽样（Bootstrap），生成多个子数据集
- 在每个子集上训练相同的基础学习器
- 结果聚合（分类：投票，回归：平均）
- 并行训练，速度快

**演示内容：**
- 乳腺癌数据集分类演示
- 不同基础学习器数量对性能的影响
- 回归任务演示
- 不同基础学习器的对比

**关键参数：**
- `n_estimators`: 基础学习器的数量
- `max_samples`: 每次采样的样本数
- `max_features`: 每次采样的特征数
- `bootstrap`: 是否使用有放回抽样

---

### 2. `2_random_forest_demo.py` - 随机森林演示
**Random Forest（随机森林）** 是 Bagging 的改进版本，同时在样本和特征两个维度进行随机化。

**主要特点：**
- 样本随机：Bootstrap 有放回抽样
- 特征随机：每个分割点只考虑特征的随机子集
- 速度快：特征采样减少计算量
- 自动特征重要性评估
- OOB (Out-of-Bag) 评分：无偏的泛化性能估计

**演示内容：**
- 分类任务（乳腺癌数据集）
- 不同树数量的影响
- OOB 评分与测试集准确率的对比
- max_features 参数的影响
- 回归任务演示

**关键参数：**
- `n_estimators`: 森林中树的数量
- `max_depth`: 每个树的最大深度
- `max_features`: 考虑的特征数（'sqrt' 或 'log2'）
- `oob_score`: 是否计算 OOB 评分

---

### 3. `3_extra_trees_demo.py` - Extra Trees 演示
**Extra Trees (Extremely Randomized Trees)** 是随机森林的变种，使用随机分割点而非最优分割。

**主要特点：**
- 随机分割点：减少计算复杂度
- 训练速度快（O(log n) vs 随机森林的 O(n log n)）
- 更强的正则化效果
- 特别适合大规模数据集

**演示内容：**
- 分类任务演示
- 随机森林 vs Extra Trees 速度对比
- 回归任务演示
- 不同树数量的影响

**关键参数：**
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `max_features`: 考虑的特征数
- `bootstrap`: 是否使用 Bootstrap

---

### 4. `4_bagging_comparison.py` - 综合对比演示
展示四种 Bagging 方法的全面对比。

**对比方法：**
1. **Bagging** - 基础 Bagging
2. **Random Forest** - 随机森林（样本+特征随机）
3. **Extra Trees** - 极端随机树（更快的随机分割）
4. **Pasting** - 不使用 Bootstrap 的 Bagging

**对比维度：**
- 训练时间
- 测试准确率
- 精确率、召回率、F1 分数
- 性能-速度权衡

**生成图表：**
- 性能指标对比（6 个子图）
- 精确率 vs 召回率
- 准确率 vs 训练时间
- 多维雷达图

---

## Bagging 方法对比表

| 方法 | 特点 | 速度 | 内存 | 准确率 | 最适合 |
|------|------|------|------|--------|---------|
| **Bagging** | 灵活，任意基础分类器 | 中 | 大 | ★★★★ | 通用场景 |
| **Random Forest** | 特征随机，平衡性好 | 中 | 大 | ★★★★★ | 通用，高精度 |
| **Extra Trees** | 快速，强正则化 | 快 | 大 | ★★★★ | 大数据集 |
| **Pasting** | 内存高效 | 快 | 小 | ★★★☆ | 内存受限 |

---

## 快速开始

### 运行单个演示：
```bash
# Bagging 演示
python 1_bagging_demo.py

# 随机森林演示
python 2_random_forest_demo.py

# Extra Trees 演示
python 3_extra_trees_demo.py

# 综合对比演示
python 4_bagging_comparison.py
```

### 所有图片保存到 `images/` 目录下：
- 混淆矩阵
- 特征重要性
- 学习曲线
- 参数对比
- 方法对比
- 雷达图

---

## 依赖库

**必需：**
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

**安装命令：**
```bash
pip install numpy matplotlib seaborn scikit-learn
```

---

## 学习路线

建议按以下顺序学习：

1. **首先运行 `1_bagging_demo.py`** - 理解 Bagging 的基本原理
2. **然后运行 `2_random_forest_demo.py`** - 学习随机森林的增强
3. **再运行 `3_extra_trees_demo.py`** - 了解快速的替代方案
4. **最后运行 `4_bagging_comparison.py`** - 全面对比各方法

---

## 关键概念

### Bootstrap 采样
- 有放回地从原始数据集中随机抽取样本
- 生成多个子数据集，可能存在重复样本
- 约 63.2% 的样本会被至少选中一次（1 - (1-1/n)^n）

### OOB (Out-of-Bag) 评分
- 对于每个样本，使用没有选中它的树进行预测
- 计算这些预测的准确率
- 是对模型泛化性能的无偏估计

### 特征重要性
- 基于特征在树中被使用的频率和贡献度
- 特征被用于分割的次数越多，重要性越高
- 随机森林自动提供特征重要性

### 样本随机 vs 特征随机
- **样本随机（Bagging）**：同一特征可以被所有树使用
- **特征随机（随机森林）**：每个分割只考虑特征的子集
- **双重随机（Extra Trees）**：样本和特征都随机化，且使用随机分割点

---

## 主要参数调优建议

### n_estimators（树的数量）
- 值越大，模型越复杂，但速度慢
- 通常范围：50-500
- 建议：从 100 开始调整

### max_features（特征数）
- 'sqrt'：使用 √n 个特征（好的起点）
- 'log2'：使用 log2(n) 个特征（更强的正则化）
- None：使用所有特征（高方差）
- 建议：'sqrt' 或 'log2'

### max_depth（树的深度）
- 限制树的深度以防止过拟合
- 通常范围：10-30
- 建议：根据验证集性能调整

### min_samples_split / min_samples_leaf
- 控制树的复杂度
- min_samples_split：分割所需的最小样本数
- min_samples_leaf：叶节点所需的最小样本数
- 建议：从默认值开始，根据需要调整

---

## 性能优化建议

1. **使用 n_jobs=-1 并行化**：
   ```python
   clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
   ```

2. **使用 OOB 评分替代交叉验证**（随机森林）：
   ```python
   clf = RandomForestClassifier(oob_score=True)
   # 访问 clf.oob_score_
   ```

3. **Early stopping 策略**：
   - 监控验证集性能
   - 当性能不再改进时停止添加树

4. **特征选择**：
   - 使用特征重要性进行特征选择
   - 去除低重要性特征，加快训练

---

## 常见问题

**Q: Bagging vs 随机森林，应该选择哪个？**
A: 随机森林通常更好，因为：
- 特征随机增加树之间的多样性
- 训练更快（特征采样减少计算）
- 自动提供特征重要性
- 大多数情况下准确率更高

**Q: 什么时候使用 Extra Trees？**
A: 当：
- 数据集很大（> 100K 样本）
- 需要快速训练
- 特征维度高（> 100）
- 追求计算效率

**Q: OOB 评分可以替代交叉验证吗？**
A: 可以。OOB 评分：
- 自动计算，无需额外成本
- 是无偏的泛化性能估计
- 推荐用于快速评估

**Q: 如何选择树的数量？**
A: 
- 通常 50-200 是合理范围
- 使用 OOB 评分监控性能
- 当 OOB 评分平稳时停止增加树

---

## 相关资源

- [sklearn Ensemble 文档](https://scikit-learn.org/stable/modules/ensemble.html)
- [Random Forest 论文](https://www.stat.berkeley.edu/~breiman/random-forests.pdf)
- [Extra Trees 论文](https://arxiv.org/pdf/1309.0238.pdf)

