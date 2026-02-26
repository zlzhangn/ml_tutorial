# 决策树方法示例

本文件夹包含决策树及其相关算法的各种应用示例。

## 文件说明

### 1. `1_classification_iris.py` - 决策树分类基础
- **数据集**: 鸢尾花数据集（Iris）
- **内容**:
  - 决策树分类器的基本使用
  - 基尼不纯度 vs 信息熵
  - 特征重要性分析
  - 决策树可视化
  - 决策路径追踪
  - 过拟合防止策略

### 2. `2_regression_advertising.py` - 决策树回归
- **数据集**: 广告数据集（Advertising.csv）
- **内容**:
  - 决策树回归器的使用
  - 不同深度模型的性能对比
  - 特征重要性分析
  - 残差分析
  - 预测值与真实值对比
  - 与线性回归的对比

### 3. `3_heart_disease_tree.py` - 实战应用
- **数据集**: 心脏病数据集
- **内容**:
  - 完整的分类流程
  - 网格搜索参数调优
  - 与逻辑回归、朴素贝叶斯对比
  - ROC曲线分析
  - 决策规则提取
  - 交叉验证

### 4. `4_random_forest.py` - 随机森林集成方法
- **数据集**: 鸢尾花数据集
- **内容**:
  - 随机森林原理和实现
  - 单棵树 vs 随机森林对比
  - OOB（袋外）评估
  - 树数量对性能的影响
  - 森林中树的深度分布
  - 集成学习优势展示

## 决策树原理

### 1. 基本概念

决策树是一种树形结构的机器学习算法：
- **根节点**: 包含所有训练样本
- **内部节点**: 表示对某个特征的判断
- **分支**: 表示判断的结果
- **叶子节点**: 表示最终的预测结果

### 2. 分裂标准

#### 基尼不纯度 (Gini Impurity)
```
Gini = 1 - Σ(p_i)²
```
- p_i 是类别 i 的概率
- 基尼不纯度越小，节点越纯
- CART算法默认使用

#### 信息熵 (Entropy)
```
Entropy = -Σ(p_i × log2(p_i))
```
- 熵越小，节点越纯
- ID3和C4.5算法使用

#### 信息增益 (Information Gain)
```
IG = 父节点熵 - 子节点加权平均熵
```
- 选择信息增益最大的特征进行分裂

### 3. 决策树算法

| 算法 | 分裂标准 | 特点 |
|------|---------|------|
| ID3 | 信息增益 | 只能处理离散特征 |
| C4.5 | 信息增益率 | 可以处理连续特征和缺失值 |
| CART | 基尼不纯度/MSE | 可用于分类和回归，二叉树 |

Scikit-learn 使用的是优化的 CART 算法。

### 4. 重要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `criterion` | 分裂标准：'gini' 或 'entropy' | 'gini' |
| `max_depth` | 树的最大深度 | None（不限制） |
| `min_samples_split` | 分裂内部节点所需的最小样本数 | 2 |
| `min_samples_leaf` | 叶子节点最小样本数 | 1 |
| `max_features` | 寻找最佳分割时考虑的特征数 | None（所有特征） |
| `max_leaf_nodes` | 最大叶子节点数 | None |

### 5. 优点与缺点

#### 优点 ✓
- 易于理解和解释（白盒模型）
- 可以可视化
- 需要的数据预处理少
- 可以处理数值和类别数据
- 可以处理多输出问题
- 可以处理缺失值
- 训练速度快

#### 缺点 ✗
- 容易过拟合
- 对训练数据的微小变化敏感
- 可能创建有偏的树（不平衡数据）
- 难以捕获特征间的线性关系
- 预测边界呈轴对齐的矩形

### 6. 防止过拟合

1. **预剪枝** (Pre-pruning)：
   - 限制树的深度 (`max_depth`)
   - 设置叶子节点最小样本数 (`min_samples_leaf`)
   - 设置分裂所需最小样本数 (`min_samples_split`)
   - 限制叶子节点数量 (`max_leaf_nodes`)

2. **后剪枝** (Post-pruning)：
   - 先构建完整的树
   - 然后向上剪掉一些子树
   - Scikit-learn 使用 `ccp_alpha` 参数

3. **集成方法**：
   - 随机森林 (Random Forest)
   - 梯度提升树 (Gradient Boosting)
   - AdaBoost
   - XGBoost, LightGBM

## 随机森林 (Random Forest)

### 原理

随机森林是基于决策树的集成学习方法：

1. **Bootstrap采样**: 从训练集中有放回地随机抽取样本
2. **特征随机**: 每次分裂时只考虑随机选择的部分特征
3. **投票/平均**: 多棵树的预测结果通过投票或平均得出

### 优势

- ✅ 准确率更高
- ✅ 抗过拟合能力强
- ✅ 能评估特征重要性
- ✅ 可以并行训练
- ✅ OOB评估（无需验证集）

### 关键参数

- `n_estimators`: 树的数量（通常100-500）
- `max_features`: 每次分裂考虑的特征数
  - 分类: `'sqrt'` 或 `'log2'`
  - 回归: `'auto'` 或 `1.0`
- `bootstrap`: 是否使用自助采样
- `oob_score`: 是否计算OOB分数

## 使用场景

### 使用决策树
- 需要可解释的模型
- 数据维度不太高
- 快速原型开发
- 特征重要性分析

### 使用随机森林
- 需要高准确率
- 数据维度高
- 对过拟合敏感的场景
- 有噪声的数据

### 不适合决策树的场景
- 线性关系明显（用线性模型）
- 数据量非常小
- 需要外推预测
- 特征之间有复杂的非轴对齐关系

## 运行示例

```bash
# 决策树分类
python ch10_decision_tree/1_classification_iris.py

# 决策树回归
python ch10_decision_tree/2_regression_advertising.py

# 实战应用
python ch10_decision_tree/3_heart_disease_tree.py

# 随机森林
python ch10_decision_tree/4_random_forest.py
```

## 数据集需求

- **内置数据集**: Iris（无需下载）
- **本地数据集**: 
  - `data/advertising.csv` - 广告数据
  - `data/heart_disease.csv` - 心脏病数据

## 依赖库

```python
- scikit-learn
- pandas
- numpy
- matplotlib
```

## 可视化输出

运行示例代码会生成以下可视化图片：
- 决策树结构图
- 特征重要性图
- 模型性能对比图
- ROC曲线
- 残差分析图
- 随机森林参数影响图

所有图片保存在 `ch10_decision_tree/` 目录下。

## 学习建议

1. **从简单开始**: 先运行 `1_classification_iris.py` 理解基本概念
2. **理解回归**: 运行 `2_regression_advertising.py` 学习决策树回归
3. **实战练习**: 运行 `3_heart_disease_tree.py` 看完整流程
4. **进阶学习**: 运行 `4_random_forest.py` 学习集成方法
5. **调参实践**: 修改参数观察效果变化
6. **可视化理解**: 仔细观察生成的决策树图，理解分裂逻辑

## 参考资料

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Feature Importances](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
