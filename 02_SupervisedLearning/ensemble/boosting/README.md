# Boosting 集成学习方法演示

本文件夹包含 Boosting 集成学习的多个演示脚本，展示了不同 Boosting 方法的应用和特点。

## 文件说明

### 1. `1_adaboost_demo.py` - AdaBoost 演示
**AdaBoost (Adaptive Boosting)** 是最基础的 Boosting 算法。

**主要特点：**
- 重点关注被错误分类的样本
- 逐步降低强分类器的训练误差
- 对异常值较为敏感

**演示内容：**
- 乳腺癌数据集分类
- 不同弱学习器数量对性能的影响
- 不同学习率对性能的影响

**关键参数：**
- `n_estimators`: 弱学习器的数量
- `learning_rate`: 学习率，控制每个弱学习器的贡献
- `estimator`: 基础学习器（通常为决策树桩）

---

### 2. `2_gradient_boosting_demo.py` - Gradient Boosting 演示
**Gradient Boosting** 是通过拟合残差来改进模型的方法。

**主要特点：**
- 基于损失函数的梯度方向进行优化
- 对各类数据集都有很好的泛化性能
- 相比 AdaBoost，对异常值不太敏感
- 通常具有更强的预测能力

**演示内容：**
- 分类任务（乳腺癌数据集）
- 回归任务（合成数据集）
- 超参数调优（n_estimators、learning_rate、max_depth）

**关键参数：**
- `n_estimators`: 树的数量
- `learning_rate`: 学习率
- `max_depth`: 每个树的最大深度
- `subsample`: 样本比例

---

### 3. `3_xgboost_demo.py` - XGBoost 演示
**XGBoost (Extreme Gradient Boosting)** 是 Gradient Boosting 的高度优化实现。

**主要特点：**
- 支持正则化（L1 和 L2）防止过拟合
- 支持缺失值处理
- 支持并行化处理
- 在竞赛中广泛使用

**演示内容：**
- 分类任务
- ROC 曲线分析
- 与其他 Boosting 方法的性能对比
- 交叉验证

**关键参数：**
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `learning_rate`: 学习率
- `reg_alpha`: L1 正则化参数
- `reg_lambda`: L2 正则化参数

**安装：**
```bash
pip install xgboost
```

---

### 4. `4_lightgbm_demo.py` - LightGBM 演示
**LightGBM (Light Gradient Boosting Machine)** 是微软开发的快速梯度提升框架。

**主要特点：**
- 训练速度快（相比 XGBoost）
- 内存占用小
- 支持分类特征处理
- 支持并行和 GPU 学习
- 更好的大型数据集处理能力

**演示内容：**
- 分类任务
- 大规模数据集性能对比
- 早停（Early Stopping）功能

**关键参数：**
- `num_leaves`: 最大叶子数（而不是深度）
- `learning_rate`: 学习率
- `n_estimators`: 树的数量
- `feature_fraction`: 特征比例
- `bagging_fraction`: 样本比例

**安装：**
```bash
pip install lightgbm
```

---

## Boosting 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **AdaBoost** | 实现简单，理论清晰 | 对异常值敏感，性能一般 | 小规模数据，快速演示 |
| **Gradient Boosting** | 性能强，泛化好 | 训练较慢，易过拟合 | 中等规模数据，高精度要求 |
| **XGBoost** | 性能优异，支持正则化 | 复杂度高，调参困难 | 竞赛，大规模数据，需要高性能 |
| **LightGBM** | 速度快，内存占用少 | 易过拟合，对小数据集不友好 | 大规模数据，在线学习 |

---

## 快速开始

### 运行单个演示：
```bash
# AdaBoost 演示
python 1_adaboost_demo.py

# Gradient Boosting 演示
python 2_gradient_boosting_demo.py

# XGBoost 演示
python 3_xgboost_demo.py

# LightGBM 演示
python 4_lightgbm_demo.py
```

### 所有图片保存到 `images/` 目录下：
- 混淆矩阵
- 特征重要性排名
- 学习曲线
- 超参数对比
- ROC 曲线
- 训练时间对比

---

## 依赖库

**必需：**
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

**可选：**
- `xgboost` - 用于 XGBoost 演示
- `lightgbm` - 用于 LightGBM 演示

**安装所有依赖：**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

---

## 学习路线

建议按以下顺序学习：

1. **首先运行 `1_adaboost_demo.py`** - 理解 Boosting 的基本概念
2. **然后运行 `2_gradient_boosting_demo.py`** - 学习梯度提升的原理
3. **再运行 `3_xgboost_demo.py`** - 了解工业级别的优化
4. **最后运行 `4_lightgbm_demo.py`** - 学习最新的高效方法

---

## 主要参数调优建议

### n_estimators（树的数量）
- 值越大，模型越复杂，但训练时间增加
- 通常范围：50-1000
- 建议：先用较小值快速迭代，找到最优值

### learning_rate（学习率）
- 控制每个树对最终结果的贡献
- 值越小，收敛越慢，但泛化性能通常更好
- 通常范围：0.01-0.3
- 建议：降低学习率，增加 n_estimators 以获得更好的模型

### max_depth / num_leaves（树的复杂度）
- 较大的值会导致过拟合
- 通常范围：5-10（max_depth）或 20-50（num_leaves）
- 建议：从较小值开始，逐步增加

### subsample / bagging_fraction（样本比例）
- 用于减少过拟合
- 通常范围：0.6-1.0
- 建议：0.8 是一个比较好的起点

---

## 注意事项

1. **特征缩放**：虽然树模型对特征缩放不敏感，但对某些参数调优仍有帮助
2. **缺失值**：XGBoost 和 LightGBM 可以直接处理缺失值
3. **类别不平衡**：可以调整 `scale_pos_weight` 或使用 `scale_neg_weight`
4. **交叉验证**：始终使用交叉验证来评估模型性能
5. **早停**：使用早停可以防止过拟合和减少训练时间

---

## 相关资源

- [sklearn Ensemble 文档](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost 官方文档](https://xgboost.readthedocs.io/)
- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)
- [Gradient Boosting 论文](https://jerryfriedman.su.domains/papers/)

