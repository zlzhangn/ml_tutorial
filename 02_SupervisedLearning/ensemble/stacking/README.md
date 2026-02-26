# Stacking 集成学习 - 完整教程

## 📖 项目概述

本项目详细演示了 **Stacking 集成学习方法**及其相关技术。Stacking 是最强大和最灵活的集成学习方法，通过让多个基础学习器进行"合作"来构建高性能的预测模型。

## 🎯 Stacking 的核心思想

### 什么是 Stacking？

Stacking（堆叠）是一种两层的集成学习方法：

**第一层（Level 0）：基础学习器**
- 多个不同的机器学习模型
- 这些模型在原始数据上进行训练
- 模型类型应该多样化（决策树、SVM、KNN 等）

**第二层（Level 1）：元学习器**
- 一个学习器，用来综合第一层的预测
- 输入是基础学习器的预测结果（元特征）
- 输出是最终的预测结果

### Stacking 的工作流程

```
原始数据
   ↓
┌──────────┬──────────┬──────────┬──────────┐
│ 模型 1   │ 模型 2   │ 模型 3   │ 模型 4   │ （基础学习器）
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     └──────────┴──────────┴──────────┘
              ↓
         元特征生成
         （各模型的预测）
              ↓
        ┌──────────────┐
        │  元学习器    │ （如：逻辑回归）
        └──────┬───────┘
               ↓
          最终预测
```

## 📚 项目内容

### 4 个完整的演示脚本

#### 1️⃣ `1_stacking_basic.py` - Stacking 基础演示
**核心内容**：
- ✅ Stacking 分类器的基本使用
- ✅ Stacking 回归器的基本使用
- ✅ 不同基础学习器组合的对比
- ✅ Stacking vs 其他集成方法

**学习目标**：
- 理解 Stacking 的基本原理
- 学会使用 StackingClassifier/Regressor
- 了解基础学习器的选择

**关键代码示例**：
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 定义基础学习器
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=15)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True))
]

# 定义元学习器
meta_learner = LogisticRegression()

# 创建 Stacking 分类器
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5  # 5 折交叉验证生成元特征
)

# 训练和预测
stacking_clf.fit(X_train, y_train)
predictions = stacking_clf.predict(X_test)
```

#### 2️⃣ `2_blending_demo.py` - Blending 演示
**Blending 是什么**：
- Stacking 的简化版本
- 不使用交叉验证，而使用验证集
- 计算量更小，适合大数据集
- 性能通常略低于 Stacking

**核心内容**：
- ✅ Blending 的手动实现（理解原理）
- ✅ Blending 回归器演示
- ✅ Blending vs Stacking 的对比

**Blending 的步骤**：
```python
# 1. 将训练集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(...)

# 2. 用训练集训练基础学习器
for clf in base_learners:
    clf.fit(X_train, y_train)

# 3. 用验证集生成元特征
meta_features_val = np.hstack([
    clf.predict_proba(X_val) for clf in base_learners
])

# 4. 用元特征训练元学习器
meta_learner.fit(meta_features_val, y_val)

# 5. 最终预测
meta_features_test = np.hstack([
    clf.predict_proba(X_test) for clf in base_learners
])
final_predictions = meta_learner.predict(meta_features_test)
```

#### 3️⃣ `3_multilevel_stacking.py` - 多层 Stacking
**多层 Stacking 的概念**：
- 不仅用一层元学习器，而是用多层
- 每一层的输出成为下一层的输入
- 可以学习更复杂的特征组合

**核心内容**：
- ✅ 两层 Stacking（标准 Stacking）
- ✅ 三层 Stacking（多层集成）
- ✅ 不同层数的性能对比

**多层 Stacking 结构**：
```
原始数据
   ↓
[基础学习器 1, 2, 3]  ← 第一层
   ↓
[元特征：3 维]
   ↓
[基础学习器 4, 5]     ← 第二层
   ↓
[元特征：2 维]
   ↓
[最终元学习器]        ← 第三层
   ↓
最终预测
```

#### 4️⃣ `4_stacking_comprehensive.py` - 综合对比分析
**核心内容**：
- ✅ 不同元学习器的影响
- ✅ 基础学习器多样性的重要性
- ✅ Stacking vs 6 种其他集成方法
- ✅ 性能与速度的权衡

**包含的集成方法**：
1. Bagging (随机森林)
2. AdaBoost
3. Gradient Boosting
4. Voting (硬投票)
5. Voting (软投票)
6. Stacking

## ⚙️ 关键参数说明

### StackingClassifier/Regressor 的主要参数

```python
StackingClassifier(
    estimators,          # 基础学习器列表，格式: [('name1', clf1), ('name2', clf2), ...]
    final_estimator,     # 元学习器，通常选择简单的模型如逻辑回归
    cv=5,               # 交叉验证折数，用于生成元特征
    stack_method='auto', # 生成元特征的方法，可选 'predict' 或 'predict_proba'
    n_jobs=None,        # 并行处理，设为 -1 使用所有 CPU
    verbose=0           # 日志级别
)
```

## 🎓 Stacking vs 其他方法对比

| 特性 | Bagging | Boosting | Voting | Stacking |
|------|---------|----------|--------|----------|
| **原理** | Bootstrap采样 | 顺序加权 | 投票平均 | 元学习器 |
| **准确率** | 中等 | 高 | 中等 | ⭐⭐⭐⭐⭐ 最高 |
| **速度** | 快 | 中等 | 快 | 慢 |
| **易用性** | 简单 | 中等 | 简单 | 复杂 |
| **过拟合风险** | 低 | 高 | 低 | 中等 |
| **适用场景** | 快速原型 | 提升性能 | 速度优先 | 性能优先 |

## 📊 输出图表

脚本运行后会生成以下图表（保存在 `./images/` 目录）：

**脚本 1 - 基础演示**：
- `1_confusion_matrix.png` - Stacking 分类的混淆矩阵
- `2_regression_predictions.png` - Stacking 回归的预测结果
- `3_base_learners_comparison.png` - 不同基础学习器组合的对比
- `4_methods_comparison.png` - Stacking vs 其他集成方法

**脚本 2 - Blending**：
- `2_blending_confusion_matrix.png` - Blending 分类的混淆矩阵
- `2_blending_regression.png` - Blending 回归的预测结果
- `3_blending_vs_stacking.png` - Blending vs Stacking 的对比

**脚本 3 - 多层 Stacking**：
- `3_two_level_stacking.png` - 两层 Stacking 的混淆矩阵
- `3_three_level_stacking.png` - 三层 Stacking 的混淆矩阵
- `3_layers_comparison.png` - 不同层数的性能对比

**脚本 4 - 综合对比**：
- `4_meta_learner_impact.png` - 元学习器选择的影响
- `4_diversity_importance.png` - 基础学习器多样性的重要性
- `4_ensemble_comparison.png` - 6 种集成方法的全面对比

## 🚀 快速开始

### 最快启动 (5 分钟)
```bash
python 1_stacking_basic.py
```

### 完整学习 (30 分钟)
```bash
python 1_stacking_basic.py        # 基础概念
python 2_blending_demo.py         # Blending 方法
python 3_multilevel_stacking.py   # 多层 Stacking
python 4_stacking_comprehensive.py # 全面对比
```

## 💡 最佳实践

### 1. 基础学习器的选择
- ✅ **选择多样化的模型** - 不要用 4 个都一样的决策树
- ✅ **推荐组合**：
  - 线性模型 (逻辑回归、Ridge)
  - 树模型 (决策树、随机森林)
  - 实例模型 (KNN)
  - 核模型 (SVM)

### 2. 元学习器的选择
- ✅ **通常用简单的模型**：
  - 分类：逻辑回归（最常用）
  - 回归：Ridge 回归或线性回归
  - 优点：防止过拟合，计算快速

### 3. 避免过拟合
- ✅ 使用交叉验证生成元特征
- ✅ 不要用复杂的元学习器
- ✅ 基础学习器不要过度拟合

### 4. 计算效率
- ✅ 数据量大时考虑用 Blending 代替 Stacking
- ✅ 使用 `n_jobs=-1` 启用并行计算
- ✅ 基础学习器数量不要超过 10 个

## ❓ 常见问题

### Q1: Stacking 何时使用？
**A**: 
- 当你需要最好的性能时
- 当有足够的计算资源时
- 当数据不是太大时（GB 级以内）

### Q2: 基础学习器越多越好吗？
**A**: 
- 不一定。通常 4-6 个多样化的学习器足够
- 超过 10 个通常没有明显改进
- 太多会增加复杂度和过拟合风险

### Q3: Blending 和 Stacking 如何选择？
**A**:
- **选择 Stacking**：性能优先，数据量中等（< 1GB）
- **选择 Blending**：速度优先，数据量很大（> 1GB）

### Q4: 元学习器很重要吗？
**A**:
- 是的。但通常简单的模型（如逻辑回归）效果最好
- 复杂的元学习器反而容易过拟合
- 元学习器的作用是"学会如何组合"

### Q5: 三层 Stacking 有必要吗？
**A**:
- 很少。大多数情况下两层 Stacking 就足够了
- 三层开始复杂度大幅增加，性能提升有限
- 除非特别需要，一般不推荐

## 📈 学习路径

### 初学者 (1-2 小时)
1. 阅读项目概述部分
2. 运行 `1_stacking_basic.py` 了解基本用法
3. 查看生成的图表

### 进阶者 (2-3 小时)
1. 详细阅读本文档
2. 按顺序运行所有脚本
3. 修改参数进行实验

### 专家 (3+ 小时)
1. 研究脚本的详细实现
2. 在自己的数据集上应用
3. 进行超参数优化
4. 与其他方法比较

## 🔗 相关资源

- [scikit-learn Stacking 文档](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- Random Forest 论文：Breiman, L. (2001)
- Stacking 论文：Wolpert, D. H. (1992)

## 📝 总结

**Stacking 的优势**：
- ✅ 通常性能最好
- ✅ 灵活，可以组合任意模型
- ✅ 理论上有坚实的基础

**Stacking 的劣势**：
- ❌ 计算复杂度高
- ❌ 需要更多代码和调试
- ❌ 容易过拟合，需要谨慎

**何时使用 Stacking**：
- ✅ 当性能是首要目标时
- ✅ 当你有时间和资源进行调优时
- ✅ 当数据是结构化的时（不是图像或文本）

---

**祝你学习愉快！** 🎉

从 `1_stacking_basic.py` 开始，或查看 QUICKSTART.md 了解快速开始方式。
