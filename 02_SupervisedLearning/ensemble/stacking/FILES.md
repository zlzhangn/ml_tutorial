# 📁 Stacking 项目文件清单

## 项目概览

本项目包含 Stacking 集成学习的完整演示和教学资源。

```
stacking/
├── 演示脚本 (Demonstration Scripts)
│   ├── 1_stacking_basic.py              # Stacking 基础演示
│   ├── 2_blending_demo.py               # Blending 方法演示
│   ├── 3_multilevel_stacking.py         # 多层 Stacking 演示
│   └── 4_stacking_comprehensive.py      # 综合对比分析
│
├── 文档 (Documentation)
│   ├── README.md                        # 完整项目文档
│   ├── QUICKSTART.md                    # 快速开始指南
│   ├── FILES.md                         # 本文件 - 文件清单
│   ├── PROJECT_SUMMARY.md               # 项目统计总结
│   ├── START_HERE.md                    # 新用户入门指南
│   └── COMPLETION_SUMMARY.md            # 项目完成报告
│
├── 交互工具 (Interactive Tools)
│   └── run_demos.py                     # 交互式菜单脚本
│
└── 输出目录 (Output Directory)
    └── images/                          # 图表输出目录
        ├── 1_confusion_matrix.png
        ├── 1_regression_predictions.png
        ├── 1_base_learners_comparison.png
        ├── 1_methods_comparison.png
        ├── 2_blending_confusion_matrix.png
        ├── 2_blending_regression.png
        ├── 2_blending_vs_stacking.png
        ├── 3_two_level_stacking.png
        ├── 3_three_level_stacking.png
        ├── 3_layers_comparison.png
        ├── 4_meta_learner_impact.png
        ├── 4_diversity_importance.png
        └── 4_ensemble_comparison.png
```

---

## 📄 详细文件说明

### 演示脚本 (4 个)

#### 1️⃣ `1_stacking_basic.py` (530 行)

**目的**：理解 Stacking 的基础概念和用法

**关键功能**：
```python
- demo_stacking_classification()      # Stacking 分类演示
- demo_stacking_regression()          # Stacking 回归演示
- compare_base_learner_combinations() # 基础学习器组合对比
- compare_with_other_methods()        # 与其他方法对比
```

**核心概念**：
- StackingClassifier/StackingRegressor API
- 基础学习器选择 (4 个不同类型的学习器)
- 元学习器角色 (逻辑回归)
- 交叉验证在元特征生成中的作用

**使用场景**：
```python
# 基本 Stacking 分类
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

base_learners = [
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier())
]
meta_learner = LogisticRegression()
clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
```

**生成的输出**：
- `images/1_confusion_matrix.png` - 混淆矩阵对比
- `images/1_regression_predictions.png` - 回归预测对比
- `images/1_base_learners_comparison.png` - 基础学习器数量影响
- `images/1_methods_comparison.png` - 方法性能对比

**预计运行时间**：5-10 分钟

---

#### 2️⃣ `2_blending_demo.py` (470 行)

**目的**：理解 Blending - Stacking 的简化版本

**核心差异**：
- Stacking：使用交叉验证生成元特征 (慢但准确)
- Blending：使用验证集生成元特征 (快但准确率略低)

**关键功能**：
```python
- demo_blending_classification()      # Blending 分类
- demo_blending_regression()          # Blending 回归
- compare_blending_vs_stacking()      # Blending vs Stacking
```

**详细流程**：
```python
# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)

# 基础学习器在训练集上训练
for base_learner in base_learners:
    base_learner.fit(X_train, y_train)

# 验证集生成元特征
meta_features = np.zeros((len(X_val), len(base_learners)))
for i, base_learner in enumerate(base_learners):
    meta_features[:, i] = base_learner.predict(X_val)

# 元学习器在验证元特征上训练
meta_learner.fit(meta_features, y_val)
```

**性能对比**：
- Stacking 速度：较慢 (需要交叉验证)
- Blending 速度：快 5-10 倍
- 准确率：Stacking 通常略高 (0.5-1%)

**生成的输出**：
- `images/2_blending_confusion_matrix.png` - Blending 混淆矩阵
- `images/2_blending_regression.png` - Blending 回归结果
- `images/2_blending_vs_stacking.png` - 速度和准确率对比

**预计运行时间**：5-10 分钟

---

#### 3️⃣ `3_multilevel_stacking.py` (480 行)

**目的**：理解多层 Stacking 和高级集成架构

**架构演示**：
```
两层 Stacking:
Level 0: 原始特征 (30 维)
Level 1: 基础学习器们 → 4 个元特征
Level 2: 元学习器 → 最终预测

三层 Stacking:
Level 0: 原始特征
Level 1: 4 个基础学习器 → 4 个元特征
Level 2: 2 个基础学习器 → 2 个元特征
Level 3: 最终元学习器 → 最终预测
```

**关键功能**：
```python
- demo_two_level_stacking()      # 两层 Stacking
- demo_three_level_stacking()    # 三层 Stacking  
- compare_stacking_layers()      # 层数影响分析
```

**重要发现**：
- 两层 Stacking 通常最有效
- 三层及以上的效果改进有限
- 过多层数会增加过拟合风险

**生成的输出**：
- `images/3_two_level_stacking.png` - 两层 Stacking 结果
- `images/3_three_level_stacking.png` - 三层 Stacking 结果
- `images/3_layers_comparison.png` - 不同层数对比

**预计运行时间**：8-12 分钟

---

#### 4️⃣ `4_stacking_comprehensive.py` (540 行)

**目的**：全面对比 Stacking 与其他集成方法

**涵盖的方法** (6 种)：
```python
1. Bagging         - 并行基础学习器
2. AdaBoost        - 串行提升
3. Gradient Boosting - 梯度提升
4. Voting Hard     - 硬投票
5. Voting Soft     - 软投票
6. Stacking        - 元学习器
```

**元学习器变体**：
```python
- LogisticRegression  # 线性元学习器
- RandomForest        # 树形元学习器
- SVM                 # 核方法元学习器
- GradientBoosting    # 提升型元学习器
```

**基础学习器多样性分析**：
```python
# 低多样性 (都是树模型)
low_diversity = [DecisionTree, RandomForest, GradientBoosting, ExtraTree]

# 高多样性 (混合类型)
high_diversity = [DecisionTree, RandomForest, KNeighbors, SVM, LinearRegression, LogisticRegression]
```

**性能指标**：
```python
- Accuracy / AUC      # 准确率
- Precision / Recall  # 精确率/召回率
- F1 Score           # F1 分数
- Training Time      # 训练时间
```

**关键发现**：
- 基础学习器多样性很关键 (影响最多 3-5%)
- 元学习器选择影响 1-2%
- Stacking 通常是最佳选择，但速度最慢
- VotingSoft 是速度/性能的最佳平衡点

**生成的输出**：
- `images/4_meta_learner_impact.png` - 元学习器影响
- `images/4_diversity_importance.png` - 多样性重要性
- `images/4_ensemble_comparison.png` - 6 方法对比

**预计运行时间**：10-15 分钟

---

### 📚 文档文件 (6 个)

#### `README.md` (350+ 行)
**内容**：完整的项目文档和理论说明
- ✅ Stacking 原理详解 (What/Why/How)
- ✅ 4 个脚本详细说明
- ✅ API 参数完整指南
- ✅ 集成方法对比表
- ✅ 最佳实践 (5 条)
- ✅ 常见问题解答 (8 个)
- ✅ 推荐学习路径

**适合阅读**：所有水平用户（初学者到高级）

#### `QUICKSTART.md` (300+ 行)
**内容**：快速开始和实用代码
- ✅ 5 分钟快速开始
- ✅ 脚本选择决策树
- ✅ 3 个完整代码示例
- ✅ 参数速查表
- ✅ 性能优化技巧 (3 条)
- ✅ 故障排查指南
- ✅ 学习顺序建议

**适合阅读**：初学者和急于上手的用户

#### `FILES.md` (本文件)
**内容**：项目文件结构和使用指南
- ✅ 项目概览
- ✅ 每个文件的详细说明
- ✅ 代码示例
- ✅ 文件间的关系

**适合阅读**：需要了解项目结构的用户

#### `PROJECT_SUMMARY.md` (400+ 行)
**内容**：项目统计和完成情况总结
- ✅ 项目统计数据
- ✅ 功能完整性检查
- ✅ 学习目标成果
- ✅ 代码质量评估
- ✅ 性能基准测试结果

**适合阅读**：要求了解项目全貌的用户

#### `START_HERE.md` (100+ 行)
**内容**：新用户的快速入门指南
- ✅ 如何开始 (3 步)
- ✅ 推荐学习路径
- ✅ 常见问题
- ✅ 文档导航

**适合阅读**：第一次使用此项目的用户

#### `COMPLETION_SUMMARY.md` (200+ 行)
**内容**：项目完成情况详细报告
- ✅ 完成清单
- ✅ 质量评估
- ✅ 关键成果
- ✅ 下一步建议

**适合阅读**：项目审查和验收

---

### 🛠️ 工具文件

#### `run_demos.py` (450+ 行)

**功能**：交互式菜单运行演示脚本

**功能特性**：
```
✓ 彩色菜单界面
✓ 依赖包检查
✓ 脚本选择和运行
✓ 批量运行所有脚本
✓ 错误处理
✓ 文档查看
```

**使用方法**：
```bash
python run_demos.py

# 按菜单提示选择:
# 1-4: 运行对应的演示脚本
# 5-6: 查看文档
# 0: 退出
# a: 运行所有脚本
```

**输出示例**：
```
╔════════════════════════════════════════════════════════════════╗
║                  Stacking 演示菜单 - 交互式                   ║
║              Interactive Menu for Stacking Methods            ║
╚════════════════════════════════════════════════════════════════╝

检查依赖包...
✓ NumPy - 已安装
✓ Pandas - 已安装
✓ Matplotlib - 已安装
✓ Seaborn - 已安装
✓ scikit-learn - 已安装

所有依赖已安装！

📋 请选择要运行的演示：

1. Stacking 基础演示
   📝 学习 Stacking 的核心概念和基本用法
   ⏱️  5-10 分钟
   📊 初级
```

---

## 🔗 文件依赖关系

```
学习推荐顺序：
↓
START_HERE.md (了解项目)
↓
QUICKSTART.md (快速示例)
↓
1_stacking_basic.py (运行基础脚本)
↓
2_blending_demo.py (学习简化版本)
↓
3_multilevel_stacking.py (理解高级用法)
↓
4_stacking_comprehensive.py (全面对比)
↓
README.md (深入理解)
↓
PROJECT_SUMMARY.md (总结收获)
```

**脚本运行时的文件关系**：
```
1_stacking_basic.py
├── 导入: numpy, pandas, matplotlib, sklearn
├── 输出: ./images/1_*.png
└── 独立运行（无依赖）

2_blending_demo.py
├── 导入: numpy, pandas, matplotlib, sklearn
├── 输出: ./images/2_*.png
└── 独立运行（无依赖）

3_multilevel_stacking.py
├── 导入: numpy, pandas, matplotlib, sklearn
├── 输出: ./images/3_*.png
└── 独立运行（无依赖）

4_stacking_comprehensive.py
├── 导入: numpy, pandas, matplotlib, sklearn
├── 输出: ./images/4_*.png
└── 独立运行（无依赖）

run_demos.py
├── 调用: 1-4 的脚本
├── 依赖检查: numpy, pandas, matplotlib, seaborn, sklearn
└── 管理: 菜单界面和脚本执行
```

---

## 📊 项目统计

| 项目 | 数量 | 行数 |
|------|------|------|
| 演示脚本 | 4 | 2,020 |
| 文档文件 | 6 | 1,650+ |
| 工具脚本 | 1 | 450+ |
| **总计** | **11** | **4,120+** |

| 脚本 | 行数 | 函数 | 耗时 |
|------|------|------|------|
| 1_stacking_basic.py | 530 | 4 | 5-10 分钟 |
| 2_blending_demo.py | 470 | 3 | 5-10 分钟 |
| 3_multilevel_stacking.py | 480 | 3 | 8-12 分钟 |
| 4_stacking_comprehensive.py | 540 | 3 | 10-15 分钟 |

---

## 🎓 学习目标覆盖

项目涵盖以下学习目标：

- ✅ 理解 Stacking 的基本概念
- ✅ 掌握 StackingClassifier/Regressor API
- ✅ 学会选择基础学习器和元学习器
- ✅ 理解 Blending 方法及其优缺点
- ✅ 掌握多层 Stacking 的构建方法
- ✅ 对比不同集成方法的性能
- ✅ 分析基础学习器多样性的重要性
- ✅ 进行集成学习的参数优化

---

## 💡 使用建议

### 第一次使用
1. 阅读 `START_HERE.md`
2. 运行 `run_demos.py` 并选择脚本 1
3. 查看生成的图表 (`images/` 目录)
4. 读 `QUICKSTART.md` 的代码示例

### 深入学习
1. 按顺序运行脚本 2-4
2. 研究代码中的注释
3. 修改参数并重新运行，观察结果变化
4. 阅读 `README.md` 的深入解释

### 实际应用
1. 参考 `QUICKSTART.md` 的代码模板
2. 使用 `run_demos.py` 快速测试想法
3. 参考 `README.md` 的最佳实践
4. 调整参数以适应你的数据集

---

## 📝 文件编码

所有文件均使用 **UTF-8** 编码，支持中文注释和文档。

---

## 🔄 维护和更新

- **最后更新**：2024 年
- **Python 版本**：3.7+
- **依赖版本**：scikit-learn 0.24+, numpy 1.19+, matplotlib 3.3+
- **维护状态**：✅ 活跃维护

---

## ❓ 快速问答

**Q: 应该从哪个脚本开始？**
A: 从 `1_stacking_basic.py` 开始，它提供了基础概念的完整介绍。

**Q: 脚本之间有依赖吗？**
A: 没有，所有脚本都可以独立运行。

**Q: 如何添加自己的数据集？**
A: 参考脚本中的 `train_test_split` 示例，替换为你的数据。

**Q: 图表保存在哪里？**
A: 所有图表都保存在 `images/` 目录下。

**Q: 如何修改模型参数？**
A: 参考 `README.md` 的参数指南章节。

---

**祝你学习愉快！🎓**
