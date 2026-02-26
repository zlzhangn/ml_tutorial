# 快速开始指南 - Bagging 演示

## 五分钟快速开始

### 1️⃣ 安装依赖（如未安装）
```bash
pip install numpy matplotlib seaborn scikit-learn
```

### 2️⃣ 进入项目目录
```bash
cd 02_SupervisedLearning/ensemble/bagging
```

### 3️⃣ 运行演示

**推荐方式：按顺序逐个运行**

```bash
# 1. 了解基础 Bagging
python 1_bagging_demo.py

# 2. 学习随机森林（最常用）
python 2_random_forest_demo.py

# 3. 学习 Extra Trees（快速版本）
python 3_extra_trees_demo.py

# 4. 全面对比所有方法
python 4_bagging_comparison.py
```

### 4️⃣ 查看结果
- **控制台输出**：性能指标和模型评估
- **图表文件**：保存在 `images/` 目录

---

## 根据需要选择运行

### 如果你想...

**快速了解 Bagging**
```bash
python 1_bagging_demo.py
```

**学习最实用的方法**
```bash
python 2_random_forest_demo.py
```

**比较所有方法的性能**
```bash
python 4_bagging_comparison.py
```

**处理大数据集**
```bash
python 3_extra_trees_demo.py
```

---

## 四种方法快速对比

| 方法 | 优点 | 缺点 | 何时使用 |
|------|------|------|---------|
| **Bagging** | 灵活 | 速度中等 | 需要自定义基础分类器 |
| **Random Forest** | 准确率高 | 内存占用大 | 大多数情况 |
| **Extra Trees** | 速度快 | 随机性大 | 大数据集 |
| **Pasting** | 内存高效 | 准确率较低 | 内存受限 |

---

## 常用参数速查

### Random Forest 最重要的参数

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,      # 树的数量（默认 100）
    max_depth=None,        # 树的最大深度（默认不限制）
    max_features='sqrt',   # 每次分割考虑的特征数（'sqrt', 'log2', None）
    oob_score=True,        # 计算 OOB 评分（推荐）
    n_jobs=-1,            # 使用所有 CPU 核心
    random_state=42       # 随机种子
)
```

### Extra Trees 参数

```python
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(
    n_estimators=100,      # 树的数量
    max_depth=None,        # 树的最大深度
    max_features='sqrt',   # 特征数
    bootstrap=True,        # 使用 Bootstrap
    n_jobs=-1,            # 并行化
    random_state=42
)
```

---

## 快速代码示例

### 最简单的用法

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练
clf.fit(X_train, y_train)

# 评估
accuracy = clf.score(X_test, y_test)
print(f"准确率: {accuracy:.4f}")

# 特征重要性
importance = clf.feature_importances_
```

### 使用 OOB 评分

```python
clf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # 启用 OOB 评分
    random_state=42
)
clf.fit(X_train, y_train)

# 获取 OOB 评分（无偏的泛化性能估计）
print(f"OOB 评分: {clf.oob_score_:.4f}")
print(f"测试准确率: {clf.score(X_test, y_test):.4f}")
```

### 获取特征重要性

```python
# 训练后获取特征重要性
importance = clf.feature_importances_

# 排序
import numpy as np
indices = np.argsort(importance)[::-1]

# 打印 Top 10 特征
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
```

---

## 性能优化技巧

### 1. 使用并行化
```python
# 使用所有 CPU 核心
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
```

### 2. 调整树的深度
```python
# 对于大数据集，限制树的深度可以加快训练
clf = RandomForestClassifier(max_depth=20, n_estimators=100)
```

### 3. 特征采样
```python
# 使用更少的特征可以加快训练
clf = RandomForestClassifier(max_features='log2', n_estimators=100)
```

### 4. 使用 OOB 评分代替交叉验证
```python
# 自动计算 OOB 评分，无需交叉验证
clf = RandomForestClassifier(oob_score=True, n_estimators=100)
clf.fit(X_train, y_train)
print(f"OOB 评分: {clf.oob_score_}")  # 快速评估
```

---

## 故障排查

**Q: 训练太慢？**
A: 
- 增加 `max_depth` 的限制
- 减少 `n_estimators`
- 使用 `max_features='log2'`
- 确保使用 `n_jobs=-1`

**Q: 内存占用过大？**
A:
- 使用 Pasting（`bootstrap=False`）
- 减少 `n_estimators`
- 限制 `max_depth`

**Q: 准确率不理想？**
A:
- 增加 `n_estimators`
- 减少 `min_samples_split` 和 `min_samples_leaf`
- 检查数据预处理是否正确

**Q: 过拟合？**
A:
- 增加 `min_samples_split` 和 `min_samples_leaf`
- 减少 `max_depth`
- 增加 `max_features` 的约束

---

## 推荐学习流程

### 初学者（1 小时）
1. 运行 `1_bagging_demo.py` - 理解基本概念
2. 查看 README.md - 了解关键参数
3. 观察 `images/` 中的图表

### 进阶者（2 小时）
1. 运行 `2_random_forest_demo.py` - 深入学习
2. 修改参数进行实验
3. 对比不同参数的效果

### 高级者（3+ 小时）
1. 运行 `3_extra_trees_demo.py` 和 `4_bagging_comparison.py`
2. 在自己的数据集上实验
3. 进行超参数优化

---

## 快速参考：何时选择哪种方法

```
你的数据集有多大？
├─ < 10K 样本
│  └─ 使用 Random Forest（平衡性好）
├─ 10K - 100K 样本
│  └─ 使用 Random Forest 或 Extra Trees
└─ > 100K 样本
   └─ 优先考虑 Extra Trees（快速）

你有多少特征？
├─ < 50 特征
│  └─ 使用 Random Forest
├─ 50 - 200 特征
│  └─ 使用 Random Forest 或 Extra Trees
└─ > 200 特征
   └─ 使用 Extra Trees（特征采样加快）

你的优先级是什么？
├─ 最高准确率
│  └─ Random Forest（尽可能多的树）
├─ 速度优先
│  └─ Extra Trees（快速分割）
├─ 内存优先
│  └─ Pasting（不使用 Bootstrap）
└─ 灵活性优先
   └─ Bagging（任意基础分类器）
```

---

## 相关资源

- 官方文档：https://scikit-learn.org/stable/modules/ensemble.html
- Random Forest 论文：https://www.stat.berkeley.edu/~breiman/random-forests.pdf
- Extra Trees 论文：https://arxiv.org/pdf/1309.0238.pdf

---

**现在就开始吧！** 🚀

```bash
python 2_random_forest_demo.py
```

