# 快速开始指南 - Stacking

## ⚡ 5 分钟快速开始

### 第 1 步: 安装依赖
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 第 2 步: 进入项目目录
```bash
cd 02_SupervisedLearning/ensemble/stacking
```

### 第 3 步: 运行演示
```bash
# 最快方式：运行基础演示
python 1_stacking_basic.py
```

### 第 4 步: 查看结果
- **控制台输出**：性能指标
- **图表文件**：保存在 `./images/` 目录

---

## 📋 按需运行

### 我想快速了解 Stacking
```bash
python 1_stacking_basic.py
```

### 我想学习 Blending 方法
```bash
python 2_blending_demo.py
```

### 我想研究多层 Stacking
```bash
python 3_multilevel_stacking.py
```

### 我想全面对比集成方法
```bash
python 4_stacking_comprehensive.py
```

### 我想一次运行所有演示
```bash
python 1_stacking_basic.py
python 2_blending_demo.py
python 3_multilevel_stacking.py
python 4_stacking_comprehensive.py
```

---

## 🔥 四种方法速查表

| 方法 | 优点 | 缺点 | 何时使用 |
|------|------|------|---------|
| **Stacking** | 性能最好 | 计算量大 | 性能优先 |
| **Blending** | 速度快 | 性能略低 | 大数据集 |
| **多层 Stacking** | 更强大 | 极其复杂 | 特殊情况 |
| **Voting** | 简单快速 | 性能一般 | 快速原型 |

---

## 💻 快速代码示例

### 最简单的 Stacking
```python
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义基础学习器
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=10)),
    ('rf', RandomForestClassifier(n_estimators=50))
]

# 定义元学习器
meta_learner = LogisticRegression()

# 创建 Stacking 分类器
clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner
)

# 训练
clf.fit(X_train, y_train)

# 预测
accuracy = clf.score(X_test, y_test)
print(f"准确率: {accuracy:.4f}")
```

### Stacking 回归
```python
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# 生成数据
X, y = make_regression(n_samples=300, n_features=20, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义基础回归器
base_learners = [
    ('dt', DecisionTreeRegressor(max_depth=10)),
    ('rf', RandomForestRegressor(n_estimators=50))
]

# 定义元回归器
meta_learner = Ridge()

# 创建 Stacking 回归器
reg = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner
)

# 训练和评估
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
print(f"R² 分数: {score:.4f}")
```

### 手动实现 Blending
```python
from sklearn.model_selection import train_test_split
import numpy as np

# 将训练集分为训练和验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42
)

# 训练基础学习器
clf1 = DecisionTreeClassifier(max_depth=10)
clf2 = RandomForestClassifier(n_estimators=50)
clf1.fit(X_train_split, y_train_split)
clf2.fit(X_train_split, y_train_split)

# 生成验证集的元特征
meta_features_val = np.hstack([
    clf1.predict_proba(X_val),
    clf2.predict_proba(X_val)
])

# 训练元学习器
meta_clf = LogisticRegression()
meta_clf.fit(meta_features_val, y_val)

# 在测试集上预测
meta_features_test = np.hstack([
    clf1.predict_proba(X_test),
    clf2.predict_proba(X_test)
])
predictions = meta_clf.predict(meta_features_test)
```

---

## ⚙️ 常用参数速查

### Stacking 关键参数
```python
StackingClassifier(
    estimators=[              # 基础学习器列表
        ('name1', clf1),
        ('name2', clf2)
    ],
    final_estimator=clf,      # 元学习器
    cv=5,                     # 交叉验证折数（通常 5 或 10）
    stack_method='predict_proba',  # 生成元特征的方法
    n_jobs=-1                 # 并行处理数（-1 表示使用所有 CPU）
)
```

### 推荐的基础学习器组合
```python
base_learners = [
    # 树模型
    ('dt', DecisionTreeClassifier(max_depth=10)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    
    # 线性模型
    ('lr', LogisticRegression(max_iter=1000)),
    
    # 实例模型
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    
    # 核模型
    ('svm', SVC(kernel='rbf', probability=True))
]
```

### 推荐的元学习器
```python
# 分类：逻辑回归（最常用）
meta_clf = LogisticRegression(random_state=42, max_iter=1000)

# 回归：Ridge 回归
meta_reg = Ridge(alpha=1.0)

# 分类：也可以用随机森林（但容易过拟合）
meta_clf = RandomForestClassifier(n_estimators=50, random_state=42)
```

---

## 🔧 性能优化技巧

### 1. 使用并行化加速训练
```python
# 使用所有 CPU 核心
clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_clf,
    n_jobs=-1  # 关键！
)
```

### 2. 减少交叉验证折数
```python
# 从 10 折改为 5 折，速度提升一倍
clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_clf,
    cv=5  # 从 10 改为 5
)
```

### 3. 简化基础学习器
```python
# 减少树的数量或深度
('rf', RandomForestClassifier(n_estimators=50, max_depth=10))
```

### 4. 数据量大时用 Blending
```python
# Blending 不需要交叉验证，速度是 Stacking 的 5-10 倍
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
# ... 手动实现 Blending
```

---

## 🐛 常见问题解决

### 问题 1: 训练太慢
**原因**：基础学习器太多或交叉验证折数太多
**解决**：
```python
# 减少基础学习器数量（保留多样性）
# 减少交叉验证折数
clf = StackingClassifier(..., cv=3)
# 使用 n_jobs=-1 启用并行
clf = StackingClassifier(..., n_jobs=-1)
```

### 问题 2: 性能没有提升
**原因**：基础学习器不够多样化
**解决**：
```python
# 使用不同类型的模型
base_learners = [
    ('tree', DecisionTreeClassifier()),     # 树
    ('forest', RandomForestClassifier()),   # 树
    ('svm', SVC(probability=True)),         # 核
    ('knn', KNeighborsClassifier()),        # 实例
    ('nb', GaussianNB())                    # 概率
]
```

### 问题 3: 过拟合
**原因**：元学习器过于复杂
**解决**：
```python
# 使用简单的元学习器
final_estimator = LogisticRegression()  # 而不是复杂的模型

# 或添加正则化
final_estimator = Ridge(alpha=10.0)
```

### 问题 4: 内存不足
**原因**：处理大数据集时内存占用过多
**解决**：
```python
# 使用 Blending 代替 Stacking（内存占用更少）
# 或减少基础学习器数量
# 或使用样本子集进行训练
```

---

## 📊 何时选择哪种方法

```
你的优先级是什么？

├─ 性能优先 (1%)
│  └─ 使用 Stacking (4-6 个多样化基础学习器)
│
├─ 速度优先 (50%)
│  └─ 使用 Blending 或 Voting
│
├─ 平衡 (45%)
│  └─ 使用 Voting Soft (快速且有效)
│
└─ 研究学习 (4%)
   └─ 尝试多层 Stacking
```

---

## 🎯 推荐学习顺序

### 第 1 天（1 小时）
1. 阅读本快速开始指南
2. 运行 `python 1_stacking_basic.py`
3. 观察输出和生成的图表

### 第 2 天（2 小时）
1. 详细阅读 README.md
2. 运行所有脚本
3. 修改参数进行实验

### 第 3 天及以后（3+ 小时）
1. 在自己的数据集上应用
2. 进行超参数优化
3. 与其他方法比较性能

---

## 📚 进一步学习

### 相关主题
- Boosting 集成学习 (ensemble/boosting/)
- Bagging 集成学习 (ensemble/bagging/)
- 超参数优化 (GridSearchCV, RandomizedSearchCV)
- 模型选择和评估

### 推荐资源
- scikit-learn 官方文档
- "Python 机器学习"书籍
- "模式识别和机器学习" (PRML)

---

## 🚀 现在就开始！

```bash
# 5 秒启动
python 1_stacking_basic.py

# 或查看完整说明
cat README.md
```

**祝你学习愉快！** 🎓
