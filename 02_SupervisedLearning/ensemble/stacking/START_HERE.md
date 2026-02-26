# 🚀 Stacking 项目快速入门

欢迎来到 **Stacking 集成学习演示项目**！

本项目提供了从基础到高级的完整 Stacking 学习资源。

---

## ⚡ 3 分钟快速开始

### 第一步：检查依赖
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 第二步：运行菜单
```bash
python run_demos.py
```

### 第三步：选择演示
```
1. 基础 Stacking      ← 推荐首选！
2. Blending 方法      
3. 多层 Stacking
4. 综合对比
```

**就这么简单！** 脚本会自动生成图表，保存到 `images/` 目录。

---

## 📚 文档地图

```
你在这里 ↓
START_HERE.md (3分钟读完)
    ↓
QUICKSTART.md (代码示例) 
    ↓
README.md (完整理论)
    ↓
PROJECT_SUMMARY.md (学习总结)
```

| 文件 | 用途 | 阅读时间 | 推荐人群 |
|------|------|---------|---------|
| **START_HERE.md** | 快速入门（你在看这个）| 3 分钟 | 所有人 |
| **QUICKSTART.md** | 代码示例+参数表 | 10 分钟 | 急于上手的人 |
| **README.md** | 完整理论+最佳实践 | 30 分钟 | 想深入理解的人 |
| **FILES.md** | 文件结构说明 | 10 分钟 | 想了解项目全貌的人 |
| **PROJECT_SUMMARY.md** | 项目统计报告 | 15 分钟 | 做项目审查的人 |

---

## 🎯 推荐学习路径

### 初学者路径（30 分钟）
```
1. 阅读本文档 (START_HERE)          ← 3 分钟
2. 运行脚本 1 (1_stacking_basic.py)  ← 5-10 分钟
3. 查看生成的图表 (images/)          ← 2 分钟
4. 读 QUICKSTART.md 代码示例        ← 10 分钟
✅ 你已掌握 Stacking 基础！
```

### 进阶学习路径（60 分钟）
```
1. 完成初学者路径                     ← 30 分钟
2. 运行脚本 2 (2_blending_demo.py)   ← 5-10 分钟
3. 运行脚本 3 (3_multilevel_stacking) ← 8-12 分钟
4. 读 README.md 深入部分              ← 20 分钟
✅ 你已掌握多层 Stacking！
```

### 专业级路径（120 分钟）
```
1. 完成进阶学习路径                          ← 60 分钟
2. 运行脚本 4 (4_stacking_comprehensive.py)  ← 10-15 分钟
3. 研究脚本代码和注释                       ← 30 分钟
4. 读 README.md 最佳实践+FAQ                 ← 20 分钟
✅ 你已成为 Stacking 专家！
```

---

## 🎓 4 个演示脚本速览

| # | 脚本名 | 难度 | 时间 | 核心内容 |
|---|--------|------|------|---------|
| 1️⃣ | `1_stacking_basic.py` | ⭐ 初级 | 5-10min | Stacking 核心概念 |
| 2️⃣ | `2_blending_demo.py` | ⭐⭐ 中级 | 5-10min | Blending 简化方法 |
| 3️⃣ | `3_multilevel_stacking.py` | ⭐⭐⭐ 高级 | 8-12min | 多层集成架构 |
| 4️⃣ | `4_stacking_comprehensive.py` | ⭐⭐⭐ 高级 | 10-15min | 6 种方法对比 |

### 脚本 1️⃣ - Stacking 基础
```python
# 学到什么：
✓ Stacking 的 What/Why/How
✓ StackingClassifier API
✓ 基础学习器选择
✓ 元学习器的作用
✓ 与其他方法的对比

# 关键代码：
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(
    estimators=[('rf', RandomForest()), ('svm', SVM())],
    final_estimator=LogisticRegression()
)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
```

### 脚本 2️⃣ - Blending 演示
```python
# 学到什么：
✓ Blending 和 Stacking 的区别
✓ 手动实现 Blending
✓ 性能/速度权衡
✓ 什么时候用 Blending

# 核心思想：
Stacking = 使用 CV 生成元特征 (准但慢)
Blending = 使用验证集生成元特征 (快但准)
```

### 脚本 3️⃣ - 多层 Stacking
```python
# 学到什么：
✓ 二层 Stacking 架构
✓ 三层 Stacking 架构
✓ 层数的影响
✓ 何时停止堆叠

# 关键发现：
- 2 层是最优选择
- 3 层以上改进有限
- 过多层数容易过拟合
```

### 脚本 4️⃣ - 综合对比
```python
# 学到什么：
✓ 6 种集成方法对比 (Bagging, Boosting, 等)
✓ 元学习器的影响
✓ 基础学习器多样性的重要性
✓ 性能 vs 速度权衡

# 对比方法：
1. Bagging
2. AdaBoost  
3. GradientBoosting
4. VotingHard
5. VotingSoft
6. Stacking ← 通常最好
```

---

## 💻 常见任务

### 任务 1：在自己的数据上使用 Stacking

```python
# 步骤 1：加载数据
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 步骤 2：创建基础学习器
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

base_learners = [
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True))
]

# 步骤 3：创建 Stacking 分类器
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5  # 5 折交叉验证
)

# 步骤 4：训练和预测
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f"准确率: {score:.4f}")
```

### 任务 2：比较 Stacking 和 Blending

```python
# Stacking（参考脚本 1）
from sklearn.ensemble import StackingClassifier
clf_stack = StackingClassifier(...)
clf_stack.fit(X_train, y_train)
stack_score = clf_stack.score(X_test, y_test)

# Blending（参考脚本 2）
# 1. 分割数据
X_train_base, X_val, y_train_base, y_val = train_test_split(
    X_train, y_train, test_size=0.5
)

# 2. 基础学习器在训练集上训练
for name, clf in base_learners:
    clf.fit(X_train_base, y_train_base)

# 3. 验证集生成元特征
meta_X = np.array([
    clf.predict_proba(X_val)[:, 1] for name, clf in base_learners
]).T

# 4. 元学习器在元特征上训练
meta_clf = LogisticRegression()
meta_clf.fit(meta_X, y_val)

blending_score = meta_clf.score(X_test, y_test)
```

### 任务 3：添加自己的基础学习器

```python
# 步骤 1：导入学习器
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 步骤 2：创建基础学习器列表（多样性很重要！）
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=10)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('xgb', XGBClassifier(n_estimators=100)),
]

# 步骤 3：使用 Stacking
clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression()
)
clf.fit(X_train, y_train)
```

---

## ❓ 常见问题

**Q1: Stacking 和 Boosting 有什么区别？**
- Stacking：并行训练多个学习器，再用元学习器整合
- Boosting：串行训练，后一个关注前一个的错误
- → Stacking 通常更准确，Boosting 通常更快

**Q2: 需要多少个基础学习器？**
- 至少 3 个
- 最多 5-7 个（再多了效果不明显）
- **多样性比数量更重要！**

**Q3: 基础学习器怎么选？**
✅ **好的选择** (多样性高)：
- DecisionTree + SVM + KNN + LogisticRegression

❌ **不好的选择** (多样性低)：
- DecisionTree + RandomForest + GradientBoosting （都是树）

**Q4: 元学习器应该选什么？**
- 通常用 LogisticRegression（简单高效）
- 也可以用 RandomForest 或 SVM
- **避免用复杂的元学习器**（容易过拟合）

**Q5: 为什么我的 Stacking 结果不好？**

检查清单：
- [ ] 基础学习器足够多样化吗？
- [ ] 交叉折数足够吗？（推荐 5-10 折）
- [ ] 数据预处理正确吗？（规范化/标准化）
- [ ] 元学习器过度拟合了吗？（尝试加正则化）

---

## 🔧 快速参考

### 导入
```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
```

### API 基础
```python
# 分类
clf = StackingClassifier(
    estimators=[...],           # 基础学习器列表
    final_estimator=...,        # 元学习器
    cv=5,                       # 交叉验证折数
    stack_method='predict_proba' # 或 'predict'
)

# 回归
reg = StackingRegressor(
    estimators=[...],
    final_estimator=...,
    cv=5
)
```

### 性能指标
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
```

---

## 📈 下一步怎么走

### ✅ 现在做什么
1. 运行 `run_demos.py` 看脚本 1
2. 观察生成的图表
3. 修改代码参数，重新运行

### 📚 然后学习什么
- 脚本 2：理解 Blending
- 脚本 3：掌握多层 Stacking
- 脚本 4：对比所有方法

### 🚀 最后怎么应用
- 在自己的项目中使用 Stacking
- 参考 QUICKSTART.md 的代码模板
- 查阅 README.md 的最佳实践

---

## 💡 学习提示

1. **代码优于理论**：先运行脚本，看到结果，再读注释
2. **参数很重要**：修改参数值，观察如何改变结果
3. **多样性至上**：最大化基础学习器的差异性
4. **迭代改进**：不要期望一次完美，多次实验

---

## 📞 需要帮助？

- 📖 读 **README.md** - 理论和最佳实践
- 💻 读 **QUICKSTART.md** - 代码示例
- 📁 读 **FILES.md** - 文件结构
- 📊 读脚本的**注释** - 代码级说明

---

## 🎉 准备好了吗？

```bash
python run_demos.py
```

**选择 1，按 Enter，享受 Stacking！** 🚀

---

**下一步：打开 `QUICKSTART.md` 或运行 `run_demos.py` 开始学习！** ✨
