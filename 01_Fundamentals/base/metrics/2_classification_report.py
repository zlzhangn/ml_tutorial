from sklearn.datasets import make_classification   # 自动生成分类数据集
from sklearn.model_selection import train_test_split   # 划分数据集
from sklearn.linear_model import LogisticRegression    # 逻辑回归（分类模型）
from sklearn.metrics import classification_report

# 完整的分类任务示例

# 1. 生成分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,random_state=42)

# 2. 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

# 3. 定义模型：逻辑回归
model = LogisticRegression()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 生成分类报告
report = classification_report(y_test, y_pred)

# print(y_pred)

# 增加 AUC 指标
from sklearn.metrics import roc_auc_score

# 预测属于哪个类的概率值
# [:1]取类别1（正类）的概率
y_pred_proba = model.predict_proba(X_test)[:,1]
# print(y_pred_proba)

auc = roc_auc_score(y_test, y_pred_proba)

# print(report)
print(auc)