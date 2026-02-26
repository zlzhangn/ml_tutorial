import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.preprocessing import MinMaxScaler    # 归一化
from sklearn.linear_model import LogisticRegression     # 逻辑回归
import os

# 1. 加载数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/train.csv")
digit = pd.read_csv(data_path)

# 图片测试
# plt.imshow(digit.iloc[10, 1:].values.reshape(28, 28), cmap='gray')
# plt.show

# 2. 划分训练集和测试集
X = digit.drop(['label'], axis=1)
y = digit['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 特征转换：归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. 创建模型并训练
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 5. 模型评估，计算准确率
print(model.score(x_test, y_test))

# 6. 预测
plt.imshow(digit.iloc[1000, 1:].values.reshape(28, 28), cmap='gray')
plt.show()

print("预测数字为：", model.predict(digit.iloc[1000, 1:].values.reshape(1, -1)))