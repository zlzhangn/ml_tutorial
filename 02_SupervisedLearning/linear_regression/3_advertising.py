import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor   # 解析法和梯度下降法求解线性回归模型
from sklearn.model_selection import train_test_split    # 切分数据集
from sklearn.metrics import mean_squared_error    # 损失函数（性能指标）
from sklearn.preprocessing import StandardScaler

# 1. 加载数据集
data = pd.read_csv('../data/advertising.csv')

# 2. 数据预处理
data.drop(data.columns[0], axis=1, inplace=True)
data.dropna(inplace=True)

data.info()
print(data.head())

# 3. 划分数据集
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 4. 特征工程：标准化
transformer = StandardScaler()
x_train = transformer.fit_transform(X_train)
x_test = transformer.transform(X_test)

# 5. 创建模型并训练
# 5.1 正规方程法
regressor_normal = LinearRegression()
regressor_normal.fit(x_train, y_train)

print("正规方程法模型系数: ", regressor_normal.coef_)
print("正规方程法模型偏置: ", regressor_normal.intercept_)

# 5.2 梯度下降法
regressor_SGD = SGDRegressor()
regressor_SGD.fit(x_train, y_train)

print("梯度下降法模型系数: ", regressor_SGD.coef_)
print("梯度下降法模型偏置: ", regressor_SGD.intercept_)

# 6. 测试（预测）
y_pred1 = regressor_normal.predict(x_test)
y_pred2 = regressor_SGD.predict(x_test)

print("正规方程法均方误差: ", mean_squared_error(y_test, y_pred1))
print("梯度下降法均方误差: ", mean_squared_error(y_test, y_pred2))
