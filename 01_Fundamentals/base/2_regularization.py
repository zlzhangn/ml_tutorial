import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型，Lasso回归，Ridge回归
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

# 配置matplotlib中全局绘图参数
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 楷体字
plt.rcParams['axes.unicode_minus'] = False
"""
本示例，演示
正则化：通过增加惩罚项，将参数往零的方向压缩，从而通过略微增加偏差，换取方差的下降，提升模型泛化能力
1. 不适用正则化
2. 使用L1正则化
3. 使用L2正则化
"""
# 由一个向量x，生成 degree列（degree个特征）的矩阵，(x, x^2, ... x^degree)
def polynomial(X, degree):
    return np.hstack([X**i for i in range(1, degree + 1)])

'''
机器学习步骤：
1. 读取数据（生成数据）
2. 划分训练集和测试集
3. 定义模型
4. 训练模型
5. 预测结果，计算误差（测试误差）
'''

# 1. 读取数据（生成数据）
# 生成随机数据，扩展成二维矩阵表示，形状(300,1)
X = np.linspace(-3, 3, 300).reshape(-1, 1)
# print(X.shape)

# 基于sinX叠加随机噪声
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 画出散点图
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, c='y')
ax[0,1].scatter(X, y, c='y')
ax[0,2].scatter(X, y, c='y')

# plt.show()

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 过拟合：20阶，转换成20个特征的线性拟合，复杂度过高
x_train = polynomial(x_train, 20)
x_test = polynomial(x_test, 20)

# 3. 定义模型，线性回归模型
model = LinearRegression()

# 4. 训练模型
model.fit(x_train, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred1 = model.predict(x_test)
# 调用均方误差函数，传入y的真实值和预测值
test_loss1 = mean_squared_error(y_test, y_pred1)

# 画出拟合曲线，并标出误差
ax[0,0].plot(X, model.predict( polynomial(X, 20) ), 'r')
ax[0,0].text(-3, 1, f"测试集均方误差：{test_loss1:.4f}")
# 画出所有系数的柱状图
ax[1,0].bar(np.arange(20), model.coef_.reshape(-1))
print("未使用正则化时的系数：", model.coef_ )


# L1正则化 —— Lasso回归
lasso = Lasso(alpha=0.01)

# 4. 训练模型
lasso.fit(x_train, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred2 = lasso.predict(x_test)
# 调用均方误差函数，传入y的真实值和预测值
test_loss2 = mean_squared_error(y_test, y_pred2)

# 画出拟合曲线，并标出误差
ax[0,1].plot(X, lasso.predict( polynomial(X, 20) ), 'r')
ax[0,1].text(-3, 1, f"测试集均方误差：{test_loss2:.4f}")
ax[0,1].text(-3, 1.3, "Lasso回归")
# 画出所有系数的柱状图
ax[1,1].bar(np.arange(20), lasso.coef_.reshape(-1))
print("Lasso回归时的系数", lasso.coef_ )


# L2正则化 —— Ridge回归
ridge = Ridge(alpha=1)

# 4. 训练模型
ridge.fit(x_train, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred3 = ridge.predict(x_test)
# 调用均方误差函数，传入y的真实值和预测值
test_loss3 = mean_squared_error(y_test, y_pred3)

# 画出拟合曲线，并标出误差
ax[0,2].plot(X, ridge.predict( polynomial(X, 20) ), 'r')
ax[0,2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[0,2].text(-3, 1.3, "Ridge回归")
# 画出所有系数的柱状图
ax[1,2].bar(np.arange(20), ridge.coef_.reshape(-1))

print("Ridge回归时的系数", ridge.coef_)

plt.show()