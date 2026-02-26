import numpy as np

# 损失函数
def J(beta):
    return np.sum((X @ beta - y)**2, axis=0).reshape(-1, 1) / n

# 计算梯度
def gradient(beta):
    return X.T @ (X @ beta - y) / n * 2

# 1. 定义数据
# 自变量
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])
# 因变量
y = np.array( [55, 65, 70, 75, 85, 50, 60, 72, 80, 58] ).reshape(-1, 1)

n = X.shape[0]

# 2. 数据处理，X 增加一列 1
# hstack垂直方向添加列
X = np.hstack([np.ones((n, 1)), X])

# print(X)

# 3. 初始化参数及超参数
alpha = 0.01
iterations = 10000

beta = np.array([[1], [1]])

beta0 = []
beta1 = []

# 循环迭代
while (np.abs(grad := gradient(beta)) > 1e-10).any() and (iterations := iterations - 1) >= 0:
    # 4. 计算梯度（略）
    # grad = gradient(beta)

    # 5. 更新参数
    beta = beta - alpha * grad
    beta0.append(beta[0, 0])
    beta1.append(beta[1, 0])

    # 每迭代 10 轮打印一次当前的参数值和损失函数值
    if iterations % 10 == 0:
        print(f"beta = {beta.reshape(-1)}\tJ = {J(beta).reshape(-1)}")

# 画出 β0、β1 变化曲线
import matplotlib.pyplot as plt

plt.plot(beta0, beta1)
plt.show()