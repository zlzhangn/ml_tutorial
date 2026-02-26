import numpy as np

# 与门
def AND0(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    res = w1 * x1 + w2 * x2
    # 判断加权和与阈值θ的大小关系，决定输出
    if res <= theta:
        return 0
    elif res > theta:
        return 1

# 改进之后的与门实现
def AND(x1, x2):
    X = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    res = np.sum(w * X) + b
    if res <= 0:
        return 0
    else:
        return 1

# 测试与门
print("测试与门：")
print(AND(0, 0))   # 0
print(AND(0, 1))   # 0
print(AND(1, 0))   # 0
print(AND(1, 1))   # 1

# 与非门
def NAND(x1, x2):
    X = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    res = np.sum(w * X) + b
    if res <= 0:
        return 0
    else:
        return 1

# 测试与非门
print("测试与非门：")
print(NAND(0, 0))   # 1
print(NAND(0, 1))   # 1
print(NAND(1, 0))   # 1
print(NAND(1, 1))   # 0

# 或门
def OR(x1, x2):
    X = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0

    res = np.sum(w * X) + b
    if res <= 0:
        return 0
    else:
        return 1

# 测试或门
print("测试或门：")
print(OR(0, 0))   # 0
print(OR(0, 1))   # 1
print(OR(1, 0))   # 1
print(OR(1, 1))   # 1

# 异或门
# 需要用多层感知机实现
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# 测试异或门
print("测试异或门：")
print(XOR(0, 0))   # 0
print(XOR(0, 1))   # 1
print(XOR(1, 0))   # 1
print(XOR(1, 1))   # 0