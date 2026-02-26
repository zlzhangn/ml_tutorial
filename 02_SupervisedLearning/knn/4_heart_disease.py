import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   # 划分数据集，网格搜索交叉验证
from sklearn.compose import ColumnTransformer   # 列转换，做特征转换
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # 独热编码和标准化

import joblib   # 用来保存对象到文件和加载对象

# 1. 加载数据集
heart_disease_data = pd.read_csv("../data/heart_disease.csv")

# 数据清洗
heart_disease_data.dropna(inplace=True)

heart_disease_data.info()
# print(heart_disease_data.head())

# 2. 数据集划分
# 定义特征
X = heart_disease_data.drop("是否患有心脏病", axis=1)
y = heart_disease_data["是否患有心脏病"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程
# 数值型特征
numerical_features = ["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
# 类别型特征（多元分类）
categorical_features = ["胸痛类型","静息心电图结果","峰值ST段的斜率","地中海贫血"]
# 二元类别特征
binary_features = ["性别","空腹血糖","运动性心绞痛"]

# 创建列转换器
transformer = ColumnTransformer(
    # (名称，操作，特征列表)
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ])

# 执行特征转换
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

print(x_train.shape, x_test.shape)

# 4. 创建模型并训练
# 使用 KNN 进行二分类
# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()

# 定义参数网格
# 尝试k最近邻取1~16，权重使用uniform（所有点权重一样）或distance（近的点权重大），p取1或2
param_grid = {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"], "p": [1, 2]}
knn = GridSearchCV(knn, param_grid=param_grid, cv=10)

# 训练模型
knn.fit(x_train, y_train)

# 5. 测试，模型评估
# print(knn.score(x_test, y_test))

# 6. 保存模型和加载模型
# 保存模型对象到二进制文件
# joblib.dump(knn, "knn_heart_disease.joblib")
#
# # 从文件中加载模型
# knn_load = joblib.load("knn_heart_disease.joblib")
#
# # 7. 预测
# y_pred = knn_load.predict(x_test[100:101])
# print(y_pred, y_test.iloc[100])

# 8. 打印验证结果
pd.set_option('display.max_columns', None)
print(pd.DataFrame(knn.cv_results_))
print(knn.best_estimator_)    # 最佳模型 （输出 最佳模型以及该模型的超参数，如果超参数没输出表示取默认值）
print(knn.best_score_)    # 最佳模型得分

# 用最佳模型做测试和预测
knn = knn.best_estimator_
print(knn.score(x_test, y_test))

