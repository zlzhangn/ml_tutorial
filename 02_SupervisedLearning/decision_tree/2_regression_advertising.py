"""
决策树回归器示例 - 广告销售预测
使用决策树进行回归任务，预测连续值
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("决策树回归器 - 广告销售预测")
print("=" * 60)

# 1. 加载数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../data/advertising.csv")

try:
    data = pd.read_csv(data_path)
    print(f"\n成功加载数据集，共 {len(data)} 条记录")
except FileNotFoundError:
    print("\n错误：找不到 advertising.csv 文件")
    print("请确保数据文件存在于 data/ 目录下")
    exit(1)

# 数据预览
print("\n数据集前5行:")
print(data.head())

print("\n数据集统计信息:")
print(data.describe())

# 2. 数据预处理
# 分离特征和目标变量
X = data[['TV', 'Radio', 'Newspaper']]  # 特征：各媒体的广告费用
y = data['Sales']  # 目标：销售额

print(f"\n特征列: {X.columns.tolist()}")
print(f"目标变量: Sales")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 创建决策树回归器
"""
决策树回归器与分类器的区别:
- 分类器预测类别，回归器预测连续值
- 叶子节点存储的是数值（通常是该节点样本的平均值）
- 使用均方误差（MSE）作为分裂标准
"""

# 创建不同深度的模型用于对比
depths = [2, 5, 10, None]  # None表示不限制深度
models = {}
predictions = {}

print("\n" + "=" * 60)
print("训练不同深度的决策树模型")
print("=" * 60)

for depth in depths:
    print(f"\n训练深度为 {depth} 的模型...")
    
    model = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    models[depth] = model
    predictions[depth] = y_pred
    
    # 评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  深度: {depth if depth else '无限制'}")
    print(f"  树的实际深度: {model.get_depth()}")
    print(f"  叶子节点数: {model.get_n_leaves()}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# 5. 选择最佳模型（这里选择深度为5的模型）
best_depth = 5
best_model = models[best_depth]
y_pred_best = predictions[best_depth]

print("\n" + "=" * 60)
print(f"最佳模型（深度={best_depth}）详细评估")
print("=" * 60)

mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)

print(f"\n均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R² 分数: {r2:.4f}")

# 6. 特征重要性
print("\n" + "=" * 60)
print("特征重要性分析")
print("=" * 60)

feature_importance = best_model.feature_importances_
print("\n各特征重要性:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance:.4f}")

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importance)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('决策树回归 - 特征重要性')
plt.tight_layout()
plt.savefig('regression_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n特征重要性图已保存到 regression_feature_importance.png")

# 7. 预测值与真实值对比
print("\n" + "=" * 60)
print("预测值与真实值对比")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'决策树回归预测结果 (R²={r2:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_prediction.png', dpi=300, bbox_inches='tight')
print("预测对比图已保存到 regression_prediction.png")

# 8. 残差分析
residuals = y_test - y_pred_best

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch10_regression_residuals.png', dpi=300, bbox_inches='tight')
print("残差分析图已保存到 ch10_regression_residuals.png")

# 9. 可视化决策树
print("\n" + "=" * 60)
print("决策树可视化")
print("=" * 60)

plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=X.columns.tolist(),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f'决策树回归结构（深度={best_depth}）', fontsize=16)
plt.tight_layout()
plt.savefig('ch10_regression_tree_structure.png', dpi=300, bbox_inches='tight')
print("决策树结构图已保存到 ch10_regression_tree_structure.png")

# 10. 预测示例
print("\n" + "=" * 60)
print("预测示例")
print("=" * 60)

# 创建测试样本
test_samples = pd.DataFrame({
    'TV': [230.1, 44.5, 17.2],
    'Radio': [37.8, 39.3, 45.9],
    'Newspaper': [69.2, 45.1, 69.3]
})

predictions_samples = best_model.predict(test_samples)

print("\n测试样本预测结果:")
for i in range(len(test_samples)):
    print(f"\n样本 {i+1}:")
    print(f"  TV广告费用: {test_samples.iloc[i]['TV']:.1f}")
    print(f"  Radio广告费用: {test_samples.iloc[i]['Radio']:.1f}")
    print(f"  Newspaper广告费用: {test_samples.iloc[i]['Newspaper']:.1f}")
    print(f"  预测销售额: {predictions_samples[i]:.2f}")

# 11. 不同深度模型的性能对比
print("\n" + "=" * 60)
print("不同深度模型性能对比")
print("=" * 60)

plt.figure(figsize=(12, 5))

# R²对比
plt.subplot(1, 2, 1)
r2_scores = []
depth_labels = []
for depth in depths:
    y_pred = predictions[depth]
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    depth_labels.append(str(depth) if depth else '无限制')

plt.bar(depth_labels, r2_scores)
plt.xlabel('树的深度')
plt.ylabel('R² 分数')
plt.title('不同深度的R²分数')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)

# RMSE对比
plt.subplot(1, 2, 2)
rmse_scores = []
for depth in depths:
    y_pred = predictions[depth]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

plt.bar(depth_labels, rmse_scores)
plt.xlabel('树的深度')
plt.ylabel('RMSE')
plt.title('不同深度的RMSE')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch10_depth_comparison.png', dpi=300, bbox_inches='tight')
print("\n深度对比图已保存到 ch10_depth_comparison.png")

# 12. 决策树回归原理说明
print("\n" + "=" * 60)
print("决策树回归原理说明")
print("=" * 60)

print("""
决策树回归的工作原理:

1. 分裂标准:
   - 使用均方误差（MSE）作为分裂标准
   - MSE = (1/n) × Σ(y_i - ȳ)²
   - 选择能最小化MSE的特征和分裂点

2. 叶子节点预测:
   - 每个叶子节点的预测值是该节点所有训练样本的平均值
   - 对于新样本，沿着决策路径到达叶子节点，返回该节点的平均值

3. 与线性回归的对比:
   决策树回归:
   ✓ 可以捕捉非线性关系
   ✓ 不需要特征缩放
   ✓ 可以自动处理特征交互
   ✗ 容易过拟合
   ✗ 预测值是分段常数（阶梯状）
   
   线性回归:
   ✓ 模型简单，易于解释
   ✓ 对新数据泛化能力较好
   ✗ 只能建模线性关系
   ✗ 对异常值敏感

4. 过拟合问题:
   - 深度过大的树会记住训练数据的噪声
   - 解决方法：
     * 限制树的深度
     * 设置最小样本分裂数
     * 设置最小叶子样本数
     * 使用集成方法（随机森林、GBDT）

5. 适用场景:
   ✓ 数据存在非线性关系
   ✓ 需要可解释的模型
   ✓ 特征之间有交互作用
   ✓ 数据存在缺失值
""")

print(f"\n当前最佳模型总结:")
print(f"树的深度: {best_model.get_depth()}")
print(f"叶子节点数: {best_model.get_n_leaves()}")
print(f"最重要的特征: {X.columns[np.argmax(feature_importance)]}")
print(f"R²分数: {r2:.4f} (解释了{r2*100:.2f}%的方差)")

plt.show()
