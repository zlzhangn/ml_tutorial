"""
使用SVM进行心脏病预测
基于真实的心脏病数据集，演示SVM在医疗诊断中的应用
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc, roc_auc_score)
import seaborn as sns

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)

print("=" * 70)
print("使用SVM进行心脏病预测")
print("=" * 70)

# 1. 加载数据
print("\n加载心脏病数据集...")
data_path = 'c:/MyWorkSpace/ML/06 ML/ml_tutorial/data/heart_disease.csv'
df = pd.read_csv(data_path)

print(f"\n数据集形状: {df.shape}")
print("\n前5行数据:")
print(df.head())

print("\n数据集信息:")
print(df.info())

print("\n数据集统计描述:")
print(df.describe())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 检查目标变量分布
print("\n目标变量分布:")
print(df['target'].value_counts())
print(f"有心脏病的比例: {df['target'].mean():.2%}")

# 2. 数据预处理
print("\n" + "=" * 70)
print("数据预处理...")

# 分离特征和标签
X = df.drop('target', axis=1)  # 特征：所有列除了target
y = df['target']  # 标签：target列（0=无心脏病，1=有心脏病）

print(f"\n特征列: {list(X.columns)}")
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 标准化特征
# SVM对特征尺度敏感，标准化可以提高模型性能
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n特征标准化完成！")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y 确保训练集和测试集中各类别的比例与原数据集一致

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集中有心脏病的比例: {y_train.mean():.2%}")
print(f"测试集中有心脏病的比例: {y_test.mean():.2%}")

# 4. 使用网格搜索找到最佳参数
print("\n" + "=" * 70)
print("使用网格搜索寻找最佳参数...")
print("=" * 70)

# 定义参数网格
# C: 正则化参数，控制对误分类的惩罚程度
# gamma: RBF核的系数，控制单个训练样本的影响范围
# kernel: 核函数类型
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# 创建网格搜索对象
# cv=5: 5折交叉验证
# scoring='accuracy': 使用准确率作为评估指标
# n_jobs=-1: 使用所有CPU核心并行计算
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
print("\n开始网格搜索...")
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\n最佳参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 5. 使用最佳参数训练最终模型
print("\n" + "=" * 70)
print("使用最佳参数训练最终模型...")
best_svm = grid_search.best_estimator_
print("模型训练完成！")

# 6. 模型预测
print("\n" + "=" * 70)
print("模型预测...")
y_pred_train = best_svm.predict(X_train)
y_pred_test = best_svm.predict(X_test)

# 获取预测概率（用于ROC曲线）
# decision_function返回样本到决策边界的距离
y_score = best_svm.decision_function(X_test)

# 7. 模型评估
print("\n" + "=" * 70)
print("模型评估结果")
print("=" * 70)

# 训练集性能
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"\n训练集准确率: {train_accuracy:.4f}")

# 测试集性能
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"测试集准确率: {test_accuracy:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['无心脏病', '有心脏病']))

# 计算AUC
roc_auc = roc_auc_score(y_test, y_score)
print(f"\nROC AUC得分: {roc_auc:.4f}")

# 8. 可视化结果
print("\n" + "=" * 70)
print("生成可视化结果...")

# 8.1 混淆矩阵
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['无心脏病', '有心脏病'],
            yticklabels=['无心脏病', '有心脏病'])
plt.title('混淆矩阵 - SVM心脏病预测')
plt.ylabel('真实标签')
plt.xlabel('预测标签')

# 在每个格子中添加百分比
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('c:/MyWorkSpace/ML/06 ML/ml_tutorial/02_SupervisedLearning/svm/heart_confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
print("混淆矩阵已保存！")
plt.show()

# 8.2 ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_score)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC曲线 - SVM心脏病预测')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('c:/MyWorkSpace/ML/06 ML/ml_tutorial/02_SupervisedLearning/svm/heart_roc_curve.png', 
            dpi=300, bbox_inches='tight')
print("ROC曲线已保存！")
plt.show()

# 8.3 特征重要性（对于线性核）
if grid_search.best_params_['kernel'] == 'linear':
    print("\n绘制特征重要性...")
    # 线性核的系数可以表示特征重要性
    feature_importance = np.abs(best_svm.coef_[0])
    feature_names = df.drop('target', axis=1).columns
    
    # 创建特征重要性DataFrame并排序
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importance
    }).sort_values('重要性', ascending=True)
    
    # 绘制水平柱状图
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['特征'], importance_df['重要性'], color='steelblue')
    plt.xlabel('特征重要性（系数绝对值）')
    plt.ylabel('特征')
    plt.title('线性SVM的特征重要性')
    plt.tight_layout()
    plt.savefig('c:/MyWorkSpace/ML/06 ML/ml_tutorial/02_SupervisedLearning/svm/feature_importance.png', 
                dpi=300, bbox_inches='tight')
    print("特征重要性图已保存！")
    plt.show()
    
    print("\n特征重要性排序（从高到低）:")
    print(importance_df.sort_values('重要性', ascending=False))

# 9. 模型分析
print("\n" + "=" * 70)
print("模型分析")
print("=" * 70)

# 支持向量数量
print(f"\n支持向量数量: {best_svm.n_support_}")
print(f"支持向量总数: {best_svm.support_vectors_.shape[0]}")
print(f"支持向量占训练集比例: {best_svm.support_vectors_.shape[0] / X_train.shape[0]:.2%}")

# 混淆矩阵详细分析
tn, fp, fn, tp = cm.ravel()
print(f"\n混淆矩阵详细分析:")
print(f"  真负例 (TN): {tn} - 正确预测为无心脏病")
print(f"  假正例 (FP): {fp} - 错误预测为有心脏病")
print(f"  假负例 (FN): {fn} - 错误预测为无心脏病")
print(f"  真正例 (TP): {tp} - 正确预测为有心脏病")

# 计算其他指标
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n其他评估指标:")
print(f"  精确率 (Precision): {precision:.4f} - 预测为有心脏病中真正有的比例")
print(f"  召回率 (Recall): {recall:.4f} - 实际有心脏病中被正确识别的比例")
print(f"  F1分数: {f1:.4f} - 精确率和召回率的调和平均")
print(f"  特异性 (Specificity): {specificity:.4f} - 实际无心脏病中被正确识别的比例")

print("\n" + "=" * 70)
print("程序执行完成！")
print("=" * 70)

# 10. 保存模型（可选）
print("\n提示: 可以使用joblib保存训练好的模型，便于后续使用")
print("示例代码:")
print("  import joblib")
print("  joblib.dump(best_svm, 'heart_disease_svm.joblib')")
print("  # 加载模型: model = joblib.load('heart_disease_svm.joblib')")
