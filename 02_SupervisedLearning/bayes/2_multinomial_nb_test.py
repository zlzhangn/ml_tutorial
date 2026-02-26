"""
多项式朴素贝叶斯分类器示例
适用于离散特征计数数据，常用于文本分类
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("=" * 50)
print("多项式朴素贝叶斯分类器 - 文本分类示例")
print("=" * 50)

# 1. 加载数据集 - 使用20类新闻组数据集的子集
# 为了快速演示，只选择4个类别
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

print(f"\n加载数据集（类别: {categories}）...")
# 使用subset='train'加载训练数据
newsgroups_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')  # 移除头部、尾部和引用
)

# 加载测试数据
newsgroups_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print(f"\n训练集大小: {len(newsgroups_train.data)}")
print(f"测试集大小: {len(newsgroups_test.data)}")
print(f"类别: {newsgroups_train.target_names}")

# 2. 特征提取 - 将文本转换为词频向量
print("\n特征提取中...")
# CountVectorizer将文本转换为词频矩阵
vectorizer = CountVectorizer(
    max_features=5000,  # 只保留出现频率最高的5000个词
    stop_words='english'  # 移除英文停用词
)

# 对训练集进行拟合和转换
X_train = vectorizer.fit_transform(newsgroups_train.data)
# 对测试集只进行转换
X_test = vectorizer.transform(newsgroups_test.data)

y_train = newsgroups_train.target
y_test = newsgroups_test.target

print(f"特征矩阵形状: {X_train.shape}")
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")

# 3. 创建多项式朴素贝叶斯模型
# alpha是拉普拉斯平滑参数，用于处理训练集中未出现的词
# alpha=1.0表示完全平滑，alpha=0表示不平滑
model = MultinomialNB(alpha=1.0)

# 4. 训练模型
print("\n训练模型中...")
model.fit(X_train, y_train)

print("模型训练完成！")
print(f"每个类别的先验对数概率: {model.class_log_prior_}")

# 5. 预测
y_pred = model.predict(X_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=newsgroups_test.target_names))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 7. 示例预测
print("\n" + "=" * 50)
print("示例预测")
print("=" * 50)

test_docs = [
    "God is love and we should follow the Bible",
    "The graphics card needs better drivers for gaming",
    "Medical research shows new treatment for cancer",
]

# 将文本转换为特征向量
test_vectors = vectorizer.transform(test_docs)
predictions = model.predict(test_vectors)
probabilities = model.predict_proba(test_vectors)

for i, doc in enumerate(test_docs):
    print(f"\n文本: {doc}")
    print(f"预测类别: {newsgroups_test.target_names[predictions[i]]}")
    print(f"各类别概率:")
    for j, prob in enumerate(probabilities[i]):
        print(f"  {newsgroups_test.target_names[j]}: {prob:.4f}")

# 8. 查看每个类别的特征词（权重最高的词）
print("\n" + "=" * 50)
print("每个类别的特征词（Top 10）")
print("=" * 50)

feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(newsgroups_test.target_names):
    # 获取该类别的特征对数概率
    top_indices = np.argsort(model.feature_log_prob_[i])[-10:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"\n{category}: {', '.join(top_features)}")
