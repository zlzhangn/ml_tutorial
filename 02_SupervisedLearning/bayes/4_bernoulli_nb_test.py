"""
伯努利朴素贝叶斯分类器示例
适用于二元特征（0/1）数据
常用于文本分类中的词汇出现/不出现判断
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("=" * 50)
print("伯努利朴素贝叶斯分类器 - 二元文本分类")
print("=" * 50)

# 1. 加载数据集 - 简化为二分类问题
categories = ['comp.graphics', 'sci.med']  # 只选择2个类别

print(f"\n加载数据集（类别: {categories}）...")
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print(f"\n数据集大小: {len(newsgroups.data)}")
print(f"类别: {newsgroups.target_names}")

# 2. 划分训练集和测试集
X_text, X_test_text, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.3, random_state=42, stratify=newsgroups.target
)

print(f"\n训练集大小: {len(X_text)}")
print(f"测试集大小: {len(X_test_text)}")

# 3. 特征提取 - 转换为二元特征（词汇出现与否）
print("\n特征提取中...")

# 对于伯努利模型，我们使用binary=True，表示只关心词是否出现
vectorizer_binary = CountVectorizer(
    max_features=3000,
    stop_words='english',
    binary=True  # 关键参数：将计数转换为0/1
)

X_train = vectorizer_binary.fit_transform(X_text)
X_test = vectorizer_binary.transform(X_test_text)

print(f"特征矩阵形状: {X_train.shape}")
print(f"特征类型: 二元 (0/1)")

# 同时创建词频向量用于对比
vectorizer_count = CountVectorizer(
    max_features=3000,
    stop_words='english',
    binary=False  # 使用词频
)

X_train_count = vectorizer_count.fit_transform(X_text)
X_test_count = vectorizer_count.transform(X_test_text)

# 4. 创建伯努利朴素贝叶斯模型
print("\n" + "=" * 50)
print("伯努利朴素贝叶斯模型（二元特征）")
print("=" * 50)

# alpha: 平滑参数
# binarize: 如果输入不是二元的，将大于该阈值的值设为1
bernoulli_model = BernoulliNB(alpha=1.0, binarize=None)
bernoulli_model.fit(X_train, y_train)

# 预测
y_pred_bernoulli = bernoulli_model.predict(X_test)

# 评估
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)
print(f"\n准确率: {accuracy_bernoulli:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_bernoulli, target_names=newsgroups.target_names))

# 5. 创建多项式朴素贝叶斯模型用于对比
print("\n" + "=" * 50)
print("多项式朴素贝叶斯模型（词频特征）")
print("=" * 50)

multinomial_model = MultinomialNB(alpha=1.0)
multinomial_model.fit(X_train_count, y_train)

# 预测
y_pred_multinomial = multinomial_model.predict(X_test_count)

# 评估
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)
print(f"\n准确率: {accuracy_multinomial:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_multinomial, target_names=newsgroups.target_names))

# 6. 模型对比
print("\n" + "=" * 50)
print("模型对比")
print("=" * 50)

print(f"""
伯努利朴素贝叶斯（二元特征）准确率: {accuracy_bernoulli:.4f}
多项式朴素贝叶斯（词频特征）准确率: {accuracy_multinomial:.4f}

区别说明:
1. 伯努利朴素贝叶斯:
   - 特征: 词汇是否出现 (0 或 1)
   - 适用场景: 短文本、关注词汇的出现与否
   - 模型假设: 特征服从伯努利分布（二元分布）

2. 多项式朴素贝叶斯:
   - 特征: 词汇出现的次数 (0, 1, 2, ...)
   - 适用场景: 较长文本、关注词频信息
   - 模型假设: 特征服从多项式分布

选择建议:
- 如果文档较短或词频信息不重要，选择伯努利
- 如果文档较长且词频有意义，选择多项式
""")

# 7. 示例预测
print("=" * 50)
print("示例预测")
print("=" * 50)

test_docs = [
    "computer graphics rendering image processing",
    "medical treatment hospital patient disease",
    "visualization algorithm display screen",
]

# 使用伯努利模型预测
test_vectors_binary = vectorizer_binary.transform(test_docs)
predictions_bernoulli = bernoulli_model.predict(test_vectors_binary)
probabilities_bernoulli = bernoulli_model.predict_proba(test_vectors_binary)

# 使用多项式模型预测
test_vectors_count = vectorizer_count.transform(test_docs)
predictions_multinomial = multinomial_model.predict(test_vectors_count)
probabilities_multinomial = multinomial_model.predict_proba(test_vectors_count)

for i, doc in enumerate(test_docs):
    print(f"\n文本: {doc}")
    print(f"\n伯努利模型预测:")
    print(f"  类别: {newsgroups.target_names[predictions_bernoulli[i]]}")
    print(f"  概率: {probabilities_bernoulli[i]}")
    print(f"\n多项式模型预测:")
    print(f"  类别: {newsgroups.target_names[predictions_multinomial[i]]}")
    print(f"  概率: {probabilities_multinomial[i]}")

# 8. 查看特征重要性
print("\n" + "=" * 50)
print("每个类别的重要特征（Top 10）")
print("=" * 50)

feature_names = vectorizer_binary.get_feature_names_out()

print("\n伯努利模型:")
for i, category in enumerate(newsgroups.target_names):
    # 获取该类别的特征对数概率
    top_indices = np.argsort(bernoulli_model.feature_log_prob_[i])[-10:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"{category}: {', '.join(top_features)}")

print("\n多项式模型:")
for i, category in enumerate(newsgroups.target_names):
    top_indices = np.argsort(multinomial_model.feature_log_prob_[i])[-10:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"{category}: {', '.join(top_features)}")
