"""
多项式朴素贝叶斯分类器示例（简化版）
使用自定义文本数据，无需下载外部数据集
适用于离散特征计数数据，常用于文本分类
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("=" * 50)
print("多项式朴素贝叶斯分类器 - 文本分类示例")
print("=" * 50)

# 1. 创建自定义数据集 - 4个类别的文本数据
# 类别：科技、体育、金融、娱乐

# 科技类文本
tech_texts = [
    "artificial intelligence machine learning deep neural networks",
    "computer programming software development coding algorithms",
    "technology innovation smartphone mobile applications",
    "data science big data analytics database",
    "cloud computing servers infrastructure network security",
    "python java programming languages frameworks libraries",
    "algorithm optimization performance computational efficiency",
    "hardware processor memory storage devices",
    "internet web development html css javascript",
    "cybersecurity encryption protocols firewall protection",
    "software engineering testing debugging deployment",
    "machine learning models training prediction accuracy",
    "neural networks deep learning tensorflow pytorch",
    "data mining pattern recognition classification",
    "cloud services aws azure deployment scalability",
]

# 体育类文本
sports_texts = [
    "football soccer match goal team player championship",
    "basketball game score points shooting dribbling",
    "tennis tournament match set serve volley",
    "baseball pitch home run batting fielding",
    "olympics athletes gold medal competition race",
    "training exercise fitness workout strength endurance",
    "coach strategy tactics formation defense attack",
    "championship final score victory win defeat",
    "athlete performance speed agility competition",
    "soccer world cup football match stadium fans",
    "basketball championship playoff finals mvp award",
    "tennis player serve ace forehand backhand",
    "olympic games athletes medals podium ceremony",
    "training regimen practice drills conditioning",
    "sports competition tournament league championship",
]

# 金融类文本
finance_texts = [
    "stock market investment trading portfolio dividend",
    "banking finance credit loan mortgage interest",
    "economy inflation recession gdp growth rate",
    "cryptocurrency bitcoin blockchain digital currency",
    "insurance policy premium coverage claim risk",
    "mutual funds hedge funds asset management",
    "financial planning retirement savings pension",
    "stock exchange dow nasdaq trading volume",
    "investment strategy diversification risk management",
    "banking services account deposit withdrawal transfer",
    "economic indicators employment inflation consumer spending",
    "cryptocurrency trading exchange wallet mining",
    "insurance company liability health auto coverage",
    "investment portfolio stocks bonds mutual funds",
    "financial markets bull bear market volatility",
]

# 娱乐类文本
entertainment_texts = [
    "movie film cinema director actor actress screenplay",
    "music concert song album artist performance",
    "television series episode drama comedy streaming",
    "celebrity fashion award show red carpet",
    "theater play drama musical performance stage",
    "game video entertainment console player level",
    "movie premiere box office blockbuster review",
    "music festival concert band singer performance",
    "tv show series season finale streaming platform",
    "celebrity gossip news fashion style trend",
    "broadway theater musical play performance",
    "gaming console playstation xbox nintendo switch",
    "cinema hollywood movie stars director producer",
    "rock pop music artist album release tour",
    "streaming netflix hulu amazon prime series",
]

# 合并所有文本和标签
all_texts = tech_texts + sports_texts + finance_texts + entertainment_texts
all_labels = (
    ['科技'] * len(tech_texts) +
    ['体育'] * len(sports_texts) +
    ['金融'] * len(finance_texts) +
    ['娱乐'] * len(entertainment_texts)
)

print(f"\n数据集大小: {len(all_texts)}")
print(f"类别: {list(set(all_labels))}")
print(f"每个类别样本数: {len(tech_texts)}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    all_texts, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

print(f"\n训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 3. 特征提取 - 将文本转换为词频向量
print("\n特征提取中...")
# CountVectorizer将文本转换为词频矩阵
vectorizer = CountVectorizer(
    max_features=200,  # 保留出现频率最高的200个词
    stop_words='english'  # 移除英文停用词
)

# 对训练集进行拟合和转换
X_train_vec = vectorizer.fit_transform(X_train)
# 对测试集只进行转换
X_test_vec = vectorizer.transform(X_test)

print(f"特征矩阵形状: {X_train_vec.shape}")
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")

# 4. 创建多项式朴素贝叶斯模型
# alpha是拉普拉斯平滑参数，用于处理训练集中未出现的词
# alpha=1.0表示完全平滑，alpha=0表示不平滑
model = MultinomialNB(alpha=1.0)

# 5. 训练模型
print("\n训练模型中...")
model.fit(X_train_vec, y_train)

print("模型训练完成！")
print(f"每个类别的先验概率:")
for i, label in enumerate(model.classes_):
    print(f"  {label}: {np.exp(model.class_log_prior_[i]):.4f}")

# 6. 预测
y_pred = model.predict(X_test_vec)

# 7. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred, labels=['科技', '体育', '金融', '娱乐'])
print("           科技  体育  金融  娱乐")
for i, label in enumerate(['科技', '体育', '金融', '娱乐']):
    print(f"{label:>4}: {cm[i]}")

# 8. 示例预测
print("\n" + "=" * 50)
print("示例预测")
print("=" * 50)

test_docs = [
    "python programming machine learning artificial intelligence",
    "football match soccer goal team championship",
    "stock market investment trading portfolio",
    "movie cinema film director actor performance",
    "basketball game player score points",
    "bitcoin cryptocurrency blockchain digital currency",
]

# 将文本转换为特征向量
test_vectors = vectorizer.transform(test_docs)
predictions = model.predict(test_vectors)
probabilities = model.predict_proba(test_vectors)

for i, doc in enumerate(test_docs):
    print(f"\n文本: {doc}")
    print(f"预测类别: {predictions[i]}")
    print(f"各类别概率:")
    for j, label in enumerate(model.classes_):
        print(f"  {label}: {probabilities[i][j]:.4f}")

# 9. 查看每个类别的特征词（权重最高的词）
print("\n" + "=" * 50)
print("每个类别的特征词（Top 10）")
print("=" * 50)

feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(model.classes_):
    # 获取该类别的特征对数概率
    top_indices = np.argsort(model.feature_log_prob_[i])[-10:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"\n{category}: {', '.join(top_features)}")

# 10. 贝叶斯原理解释
print("\n" + "=" * 50)
print("多项式朴素贝叶斯原理")
print("=" * 50)

print("""
多项式朴素贝叶斯基于贝叶斯定理:
P(类别|文档) = P(文档|类别) × P(类别) / P(文档)

对于文本分类:
- 文档表示为词频向量: (w₁, w₂, ..., wₙ)
- P(类别|文档) ∝ P(类别) × ∏ P(词|类别)^词频

特点:
1. 假设词的出现次数服从多项式分布
2. 适合文本分类任务（词袋模型）
3. 词频信息被考虑在内（与伯努利模型的区别）

拉普拉斯平滑（alpha参数）:
- 防止零概率问题
- 即使某个词在训练集的某个类别中未出现，也能给它一个小的概率
- alpha=1.0是最常用的设置
""")
