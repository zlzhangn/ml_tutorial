# 支持向量机 (Support Vector Machine, SVM)

## 简介
支持向量机是一种强大的监督学习算法，用于分类和回归问题。

## 核心概念
- **支持向量**: 距离超平面最近的数据点
- **超平面**: 在n维空间中用于分类的(n-1)维子空间
- **核函数**: 将数据映射到高维空间，实现非线性分类

## 示例文件
1. `1_basic_classification.py` - 基础分类示例
2. `2_kernel_comparison.py` - 不同核函数的对比
3. `3_heart_disease_svm.py` - 使用心脏病数据集的实际应用

## 常用核函数
- **linear**: 线性核，适用于线性可分数据
- **rbf**: 径向基函数（高斯核），最常用的核函数
- **poly**: 多项式核
- **sigmoid**: sigmoid核
