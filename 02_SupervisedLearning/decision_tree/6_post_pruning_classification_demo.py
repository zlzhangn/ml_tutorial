"""决策树分类的后剪枝（代价复杂度剪枝）示例。

说明：
- 使用 sklearn 自带的鸢尾花数据集（无需外部文件）。
- 通过验证集选择最优 ccp_alpha。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """加载鸢尾花数据集。"""
    data = load_iris()
    X = data.data
    y = data.target
    return X, y


def select_alpha_by_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> float:
    """基于验证集选择最优 alpha（后剪枝参数）。"""
    base_tree = DecisionTreeClassifier(random_state=random_state)
    base_tree.fit(X_train, y_train)

    # 代价复杂度剪枝路径
    path = base_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # 去掉最后一个（通常会剪成单节点树）
    ccp_alphas = ccp_alphas[:-1]

    best_alpha = 0.0
    best_acc = -1.0

    for alpha in ccp_alphas:
        pruned_tree = DecisionTreeClassifier(random_state=random_state, ccp_alpha=alpha)
        pruned_tree.fit(X_train, y_train)
        y_val_pred = pruned_tree.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    return best_alpha


def main() -> None:
    # 1. 加载数据
    X, y = load_data()

    # 2. 划分训练/验证/测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    # 3. 未剪枝模型
    print("=== 未剪枝模型 ===")
    unpruned_tree = DecisionTreeClassifier(random_state=42)
    unpruned_tree.fit(X_train_full, y_train_full)
    y_test_pred = unpruned_tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    # 4. 选择最优 alpha 并训练后剪枝模型
    best_alpha = select_alpha_by_validation(X_train, y_train, X_val, y_val)
    print(f"\n选择的最优 alpha: {best_alpha:.6f}")

    print("=== 后剪枝模型 ===")
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    pruned_tree.fit(X_train_full, y_train_full)
    y_test_pred = pruned_tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    # 5. 输出分类报告
    print("\n分类报告：")
    print(classification_report(y_test, y_test_pred))


if __name__ == "__main__":
    main()
