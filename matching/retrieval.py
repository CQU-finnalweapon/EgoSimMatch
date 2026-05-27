"""检索与匹配模块。"""

import numpy as np


class Retriever:
    """基于 Embedding 相似度的检索器。"""

    def __init__(self, index: np.ndarray = None):
        self.index = index  # (N, D) 的 Embedding 库

    def build_index(self, embeddings: np.ndarray):
        """构建检索索引库。"""
        self.index = embeddings

    def search(self, query: np.ndarray, top_k: int = 5) -> tuple:
        """检索最相似的 top_k 个结果。"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        sims = np.dot(self.index, query)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return top_indices, sims[top_indices]
