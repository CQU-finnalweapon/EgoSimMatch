"""相似度度量模块。"""

import numpy as np


class SimilarityCalculator:
    """支持多种相似度度量方式。"""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / norm if norm != 0 else 0.0

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """计算欧氏距离。"""
        return np.linalg.norm(a - b)

    @staticmethod
    def cross_modal_similarity(
        ego_embedding: np.ndarray,
        robot_embedding: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """跨模态相似度计算。"""
        if metric == "cosine":
            return SimilarityCalculator.cosine_similarity(ego_embedding, robot_embedding)
        elif metric == "euclidean":
            return SimilarityCalculator.euclidean_distance(ego_embedding, robot_embedding)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
