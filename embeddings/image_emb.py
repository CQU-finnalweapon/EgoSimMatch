"""图像 Embedding 提取模块。"""

import numpy as np


class ImageEmbedder:
    """图像特征提取器基类。"""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name

    def encode(self, image_path: str) -> np.ndarray:
        """将单张图像编码为 Embedding 向量。"""
        raise NotImplementedError

    def encode_batch(self, image_paths: list) -> np.ndarray:
        """批量编码图像。"""
        raise NotImplementedError
