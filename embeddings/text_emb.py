"""文本 Embedding 提取模块。"""

import numpy as np


class TextEmbedder:
    """文本特征提取器基类。"""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name

    def encode(self, text: str) -> np.ndarray:
        """将单段文本编码为 Embedding 向量。"""
        raise NotImplementedError

    def encode_batch(self, texts: list) -> np.ndarray:
        """批量编码文本。"""
        raise NotImplementedError
