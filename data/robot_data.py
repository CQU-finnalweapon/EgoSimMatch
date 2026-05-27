"""机器人数据加载接口。"""

from typing import List, Tuple


class RobotDataLoader:
    """加载和管理机器人采集数据。"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_images(self, split: str = "train") -> List[str]:
        """加载机器人图像数据路径。"""
        raise NotImplementedError

    def load_texts(self, split: str = "train") -> List[str]:
        """加载机器人文本描述数据。"""
        raise NotImplementedError

    def get_pair(self, idx: int) -> Tuple[str, str]:
        """返回第 idx 个样本的 (image_path, text_description)。"""
        raise NotImplementedError
