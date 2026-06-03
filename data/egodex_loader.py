"""EgoDex DataLoader — webdataset (tar + msgpack) 格式。

数据路径：
  .../EgoDex_lerobot_metis_gripper_v1_retask_full_window_with_hand/webdataset/{split}/
    eval/webdataset/dataset-XXXXXX.tar
    train/part-XXX/webdataset/dataset-XXXXXX.tar

每个 tar 内文件命名规则：
  ep{episode_index:06d}_fr{frame_index:06d}.cam_high.jpg
  ep{episode_index:06d}_fr{frame_index:06d}.cam_left_wrist.jpg
  ep{episode_index:06d}_fr{frame_index:06d}.cam_right_wrist.jpg
  ep{episode_index:06d}_fr{frame_index:06d}.meta.msgpack
"""

import io
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import msgpack
from PIL import Image

EGODEX_ROOT = (
    "/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/code/"
    "mozbrain_main_0402_ego_grouploss/outputs/"
    "EgoDex_lerobot_metis_gripper_v1_retask_full_window_with_hand/webdataset"
)

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


@dataclass
class EgoDexSample:
    episode_index: int
    frame_index: int
    task: str
    images: Dict[str, Image.Image]  # cam_name -> PIL Image
    source: str = "egodex"


class EgoDexDataLoader:
    """从 EgoDex webdataset tar 文件中加载样本。

    Args:
        split: "train" 或 "eval"
        cameras: 要加载的摄像头列表，默认全部
        max_tars: 最多加载多少个 tar 文件，None 表示全部（用于快速测试）
    """

    def __init__(
        self,
        root: str = EGODEX_ROOT,
        split: str = "eval",
        cameras: Optional[List[str]] = None,
        max_tars: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.cameras = cameras or CAMERAS
        self.max_tars = max_tars
        self._tar_paths = self._collect_tars()

    def _collect_tars(self) -> List[Path]:
        split_dir = self.root / self.split
        tars = []
        if self.split == "eval":
            wds_dir = split_dir / "webdataset"
            if wds_dir.exists():
                tars = sorted(wds_dir.glob("dataset-*.tar"))
        else:
            # train 有多个 part-XXX 子目录
            for part in sorted(split_dir.glob("part-*")):
                tars.extend(sorted((part / "webdataset").glob("dataset-*.tar")))
        if self.max_tars is not None:
            tars = tars[: self.max_tars]
        return tars

    def __len__(self) -> int:
        return len(self._tar_paths)

    def iter_samples(self) -> Iterator[EgoDexSample]:
        """逐样本迭代，每个 (episode, frame) 为一个样本。"""
        for tar_path in self._tar_paths:
            yield from self._iter_tar(tar_path)

    def _iter_tar(self, tar_path: Path) -> Iterator[EgoDexSample]:
        with tarfile.open(tar_path, "r") as tar:
            members_by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
            for m in tar.getmembers():
                # key = "ep{ep}_fr{fr}"
                key, _, ext = m.name.partition(".")
                members_by_key.setdefault(key, {})[ext] = m

            for key, files in members_by_key.items():
                if "meta.msgpack" not in files:
                    continue
                meta_raw = tar.extractfile(files["meta.msgpack"]).read()
                meta = msgpack.unpackb(meta_raw, raw=False)

                images = {}
                for cam in self.cameras:
                    ext_key = f"{cam}.jpg"
                    if ext_key in files:
                        raw = tar.extractfile(files[ext_key]).read()
                        images[cam] = Image.open(io.BytesIO(raw)).convert("RGB")

                yield EgoDexSample(
                    episode_index=meta["episode_index"],
                    frame_index=meta["frame_index"],
                    task=meta["task"],
                    images=images,
                )

    def load_by_task(self, task: str, max_samples: int = 10) -> List[EgoDexSample]:
        """加载指定 task 的样本（精确匹配，不区分大小写）。"""
        task_lower = task.lower().strip()
        results = []
        for sample in self.iter_samples():
            if sample.task.lower().strip() == task_lower:
                results.append(sample)
                if len(results) >= max_samples:
                    break
        return results

    def get_task_index(self) -> Dict[str, List[str]]:
        """扫描 eval meta/tasks.jsonl 建立 task -> [episode_key] 索引（快速）。"""
        import json

        tasks_file = self.root / self.split / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            return {}
        index: Dict[str, List[str]] = {}
        with open(tasks_file) as f:
            for line in f:
                d = json.loads(line)
                index[d["task"]] = index.get(d["task"], [])
        return index
