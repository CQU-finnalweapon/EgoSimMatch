"""EgoTel DataLoader — lerobot parquet + MP4 格式，按 task_index 切分 segment。

数据路径：
  .../EgoTel_aligned/data/raw_retask_split/{split}/task_{id}/{dataset_dir}/
    meta/tasks.jsonl          task_index -> task 文本
    data/chunk-000/episode_XXXXXX.parquet   含 task_index 字段（逐帧）
    videos/chunk-000/{cam}/episode_XXXXXX.mp4

切分逻辑：
  每个 episode 按 task_index 字段切分成多个 segment（通常 120 帧），
  每个 segment 是一个独立的动作片段，取中间帧作为代表帧。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import pyarrow.parquet as pq
from PIL import Image

EGOTEL_ROOT = (
    "/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/code/"
    "mozbrain_main_0402_ego_grouploss/outputs/converted_data/"
    "EgoTel_aligned/data/raw_retask_split"
)

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


@dataclass
class EgoTelSegment:
    task_dir: str
    dataset_dir: str
    episode_index: int
    task_index: int
    task: str
    frame_start: int
    frame_end: int
    images: Dict[str, Image.Image]
    source: str = "egotel"


@dataclass
class EgoTelDataset:
    dataset_dir: Path
    task_dir_name: str
    dataset_dir_name: str
    tasks: Dict[int, str]
    video_dirs: Dict[str, Path]
    parquet_dir: Path


class EgoTelDataLoader:
    """从 EgoTel lerobot 数据中加载 task segment 样本。

    Args:
        split: "train" 或 "eval"
        cameras: 要加载的摄像头列表
        max_datasets: 最多加载多少个 dataset 目录（用于快速测试）
        min_segment_frames: 最短 segment 帧数
    """

    def __init__(
        self,
        root: str = EGOTEL_ROOT,
        split: str = "eval",
        cameras: Optional[List[str]] = None,
        max_datasets: Optional[int] = None,
        min_segment_frames: int = 10,
    ):
        self.root = Path(root)
        self.split = split
        self.cameras = cameras or CAMERAS
        self.max_datasets = max_datasets
        self.min_segment_frames = min_segment_frames
        self._datasets = self._collect_datasets()

    def _collect_datasets(self) -> List[EgoTelDataset]:
        split_dir = self.root / self.split
        datasets = []
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for dataset_dir in sorted(task_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                ds = self._load_dataset_meta(task_dir.name, dataset_dir)
                if ds is not None:
                    datasets.append(ds)
                    if self.max_datasets and len(datasets) >= self.max_datasets:
                        return datasets
        return datasets

    def _load_dataset_meta(self, task_dir_name: str, dataset_dir: Path) -> Optional[EgoTelDataset]:
        tasks_file = dataset_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            return None

        tasks = {}
        with open(tasks_file) as f:
            for line in f:
                d = json.loads(line)
                tasks[d["task_index"]] = d["task"]

        video_dirs = {}
        for cam in self.cameras:
            cam_dir = dataset_dir / "videos" / "chunk-000" / cam
            if cam_dir.exists():
                video_dirs[cam] = cam_dir

        parquet_dir = dataset_dir / "data" / "chunk-000"

        return EgoTelDataset(
            dataset_dir=dataset_dir,
            task_dir_name=task_dir_name,
            dataset_dir_name=dataset_dir.name,
            tasks=tasks,
            video_dirs=video_dirs,
            parquet_dir=parquet_dir,
        )

    def __len__(self) -> int:
        return len(self._datasets)

    def iter_segments(self) -> Iterator[EgoTelSegment]:
        for ds in self._datasets:
            yield from self._iter_dataset_segments(ds)

    def _iter_dataset_segments(self, ds: EgoTelDataset) -> Iterator[EgoTelSegment]:
        for parquet_path in sorted(ds.parquet_dir.glob("episode_*.parquet")):
            ep_idx = int(parquet_path.stem.split("_")[1])
            yield from self._extract_segments(ds, ep_idx, parquet_path)

    def _extract_segments(
        self, ds: EgoTelDataset, episode_index: int, parquet_path: Path
    ) -> Iterator[EgoTelSegment]:
        try:
            table = pq.read_table(parquet_path, columns=["task_index", "frame_index"])
        except Exception:
            return

        task_indices = table["task_index"].to_pylist()
        frame_indices = table["frame_index"].to_pylist()
        segments = self._find_segments(task_indices, frame_indices)

        caps: Dict[str, cv2.VideoCapture] = {}
        for cam, cam_dir in ds.video_dirs.items():
            video_path = cam_dir / f"episode_{episode_index:06d}.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    caps[cam] = cap

        for task_idx, frame_start, frame_end in segments:
            if frame_end - frame_start + 1 < self.min_segment_frames:
                continue
            task = ds.tasks.get(task_idx, "")
            if not task:
                continue

            mid_frame = (frame_start + frame_end) // 2
            images = self._read_frame(caps, mid_frame)
            if not images:
                continue

            yield EgoTelSegment(
                task_dir=ds.task_dir_name,
                dataset_dir=ds.dataset_dir_name,
                episode_index=episode_index,
                task_index=task_idx,
                task=task,
                frame_start=frame_start,
                frame_end=frame_end,
                images=images,
            )

        for cap in caps.values():
            cap.release()

    def _find_segments(
        self, task_indices: List[int], frame_indices: List[int]
    ) -> List[Tuple[int, int, int]]:
        if not task_indices:
            return []
        segments = []
        cur_task = task_indices[0]
        cur_start = frame_indices[0]
        cur_end = frame_indices[0]
        for task_idx, frame_idx in zip(task_indices[1:], frame_indices[1:]):
            if task_idx == cur_task:
                cur_end = frame_idx
            else:
                segments.append((cur_task, cur_start, cur_end))
                cur_task = task_idx
                cur_start = frame_idx
                cur_end = frame_idx
        segments.append((cur_task, cur_start, cur_end))
        return segments

    def _read_frame(
        self, caps: Dict[str, cv2.VideoCapture], frame_idx: int
    ) -> Dict[str, Image.Image]:
        images = {}
        for cam, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                images[cam] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return images

    def get_task_to_segments(self) -> Dict[str, List[Tuple[str, str, int, int, int]]]:
        """返回 task -> [(task_dir, dataset_dir, episode_index, frame_start, frame_end)] 索引。"""
        index: Dict[str, List[Tuple[str, str, int, int, int]]] = {}
        for ds in self._datasets:
            for parquet_path in sorted(ds.parquet_dir.glob("episode_*.parquet")):
                ep_idx = int(parquet_path.stem.split("_")[1])
                try:
                    table = pq.read_table(parquet_path, columns=["task_index", "frame_index"])
                except Exception:
                    continue
                task_indices = table["task_index"].to_pylist()
                frame_indices = table["frame_index"].to_pylist()
                for task_idx, frame_start, frame_end in self._find_segments(task_indices, frame_indices):
                    if frame_end - frame_start + 1 < self.min_segment_frames:
                        continue
                    task = ds.tasks.get(task_idx, "")
                    if task:
                        index.setdefault(task, []).append(
                            (ds.task_dir_name, ds.dataset_dir_name, ep_idx, frame_start, frame_end)
                        )
        return index
