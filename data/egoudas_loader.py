"""EgouDas DataLoader — lerobot parquet + MP4 格式，按 task_index 切分 segment。

数据路径（使用 retask 版本，parquet 可读）：
  .../ego_data_retask/{date}/{session}/
    meta/tasks.jsonl          task_index -> task 文本
    meta/episodes.jsonl       episode_index -> task_index（多个）
    data/chunk-000/episode_XXXXXX.parquet   含 task_index 字段（逐帧）
    videos/chunk-000/{cam}/episode_XXXXXX.mp4

切分逻辑：
  每个 episode 按 task_index 字段切分成多个 segment，
  每个 segment 对应一个独立的动作片段（通常 120 帧）。
  一个 segment = 一个训练样本。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

EGOUDAS_RETASK_ROOT = (
    "/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/code/"
    "mozbrain_main_0402_ego_grouploss/outputs/converted_data/"
    "ego_uDas/moz_lerobot/ego_data_retask"
)

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


@dataclass
class EgouDasSegment:
    """一个 task segment（episode 中连续属于同一 task_index 的帧）。"""
    session: str
    episode_index: int
    task_index: int
    task: str
    frame_start: int
    frame_end: int      # inclusive
    images: Dict[str, Image.Image]  # cam_name -> 中间帧图像
    source: str = "egoudas"


@dataclass
class EgouDasSession:
    session_dir: Path
    session_name: str
    tasks: Dict[int, str]       # task_index -> task text
    video_dirs: Dict[str, Path] # cam_name -> videos/chunk-000/{cam}/
    parquet_dir: Path           # data/chunk-000/


class EgouDasDataLoader:
    """从 EgouDas retask 数据中加载 task segment 样本。

    每个样本是 episode 中一个连续的 task segment，取中间帧作为代表帧。

    Args:
        split: "train" 或 "eval"（通过 split_gripper 目录的 split_info.json 过滤）
        cameras: 要加载的摄像头列表
        max_sessions: 最多加载多少个 session（用于快速测试）
        min_segment_frames: 最短 segment 帧数（过滤过短的片段）
    """

    def __init__(
        self,
        root: str = EGOUDAS_RETASK_ROOT,
        split: str = "eval",
        cameras: Optional[List[str]] = None,
        max_sessions: Optional[int] = None,
        min_segment_frames: int = 10,
    ):
        self.root = Path(root)
        self.split = split
        self.cameras = cameras or CAMERAS
        self.max_sessions = max_sessions
        self.min_segment_frames = min_segment_frames
        self._sessions = self._collect_sessions()

    def _collect_sessions(self) -> List[EgouDasSession]:
        # 用 split_gripper 的 split_info.json 来确定哪些 session 属于 train/eval
        split_gripper_root = self.root.parent / "ego_data_retask_split_gripper" / self.split
        sessions = []

        for date_dir in sorted(self.root.iterdir()):
            if not date_dir.is_dir():
                continue
            for session_dir in sorted(date_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                # 检查对应的 split_gripper 目录是否存在（确认该 session 属于此 split）
                split_check = split_gripper_root / date_dir.name / session_dir.name
                if not split_check.exists():
                    continue

                session = self._load_session_meta(session_dir)
                if session is not None:
                    sessions.append(session)
                    if self.max_sessions and len(sessions) >= self.max_sessions:
                        return sessions
        return sessions

    def _load_session_meta(self, session_dir: Path) -> Optional[EgouDasSession]:
        tasks_file = session_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            return None

        tasks = {}
        with open(tasks_file) as f:
            for line in f:
                d = json.loads(line)
                tasks[d["task_index"]] = d["task"]

        video_dirs = {}
        for cam in self.cameras:
            cam_dir = session_dir / "videos" / "chunk-000" / cam
            if cam_dir.exists():
                video_dirs[cam] = cam_dir

        parquet_dir = session_dir / "data" / "chunk-000"

        return EgouDasSession(
            session_dir=session_dir,
            session_name=session_dir.name,
            tasks=tasks,
            video_dirs=video_dirs,
            parquet_dir=parquet_dir,
        )

    def __len__(self) -> int:
        return len(self._sessions)

    def iter_segments(self) -> Iterator[EgouDasSegment]:
        """逐 segment 迭代，每个 segment 是一个独立的 task 片段。"""
        for session in self._sessions:
            yield from self._iter_session_segments(session)

    def _iter_session_segments(self, session: EgouDasSession) -> Iterator[EgouDasSegment]:
        for parquet_path in sorted(session.parquet_dir.glob("episode_*.parquet")):
            ep_idx = int(parquet_path.stem.split("_")[1])
            yield from self._extract_segments(session, ep_idx, parquet_path)

    def _extract_segments(
        self, session: EgouDasSession, episode_index: int, parquet_path: Path
    ) -> Iterator[EgouDasSegment]:
        try:
            table = pq.read_table(parquet_path, columns=["task_index", "frame_index"])
        except Exception:
            return

        task_indices = table["task_index"].to_pylist()
        frame_indices = table["frame_index"].to_pylist()

        # 找连续相同 task_index 的 segment
        segments = self._find_segments(task_indices, frame_indices)

        # 打开视频
        caps: Dict[str, cv2.VideoCapture] = {}
        for cam, cam_dir in session.video_dirs.items():
            video_path = cam_dir / f"episode_{episode_index:06d}.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    caps[cam] = cap

        for task_idx, frame_start, frame_end in segments:
            if frame_end - frame_start + 1 < self.min_segment_frames:
                continue
            task = session.tasks.get(task_idx, "")
            if not task:
                continue

            # 取中间帧
            mid_frame = (frame_start + frame_end) // 2
            images = self._read_frame(caps, mid_frame)
            if not images:
                continue

            yield EgouDasSegment(
                session=session.session_name,
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
        """返回 [(task_index, frame_start, frame_end)] 列表。"""
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

    def get_task_to_segments(self) -> Dict[str, List[Tuple[str, int, int, int]]]:
        """返回 task -> [(session, episode_index, frame_start, frame_end)] 索引。"""
        index: Dict[str, List[Tuple[str, int, int, int]]] = {}
        for session in self._sessions:
            for parquet_path in sorted(session.parquet_dir.glob("episode_*.parquet")):
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
                    task = session.tasks.get(task_idx, "")
                    if task:
                        index.setdefault(task, []).append(
                            (session.session_name, ep_idx, frame_start, frame_end)
                        )
        return index
