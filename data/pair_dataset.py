"""PairDataset — 基于 task 语义相似度构建 EgouDas × EgoTel 正负样本对。

核心设计：
  - 正例：EgouDas segment 和 EgoTel segment 的 task 语义相似度 >= pos_threshold
  - 负例：task 语义相似度 <= neg_threshold
  - batch 构建：一半 ego（EgouDas），一半 tele（EgoTel），task 尽量匹配

用法：
    ds = PairDataset(split="eval", max_sessions=5, max_datasets=5)
    test_set = ds.build_test_set(n_pos=50, n_neg=50)

    # 构建训练 batch（一半 ego，一半 tele，task 匹配）
    batch = ds.build_matched_batch(batch_size=16)
"""

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from data.egoudas_loader import EgouDasDataLoader, EgouDasSegment
from data.egotel_loader import EgoTelDataLoader, EgoTelSegment

QWEN3_EMB_PATH = (
    "/mnt/vepfs01/output/klayzhou/EgoSimMatch/models/Qwen3-Embedding-8B"
)


@dataclass
class SamplePair:
    label: int                      # 1=正例, 0=负例
    task_sim: float
    ego_segment: EgouDasSegment     # EgouDas (ego)
    tele_segment: EgoTelSegment     # EgoTel (tele)
    ego_task: str
    tele_task: str


@dataclass
class MatchedBatch:
    """一个 batch：一半 ego，一半 tele，按 task 相似度配对。"""
    ego_segments: List[EgouDasSegment]
    tele_segments: List[EgoTelSegment]
    task_sim_matrix: np.ndarray     # (n_ego, n_tele) 相似度矩阵
    pair_labels: np.ndarray         # (n_ego, n_tele) 1=正例, 0=负例


class PairDataset:
    """构建 EgouDas × EgoTel 的正负样本对，支持 matched batch 构建。

    Args:
        split: "train" 或 "eval"
        pos_threshold: task 相似度 >= 此值视为正例（默认 0.85）
        neg_threshold: task 相似度 <= 此值视为负例（默认 0.70）
        neg_ratio: 负例与正例的比例
        max_sessions: EgouDas 最多加载的 session 数
        max_datasets: EgoTel 最多加载的 dataset 数
        seed: 随机种子
        cameras: 要加载的摄像头
        emb_model_path: Qwen3-Embedding 模型路径
        emb_device: embedding 模型设备
    """

    def __init__(
        self,
        split: str = "eval",
        pos_threshold: float = 0.85,
        neg_threshold: float = 0.70,
        neg_ratio: float = 1.0,
        max_sessions: Optional[int] = None,
        max_datasets: Optional[int] = None,
        seed: int = 42,
        cameras: Optional[List[str]] = None,
        emb_model_path: str = QWEN3_EMB_PATH,
        emb_device: str = "cuda:0",
    ):
        self.split = split
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        print(f"[PairDataset] Loading EgouDas ({split})...")
        self.udas_loader = EgouDasDataLoader(
            split=split, cameras=cameras, max_sessions=max_sessions
        )

        print(f"[PairDataset] Loading EgoTel ({split})...")
        self.tel_loader = EgoTelDataLoader(
            split=split, cameras=cameras, max_datasets=max_datasets
        )

        print("[PairDataset] Building segment index...")
        self._udas_index = self.udas_loader.get_task_to_segments()
        self._tel_index = self.tel_loader.get_task_to_segments()

        udas_tasks = sorted(self._udas_index.keys())
        tel_tasks = sorted(self._tel_index.keys())
        print(f"[PairDataset] EgouDas tasks: {len(udas_tasks)}, EgoTel tasks: {len(tel_tasks)}")

        print("[PairDataset] Computing task embeddings with Qwen3-Embedding...")
        self._task_embs = self._compute_task_embeddings(
            udas_tasks, tel_tasks, emb_model_path, emb_device
        )
        self._udas_tasks = udas_tasks
        self._tel_tasks = tel_tasks

        print("[PairDataset] Computing similarity matrix...")
        self._udas_embs = np.array([self._task_embs[t] for t in udas_tasks])
        self._tel_embs = np.array([self._task_embs[t] for t in tel_tasks])
        self._sim_matrix = self._udas_embs @ self._tel_embs.T  # (n_udas, n_tel)

        self._pos_pairs, self._neg_pairs = self._find_task_pairs()
        print(
            f"[PairDataset] Positive task pairs: {len(self._pos_pairs)}, "
            f"Negative task pairs: {len(self._neg_pairs)}"
        )

        print("[PairDataset] Caching segments...")
        self._udas_segments = self._cache_segments(self.udas_loader)
        self._tel_segments = self._cache_segments(self.tel_loader)
        print(
            f"[PairDataset] Cached {len(self._udas_segments)} ego segments, "
            f"{len(self._tel_segments)} tele segments"
        )

    def _compute_task_embeddings(
        self, udas_tasks, tel_tasks, model_path, device
    ) -> Dict[str, np.ndarray]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
        all_tasks = list(set(udas_tasks + tel_tasks))
        embs = model.encode(all_tasks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        return {task: emb for task, emb in zip(all_tasks, embs)}

    def _find_task_pairs(self):
        pos_pairs, neg_pairs = [], []
        for i, udas_task in enumerate(self._udas_tasks):
            for j, tel_task in enumerate(self._tel_tasks):
                sim = float(self._sim_matrix[i, j])
                if sim >= self.pos_threshold:
                    pos_pairs.append((udas_task, tel_task, sim))
                elif sim <= self.neg_threshold:
                    neg_pairs.append((udas_task, tel_task, sim))
        return pos_pairs, neg_pairs

    def _cache_segments(self, loader) -> List:
        return list(loader.iter_segments())

    def _get_udas_segment(self, task: str) -> Optional[EgouDasSegment]:
        entries = self._udas_index.get(task, [])
        if not entries:
            return None
        session, ep_idx, frame_start, frame_end = self.rng.choice(entries)
        # 从缓存中找匹配的 segment
        matches = [
            s for s in self._udas_segments
            if s.session == session and s.episode_index == ep_idx
            and s.frame_start == frame_start
        ]
        return self.rng.choice(matches) if matches else None

    def _get_tel_segment(self, task: str) -> Optional[EgoTelSegment]:
        entries = self._tel_index.get(task, [])
        if not entries:
            return None
        task_dir, dataset_dir, ep_idx, frame_start, frame_end = self.rng.choice(entries)
        matches = [
            s for s in self._tel_segments
            if s.dataset_dir == dataset_dir and s.episode_index == ep_idx
            and s.frame_start == frame_start
        ]
        return self.rng.choice(matches) if matches else None

    # ── 正负例对迭代 ──────────────────────────────────────────────────────────

    def iter_pairs(self) -> Iterator[SamplePair]:
        yield from self._iter_positive_pairs()
        yield from self._iter_negative_pairs()

    def _iter_positive_pairs(self) -> Iterator[SamplePair]:
        for udas_task, tel_task, sim in self._pos_pairs:
            ego = self._get_udas_segment(udas_task)
            tele = self._get_tel_segment(tel_task)
            if ego is None or tele is None:
                continue
            yield SamplePair(label=1, task_sim=sim, ego_segment=ego, tele_segment=tele,
                             ego_task=udas_task, tele_task=tel_task)

    def _iter_negative_pairs(self) -> Iterator[SamplePair]:
        n_neg = int(len(self._pos_pairs) * self.neg_ratio)
        pool = self._neg_pairs.copy()
        self.rng.shuffle(pool)
        generated = 0
        for udas_task, tel_task, sim in pool:
            if generated >= n_neg:
                break
            ego = self._get_udas_segment(udas_task)
            tele = self._get_tel_segment(tel_task)
            if ego is None or tele is None:
                continue
            yield SamplePair(label=0, task_sim=sim, ego_segment=ego, tele_segment=tele,
                             ego_task=udas_task, tele_task=tel_task)
            generated += 1

    def build_test_set(self, n_pos: int = 50, n_neg: int = 50) -> List[SamplePair]:
        """构建固定大小的测试集（正例 + 负例）。"""
        pairs = []
        pos_count = neg_count = 0
        for pair in self.iter_pairs():
            if pair.label == 1 and pos_count < n_pos:
                pairs.append(pair)
                pos_count += 1
            elif pair.label == 0 and neg_count < n_neg:
                pairs.append(pair)
                neg_count += 1
            if pos_count >= n_pos and neg_count >= n_neg:
                break
        self.rng.shuffle(pairs)
        print(f"[PairDataset] Test set: {pos_count} pos + {neg_count} neg = {len(pairs)} pairs")
        return pairs

    # ── Matched Batch 构建 ────────────────────────────────────────────────────

    def build_matched_batch(self, batch_size: int = 16) -> MatchedBatch:
        """构建一个 batch：一半 ego，一半 tele，task 尽量匹配。

        策略：
          1. 从正例 task 对中随机选 batch_size//2 个 ego segment
          2. 为每个 ego segment 找相似度最高的 tele segment
          3. 计算 batch 内的 (n_ego, n_tele) 相似度矩阵和 pair_labels
        """
        half = batch_size // 2

        # 从正例 task 对中采样
        pos_pool = [(u, t, s) for u, t, s in self._pos_pairs
                    if self._udas_index.get(u) and self._tel_index.get(t)]
        if not pos_pool:
            raise ValueError("No positive task pairs with available segments")

        selected_pos = self.rng.choices(pos_pool, k=half)

        ego_segs: List[EgouDasSegment] = []
        tele_segs: List[EgoTelSegment] = []
        used_tele_tasks = set()

        for udas_task, tel_task, _ in selected_pos:
            ego = self._get_udas_segment(udas_task)
            tele = self._get_tel_segment(tel_task)
            if ego is not None and tele is not None:
                ego_segs.append(ego)
                tele_segs.append(tele)
                used_tele_tasks.add(tel_task)

        # 补充到 half 个（如果有缺失）
        while len(ego_segs) < half and self._udas_segments:
            seg = self.rng.choice(self._udas_segments)
            ego_segs.append(seg)
        while len(tele_segs) < half and self._tel_segments:
            seg = self.rng.choice(self._tel_segments)
            tele_segs.append(seg)

        ego_segs = ego_segs[:half]
        tele_segs = tele_segs[:half]

        # 计算 batch 内相似度矩阵
        ego_task_embs = np.array([self._task_embs.get(s.task, np.zeros(len(next(iter(self._task_embs.values()))))) for s in ego_segs])
        tele_task_embs = np.array([self._task_embs.get(s.task, np.zeros(len(next(iter(self._task_embs.values()))))) for s in tele_segs])
        sim_matrix = ego_task_embs @ tele_task_embs.T

        pair_labels = (sim_matrix >= self.pos_threshold).astype(np.int32)

        return MatchedBatch(
            ego_segments=ego_segs,
            tele_segments=tele_segs,
            task_sim_matrix=sim_matrix,
            pair_labels=pair_labels,
        )
