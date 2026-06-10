"""
全量 Embedding 构建脚本：从 TOS 读取 EgouDas / EgoTel 的 webdataset tar → 编码 → 存盘。

默认行为（开箱即用）:
  python scripts/build_full_embeddings.py --device cuda:0 --batch_size 24
  等价于:
  python scripts/build_full_embeddings.py --dataset all --use_webdataset --stream --device cuda:0 --batch_size 24

用法:
  # --- 默认：TOS 流式读取两个数据集 ---
  python scripts/build_full_embeddings.py --device cuda:0 --batch_size 24

  # --- 只跑一个数据集 ---
  python scripts/build_full_embeddings.py --dataset egotel --device cuda:0 --batch_size 24

  # --- 旧模式（本地 parquet + MP4，需本地数据）---
  python scripts/build_full_embeddings.py --dataset egotel --device cuda:0 --batch_size 24 --local

输出:
  outputs/embeddings/{dataset}_embeddings.npy   # [N, D] float32
  outputs/embeddings/{dataset}_metadata.json    # 每条 segment 的元信息
  outputs/embeddings/{dataset}_checkpoint.pkl   # 断点续跑状态

数据源（TOS webdataset）:
  EgoTel:  tos://ai-dev/.../EgoTel_aligned/raw_0521_anno_v2/  (~5.3TB, 8115 tar)
  EgouDas: tos://ai-dev/.../ego_uDas/0513_anno_v2/           (~1.1TB, ~ tars)
"""

import json
import os
import pickle
import sys
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.egotel_loader import EgoTelDataLoader
from data.egoudas_loader import EgouDasDataLoader
from data.webdataset_loader import (
    DATASET_ROOTS as WDS_ROOTS,
    list_webdataset_tars,
    download_tar,
    extract_segments_from_tar,
    extract_segments_from_tar_bytes,
    read_tar_via_http,
    load_frames_for_segments,
    DEFAULT_CACHE_DIR,
)

# ── 配置 ──────────────────────────────────────────────
QWEN3_VL_PATH = "/mnt/vepfs01/output/klayzhou/EgoSimMatch/models/Qwen3-VL-Embedding-8B"
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/embeddings")
N_FRAMES = 10                     # 每个 segment 采样帧数
VIDEO_READ_WORKERS = 8            # 并行读取视频的线程数（适度，避免抢 CPU）
CHECKPOINT_EVERY = 2000           # 每 N 个 segment 存一次 checkpoint
CAMERA = "cam_high"               # 使用的摄像头
DEFAULT_CHUNK_SEGMENTS = 500      # 默认每 chunk 加载 500 个 segment 的帧（控制内存）

# ── 辅助函数 ──────────────────────────────────────────


def preflight_check(device: str, num_gpus: int):
    """在数据收集前快速验证模型是否能正常加载，避免跑完收集才报错。"""
    print("=" * 60)
    print("  [预检] 测试模型加载 ...")
    print("=" * 60)
    try:
        t0 = time.time()
        model = SentenceTransformer(
            QWEN3_VL_PATH,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": "auto"},
        )
        load_time = time.time() - t0
        print(f"  ✅ 模型加载成功 ({load_time:.1f}s)")

        if num_gpus > 1:
            target_devices = [f"cuda:{i}" for i in range(num_gpus)]
            try:
                pool = model.start_multi_process_pool(target_devices=target_devices)
                model.stop_multi_process_pool(pool)
                print(f"  ✅ 多卡 {num_gpus} 卡池测试通过: {target_devices}")
            except Exception as e:
                print(f"  ⚠️  多卡池测试失败: {e}")
                print(f"  ⚠️  将使用单卡 {device} 继续")

        # 构造一个 dummy 输入测试编码
        from PIL import Image
        dummy_img = Image.new("RGB", (224, 224), color=255)
        dummy_msg = [
            {"role": "user", "content": [{"type": "image", "image": dummy_img}, {"type": "text", "text": "test"}]}
        ]
        test_emb = model.encode(dummy_msg, normalize_embeddings=True)
        print(f"  ✅ 编码测试通过 (embedding dim={test_emb.shape[-1]})")
        del model, dummy_img, dummy_msg, test_emb
        import gc
        gc.collect()
        print(f"  ✅ 预检全部通过，开始正式处理\n")
    except Exception as e:
        print(f"\n  ❌ 预检失败！")
        print(f"     错误: {e}")
        print(f"\n     请修复后再运行，避免浪费收集时间。")
        print(f"     常见原因: transformers 版本过旧，不支持 qwen3_vl")
        print(f"     解决: pip install --upgrade transformers\n")
        sys.exit(1)


def sample_frame_indices(frame_start: int, frame_end: int, n: int) -> List[int]:
    """均匀采样 n 个帧索引（inclusive end）。"""
    total = frame_end - frame_start + 1
    if total <= n:
        return list(range(frame_start, frame_end + 1))
    return [frame_start + int(i * (total - 1) / (n - 1)) for i in range(n)]


def build_message(frames: List[Image.Image], task: str) -> dict:
    """构造 Qwen3-VL-Embedding 的输入 message 格式。"""
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": task})
    return {"role": "user", "content": content}


# ── Segment 收集（旧模式：parquet + MP4）─────────────


def collect_egoudas_segments(split: str = "eval") -> List[dict]:
    """收集 EgouDas 所有 segment 的元信息（不含帧图像）。"""
    loader = EgouDasDataLoader(split=split, cameras=[CAMERA])
    segments = []
    for ds in loader._datasets:
        for parquet_dir in ds.parquet_dirs:
            for parquet_path in sorted(parquet_dir.glob("episode_*.parquet")):
                ep_idx = int(parquet_path.stem.split("_")[1])
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(parquet_path, columns=["task_index", "frame_index"])
                except Exception:
                    continue
                task_indices = table["task_index"].to_pylist()
                frame_indices = table["frame_index"].to_pylist()
                for task_idx, frame_start, frame_end in loader._find_segments(task_indices, frame_indices):
                    if frame_end - frame_start + 1 < loader.min_segment_frames:
                        continue
                    task = ds.tasks.get(task_idx, "")
                    if not task:
                        continue
                    # 视频在第一个 chunk 的 cam_high 下
                    first_chunk = ds.parquet_dirs[0] if ds.parquet_dirs else None
                    chunk_name = parquet_dir.name  # e.g. "chunk-000"
                    video_path = (
                        ds.dataset_dir / "videos" / chunk_name / CAMERA /
                        f"episode_{ep_idx:06d}.mp4"
                    )
                    segments.append({
                        "task": task,
                        "task_name": ds.task_name,
                        "dataset_dir": ds.dataset_dir_name,
                        "episode": ep_idx,
                        "task_index": task_idx,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "video_path": str(video_path),
                        "dataset": "egoudas",
                    })
    return segments


def collect_egotel_segments(split: str = "eval") -> List[dict]:
    """收集 EgoTel 所有 segment 的元信息（不含帧图像）。"""
    loader = EgoTelDataLoader(split=split, cameras=[CAMERA])
    segments = []
    for ds in loader._datasets:
        for parquet_dir in ds.parquet_dirs:
            for parquet_path in sorted(parquet_dir.glob("episode_*.parquet")):
                ep_idx = int(parquet_path.stem.split("_")[1])
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(parquet_path, columns=["task_index", "frame_index"])
                except Exception:
                    continue
                task_indices = table["task_index"].to_pylist()
                frame_indices = table["frame_index"].to_pylist()
                for task_idx, frame_start, frame_end in loader._find_segments(task_indices, frame_indices):
                    if frame_end - frame_start + 1 < loader.min_segment_frames:
                        continue
                    task = ds.tasks.get(task_idx, "")
                    if not task:
                        continue
                    chunk_name = parquet_dir.name  # e.g. "chunk-000"
                    video_path = (
                        ds.dataset_dir / "videos" / chunk_name / CAMERA /
                        f"episode_{ep_idx:06d}.mp4"
                    )
                    segments.append({
                        "task": task,
                        "task_name": ds.task_name,
                        "dataset_dir": ds.dataset_dir_name,
                        "episode": ep_idx,
                        "task_index": task_idx,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "video_path": str(video_path),
                        "dataset": "egotel",
                    })
    return segments


# ── 核心优化：按视频分组加载帧 ──────────────────────────
#
#  关键洞察：20 万 segments 分布在几千个 MP4 文件中，
#  每个 MP4 包含一个 episode 的多个 task segment。
#  按视频分组 → 每个 MP4 只 open/seek/decode 一次 →
#  避免网络存储上的大量随机 I/O。


def load_all_frames_by_video(
    seg_metas: List[dict],
    n_frames: int = N_FRAMES,
    num_workers: int = VIDEO_READ_WORKERS,
) -> Tuple[List[dict], List[dict]]:
    """
    按视频文件分组加载所有 segment 的帧。

    策略:
      1. 将 segments 按 video_path 分组
      2. 每组用一个线程处理：打开视频 → seek 到各 segment 的帧 → 解码
      3. 按原始索引排序返回

    Returns:
      valid_metas: 成功加载帧的 segment 元信息
      messages:    对应的 Qwen-VL message 列表（保持原序）
    """
    # ── 按视频分组 ──
    print("  按视频路径分组 segments ...")
    video_groups = defaultdict(list)  # video_path -> [(seg_idx, seg_meta)]
    for i, meta in enumerate(seg_metas):
        video_groups[meta["video_path"]].append((i, meta))

    unique_videos = len(video_groups)
    print(f"  {len(seg_metas)} segments 分布在 {unique_videos} 个视频文件中")

    # ── 并行处理每个视频 ──
    results: Dict[int, Tuple[dict, dict]] = {}  # seg_idx -> (meta, message)
    failed_count = 0

    def _process_video(item):
        """处理一个视频文件：打开一次，读取所有 segment 的帧。"""
        video_path_str, seg_list = item
        video_path = Path(video_path_str)
        local_results = {}

        if not video_path.exists():
            return local_results, len(seg_list)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return local_results, len(seg_list)

        try:
            for seg_idx, meta in seg_list:
                indices = sample_frame_indices(
                    meta["frame_start"], meta["frame_end"], n_frames
                )
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )
                if frames:
                    local_results[seg_idx] = (
                        meta,
                        build_message(frames, meta["task"]),
                    )
        finally:
            cap.release()

        failed = len(seg_list) - len(local_results)
        return local_results, failed

    # 启动多线程处理
    video_items = list(video_groups.items())
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_process_video, item): item[0]
            for item in video_items
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="  读取视频帧(按视频分组)",
            unit="video",
        ):
            local_results, failed = future.result()
            results.update(local_results)
            failed_count += failed

    # ── 按原始 seg_idx 排序 ──
    valid_metas = []
    messages = []
    for i in range(len(seg_metas)):
        if i in results:
            meta, msg = results[i]
            valid_metas.append(meta)
            messages.append(msg)

    if failed_count:
        print(f"  ⚠️  {failed_count} 个 segment 帧加载失败")

    print(f"  成功加载 {len(messages)}/{len(seg_metas)} segments 的帧")
    return valid_metas, messages


def encode_and_save(
    dataset_name: str,
    seg_metas: List[dict],
    device: str = "cuda:0",
    batch_size: int = 24,
    chunk_segments: int = DEFAULT_CHUNK_SEGMENTS,   # 每次加载多少 segment 的帧到内存
):
    """
    主流程：按视频分组加载帧 → GPU 批量编码 → 存盘，支持断点续跑。

    pipeline:
      Step 1: 按 chunk 分批 → 避免 20 万帧同时加载
      Step 2: 每 chunk 内按视频分组加载帧（每个 MP4 只 open 一次）
      Step 3: GPU 批量编码
      Step 4: 增量存盘
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / f"{dataset_name}_embeddings.npy"
    meta_path = OUTPUT_DIR / f"{dataset_name}_metadata.json"
    ckpt_path = OUTPUT_DIR / f"{dataset_name}_checkpoint.pkl"

    # ── 断点续跑 ──
    all_embeddings: List[np.ndarray] = []
    all_metas: List[dict] = []
    start_idx = 0

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt.get("next_idx", 0)
        if emb_path.exists():
            existing = np.load(emb_path)
            all_embeddings = [existing[i] for i in range(existing.shape[0])]
        if meta_path.exists():
            with open(meta_path) as f:
                all_metas = json.load(f)
        print(f"[resume] 从第 {start_idx} 个 segment 继续，已编码 {len(all_metas)} 个")

    total = len(seg_metas)
    if start_idx >= total:
        print(f"[{dataset_name}] 全部完成！共 {total} 个 segments")
        return

    print(f"[{dataset_name}] 共 {total} 个 segments，从第 {start_idx} 个开始")

    # ── 加载模型（只加载一次） ──
    print(f"加载 Qwen3-VL-Embedding 模型到 {device} ...")
    t0 = time.time()
    model = SentenceTransformer(
        QWEN3_VL_PATH,
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": "auto"},
    )
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 按 chunk 处理 ──
    idx = start_idx
    pbar = tqdm(total=total, initial=start_idx, desc=f"[{dataset_name}] 总进度", unit="seg")

    while idx < total:
        chunk_end = min(idx + chunk_segments, total)
        chunk = seg_metas[idx:chunk_end]
        chunk_size = chunk_end - idx

        # Step 1: 按视频分组加载帧（核心优化！每个 MP4 只 open 一次）
        t_load = time.time()
        valid_metas, messages = load_all_frames_by_video(chunk, n_frames=N_FRAMES)
        load_time = time.time() - t_load

        if not messages:
            print(f"  警告: chunk [{idx}:{chunk_end}] 无有效 segment，跳过")
            idx = chunk_end
            pbar.update(chunk_size)
            continue

        # Step 2: GPU 批量编码
        t_enc = time.time()
        embs = model.encode(
            messages,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        enc_time = time.time() - t_enc

        # Step 3: 累积
        all_embeddings.extend(embs[i] for i in range(len(embs)))
        all_metas.extend(valid_metas)
        idx = chunk_end
        pbar.update(len(valid_metas))

        # Step 4: 日志
        n_valid = len(valid_metas)
        print(
            f"  chunk [{idx - n_valid}..{idx}] "
            f"有效={n_valid}/{chunk_size} "
            f"帧加载={load_time:.1f}s 编码={enc_time:.1f}s "
            f"({n_valid / enc_time:.1f} seg/s GPU)"
        )

        # Step 5: checkpoint
        if (idx % CHECKPOINT_EVERY < chunk_size) or idx >= total:
            _save_checkpoint(all_embeddings, all_metas, emb_path, meta_path, ckpt_path, idx)

    pbar.close()

    # ── 最终存盘 ──
    _save_checkpoint(all_embeddings, all_metas, emb_path, meta_path, ckpt_path, total, final=True)

    emb_dim = all_embeddings[0].shape[0] if all_embeddings else 0
    print(f"\n[{dataset_name}] ✅ 完成！共 {len(all_metas)} 个 segments")
    print(f"  Embedding 维度: {emb_dim}")
    print(f"  存储路径: {emb_path}")


def _save_checkpoint(
    all_embeddings: List[np.ndarray],
    all_metas: List[dict],
    emb_path: Path,
    meta_path: Path,
    ckpt_path: Path,
    next_idx: int,
    final: bool = False,
):
    """保存 embedding、metadata 和 checkpoint。"""
    if all_embeddings:
        emb_matrix = np.stack(all_embeddings, axis=0)
        np.save(emb_path, emb_matrix)
    if all_metas:
        # 移除不可 JSON 序列化的字段（如 PIL Image）
        metas_clean = []
        for m in all_metas:
            m_clean = {k: v for k, v in m.items() if k != "cam_images"}
            metas_clean.append(m_clean)
        with open(meta_path, "w") as f:
            json.dump(metas_clean, f, ensure_ascii=False, indent=2)
    with open(ckpt_path, "wb") as f:
        pickle.dump({"next_idx": next_idx}, f)

    status = "最终" if final else "checkpoint"
    print(f"  [{status}] 已保存 {len(all_metas)} 条 → {emb_path}")


# ── Webdataset 模式：encode（segment 已含图像）─────────


def encode_and_save_webdataset(
    dataset_name: str,
    seg_metas: List[dict],
    device: str = "cuda:0",
    batch_size: int = 24,
    chunk_segments: int = DEFAULT_CHUNK_SEGMENTS,
    num_gpus: int = 1,
):
    """
    Webdataset 模式的主编码流程。

    与 encode_and_save 的区别：
      - seg_metas 已包含 cam_images（图像已加载）
      - 无需调用 load_all_frames_by_video
      - 直接从 cam_images 构建 message

    流程:
      Step 1: 分批处理（控制内存）
      Step 2: 直接从 seg_metas 构建 messages
      Step 3: GPU 批量编码（支持多卡 DataParallel）
      Step 4: 增量存盘
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / f"{dataset_name}_embeddings.npy"
    meta_path = OUTPUT_DIR / f"{dataset_name}_metadata.json"
    ckpt_path = OUTPUT_DIR / f"{dataset_name}_checkpoint.pkl"

    # ── 断点续跑 ──
    all_embeddings: List[np.ndarray] = []
    all_metas: List[dict] = []
    start_idx = 0

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt.get("next_idx", 0)
        if emb_path.exists():
            existing = np.load(emb_path)
            all_embeddings = [existing[i] for i in range(existing.shape[0])]
        if meta_path.exists():
            with open(meta_path) as f:
                all_metas = json.load(f)
        print(f"[resume] 从第 {start_idx} 个 segment 继续，已编码 {len(all_metas)} 个")

    total = len(seg_metas)
    if start_idx >= total:
        print(f"[{dataset_name}] 全部完成！共 {total} 个 segments")
        return

    print(f"[{dataset_name}] 共 {total} 个 segments，从第 {start_idx} 个开始")

    # ── 加载模型（只加载一次） ──
    print(f"加载 Qwen3-VL-Embedding 模型到 {device} ...")
    t0 = time.time()
    model = SentenceTransformer(
        QWEN3_VL_PATH,
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": "auto"},
    )
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 多卡启动 ──
    pool = None
    if num_gpus > 1:
        target_devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"启动多卡 pool: {target_devices}")
        pool = model.start_multi_process_pool(target_devices=target_devices)

    # ── 按 chunk 处理 ──
    idx = start_idx
    pbar = tqdm(total=total, initial=start_idx, desc=f"[{dataset_name}] 总进度", unit="seg")

    while idx < total:
        chunk_end = min(idx + chunk_segments, total)
        chunk = seg_metas[idx:chunk_end]
        chunk_size = chunk_end - idx

        # Step 1: 直接从 segment 的 cam_images 构建 messages
        t_load = time.time()
        valid_metas, messages = load_frames_for_segments(chunk)
        load_time = time.time() - t_load

        if not messages:
            print(f"  警告: chunk [{idx}:{chunk_end}] 无有效 segment，跳过")
            idx = chunk_end
            pbar.update(chunk_size)
            continue

        # Step 2: GPU 批量编码（单卡 / 多卡）
        t_enc = time.time()
        if pool is not None:
            per_gpu_batch = max(1, batch_size // num_gpus)
            embs = model.encode_multi_process(
                messages, pool,
                batch_size=per_gpu_batch,
                normalize_embeddings=True,
            )
        else:
            embs = model.encode(
                messages,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        enc_time = time.time() - t_enc

        # Step 3: 累积
        all_embeddings.extend(embs[i] for i in range(len(embs)))
        all_metas.extend(valid_metas)
        idx = chunk_end
        pbar.update(len(valid_metas))

        # Step 4: 日志
        n_valid = len(valid_metas)
        gpu_info = f"{num_gpus}×GPU " if num_gpus > 1 else ""
        print(
            f"  chunk [{idx - n_valid}..{idx}] "
            f"有效={n_valid}/{chunk_size} "
            f"构建消息={load_time:.1f}s 编码={enc_time:.1f}s "
            f"({n_valid / enc_time:.1f} seg/s {gpu_info})"
        )

        # Step 5: checkpoint
        if (idx % CHECKPOINT_EVERY < chunk_size) or idx >= total:
            _save_checkpoint(all_embeddings, all_metas, emb_path, meta_path, ckpt_path, idx)

    pbar.close()

    # ── 关闭多卡 pool ──
    if pool is not None:
        model.stop_multi_process_pool(pool)

    # ── 最终存盘 ──
    _save_checkpoint(all_embeddings, all_metas, emb_path, meta_path, ckpt_path, total, final=True)

    emb_dim = all_embeddings[0].shape[0] if all_embeddings else 0
    print(f"\n[{dataset_name}] ✅ 完成！共 {len(all_metas)} 个 segments")
    print(f"  Embedding 维度: {emb_dim}")
    print(f"  存储路径: {emb_path}")


# ── Webdataset 模式下的 segment 收集 ──────────────────


def collect_webdataset_segments(
    ds_name: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    stream: bool = False,
) -> List[dict]:
    """
    从 TOS webdataset tar 包中收集所有 segment 元信息（含帧图像）。

    两种模式:
      - stream=False (默认): 下载 tar 到本地缓存再读取
      - stream=True: 通过 presigned URL + HTTP 流式读取到内存（零磁盘占用）

    Returns:
        [{
            "task": str,
            "task_name": str,
            "dataset_dir": str,
            "episode": int,
            "task_index": int,
            "frame_indices": [int],
            "cam_images": {cam: [Image]},
            "dataset": str,
        }, ...]
    """
    tos_root = WDS_ROOTS[ds_name]
    print(f"  扫描 TOS: {tos_root}/eval/ ...")
    t0 = time.time()

    tar_paths = list_webdataset_tars(tos_root)
    print(f"  找到 {len(tar_paths)} 个 tar 文件，耗时 {time.time() - t0:.1f}s")

    if not tar_paths:
        return []

    # 逐个处理 tar
    all_segments = []
    t0 = time.time()
    for i, tar_path in enumerate(tar_paths):
        # 解析 task_name 和 dataset_dir
        parts = tar_path.split("/")
        try:
            wds_idx = parts.index("webdataset")
            ds_dir = parts[wds_idx - 1]
            t_name = parts[wds_idx - 2]
        except (ValueError, IndexError):
            continue

        if stream:
            # 流式模式：HTTP GET → 内存 → 解析（零磁盘）
            tar_bytes = read_tar_via_http(tar_path)
            segments = extract_segments_from_tar_bytes(
                tar_bytes, t_name, ds_dir, ds_name, cameras=[CAMERA]
            )
        else:
            # 缓存模式：下载到本地磁盘再读取
            local_path = download_tar(tar_path, cache_dir)
            segments = extract_segments_from_tar(
                local_path, t_name, ds_dir, ds_name, cameras=[CAMERA]
            )
        all_segments.extend(segments)

        if (i + 1) % 50 == 0 or (i + 1) == len(tar_paths):
            elapsed = time.time() - t0
            print(f"    已处理 {i+1}/{len(tar_paths)} tar, "
                  f"收集 {len(all_segments)} segments, "
                  f"耗时 {elapsed:.1f}s")

    print(f"  共收集 {len(all_segments)} 个 segments")
    return all_segments


# ── 入口 ───────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建全量 Embedding")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["egoudas", "egotel", "all"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="使用的 GPU 数量（多卡用 start_multi_process_pool）")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="每 GPU batch size（默认 24，多卡时自动放大）")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SEGMENTS,
                        help=f"每 chunk 加载 segment 数（控制内存，默认{DEFAULT_CHUNK_SEGMENTS}）")
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计 segment 数量，不实际编码")
    parser.add_argument("--local", action="store_true",
                        help="使用本地 parquet + MP4 模式（默认走 TOS webdataset 流式）")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="webdataset tar 本地缓存目录 (仅非流式模式)")
    parser.add_argument("--no_stream", action="store_true",
                        help="关闭流式模式，下载 tar 到缓存再读取")
    parser.add_argument("--force_recollect", action="store_true",
                        help="强制重新从 TOS 收集 segments（忽略本地缓存）")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("egoudas", "all"):
        datasets.append("egoudas")
    if args.dataset in ("egotel", "all"):
        datasets.append("egotel")

    # 自动设置 batch_size：多卡时每卡 24
    if args.batch_size == 0:
        args.batch_size = 24 * max(1, args.num_gpus)
    elif args.num_gpus > 1:
        args.batch_size = args.batch_size * args.num_gpus

    use_stream = not args.no_stream
    use_webdataset = not args.local

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  数据集: {ds_name}")
        if use_webdataset:
            mode_str = "TOS webdataset"
            if use_stream:
                mode_str += " (流式, 零磁盘)"
            print(f"  模式: {mode_str}")
        else:
            print(f"  模式: 本地 parquet + MP4")
        print(f"{'='*60}")

        # 预检：在收集数据前先测试模型能否正常加载
        if not args.dry_run and use_webdataset:
            preflight_check(args.device, args.num_gpus)

        t0 = time.time()

        if use_webdataset:
            # ── Webdataset 模式 ──
            if args.dry_run:
                tos_root = WDS_ROOTS[ds_name]
                tar_paths = list_webdataset_tars(tos_root)
                collect_time = time.time() - t0
                print(f"TOS 上找到 {len(tar_paths)} 个 webdataset tar 文件")
                print(f"收集完成，耗时 {collect_time:.1f}s (dry run, 未下载)")
                continue

            # 尝试从缓存加载，避免重新收集 40 分钟
            seg_cache_path = OUTPUT_DIR / f"{ds_name}_segments_cache.pkl"
            if seg_cache_path.exists() and not args.force_recollect:
                with open(seg_cache_path, "rb") as f:
                    seg_metas = pickle.load(f)
                collect_time = time.time() - t0
                print(f"  从本地缓存加载 {len(seg_metas)} 个 segments，耗时 {collect_time:.1f}s")
            else:
                seg_metas = collect_webdataset_segments(
                    ds_name, cache_dir=args.cache_dir,
                    stream=use_stream,
                )
                # 收集完成后立即缓存到本地 pickle
                with open(seg_cache_path, "wb") as f:
                    pickle.dump(seg_metas, f)
                print(f"  已缓存 {len(seg_metas)} 个 segments → {seg_cache_path}")
        else:
            # ── 旧模式：parquet 扫描 ──
            if ds_name == "egoudas":
                seg_metas = collect_egoudas_segments(split=args.split)
            else:
                seg_metas = collect_egotel_segments(split=args.split)

        collect_time = time.time() - t0
        print(f"收集完成: {len(seg_metas)} 个 segments，耗时 {collect_time:.1f}s")

        if args.dry_run:
            if not use_webdataset:
                tasks = set(m["task"] for m in seg_metas)
                print(f"  唯一 task 数: {len(tasks)}")
            continue

        if len(seg_metas) == 0:
            print("  无 segment，跳过")
            continue

        if use_webdataset:
            encode_and_save_webdataset(
                dataset_name=ds_name,
                seg_metas=seg_metas,
                device=args.device,
                batch_size=args.batch_size,
                chunk_segments=args.chunk_size,
                num_gpus=args.num_gpus,
            )
        else:
            encode_and_save(
                dataset_name=ds_name,
                seg_metas=seg_metas,
                device=args.device,
                batch_size=args.batch_size,
                chunk_segments=args.chunk_size,
            )


if __name__ == "__main__":
    main()
