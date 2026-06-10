"""
Action Trajectory Similarity Operator — 从 TOS webdataset 计算 segment 间动作轨迹相似度。

数据源: TOS webdataset tar 包中的 msgpack 元数据 (与 build_full_embeddings.py 同源)

相似度度量:
  1. DTW (Dynamic Time Warping) — 处理变长序列的标准方法
  2. DTW Normalized — DTW 距离除以路径长度
  3. Interpolated Euclidean — 线性插值到相同长度后逐帧欧氏距离

动作空间 (EgoTel 与 EgouDas 共有 state 列, 23 维):
  - leftarm_state_cart_pos   (6D)  左臂末端笛卡尔位姿
  - rightarm_state_cart_pos  (6D)  右臂末端笛卡尔位姿
  - torso_state_cart_pos     (6D)  躯干笛卡尔位姿
  - leftarm_gripper_state_pos (1D) 左夹爪开度
  - rightarm_gripper_state_pos (1D) 右夹爪开度
  - base_state_speed         (3D)  基座速度

用法:
  # 流式模式 (零磁盘, HTTP 直读)
  python scripts/action_trajectory_similarity.py \
    --pairs outputs/batches_greedy_n50000/pairs_all.json \
    --output outputs/trajectory_sim.json --stream --max-pairs 100

  # 缓存模式 (下载 tar 到本地, 重复访问更快)
  python scripts/action_trajectory_similarity.py \
    --pairs outputs/batches_greedy_n50000/pairs_all.json \
    --output outputs/trajectory_sim.json --cache-dir /tmp/wds_cache --max-pairs 1000

  # 全量计算 (多进程)
  python scripts/action_trajectory_similarity.py \
    --pairs outputs/batches_greedy_n50000/pairs_all.json \
    --output outputs/trajectory_sim.json --workers 8
"""

import argparse
import io
import json
import msgpack
import os
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── TOS webdataset 配置 ──────────────────────────────
WDS_ROOTS = {
    "egotel": "tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/EgoTel_aligned/raw_0521_anno_v2",
    "egoudas": "tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/ego_uDas/0513_anno_v2",
}

_TOSUTIL_CANDIDATES = [
    "/mnt/vepfs01/output/klayzhou/trainning/mozbrain/tosutil",
    "/usr/local/bin/tosutil",
    "/usr/bin/tosutil",
]

DEFAULT_CACHE_DIR = "/tmp/wds_action_cache"
PRESIGNED_URL_CACHE: Dict[str, str] = {}

# ── 动作空间配置 ──────────────────────────────────────
ACTION_COLUMNS = [
    "leftarm_state_cart_pos",
    "rightarm_state_cart_pos",
    "torso_state_cart_pos",
    "leftarm_gripper_state_pos",
    "rightarm_gripper_state_pos",
    "base_state_speed",
]

COLUMN_DIMS = {
    "leftarm_state_cart_pos": 6,
    "rightarm_state_cart_pos": 6,
    "torso_state_cart_pos": 6,
    "leftarm_gripper_state_pos": 1,
    "rightarm_gripper_state_pos": 1,
    "base_state_speed": 3,
}

PART_WEIGHTS = {
    "leftarm_state_cart_pos": 1.0,
    "rightarm_state_cart_pos": 1.0,
    "torso_state_cart_pos": 0.3,
    "leftarm_gripper_state_pos": 0.5,
    "rightarm_gripper_state_pos": 0.5,
    "base_state_speed": 0.2,
}


# ══════════════════════════════════════════════════════
#  TOS / Webdataset 工具
# ══════════════════════════════════════════════════════

def _find_tosutil() -> str:
    for path in _TOSUTIL_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    found = shutil.which("tosutil")
    if found:
        return found
    raise RuntimeError("tosutil 未找到")


def _tosutil_ls(tos_path: str, limit: int = 20000) -> List[str]:
    tosutil = _find_tosutil()
    cmd = [tosutil, "ls", tos_path, f"-limit={limit}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"tosutil ls 失败: {result.stderr[:200]}")
    paths = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("tos://") and not line.endswith("/"):
            paths.append(line.split()[0] if " " in line else line)
    return paths


def _get_presigned_url(tos_path: str) -> str:
    if tos_path in PRESIGNED_URL_CACHE:
        return PRESIGNED_URL_CACHE[tos_path]
    tosutil = _find_tosutil()
    result = subprocess.run(
        [tosutil, "presign", tos_path, "-vp=1h"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"presign 失败: {result.stderr[:200]}")
    url = result.stdout.strip()
    PRESIGNED_URL_CACHE[tos_path] = url
    return url


def _download_tar(tos_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = tos_path.replace("tos://", "").replace("/", "_")
    local_path = os.path.join(cache_dir, safe_name)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        try:
            with tarfile.open(local_path) as _:
                return local_path
        except (tarfile.ReadError, tarfile.TarError):
            os.remove(local_path)

    tosutil = _find_tosutil()
    for attempt in range(3):
        result = subprocess.run(
            [tosutil, "cp", tos_path, local_path, "-p", "4"],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            return local_path
        if os.path.exists(local_path):
            os.remove(local_path)
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError(f"下载失败: {tos_path}")


def list_wds_tars(dataset: str) -> List[str]:
    root = WDS_ROOTS[dataset]
    all_objs = _tosutil_ls(root + "/eval/", limit=20000)
    return sorted([p for p in all_objs if "/webdataset/" in p and p.endswith(".tar")])


def build_tar_index(dataset: str) -> Dict[Tuple[str, str], List[str]]:
    """构建 (task_name, dataset_dir) → [tar_paths] 索引。"""
    print(f"  Building tar index for {dataset} ...")
    t0 = time.time()
    tars = list_wds_tars(dataset)
    index = defaultdict(list)
    for tar_path in tars:
        parts = tar_path.split("/")
        try:
            wds_idx = parts.index("webdataset")
            ds_dir = parts[wds_idx - 1]
            t_name = parts[wds_idx - 2]
        except (ValueError, IndexError):
            continue
        index[(t_name, ds_dir)].append(tar_path)
    elapsed = time.time() - t0
    print(f"  Found {len(tars)} tars across {len(index)} groups ({elapsed:.1f}s)")
    return dict(index)


def build_all_tar_index(datasets: List[str]) -> Dict[str, Dict]:
    """为所有数据集构建 tar 索引。"""
    return {ds: build_tar_index(ds) for ds in datasets}


# ══════════════════════════════════════════════════════
#  轨迹提取
# ══════════════════════════════════════════════════════

def extract_trajectory(
    meta: dict,
    tar_index: Dict[Tuple[str, str], List[str]],
    cache_dir: str,
    stream: bool = False,
    columns: List[str] = ACTION_COLUMNS,
) -> Optional[np.ndarray]:
    """
    从 webdataset tar 的 msgpack 中提取 segment 的动作轨迹。

    Args:
        meta: segment 元数据 (task_name, dataset_dir, episode, frame_indices, dataset)
        tar_index: (task_name, dataset_dir) → [tar_paths]
        cache_dir: tar 本地缓存目录
        stream: True=HTTP 流式, False=下载缓存
        columns: 动作列名

    Returns:
        (T, D) numpy array, 失败返回 None
    """
    task_name = meta["task_name"]
    dataset_dir = meta["dataset_dir"]
    episode = meta["episode"]
    frame_indices = meta.get("frame_indices", [])

    if not frame_indices:
        return None

    tar_paths = tar_index.get((task_name, dataset_dir), [])
    if not tar_paths:
        return None

    target_frames = set(frame_indices)
    collected: Dict[int, np.ndarray] = {}

    for tar_path in tar_paths:
        if not target_frames:
            break
        try:
            if stream:
                url = _get_presigned_url(tar_path)
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                tar_file = tarfile.open(fileobj=io.BytesIO(resp.content))
            else:
                local_path = _download_tar(tar_path, cache_dir)
                tar_file = tarfile.open(local_path)

            with tar_file:
                for member in tar_file:
                    if not member.name.endswith(".meta.msgpack"):
                        continue
                    name = member.name
                    if not name.startswith("ep"):
                        continue
                    try:
                        parts = name.split("_fr")
                        ep = int(parts[0][2:])
                        fr = int(parts[1].split(".")[0])
                    except (ValueError, IndexError):
                        continue

                    if ep != episode or fr not in target_frames:
                        continue

                    raw = tar_file.extractfile(member)
                    if raw is None:
                        continue
                    data = msgpack.load(raw)

                    vec_parts = []
                    for col in columns:
                        val = data.get(col)
                        if val is None:
                            vec_parts = None
                            break
                        if isinstance(val, list) and len(val) > 0:
                            item = val[0]
                            if isinstance(item, list):
                                arr = np.asarray(item, dtype=np.float32).flatten()
                            else:
                                arr = np.array([float(item)], dtype=np.float32)
                            dim = COLUMN_DIMS.get(col, len(arr))
                            if len(arr) < dim:
                                arr = np.pad(arr, (0, dim - len(arr)))
                            elif len(arr) > dim:
                                arr = arr[:dim]
                            vec_parts.append(arr)
                        else:
                            vec_parts = None
                            break

                    if vec_parts is not None:
                        collected[fr] = np.concatenate(vec_parts).astype(np.float32)
                        target_frames.discard(fr)
        except Exception:
            continue

    if not collected:
        return None

    ordered = [collected[fi] for fi in frame_indices if fi in collected]
    if not ordered:
        return None
    return np.stack(ordered, axis=0)


# ══════════════════════════════════════════════════════
#  归一化
# ══════════════════════════════════════════════════════

def normalize_trajectory(traj: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "none" or traj is None:
        return traj
    if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
        traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
    if len(traj) < 2:
        return traj

    if method == "zscore":
        std = np.std(traj, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        mean = np.mean(traj, axis=0, keepdims=True)
        return (traj - mean) / std
    elif method == "minmax":
        t_min = np.min(traj, axis=0, keepdims=True)
        t_max = np.max(traj, axis=0, keepdims=True)
        denom = np.where(t_max - t_min < 1e-8, 1.0, t_max - t_min)
        return (traj - t_min) / denom
    return traj


# ══════════════════════════════════════════════════════
#  相似度计算
# ══════════════════════════════════════════════════════

def _build_weight_vector() -> np.ndarray:
    weights = []
    for col in ACTION_COLUMNS:
        dim = COLUMN_DIMS[col]
        w = PART_WEIGHTS.get(col, 1.0)
        weights.extend([w] * dim)
    return np.array(weights, dtype=np.float32)


def dtw_distance(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    weights: np.ndarray,
    window: int = 0,
) -> Tuple[float, int]:
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("inf"), 0

    diff = traj_a[:, None, :] - traj_b[None, :, :]
    weighted_diff = diff * weights[None, None, :]
    cost = np.sqrt(np.sum(weighted_diff * diff, axis=2))
    cost = np.nan_to_num(cost, nan=0.0, posinf=1e10, neginf=0.0)

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    if window <= 0:
        window = max(n, m)
    window = max(window, abs(n - m))

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window) + 1
        for j in range(j_start, j_end):
            local_cost = cost[i - 1, j - 1]
            dtw[i, j] = local_cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    path_len = 0
    i, j = n, m
    while i > 0 and j > 0:
        path_len += 1
        diag, up, left = dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]
        if diag <= up and diag <= left:
            i -= 1; j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    path_len += i + j

    return float(dtw[n, m]), path_len


def interpolated_euclidean(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    weights: np.ndarray,
    target_len: int = 100,
) -> float:
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("inf")

    t_common = np.linspace(0, 1, target_len)
    interp_a = np.array([np.interp(t_common, np.linspace(0, 1, n), traj_a[:, d]) for d in range(traj_a.shape[1])]).T
    interp_b = np.array([np.interp(t_common, np.linspace(0, 1, m), traj_b[:, d]) for d in range(traj_b.shape[1])]).T

    diff = interp_a - interp_b
    weighted_diff = diff * weights[None, :]
    frame_dists = np.sqrt(np.sum(weighted_diff * diff, axis=1))
    return float(np.mean(frame_dists))


def trajectory_similarity(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    normalize: str = "zscore",
    window_ratio: float = 0.2,
) -> Dict[str, float]:
    if traj_a is None or traj_b is None:
        return {"dtw_distance": float("inf"), "dtw_normalized": float("inf"),
                "interp_euclidean": float("inf"), "len_a": 0, "len_b": 0, "dim": 0}

    weights = _build_weight_vector()
    traj_a_norm = normalize_trajectory(traj_a, method=normalize)
    traj_b_norm = normalize_trajectory(traj_b, method=normalize)

    if traj_a_norm is None or traj_b_norm is None:
        return {"dtw_distance": float("inf"), "dtw_normalized": float("inf"),
                "interp_euclidean": float("inf"), "len_a": len(traj_a), "len_b": len(traj_b),
                "dim": traj_a.shape[1]}

    window = int(max(len(traj_a_norm), len(traj_b_norm)) * window_ratio)

    try:
        dtw_dist, path_len = dtw_distance(traj_a_norm, traj_b_norm, weights, window)
        dtw_norm = dtw_dist / path_len if path_len > 0 else float("inf")
    except Exception:
        dtw_dist, dtw_norm = float("inf"), float("inf")

    try:
        ie_dist = interpolated_euclidean(traj_a_norm, traj_b_norm, weights)
    except Exception:
        ie_dist = float("inf")

    return {
        "dtw_distance": round(dtw_dist, 6),
        "dtw_normalized": round(dtw_norm, 6),
        "interp_euclidean": round(ie_dist, 6),
        "len_a": len(traj_a),
        "len_b": len(traj_b),
        "dim": traj_a.shape[1],
    }


# ══════════════════════════════════════════════════════
#  并行计算
# ══════════════════════════════════════════════════════

_GLOBAL_TAR_INDEX: Dict[str, Dict] = {}
_GLOBAL_CACHE_DIR: str = ""
_GLOBAL_STREAM: bool = False


def _worker_init(tar_index: dict, cache_dir: str, stream: bool):
    global _GLOBAL_TAR_INDEX, _GLOBAL_CACHE_DIR, _GLOBAL_STREAM
    _GLOBAL_TAR_INDEX = tar_index
    _GLOBAL_CACHE_DIR = cache_dir
    _GLOBAL_STREAM = stream


def _compute_single_pair(pair: dict, normalize: str, window_ratio: float) -> dict:
    try:
        tar_index = _GLOBAL_TAR_INDEX
        cache_dir = _GLOBAL_CACHE_DIR
        stream = _GLOBAL_STREAM

        traj_a = extract_trajectory(pair["egotel_meta"], tar_index.get("egotel", {}), cache_dir, stream)
        traj_b = extract_trajectory(pair["egoudas_meta"], tar_index.get("egoudas", {}), cache_dir, stream)
        sim = trajectory_similarity(traj_a, traj_b, normalize=normalize, window_ratio=window_ratio)

        return {
            "pair_id": pair["pair_id"],
            "egotel_idx": pair["egotel_idx"],
            "egoudas_idx": pair["egoudas_idx"],
            "visual_similarity": pair.get("similarity", None),
            "action_dtw_distance": sim["dtw_distance"],
            "action_dtw_normalized": sim["dtw_normalized"],
            "action_interp_euclidean": sim["interp_euclidean"],
            "traj_len_egotel": sim["len_a"],
            "traj_len_egoudas": sim["len_b"],
            "action_dim": sim["dim"],
        }
    except Exception as e:
        return {
            "pair_id": pair.get("pair_id", -1),
            "egotel_idx": pair.get("egotel_idx", -1),
            "egoudas_idx": pair.get("egoudas_idx", -1),
            "visual_similarity": pair.get("similarity", None),
            "action_dtw_distance": float("inf"),
            "action_dtw_normalized": float("inf"),
            "action_interp_euclidean": float("inf"),
            "traj_len_egotel": 0,
            "traj_len_egoudas": 0,
            "action_dim": 0,
            "error": str(e)[:200],
        }


def compute_all_pairs(
    pairs: List[dict],
    tar_index: dict,
    cache_dir: str,
    stream: bool = False,
    normalize: str = "zscore",
    window_ratio: float = 0.2,
    n_workers: int = 1,
) -> List[dict]:
    total = len(pairs)
    results = [None] * total

    if n_workers <= 1:
        global _GLOBAL_TAR_INDEX, _GLOBAL_CACHE_DIR, _GLOBAL_STREAM
        _GLOBAL_TAR_INDEX = tar_index
        _GLOBAL_CACHE_DIR = cache_dir
        _GLOBAL_STREAM = stream

        for idx, pair in enumerate(tqdm(pairs, desc="Computing trajectory similarity")):
            try:
                results[idx] = _compute_single_pair(pair, normalize, window_ratio)
            except Exception as e:
                results[idx] = {"pair_id": pair.get("pair_id", idx), "error": str(e)[:200]}
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(tar_index, cache_dir, stream),
        ) as executor:
            futures = {
                executor.submit(_compute_single_pair, pair, normalize, window_ratio): idx
                for idx, pair in enumerate(pairs)
            }
            for future in tqdm(as_completed(futures), total=total, desc="Computing trajectory similarity"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"pair_id": pairs[idx].get("pair_id", idx), "error": str(e)[:200]}

    return results


# ══════════════════════════════════════════════════════
#  统计
# ══════════════════════════════════════════════════════

def summarize_results(results: List[dict]) -> dict:
    dtw_vals = [r["action_dtw_distance"] for r in results
                if r.get("action_dtw_distance", float("inf")) != float("inf")]
    dtw_norm_vals = [r["action_dtw_normalized"] for r in results
                     if r.get("action_dtw_normalized", float("inf")) != float("inf")]
    ie_vals = [r["action_interp_euclidean"] for r in results
               if r.get("action_interp_euclidean", float("inf")) != float("inf")]
    errors = sum(1 for r in results if "error" in r)

    summary = {
        "total_pairs": len(results),
        "valid_pairs": len(dtw_vals),
        "error_pairs": errors,
    }

    for name, vals in [("dtw_distance", dtw_vals), ("dtw_normalized", dtw_norm_vals),
                        ("interp_euclidean", ie_vals)]:
        if vals:
            arr = np.array(vals)
            for stat in ["mean", "std", "min", "max", "median"]:
                summary[f"{name}_{stat}"] = round(float(getattr(np, stat)(arr)), 6)
            summary[f"{name}_p25"] = round(float(np.percentile(arr, 25)), 6)
            summary[f"{name}_p75"] = round(float(np.percentile(arr, 75)), 6)

    vis_vals = [r["visual_similarity"] for r in results
                if r.get("visual_similarity") is not None
                and r.get("action_dtw_distance", float("inf")) != float("inf")]
    if vis_vals and dtw_vals:
        dtw_for_corr = [r["action_dtw_distance"] for r in results
                        if r.get("visual_similarity") is not None
                        and r.get("action_dtw_distance", float("inf")) != float("inf")]
        if len(dtw_for_corr) > 2:
            summary["visual_vs_dtw_correlation"] = round(float(np.corrcoef(vis_vals, dtw_for_corr)[0, 1]), 6)

    return summary


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Action Trajectory Similarity — webdataset 源")
    parser.add_argument("--pairs", required=True, help="pairs_all.json 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="tar 本地缓存目录")
    parser.add_argument("--stream", action="store_true", help="HTTP 流式读取 (零磁盘)")
    parser.add_argument("--max-pairs", type=int, default=0, help="最多计算多少对 (0=全部)")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数")
    parser.add_argument("--normalize", choices=["zscore", "minmax", "none"], default="zscore",
                        help="归一化方法")
    parser.add_argument("--window-ratio", type=float, default=0.2, help="DTW 窗口比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("  Action Trajectory Similarity Operator (webdataset)")
    print("=" * 60)
    print(f"  Pairs:  {args.pairs}")
    print(f"  Output: {args.output}")
    print(f"  Mode:   {'stream (HTTP)' if args.stream else f'cache ({args.cache_dir})'}")
    t0 = time.time()

    with open(args.pairs) as f:
        data = json.load(f)

    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
        meta_info = {k: v for k, v in data.items() if k != "pairs"}
    elif isinstance(data, list):
        pairs = data
        meta_info = {}
    else:
        print("Error: unknown pairs format")
        sys.exit(1)

    print(f"  Loaded: {len(pairs)} pairs")

    if args.max_pairs > 0 and args.max_pairs < len(pairs):
        np.random.seed(args.seed)
        indices = np.random.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]
        print(f"  Sampled: {len(pairs)} pairs (seed={args.seed})")

    # 构建 tar 索引
    print()
    datasets = set()
    for p in pairs:
        datasets.add(p.get("egotel_meta", {}).get("dataset", "egotel"))
        datasets.add(p.get("egoudas_meta", {}).get("dataset", "egoudas"))
    tar_index = build_all_tar_index(list(datasets))

    # 动作空间
    print(f"\n  Action space: {sum(COLUMN_DIMS.values())}D")
    for col in ACTION_COLUMNS:
        print(f"    {col}: {COLUMN_DIMS[col]}D (weight={PART_WEIGHTS[col]})")
    print(f"  Normalize: {args.normalize}  |  DTW window: {args.window_ratio}  |  Workers: {args.workers}")
    print()

    results = compute_all_pairs(
        pairs,
        tar_index=tar_index,
        cache_dir=args.cache_dir,
        stream=args.stream,
        normalize=args.normalize,
        window_ratio=args.window_ratio,
        n_workers=args.workers,
    )

    summary = summarize_results(results)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Total:   {summary['total_pairs']}")
    print(f"  Valid:   {summary['valid_pairs']}")
    print(f"  Errors:  {summary['error_pairs']}")
    print(f"  Time:    {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if summary['valid_pairs'] > 0:
        for metric in ["dtw_distance", "dtw_normalized", "interp_euclidean"]:
            if f"{metric}_mean" in summary:
                print(f"\n  {metric}:")
                print(f"    mean={summary[f'{metric}_mean']:.4f}  std={summary[f'{metric}_std']:.4f}")
                print(f"    min={summary[f'{metric}_min']:.4f}  max={summary[f'{metric}_max']:.4f}")
                print(f"    median={summary[f'{metric}_median']:.4f}  "
                      f"p25={summary[f'{metric}_p25']:.4f}  p75={summary[f'{metric}_p75']:.4f}")
        if "visual_vs_dtw_correlation" in summary:
            print(f"\n  Visual vs DTW correlation: {summary['visual_vs_dtw_correlation']:.4f}")

    output = {
        "config": {
            "pairs_file": args.pairs,
            "cache_dir": args.cache_dir,
            "stream": args.stream,
            "normalize": args.normalize,
            "window_ratio": args.window_ratio,
            "action_columns": ACTION_COLUMNS,
            "column_dims": COLUMN_DIMS,
            "part_weights": PART_WEIGHTS,
            "n_pairs": len(pairs),
            "n_workers": args.workers,
            "elapsed_seconds": round(elapsed, 1),
        },
        **meta_info,
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\n  Saved: {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
