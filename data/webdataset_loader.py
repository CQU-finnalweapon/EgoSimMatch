"""
WebDataset Loader — 从 TOS 读取 webdataset tar 格式数据，用于 Embedding 构建。

数据路径（TOS）:
  EgoTel:
    tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/EgoTel_aligned/raw_0521_anno_v2/
      eval/task_{id}/{dataset_dir}/webdataset/dataset-*.tar
  EgouDas:
    tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/ego_uDas/0513_anno_v2/
      eval/task_{id}/{dataset_dir}/webdataset/dataset-*.tar

Tar 内部格式:
  ep{episode:06d}_fr{frame:06d}.cam_high.jpg         # 主摄像头帧
  ep{episode:06d}_fr{frame:06d}.cam_left_wrist.jpg   # 左手腕摄像头
  ep{episode:06d}_fr{frame:06d}.cam_right_wrist.jpg  # 右手腕摄像头
  ep{episode:06d}_fr{frame:06d}.meta.msgpack         # 元数据

Segment = 同一 (episode_index, task_index) 的所有帧。
每个 segment 对应一个动作片段，task 文本来自 msgpack 的 "task" 字段。

读取模式:
  1. CACHE 模式: 下载 tar 到本地缓存再读取（适合有磁盘空间的环境）
  2. STREAM 模式: 通过 presigned URL + HTTP 流式读取到内存（零磁盘占用）
     使用 --use_webdataset=stream 或设置 stream=True
"""

import io
import msgpack
import os
import shutil
import subprocess
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
from PIL import Image

# ── 配置 ──────────────────────────────────────────────

# 只使用 cam_high（与旧版 embedding 脚本保持一致）
CAMERAS = ["cam_high"]

DATASET_ROOTS = {
    "egotel": "tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/EgoTel_aligned/raw_0521_anno_v2",
    "egoudas": "tos://ai-dev/moz-datasets/pretrain-v2/egocentric_data/ego_uDas/0513_anno_v2",
}

DEFAULT_CACHE_DIR = "/tmp/webdataset_cache"
TOSUTIL_CP_TIMEOUT = 180  # 单个 tar 下载超时（秒）
PRESIGNED_URL_CACHE = {}  # 缓存 presigned URL，避免重复调用 tosutil

# tosutil 二进制路径 — 可手动指定，默认在 PATH 中查找
TOSUTIL_PATH: str = "tosutil"

# tosutil 已知候选路径（当 PATH 中找不到时自动搜索）
_TOSUTIL_CANDIDATES = [
    "/mnt/vepfs01/output/klayzhou/trainning/mozbrain/tosutil",
    "/usr/local/bin/tosutil",
    "/usr/bin/tosutil",
]


# ── TOS 工具函数 ──────────────────────────────────────


def _find_tosutil() -> str:
    """查找 tosutil 二进制路径。优先返回已知候选路径，最后用 shutil.which。"""
    global TOSUTIL_PATH
    if TOSUTIL_PATH != "tosutil" and os.path.isfile(TOSUTIL_PATH):
        return TOSUTIL_PATH  # 已手动设置
    for path in _TOSUTIL_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            TOSUTIL_PATH = path
            return path
    found = shutil.which("tosutil")
    if found:
        TOSUTIL_PATH = found
        return found
    raise RuntimeError(
        "tosutil 未安装。请在当前机器上安装:\n"
        "  1. 从开发机复制二进制:\n"
        "     rsync -avP <开发机IP>:/usr/local/bin/tosutil /usr/local/bin/tosutil\n"
        "     chmod +x /usr/local/bin/tosutil\n"
        "  2. 复制或创建配置文件 ~/.tosutilconfig\n"
        "     配置内容可从开发机的 ~/.tosutilconfig 获取"
    )


def _tosutil_ls(tos_path: str, limit: int = 5000) -> List[str]:
    """运行 tosutil ls，返回对象路径列表。"""
    tosutil = _find_tosutil()
    cmd = [tosutil, "ls", tos_path, f"-limit={limit}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        stderr_short = result.stderr[:200] if result.stderr else "(empty)"
        raise RuntimeError(f"tosutil ls 失败 (code={result.returncode}): {stderr_short}")
    paths = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("tos://") and not line.endswith("/"):
            # 路径独立一行（无空格开头），或带元信息
            paths.append(line.split()[0] if " " in line else line)
    return paths


# ── Presigned URL + HTTP 流式读取 ────────────────────


def get_presigned_url(tos_path: str, validity: str = "7d") -> str:
    """生成 TOS 对象的 presigned URL（有效期内可重复使用）。"""
    if tos_path in PRESIGNED_URL_CACHE:
        return PRESIGNED_URL_CACHE[tos_path]

    tosutil = _find_tosutil()
    result = subprocess.run(
        [tosutil, "presign", tos_path, f"-vp={validity}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"presign 失败 {tos_path}: {result.stderr[:200]}")

    url = result.stdout.strip()
    PRESIGNED_URL_CACHE[tos_path] = url
    return url


def read_tar_via_http(tos_path: str) -> bytes:
    """通过 presigned URL + HTTP GET 读取整个 tar 文件到内存。"""
    url = get_presigned_url(tos_path)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.content


def extract_segments_from_tar_bytes(
    tar_bytes: bytes,
    task_name: str,
    dataset_dir: str,
    dataset: str,
    cameras: Optional[List[str]] = None,
) -> List[dict]:
    """从内存中的 tar 字节数据提取所有 segment。"""
    return _extract_segments_from_tar_inner(
        io.BytesIO(tar_bytes), task_name, dataset_dir, dataset, cameras
    )


def _extract_segments_from_tar_inner(
    source, task_name: str, dataset_dir: str, dataset: str,
    cameras: Optional[List[str]] = None,
) -> List[dict]:
    """内部实现：从 tarfile 兼容的源提取 segments。"""
    cameras = cameras or CAMERAS
    frame_to_seg: Dict[Tuple[int, int], dict] = {}

    with tarfile.open(fileobj=source) as tar:
        for member in tar:
            if not member.name.endswith(".meta.msgpack"):
                continue
            meta = msgpack.load(tar.extractfile(member))
            ep = meta.get("episode_index")
            fr = meta.get("frame_index")
            if ep is None or fr is None:
                continue
            frame_to_seg[(ep, fr)] = {
                "task_index": meta.get("task_index"),
                "task": meta.get("task", ""),
            }

        segments: Dict[Tuple[int, int], dict] = defaultdict(lambda: {
            "task": "", "task_name": task_name, "dataset_dir": dataset_dir,
            "episode": None, "task_index": None,
            "frame_indices": set(),
            "cam_images": defaultdict(list),
            "dataset": dataset,
        })

        for member in tar:
            if not member.name.endswith(".jpg"):
                continue
            name_no_ext = member.name.rsplit(".", 2)
            if len(name_no_ext) < 3:
                continue
            base, cam, _ = name_no_ext
            if cam not in cameras:
                continue

            key = parse_frame_key(base)
            if key is None:
                continue
            ep, fr = key

            seg_info = frame_to_seg.get((ep, fr))
            if seg_info is None:
                continue

            seg_key = (ep, seg_info["task_index"])
            seg = segments[seg_key]
            seg["task"] = seg_info["task"] or seg["task"]
            seg["task_index"] = seg_info["task_index"]
            seg["episode"] = ep
            seg["frame_indices"].add(fr)

            img = Image.open(tar.extractfile(member))
            seg["cam_images"][cam].append((fr, img.convert("RGB")))

    result = []
    for seg_key, seg in segments.items():
        sorted_images = {}
        for cam in cameras:
            if cam in seg["cam_images"]:
                seg["cam_images"][cam].sort(key=lambda x: x[0])
                sorted_images[cam] = [img for _, img in seg["cam_images"][cam]]
        sorted_frames = sorted(seg["frame_indices"])
        result.append({
            "task": seg["task"], "task_name": seg["task_name"],
            "dataset_dir": seg["dataset_dir"], "episode": seg["episode"],
            "task_index": seg["task_index"], "frame_indices": sorted_frames,
            "cam_images": sorted_images, "dataset": seg["dataset"],
        })

    return result


# ── CACHE 模式 ───────────────────────────────────────


def download_tar(tos_path: str, cache_dir: str) -> str:
    """从 TOS 下载 tar 到本地缓存。返回本地路径。"""
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = tos_path.replace("tos://", "").replace("/", "_")
    local_path = os.path.join(cache_dir, safe_name)

    # 如果已缓存且文件有效
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        # 快速验证是否为有效 tar
        try:
            with tarfile.open(local_path) as _:
                return local_path
        except (tarfile.ReadError, tarfile.TarError):
            print(f"  缓存文件损坏，重新下载: {safe_name[:60]}...")
            os.remove(local_path)

    # 最多重试 2 次
    tosutil = _find_tosutil()
    for attempt in range(3):
        result = subprocess.run(
            [tosutil, "cp", tos_path, local_path, "-p", "4"],
            capture_output=True, text=True, timeout=TOSUTIL_CP_TIMEOUT,
        )
        if result.returncode == 0:
            return local_path
        # 下载失败，清理后重试
        if os.path.exists(local_path):
            os.remove(local_path)
        if attempt < 2:
            print(f"  下载重试 {attempt+1}/3: {safe_name[:60]}...")
            time.sleep(2)

    raise RuntimeError(f"下载失败（已重试3次）{tos_path}")
    return local_path


def cleanup_cache(cache_dir: str, max_size_gb: int = 50):
    """清理缓存目录，超过 max_size_gb 时删除最旧的文件。"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return
    files = sorted(cache_path.iterdir(), key=lambda p: p.stat().st_mtime)
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    while total_size > max_size_gb * 1024 ** 3 and files:
        oldest = files.pop(0)
        if oldest.is_file():
            total_size -= oldest.stat().st_size
            oldest.unlink()


# ── Tar 解析 ──────────────────────────────────────────


def parse_frame_key(filename: str) -> Optional[Tuple[int, int]]:
    """从文件名解析 (episode, frame)。e.g. 'ep000006_fr000332' -> (6, 332)"""
    if not filename.startswith("ep"):
        return None
    parts = filename.split("_fr")
    if len(parts) != 2:
        return None
    try:
        episode = int(parts[0][2:])  # "ep000006" -> 6
        frame = int(parts[1])        # "000332" -> 332
        return episode, frame
    except ValueError:
        return None


def extract_segments_from_tar(
    local_tar_path: str,
    task_name: str,
    dataset_dir: str,
    dataset: str,
    cameras: Optional[List[str]] = None,
) -> List[dict]:
    """从本地 tar 文件路径提取所有 segment（委托给内部实现）。"""
    with open(local_tar_path, "rb") as f:
        return _extract_segments_from_tar_inner(
            io.BytesIO(f.read()), task_name, dataset_dir, dataset, cameras
        )


def list_webdataset_tars(tos_root: str) -> List[str]:
    """列出 TOS 根路径下所有 webdataset tar 文件。"""
    all_objects = _tosutil_ls(tos_root + "/eval/", limit=10000)
    # 只保留 webdataset/ 目录下的 .tar 文件
    tars = [p for p in all_objects if "/webdataset/" in p and p.endswith(".tar")]
    return tars


def collect_segments_from_tars(
    tar_paths: List[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    cameras: Optional[List[str]] = None,
    dataset: Optional[str] = None,
    on_tar_complete: Optional[Callable] = None,
) -> List[dict]:
    """
    从多个 tar 文件中收集所有 segment（依次下载 -> 解析 -> 缓存管理）。

    Args:
        tar_paths: TOS tar 路径列表
        cache_dir: 本地缓存目录
        cameras: 要加载的摄像头列表
        dataset: 数据集名称（自动推断）
        on_tar_complete: 每个 tar 处理完毕后的回调（用于进度追踪）

    Returns:
        所有 segment 的元信息列表
    """
    cameras = cameras or CAMERAS
    all_segments = []

    for tar_path in tar_paths:
        # 解析路径获取 task_name 和 dataset_dir
        parts = tar_path.split("/")
        try:
            wds_idx = parts.index("webdataset")
            ds_dir = parts[wds_idx - 1]
            t_name = parts[wds_idx - 2]
        except (ValueError, IndexError):
            continue

        inferred_dataset = dataset or ("egotel" if "EgoTel" in tar_path else "egoudas")

        # 下载 tar（如已缓存则跳过）
        local_path = download_tar(tar_path, cache_dir)

        # 提取 segments
        segments = extract_segments_from_tar(
            local_path, t_name, ds_dir, inferred_dataset, cameras
        )
        all_segments.extend(segments)

        if on_tar_complete:
            on_tar_complete(tar_path, len(segments))

    return all_segments


def load_frames_for_segments(
    segments: List[dict],
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_workers: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """
    按 TOS tar 文件分组加载 segment 的帧图像。

    策略:
      1. 将 segments 按 tar_tos_path（需手动补充）分组
      2. 每组下载对应的 tar
      3. 提取各 segment 的帧图像

    注意：此函数假设 segments 包含 "tar_tos_path" 字段。
    如果 segments 来自 collect_segments_from_tars，则不会包含该字段，
    因为 extract_segments_from_tar 当时就已经提取了图像。

    这里提供一个替代方案：直接从已有 cam_images 的 segments 构建消息。
    """
    valid_segments = []
    messages = []

    for seg in segments:
        if "cam_images" not in seg or not seg["cam_images"]:
            continue

        # 取 cam_high 的图像（保持与旧版一致）
        cam_high_frames = seg["cam_images"].get("cam_high", [])
        if not cam_high_frames:
            continue

        # 构建 Qwen-VL message
        content = []
        for img in cam_high_frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": seg["task"]})
        message = {"role": "user", "content": content}

        valid_segments.append(seg)
        messages.append(message)

    return valid_segments, messages


def load_all_frames_by_tar(
    seg_metas: List[dict],
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_workers: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """
    从 segment 元信息加载帧图像（兼容旧版接口）。

    输入的 seg_metas 需要包含 tar_tos_path 字段。
    按 tar 文件分组，每个 tar 只下载一次。

    Args:
        seg_metas: [{"task", "task_name", "dataset_dir", "episode", "task_index",
                     "frame_indices", "tar_tos_path", "dataset"}, ...]
        cache_dir: 缓存目录
        num_workers: 并行下载线程数

    Returns:
        (valid_metas, messages) 保持原序
    """
    # 按 tar 分组
    tar_groups = defaultdict(list)
    for i, meta in enumerate(seg_metas):
        tar_path = meta.get("tar_tos_path", "")
        if tar_path:
            tar_groups[tar_path].append((i, meta))

    results: Dict[int, Tuple[dict, list]] = {}  # seg_idx -> (meta, [cam_images])

    def _process_tar(item):
        """处理一个 tar 文件。"""
        tar_path, seg_list = item
        local_path = download_tar(tar_path, cache_dir)
        task_name = seg_list[0][1].get("task_name", "")
        dataset_dir = seg_list[0][1].get("dataset_dir", "")
        dataset = seg_list[0][1].get("dataset", "egotel")

        # 提取所有 segments
        tar_segments = extract_segments_from_tar(local_path, task_name, dataset_dir, dataset)

        # 建立 (episode, task_index) -> cam_images 索引
        seg_index = {}
        for ts in tar_segments:
            key = (ts["episode"], ts["task_index"])
            seg_index[key] = ts["cam_images"]

        local_results = {}
        for seg_idx, meta in seg_list:
            key = (meta["episode"], meta["task_index"])
            cam_images = seg_index.get(key, {})
            if cam_images:
                local_results[seg_idx] = (meta, cam_images)
        return local_results

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_tar, item): item[0] for item in tar_groups.items()}
        for future in as_completed(futures):
            results.update(future.result())

    # 按原序排列
    valid_metas = []
    messages = []
    for i in range(len(seg_metas)):
        if i in results:
            meta, cam_images = results[i]
            cam_high_frames = cam_images.get("cam_high", [])
            if cam_high_frames:
                content = []
                for img in cam_high_frames:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": meta["task"]})
                messages.append({"role": "user", "content": content})
                valid_metas.append(meta)

    return valid_metas, messages
