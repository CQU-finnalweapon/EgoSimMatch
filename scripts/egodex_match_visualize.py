"""EgoDex 内部相似度匹配 + 可视化。

流程：
  1. 从 EgoDex eval 采样 N 个 episode，按 task_index 切分 segment
  2. 每个 segment 均匀采样 10 帧图像 + task 文本
  3. 用 Qwen3-VL-Embedding-8B 编码（图像 + 文本联合）
  4. 计算所有 segment 对的相似度，找出最相似的 Top-K 对
  5. 可视化：左右各一个 segment 的 10 帧，下方标注 task 文本
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

EGODEX_EVAL = Path(
    "/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/data/"
    "egocentric_data/EgoDex_lerobot/EgoDex_lerobot_metis_gripper_arm_v1_retask/eval"
)
QWEN3_VL_PATH = "/mnt/vepfs01/output/klayzhou/EgoSimMatch/models/Qwen3-VL-Embedding-8B"
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/egodex_matching")


@dataclass
class EgoDexSegment:
    episode_index: int
    task_index: int
    task: str
    frame_start: int
    frame_end: int
    frames: List[Image.Image]   # 10 帧图像
    embedding: np.ndarray = None


# ── 1. 数据加载 ────────────────────────────────────────────────────────────────

def load_tasks(base: Path) -> Dict[int, str]:
    tasks = {}
    with open(base / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_index"]] = d["task"]
    return tasks


def find_segments(task_indices: List[int], frame_indices: List[int]) -> List[Tuple[int, int, int]]:
    """返回 [(task_index, frame_start, frame_end)]。"""
    if not task_indices:
        return []
    segments = []
    cur_task = task_indices[0]
    cur_start = frame_indices[0]
    cur_end = frame_indices[0]
    for ti, fi in zip(task_indices[1:], frame_indices[1:]):
        if ti == cur_task:
            cur_end = fi
        else:
            segments.append((cur_task, cur_start, cur_end))
            cur_task = ti
            cur_start = fi
            cur_end = fi
    segments.append((cur_task, cur_start, cur_end))
    return segments


def sample_frames(cap: cv2.VideoCapture, frame_start: int, frame_end: int, n: int = 10) -> List[Image.Image]:
    """从视频中均匀采样 n 帧。"""
    total = frame_end - frame_start + 1
    if total <= n:
        indices = list(range(frame_start, frame_end + 1))
    else:
        indices = [frame_start + int(i * (total - 1) / (n - 1)) for i in range(n)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return frames


def load_segments(
    base: Path,
    tasks: Dict[int, str],
    max_episodes: int = 200,
    min_frames: int = 20,
    n_sample_frames: int = 10,
) -> List[EgoDexSegment]:
    """加载 EgoDex eval 的 segment 列表。"""
    segments = []
    parquet_dir = base / "data" / "chunk-000"
    video_dir = base / "videos" / "chunk-000" / "cam_high"

    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))[:max_episodes]
    print(f"Loading {len(parquet_files)} episodes...")

    for pf in parquet_files:
        ep_idx = int(pf.stem.split("_")[1])
        video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
        if not video_path.exists():
            continue

        try:
            table = pq.read_table(str(pf), columns=["task_index", "frame_index"])
        except Exception:
            continue

        task_indices = table["task_index"].to_pylist()
        frame_indices = table["frame_index"].to_pylist()
        segs = find_segments(task_indices, frame_indices)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        for task_idx, frame_start, frame_end in segs:
            if frame_end - frame_start + 1 < min_frames:
                continue
            task = tasks.get(task_idx, "")
            if not task:
                continue

            frames = sample_frames(cap, frame_start, frame_end, n_sample_frames)
            if len(frames) < 3:
                continue

            segments.append(EgoDexSegment(
                episode_index=ep_idx,
                task_index=task_idx,
                task=task,
                frame_start=frame_start,
                frame_end=frame_end,
                frames=frames,
            ))

        cap.release()

    print(f"Loaded {len(segments)} segments from {len(parquet_files)} episodes")
    return segments


# ── 2. Embedding ───────────────────────────────────────────────────────────────

def encode_segments(
    segments: List[EgoDexSegment],
    device: str = "cuda:0",
    batch_size: int = 4,
) -> List[EgoDexSegment]:
    """用 Qwen3-VL-Embedding 批量编码 segment（图像 + 文本）。

    使用 sentence-transformers 的 message 格式，支持多图像 + 文本联合编码。
    batch_size=4 在 80GB GPU 上约 16s/batch，585 segments ≈ 10 分钟。
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading Qwen3-VL-Embedding from {QWEN3_VL_PATH}...")
    model = SentenceTransformer(QWEN3_VL_PATH, trust_remote_code=True, device=device)
    print(f"Model loaded. Encoding {len(segments)} segments (batch_size={batch_size})...")

    # 构建 message 列表：每个 segment 是一条多图像 + 文本的 message
    messages = []
    for seg in segments:
        content = [{"type": "image", "image": img} for img in seg.frames]
        content.append({"type": "text", "text": seg.task})
        messages.append({"role": "user", "content": content})

    # 批量编码
    embs = model.encode(
        messages,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    for seg, emb in zip(segments, embs):
        seg.embedding = emb

    return segments


# ── 3. 相似度计算 ──────────────────────────────────────────────────────────────

def find_top_k_pairs(
    segments: List[EgoDexSegment], top_k: int = 10, exclude_same_episode: bool = True
) -> List[Tuple[int, int, float]]:
    """计算所有 segment 对的相似度，返回 Top-K 对 [(i, j, sim)]。"""
    embs = np.array([s.embedding for s in segments])
    sim_matrix = embs @ embs.T  # (N, N)

    # 排除自身和同 episode 的对
    pairs = []
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            if exclude_same_episode and segments[i].episode_index == segments[j].episode_index:
                continue
            pairs.append((i, j, float(sim_matrix[i, j])))

    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


# ── 4. 可视化 ──────────────────────────────────────────────────────────────────

def wrap_text(text: str, max_chars: int = 60) -> List[str]:
    """简单换行。"""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_pair_image(
    seg_a: EgoDexSegment,
    seg_b: EgoDexSegment,
    sim: float,
    frame_size: Tuple[int, int] = (160, 120),
    n_frames: int = 10,
) -> Image.Image:
    """生成一张对比图：左边 seg_a 的帧，右边 seg_b 的帧，下方标注 task。"""
    fw, fh = frame_size
    padding = 8
    text_height = 80
    total_w = n_frames * fw + (n_frames - 1) * padding
    total_h = fh + padding + text_height

    img = Image.new("RGB", (total_w * 2 + padding * 3, total_h + 40), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = font_small = font_title = ImageFont.load_default()

    # 标题行：相似度
    draw.text((total_w + padding, 4), f"Similarity: {sim:.4f}", fill=(255, 220, 50), font=font_title)

    def paste_frames(seg: EgoDexSegment, x_offset: int):
        frames = seg.frames[:n_frames]
        for k, frame in enumerate(frames):
            frame_resized = frame.resize(frame_size, Image.LANCZOS)
            x = x_offset + k * (fw + padding)
            y = 40
            img.paste(frame_resized, (x, y))

        # task 文本
        lines = wrap_text(seg.task, max_chars=total_w // 7)
        y_text = 40 + fh + padding
        for line in lines[:4]:
            draw.text((x_offset, y_text), line, fill=(200, 200, 200), font=font_small)
            y_text += 14

        # episode 信息
        info = f"ep{seg.episode_index:05d} | frames {seg.frame_start}-{seg.frame_end}"
        draw.text((x_offset, 40 + fh + padding + 60), info, fill=(120, 120, 120), font=font_small)

    paste_frames(seg_a, padding)
    paste_frames(seg_b, total_w + padding * 2)

    # 中间分隔线
    draw.line([(total_w + padding, 40), (total_w + padding, total_h + 40)], fill=(80, 80, 80), width=2)

    return img


def visualize_top_pairs(
    segments: List[EgoDexSegment],
    pairs: List[Tuple[int, int, float]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成每对的对比图
    pair_images = []
    for rank, (i, j, sim) in enumerate(pairs):
        img = make_pair_image(segments[i], segments[j], sim)
        pair_images.append(img)
        img.save(output_dir / f"pair_{rank+1:02d}_sim{sim:.3f}.png")
        print(f"  Pair {rank+1}: sim={sim:.4f}")
        print(f"    A: ep{segments[i].episode_index} | {segments[i].task[:60]}")
        print(f"    B: ep{segments[j].episode_index} | {segments[j].task[:60]}")

    # 合并成一张总览图
    if pair_images:
        total_h = sum(img.height for img in pair_images) + len(pair_images) * 4
        total_w = max(img.width for img in pair_images)
        overview = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        y = 0
        for img in pair_images:
            overview.paste(img, (0, y))
            y += img.height + 4
        overview.save(output_dir / "overview_top10.png")
        print(f"\nSaved overview to {output_dir / 'overview_top10.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for encoding (default: 16, larger=faster but more VRAM)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load EgoDex segments ===")
    tasks = load_tasks(EGODEX_EVAL)
    segments = load_segments(
        EGODEX_EVAL, tasks,
        max_episodes=args.max_episodes,
        n_sample_frames=args.n_frames,
    )

    print(f"\n=== Step 2: Encode {len(segments)} segments with Qwen3-VL-Embedding ===")
    segments = encode_segments(segments, device=args.device, batch_size=args.batch_size)

    print(f"\n=== Step 3: Find Top-{args.top_k} similar pairs ===")
    pairs = find_top_k_pairs(segments, top_k=args.top_k)

    print(f"\n=== Step 4: Visualize ===")
    visualize_top_pairs(segments, pairs, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
