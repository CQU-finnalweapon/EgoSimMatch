"""EgouDas × EgoTel 跨数据集 pair 匹配可视化。

从精确重叠的 task 中随机抽取 20 个 pair（每对一个 EgouDas segment + 一个 EgoTel segment），
用 Qwen3-VL-Embedding 编码后计算相似度，可视化结果。
"""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.egoudas_loader import EgouDasDataLoader, EgouDasSegment
from data.egotel_loader import EgoTelDataLoader, EgoTelSegment

QWEN3_VL_PATH = "/mnt/vepfs01/output/klayzhou/EgoSimMatch/models/Qwen3-VL-Embedding-8B"
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/udas_tel_pairs")
N_FRAMES = 10
N_PAIRS = 20
SEED = 42


# ── 采样帧 ─────────────────────────────────────────────────────────────────────

def sample_frames_from_video(
    video_dir: Path, episode_index: int, frame_start: int, frame_end: int, n: int = 10
) -> List[Image.Image]:
    """从视频中均匀采样 n 帧。"""
    cam_dir = video_dir / "chunk-000" / "cam_high"
    video_path = cam_dir / f"episode_{episode_index:06d}.mp4"
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = frame_end - frame_start + 1
    indices = (
        list(range(frame_start, frame_end + 1))
        if total <= n
        else [frame_start + int(i * (total - 1) / (n - 1)) for i in range(n)]
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ── 构建 pair 列表 ─────────────────────────────────────────────────────────────

def build_pairs(n_pairs: int = 20, seed: int = 42) -> List[Dict]:
    """从精确重叠的 task 中随机抽取 n_pairs 个 pair，加载图像帧。"""
    rng = random.Random(seed)

    print("Loading EgouDas index...")
    udas_loader = EgouDasDataLoader(split="eval")
    udas_index = udas_loader.get_task_to_segments()

    print("Loading EgoTel index...")
    tel_loader = EgoTelDataLoader(split="eval")
    tel_index = tel_loader.get_task_to_segments()

    overlap_tasks = sorted(set(udas_index.keys()) & set(tel_index.keys()))
    print(f"Exact overlap tasks: {len(overlap_tasks)}")

    # 随机选 n_pairs 个 task（每个 task 选 1 对）
    selected_tasks = rng.sample(overlap_tasks, min(n_pairs, len(overlap_tasks)))

    pairs = []
    for task in selected_tasks:
        # 从 EgouDas 随机选一个 segment
        udas_entry = rng.choice(udas_index[task])
        udas_session, udas_ep, udas_fs, udas_fe = udas_entry

        # 找对应的 session 目录
        udas_session_dir = None
        for date_dir in udas_loader.root.iterdir():
            if not date_dir.is_dir():
                continue
            candidate = date_dir / udas_session
            if candidate.exists():
                udas_session_dir = candidate
                break

        if udas_session_dir is None:
            continue

        udas_frames = sample_frames_from_video(
            udas_session_dir / "videos", udas_ep, udas_fs, udas_fe, N_FRAMES
        )
        if not udas_frames:
            continue

        # 从 EgoTel 随机选一个 segment
        tel_entry = rng.choice(tel_index[task])
        tel_task_dir, tel_dataset_dir, tel_ep, tel_fs, tel_fe = tel_entry

        tel_dataset_path = tel_loader.root / tel_loader.split / tel_task_dir / tel_dataset_dir
        tel_frames = sample_frames_from_video(
            tel_dataset_path / "videos", tel_ep, tel_fs, tel_fe, N_FRAMES
        )
        if not tel_frames:
            continue

        pairs.append({
            "task": task,
            "udas_session": udas_session,
            "udas_ep": udas_ep,
            "udas_frames": udas_frames,
            "tel_task_dir": tel_task_dir,
            "tel_ep": tel_ep,
            "tel_frames": tel_frames,
        })

        if len(pairs) >= n_pairs:
            break

    print(f"Built {len(pairs)} pairs")
    return pairs


# ── Embedding ──────────────────────────────────────────────────────────────────

def encode_pairs(pairs: List[Dict], device: str = "cuda:0", batch_size: int = 4) -> List[Dict]:
    """对每个 pair 的 ego 和 tele segment 分别编码。"""
    from sentence_transformers import SentenceTransformer

    print(f"Loading Qwen3-VL-Embedding...")
    model = SentenceTransformer(QWEN3_VL_PATH, trust_remote_code=True, device=device)

    # 构建所有 message（ego 和 tele 交替）
    messages = []
    for p in pairs:
        for side in ["udas", "tel"]:
            frames = p[f"{side}_frames"]
            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": p["task"]})
            messages.append({"role": "user", "content": content})

    print(f"Encoding {len(messages)} segments (batch_size={batch_size})...")
    embs = model.encode(messages, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)

    for i, p in enumerate(pairs):
        p["udas_emb"] = embs[i * 2]
        p["tel_emb"] = embs[i * 2 + 1]
        p["sim"] = float(np.dot(p["udas_emb"], p["tel_emb"]))

    pairs.sort(key=lambda x: -x["sim"])
    return pairs


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def wrap_text(text: str, max_chars: int = 55) -> List[str]:
    words = text.split()
    lines, cur = [], ""
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


def make_pair_image(pair: Dict, rank: int, frame_size=(180, 135)) -> Image.Image:
    """每个 pair 取10帧中的9帧（均匀间隔选9帧），排列为左边EgouDas 3×3，右边EgoTel 3×3。"""
    fw, fh = frame_size
    pad = 5
    border = 3
    header_h = 32
    footer_h = 52

    # 从10帧中均匀选9帧
    def pick_9(frames):
        total = len(frames)
        if total >= 10:
            # 从10帧里均匀选9帧（跳过第5帧，保留头尾）
            indices = [int(i * (total - 1) / 8) for i in range(9)]
        else:
            indices = list(range(min(9, total)))
        return [frames[i] for i in indices if i < len(frames)]

    udas_frames = pick_9(pair["udas_frames"])
    tel_frames = pick_9(pair["tel_frames"])

    # 3×3 网格尺寸
    grid_w = fw * 3 + pad * 2
    grid_h = fh * 3 + pad * 2

    # 整体布局：[header] [left_grid | divider | right_grid] [footer]
    divider_w = 10
    total_w = grid_w * 2 + divider_w + border * 4
    total_h = header_h + grid_h + footer_h

    bg_color = (20, 22, 28)
    img = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font_title = font_info = font_label = ImageFont.load_default()

    # ── Header ──
    draw.rectangle([(0, 0), (total_w, header_h - 1)], fill=(30, 33, 42))
    # rank + sim
    sim_color = (
        (80, 255, 120) if pair['sim'] > 0.7 else
        (255, 220, 50) if pair['sim'] > 0.5 else
        (255, 100, 80)
    )
    draw.text((8, 8), f"#{rank:02d}", fill=(160, 160, 180), font=font_title)
    draw.text((42, 8), f"Sim: {pair['sim']:.4f}", fill=sim_color, font=font_title)

    # dataset labels in header
    left_cx = border + grid_w // 2
    right_cx = border * 2 + grid_w + divider_w + grid_w // 2
    draw.text((left_cx - 35, 9), "EgouDas (ego)", fill=(100, 180, 255), font=font_label)
    draw.text((right_cx - 35, 9), "EgoTel (tele)", fill=(80, 220, 140), font=font_label)

    # ── 3×3 grid helper ──
    def paste_3x3(frames_9, x_off, y_off, border_color):
        # 网格外边框
        draw.rectangle(
            [(x_off - border, y_off - border),
             (x_off + grid_w + border, y_off + grid_h + border)],
            outline=border_color, width=border
        )
        for idx, frame in enumerate(frames_9):
            row = idx // 3
            col = idx % 3
            x = x_off + col * (fw + pad)
            y = y_off + row * (fh + pad)
            # 帧间隔填充
            if col < 2:
                draw.rectangle([(x + fw, y_off), (x + fw + pad, y_off + grid_h)], fill=(12, 12, 18))
            if row < 2:
                draw.rectangle([(x_off, y + fh), (x_off + grid_w, y + fh + pad)], fill=(12, 12, 18))
            thumb = frame.resize(frame_size, Image.LANCZOS)
            img.paste(thumb, (x, y))

    y_grid = header_h + 2
    left_x = border
    right_x = border * 2 + grid_w + divider_w

    paste_3x3(udas_frames, left_x, y_grid, (70, 130, 200))
    paste_3x3(tel_frames, right_x, y_grid, (60, 180, 100))

    # ── 中间分隔 ──
    div_x = border + grid_w + border + 1
    draw.rectangle([(div_x, header_h), (div_x + divider_w - 3, header_h + grid_h + 4)], fill=(14, 16, 22))
    draw.line([(div_x + divider_w // 2, header_h + 4),
               (div_x + divider_w // 2, header_h + grid_h)],
              fill=(55, 60, 75), width=1)

    # ── Footer ──
    fy = header_h + grid_h + 6
    task_lines = wrap_text(pair["task"], max_chars=60)
    y_text = fy
    for line in task_lines[:2]:
        draw.text((8, y_text), line, fill=(170, 170, 185), font=font_info)
        y_text += 13
    ep_l = f"ep{pair['udas_ep']:05d}"
    ep_r = f"ep{pair['tel_ep']:05d} · {pair['tel_task_dir']}"
    draw.text((8, fy + 28), ep_l, fill=(80, 110, 160), font=font_info)
    draw.text((right_x, fy + 28), ep_r, fill=(60, 140, 90), font=font_info)

    return img


def make_grid_9(pair_images: List[Image.Image], title: str = "") -> Image.Image:
    """将 9 张 pair 图拼成 3×3 九宫格。"""
    assert len(pair_images) == 9
    cell_w = pair_images[0].width
    cell_h = pair_images[0].height
    gap = 8
    title_h = 36 if title else 0
    grid_w = cell_w * 3 + gap * 4
    grid_h = cell_h * 3 + gap * 4 + title_h

    grid = Image.new("RGB", (grid_w, grid_h), (12, 12, 12))
    draw = ImageDraw.Draw(grid)

    if title:
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font_title = ImageFont.load_default()
        draw.text((gap, 8), title, fill=(220, 220, 220), font=font_title)

    for idx, img in enumerate(pair_images):
        row = idx // 3
        col = idx % 3
        x = gap + col * (cell_w + gap)
        y = title_h + gap + row * (cell_h + gap)
        grid.paste(img, (x, y))

    return grid


def visualize(pairs: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取前 18 对（两个九宫格），每格取 10 张图里选 9 张（去掉最后一张）
    use_pairs = pairs[:18]
    pair_images = []

    for rank, pair in enumerate(use_pairs, 1):
        img = make_pair_image(pair, rank)
        path = output_dir / f"pair_{rank:02d}_sim{pair['sim']:.3f}.png"
        img.save(path)
        pair_images.append(img)
        print(f"  #{rank:2d} sim={pair['sim']:.4f} | {pair['task'][:60]}")

    # 保证至少 18 张
    while len(pair_images) < 18:
        pair_images.append(Image.new("RGB", pair_images[0].size, (20, 20, 20)))

    # 两个九宫格
    grid_left = make_grid_9(pair_images[:9], title="Top 1–9  (sorted by similarity ↓)")
    grid_right = make_grid_9(pair_images[9:18], title="Top 10–18")

    # 左右拼合
    gap = 16
    total_w = grid_left.width + gap + grid_right.width
    total_h = max(grid_left.height, grid_right.height)
    canvas = Image.new("RGB", (total_w, total_h), (8, 8, 8))
    canvas.paste(grid_left, (0, 0))
    canvas.paste(grid_right, (grid_left.width + gap, 0))

    overview_path = output_dir / "overview_dual_grid.png"
    canvas.save(overview_path)
    print(f"\nSaved dual 3×3 grid → {overview_path}")
    print(f"All individual pairs saved to {output_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=== Step 1: Build pairs ===")
    pairs = build_pairs(n_pairs=args.n_pairs, seed=args.seed)

    print(f"\n=== Step 2: Encode {len(pairs) * 2} segments ===")
    pairs = encode_pairs(pairs, device=args.device, batch_size=args.batch_size)

    print(f"\n=== Step 3: Visualize ===")
    visualize(pairs, OUTPUT_DIR)

    print("\nDone!")
