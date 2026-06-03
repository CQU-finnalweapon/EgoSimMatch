"""EgouDas × EgoTel 交叉匹配：计算所有 18×18 组合的相似度，找出最高匹配的配对。"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.egoudas_loader import EgouDasDataLoader
from data.egotel_loader import EgoTelDataLoader

QWEN3_VL_PATH = "/mnt/vepfs01/output/klayzhou/EgoSimMatch/models/Qwen3-VL-Embedding-8B"
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/udas_tel_cross")
N_FRAMES = 10
SEED = 42


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


def build_segments(n_tasks: int = 18, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """加载 18 个 task 的 EgouDas 和 EgoTel segments（用 seed=42 复现之前的选择）。"""
    rng = random.Random(seed)

    print("Loading EgouDas index...")
    udas_loader = EgouDasDataLoader(split="eval")
    udas_index = udas_loader.get_task_to_segments()

    print("Loading EgoTel index...")
    tel_loader = EgoTelDataLoader(split="eval")
    tel_index = tel_loader.get_task_to_segments()

    overlap_tasks = sorted(set(udas_index.keys()) & set(tel_index.keys()))
    print(f"Exact overlap tasks: {len(overlap_tasks)}")

    # 复现之前的 20 个 task 选择，取前 18 个
    selected_tasks = rng.sample(overlap_tasks, min(20, len(overlap_tasks)))[:n_tasks]

    udas_segs, tel_segs = [], []

    for task_idx, task in enumerate(selected_tasks):
        # EgouDas segment
        udas_entry = rng.choice(udas_index[task])
        udas_session, udas_ep, udas_fs, udas_fe = udas_entry

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

        udas_segs.append({
            "task": task,
            "dataset": "EgouDas",
            "session": udas_session,
            "ep": udas_ep,
            "frames": udas_frames,
        })

        # EgoTel segment
        tel_entry = rng.choice(tel_index[task])
        tel_task_dir, tel_dataset_dir, tel_ep, tel_fs, tel_fe = tel_entry

        tel_dataset_path = tel_loader.root / tel_loader.split / tel_task_dir / tel_dataset_dir
        tel_frames = sample_frames_from_video(
            tel_dataset_path / "videos", tel_ep, tel_fs, tel_fe, N_FRAMES
        )
        if not tel_frames:
            continue

        tel_segs.append({
            "task": task,
            "dataset": "EgoTel",
            "task_dir": tel_task_dir,
            "ep": tel_ep,
            "frames": tel_frames,
        })

    print(f"Built {len(udas_segs)} EgouDas segments and {len(tel_segs)} EgoTel segments")
    return udas_segs, tel_segs


def encode_segments(segs: List[Dict], device: str = "cuda:0", batch_size: int = 8) -> np.ndarray:
    """对所有 segment 编码，返回 embedding 矩阵。"""
    from sentence_transformers import SentenceTransformer

    print(f"Loading Qwen3-VL-Embedding...")
    model = SentenceTransformer(QWEN3_VL_PATH, trust_remote_code=True, device=device)

    messages = []
    for seg in segs:
        content = [{"type": "image", "image": img} for img in seg["frames"]]
        content.append({"type": "text", "text": seg["task"]})
        messages.append({"role": "user", "content": content})

    print(f"Encoding {len(messages)} segments (batch_size={batch_size})...")
    embs = model.encode(messages, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    return embs


def compute_cross_similarity(udas_embs: np.ndarray, tel_embs: np.ndarray) -> np.ndarray:
    """计算交叉相似度矩阵 (18×18)，udas_embs[i] @ tel_embs[j].T"""
    return udas_embs @ tel_embs.T


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


def make_cross_pair_image(
    udas_seg: Dict,
    tel_seg: Dict,
    rank: int,
    sim: float,
    frame_size=(180, 135)
) -> Image.Image:
    """和之前类似，但 task 文本分开写在各自下方。"""
    fw, fh = frame_size
    pad = 5
    border = 3
    header_h = 32
    footer_h = 72  # 增加高度以容纳两行 task

    def pick_9(frames):
        total = len(frames)
        if total >= 10:
            indices = [int(i * (total - 1) / 8) for i in range(9)]
        else:
            indices = list(range(min(9, total)))
        return [frames[i] for i in indices if i < len(frames)]

    udas_frames = pick_9(udas_seg["frames"])
    tel_frames = pick_9(tel_seg["frames"])

    grid_w = fw * 3 + pad * 2
    grid_h = fh * 3 + pad * 2

    divider_w = 10
    total_w = grid_w * 2 + divider_w + border * 4
    total_h = header_h + grid_h + footer_h

    bg_color = (20, 22, 28)
    img = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font_title = font_info = font_label = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (total_w, header_h - 1)], fill=(30, 33, 42))
    sim_color = (
        (80, 255, 120) if sim > 0.7 else
        (255, 220, 50) if sim > 0.5 else
        (255, 100, 80)
    )
    draw.text((8, 8), f"#{rank:02d}", fill=(160, 160, 180), font=font_title)
    draw.text((42, 8), f"Sim: {sim:.4f}", fill=sim_color, font=font_title)

    left_cx = border + grid_w // 2
    right_cx = border * 2 + grid_w + divider_w + grid_w // 2
    draw.text((left_cx - 35, 9), "EgouDas (ego)", fill=(100, 180, 255), font=font_label)
    draw.text((right_cx - 35, 9), "EgoTel (tele)", fill=(80, 220, 140), font=font_label)

    # 3×3 grid
    def paste_3x3(frames_9, x_off, y_off, border_color):
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

    # 中间分隔
    div_x = border + grid_w + border + 1
    draw.rectangle([(div_x, header_h), (div_x + divider_w - 3, header_h + grid_h + 4)], fill=(14, 16, 22))
    draw.line([(div_x + divider_w // 2, header_h + 4),
               (div_x + divider_w // 2, header_h + grid_h)],
              fill=(55, 60, 75), width=1)

    # Footer: 两边分别显示各自的 task
    fy = header_h + grid_h + 6

    # 左边 EgouDas task
    udas_lines = wrap_text(udas_seg["task"], max_chars=50)
    y_text = fy
    for line in udas_lines[:2]:
        draw.text((left_x, y_text), line, fill=(150, 180, 220), font=font_info)
        y_text += 11
    ep_l = f"ep{udas_seg['ep']:05d}"
    draw.text((left_x, fy + 25), ep_l, fill=(80, 110, 160), font=font_info)

    # 右边 EgoTel task
    tel_lines = wrap_text(tel_seg["task"], max_chars=50)
    y_text = fy
    for line in tel_lines[:2]:
        draw.text((right_x, y_text), line, fill=(120, 200, 150), font=font_info)
        y_text += 11
    ep_r = f"ep{tel_seg['ep']:05d} · {tel_seg['task_dir']}"
    draw.text((right_x, fy + 25), ep_r, fill=(60, 140, 90), font=font_info)

    # 相同 task 标记
    if udas_seg["task"] == tel_seg["task"]:
        draw.text((total_w // 2 - 60, fy + 55), "✓ Same Task", fill=(80, 255, 120), font=font_label)
    else:
        draw.text((total_w // 2 - 70, fy + 55), "✗ Different Tasks", fill=(255, 120, 80), font=font_label)

    return img


def visualize_top_and_bottom(
    udas_segs: List[Dict],
    tel_segs: List[Dict],
    sim_matrix: np.ndarray,
    output_dir: Path
):
    """可视化最高和最低相似度的配对。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 展平相似度矩阵，找出 top 18 和 bottom 10
    n = len(udas_segs)
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append((i, j, sim_matrix[i, j]))

    pairs.sort(key=lambda x: -x[2])

    # Top 18
    print("\n=== Top 18 Highest Similarity Pairs ===")
    for rank, (i, j, sim) in enumerate(pairs[:18], 1):
        img = make_cross_pair_image(udas_segs[i], tel_segs[j], rank, sim)
        path = output_dir / f"top_{rank:02d}_sim{sim:.3f}.png"
        img.save(path)
        same = "✓" if udas_segs[i]["task"] == tel_segs[j]["task"] else "✗"
        print(f"  #{rank:2d} sim={sim:.4f} {same} | {udas_segs[i]['task'][:60]}")

    # Bottom 10
    print("\n=== Bottom 10 Lowest Similarity Pairs ===")
    bottom_pairs = pairs[-10:]
    for rank, (i, j, sim) in enumerate(bottom_pairs, 1):
        img = make_cross_pair_image(udas_segs[i], tel_segs[j], rank, sim)
        path = output_dir / f"bottom_{rank:02d}_sim{sim:.3f}.png"
        img.save(path)
        same = "✓" if udas_segs[i]["task"] == tel_segs[j]["task"] else "✗"
        print(f"  #{rank:2d} sim={sim:.4f} {same}")
        print(f"    EgouDas: {udas_segs[i]['task']}")
        print(f"    EgoTel:  {tel_segs[j]['task']}")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print("=== Step 1: Load 18 segments ===")
    udas_segs, tel_segs = build_segments(n_tasks=18, seed=SEED)

    print(f"\n=== Step 2: Encode {len(udas_segs)} EgouDas segments ===")
    udas_embs = encode_segments(udas_segs, device=args.device, batch_size=args.batch_size)

    print(f"\n=== Step 3: Encode {len(tel_segs)} EgoTel segments ===")
    tel_embs = encode_segments(tel_segs, device=args.device, batch_size=args.batch_size)

    print(f"\n=== Step 4: Compute cross similarity ===")
    sim_matrix = compute_cross_similarity(udas_embs, tel_embs)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Max sim: {sim_matrix.max():.4f}, Min sim: {sim_matrix.min():.4f}")

    print(f"\n=== Step 5: Visualize ===")
    visualize_top_and_bottom(udas_segs, tel_segs, sim_matrix, OUTPUT_DIR)

    print("\nDone!")
