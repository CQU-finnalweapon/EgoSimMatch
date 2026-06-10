"""
Pair Matching (Per-EgoTel Sampled) — 每条 EgoTel 从 EgouDas 随机抽 N 条找最优匹配。

算法流程:
  1. 对每条 EgoTel (全部 ~321K)，随机采样 n_sample 条 EgouDas
  2. 矩阵乘计算相似度，取最优 EgouDas → 1 对
  3. 覆盖全部 EgoTel，共 ~321K 对
  4. 可视化大图：top N 对带首帧图片对比，每张 20 对

用法:
  python scripts/batch_pairing_greedy.py                              # 默认
  python scripts/batch_pairing_greedy.py --n_sample 500 --viz_top 500 # 采样500, 可视化top500
  python scripts/batch_pairing_greedy.py --no_viz                     # 跳过可视化
  python scripts/batch_pairing_greedy.py --viz_only                   # 仅重新生成可视化 (从已有JSON)

输出:
  outputs/batches_greedy/pairs_all.json          # 全部 ~321K 对
  outputs/batches_greedy/viz/                    # 可视化大图 (带首帧图片)
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 配置 ──────────────────────────────────────────────
EMBEDDING_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/embeddings")
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/batches_greedy")

EGOTEL_ROOT = ("/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/"
               "data/moz_data/EgoTel_aligned/data/raw_0521_anno_v2")
EGOUDAS_ROOT = ("/mnt/vepfs01/output/yuhang/spirit_VLA/egocentric/"
                "data/egocentric_data/ego_uDas/data_collection/moz_lerobot/0513_anno_v2")
CAMERA = "cam_high"

DEFAULT_N_SAMPLE = 500       # 每条 EgoTel 抽多少 EgouDas
DEFAULT_VIZ_TOP = 500        # 可视化 top N 对（带首帧图片）
PAIRS_PER_IMAGE = 20         # 每张大图包含的对数
GRID_COLS = 5                # 每行 5 对 (20对 = 5×4)
TEL_CHUNK_SIZE = 512         # 每批计算多少条 EgoTel
FRAME_MAX_SIZE = 320         # 首帧图片缩放到的最大边长 (像素)


# ══════════════════════════════════════════════════════
#  1. 数据加载
# ══════════════════════════════════════════════════════

def load_data(emb_dir: Path):
    """加载 embedding 和 metadata。"""
    print("=" * 60)
    print("  加载 embedding 和 metadata ...")
    print("=" * 60)
    t0 = time.time()
    udas_emb = np.load(emb_dir / "egoudas_embeddings.npy")
    tel_emb = np.load(emb_dir / "egotel_embeddings.npy")
    with open(emb_dir / "egoudas_metadata.json") as f:
        udas_meta = json.load(f)
    with open(emb_dir / "egotel_metadata.json") as f:
        tel_meta = json.load(f)
    t = time.time() - t0
    print(f"  egoudas: {udas_emb.shape}  {len(udas_meta)} segments  ({t:.1f}s)")
    print(f"  egotel:  {tel_emb.shape}  {len(tel_meta)} segments")
    return udas_emb, tel_emb, udas_meta, tel_meta


# ══════════════════════════════════════════════════════
#  2. 帧图片加载
# ══════════════════════════════════════════════════════

def _find_video(meta: dict, cache: dict = None):
    """根据 metadata 找到 MP4 视频路径。返回 (video_path, frame_idx)。"""
    if cache is None:
        cache = {}

    dataset = meta["dataset"]
    task_name = meta["task_name"]
    dataset_dir = meta["dataset_dir"]
    episode = meta["episode"]
    frame_idx = meta["frame_indices"][0]

    root = EGOTEL_ROOT if dataset == "egotel" else EGOUDAS_ROOT
    key = (root, task_name, dataset_dir, episode)

    if key in cache:
        return cache[key], frame_idx

    video_dir = Path(root) / task_name / dataset_dir / "videos"
    if not video_dir.exists():
        cache[key] = None
        return None, frame_idx

    for chunk_dir in sorted(video_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        cam_dir = chunk_dir / CAMERA
        video = cam_dir / f"episode_{episode:06d}.mp4"
        if video.exists():
            cache[key] = str(video)
            return str(video), frame_idx

    cache[key] = None
    return None, frame_idx


def load_first_frames(pairs, max_frames=1000, n_workers=1):
    """为 pairs 的前 max_frames 对加载首帧图片 (每个视频读第 0 帧, RGB numpy array)。"""
    video_cache = {}
    loaded = 0
    print(f"\n  加载首帧图片 (最多 {max_frames} 对)...")
    t0 = time.time()

    for pair in pairs:
        if loaded >= max_frames:
            break
        try:
            vp_u, _ = _find_video(pair["egoudas_meta"], video_cache)
            vp_t, _ = _find_video(pair["egotel_meta"], video_cache)
            img_u = img_t = None

            if vp_u:
                cap = cv2.VideoCapture(vp_u)
                # 直接读第 0 帧，不用 frame_indices（chunk 视频帧号从 0 开始）
                ret, frame = cap.read()
                cap.release()
                if ret:
                    img_u = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if vp_t:
                cap = cv2.VideoCapture(vp_t)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    img_t = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pair["img_egoudas"] = img_u
            pair["img_egotel"] = img_t
            loaded += 1
        except Exception:
            pair["img_egoudas"] = None
            pair["img_egotel"] = None
            loaded += 1

    elapsed = time.time() - t0
    has_img = sum(1 for p in pairs[:max_frames] if p.get("img_egoudas") is not None)
    print(f"  加载完成: {has_img}/{min(max_frames, len(pairs))} 对含图片 ({elapsed:.1f}s)")
    return pairs


# ══════════════════════════════════════════════════════
#  3. 逐条 EgoTel 采样配对
# ══════════════════════════════════════════════════════

def per_egotel_sampled_pairing(udas_emb, tel_emb, udas_meta, tel_meta,
                               n_sample=500, seed=42):
    """
    对每条 EgoTel，随机采样 n_sample 条 EgouDas，找最优匹配。
    输出 ~321K 对 (覆盖全部 EgoTel)。

    每 chunk (TEL_CHUNK_SIZE 条 EgoTel) 抽一批 EgouDas，矩阵乘取 argmax。
    """
    N_U, D = udas_emb.shape
    N_T = tel_emb.shape[0]
    n_sample = min(n_sample, N_U)

    print(f"\n{'=' * 60}")
    print(f"  逐条 EgoTel 采样配对")
    print(f"  每条 EgoTel 从 {N_U} 条 EgouDas 中随机抽 {n_sample} 条取最优")
    print(f"  总 EgoTel: {N_T}, 预计产出 {N_T} 对")
    print(f"{'=' * 60}")

    t0 = time.time()
    n_chunks = (N_T + TEL_CHUNK_SIZE - 1) // TEL_CHUNK_SIZE
    rng = np.random.default_rng(seed)

    pairs = []
    for chunk_idx in range(n_chunks):
        cs = chunk_idx * TEL_CHUNK_SIZE
        ce = min(cs + TEL_CHUNK_SIZE, N_T)
        chunk_tel = tel_emb[cs:ce]                     # (chunk_n, D)
        chunk_n = len(chunk_tel)

        # 这个 chunk 抽一批 EgouDas（每 chunk 不同，近似每条 EgoTel 独立采样）
        sample_idx = rng.choice(N_U, size=n_sample, replace=False)
        sampled_emb = udas_emb[sample_idx]              # (n_sample, D)

        sim_mat = chunk_tel @ sampled_emb.T             # (chunk_n, n_sample)
        best_u_local = np.argmax(sim_mat, axis=1)       # (chunk_n,)
        best_s = sim_mat[np.arange(chunk_n), best_u_local]

        for i in range(chunk_n):
            t_idx = cs + i
            u_idx = int(sample_idx[best_u_local[i]])
            pairs.append({
                "pair_id": t_idx,
                "egotel_idx": t_idx,
                "egotel_meta": tel_meta[t_idx],
                "egoudas_idx": u_idx,
                "egoudas_meta": udas_meta[u_idx],
                "similarity": float(best_s[i]),
            })

        if chunk_idx % 50 == 0 or chunk_idx == n_chunks - 1:
            elapsed = time.time() - t0
            eta = elapsed / (chunk_idx + 1) * n_chunks - elapsed if chunk_idx > 0 else 0
            print(f"  chunk {chunk_idx}/{n_chunks}, {len(pairs)} 对, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    scores = np.array([p["similarity"] for p in pairs])
    print(f"\n  ✅ 配对完成: {len(pairs)} 对, 耗时 {elapsed:.1f}s")
    print(f"  相似度: mean={scores.mean():.4f}  min={scores.min():.4f}  "
          f"median={np.median(scores):.4f}  max={scores.max():.4f}")
    return pairs


# ══════════════════════════════════════════════════════
#  4. 可视化 — 带首帧图片的网格大图
# ══════════════════════════════════════════════════════

def visualize_pairs_with_frames(pairs, output_dir: Path,
                                n_per_image=PAIRS_PER_IMAGE,
                                grid_cols=GRID_COLS,
                                frame_max_size=FRAME_MAX_SIZE):
    """
    生成带首帧图片对比的大图网格。
    每格: [EgouDas 首帧] [EgoTel 首帧]
          相似度 (彩色) + task 描述在图片下方。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    n_images = (total + n_per_image - 1) // n_per_image

    print(f"\n{'=' * 60}")
    print(f"  可视化 (带首帧图片): {n_images} 张大图, 每张 {n_per_image} 对")
    print(f"  帧缩放最大边长: {frame_max_size}px")
    print(f"{'=' * 60}")

    for img_idx in range(n_images):
        start = img_idx * n_per_image
        end = min(start + n_per_image, total)
        chunk = pairs[start:end]
        n = len(chunk)

        cols = min(grid_cols, n)
        rows = (n + cols - 1) // cols

        fig_w = cols * 4.6
        fig_h = rows * 4.5       # 给底部 task 文字留空间
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

        for j, pair in enumerate(chunk):
            r, c = divmod(j, cols)
            ax = axes[r][c]
            ax.axis("off")

            sim = pair["similarity"]
            # 相似度颜色和标签 (纯 ASCII 避免字体问题)
            if sim >= 0.85:
                sim_color = "#2e7d32"   # 深绿
                sim_label = "HIGH"
            elif sim >= 0.70:
                sim_color = "#f9a825"   # 黄
                sim_label = "MID"
            elif sim >= 0.55:
                sim_color = "#ef6c00"   # 橙
                sim_label = "LOW"
            else:
                sim_color = "#c62828"   # 红
                sim_label = "BAD"

            img_u = pair.get("img_egoudas")
            img_t = pair.get("img_egotel")

            if img_u is not None and img_t is not None:
                # 缩放帧到统一大小，减少内存占用
                h_u, w_u = img_u.shape[:2]
                h_t, w_t = img_t.shape[:2]
                scale_u = frame_max_size / max(h_u, w_u)
                scale_t = frame_max_size / max(h_t, w_t)
                if scale_u < 1.0:
                    img_u = cv2.resize(img_u, (int(w_u * scale_u), int(h_u * scale_u)))
                if scale_t < 1.0:
                    img_t = cv2.resize(img_t, (int(w_t * scale_t), int(h_t * scale_t)))

                # 左右并排两张首帧图片
                combined = np.hstack([img_u, img_t])
                ax.imshow(combined)

                # 图片上方左侧：相似度 (彩色)
                ax.text(0.02, 0.98, f"{sim_label} {sim:.4f}",
                        transform=ax.transAxes, fontsize=6.5, fontweight="bold",
                        ha="left", va="top", color=sim_color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=sim_color, alpha=0.85, linewidth=0.8))
            else:
                ax.text(0.5, 0.5, "NO\nFRAME",
                        transform=ax.transAxes, fontsize=14, fontweight="bold",
                        ha="center", va="center", color="#cc0000")

            # task 描述放在图片下方 (clip_on=False 允许超出 axes 边界)
            uda_task = pair["egoudas_meta"].get("task", "?")
            tel_task = pair["egotel_meta"].get("task", "?")
            # 截断过长文本
            max_len = 55
            uda_short = uda_task[:max_len] + ("…" if len(uda_task) > max_len else "")
            tel_short = tel_task[:max_len] + ("…" if len(tel_task) > max_len else "")
            ax.text(0.5, -0.08, f"U: {uda_short}",
                    transform=ax.transAxes, fontsize=4.8, ha="center", va="top",
                    color="#444", clip_on=False, linespacing=1.1)
            ax.text(0.5, -0.18, f"T: {tel_short}",
                    transform=ax.transAxes, fontsize=4.8, ha="center", va="top",
                    color="#222", clip_on=False, linespacing=1.1,
                    fontstyle="italic")

        # 隐藏多余格
        for j in range(n, rows * cols):
            axes.flat[j].set_visible(False)

        fig.suptitle(f"Top {total} Pairs — Page {img_idx+1}/{n_images}  "
                     f"(#{start}–#{end-1})",
                     fontsize=10, fontweight="bold", y=1.002)
        fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

        img_path = output_dir / f"pairs_{start:06d}_{end-1:06d}.png"
        fig.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{img_idx+1}/{n_images}] {img_path.name}")

    print(f"  ✅ 可视化完成 → {output_dir}/")


# ══════════════════════════════════════════════════════
#  5. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-EgoTel Sampled Pair Matching")
    parser.add_argument("--n_sample", type=int, default=DEFAULT_N_SAMPLE,
                        help=f"每条 EgoTel 抽多少 EgouDas (default: {DEFAULT_N_SAMPLE})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz_top", type=int, default=DEFAULT_VIZ_TOP,
                        help=f"可视化 top N 对，带首帧图片 (default: {DEFAULT_VIZ_TOP})")
    parser.add_argument("--pairs_per_image", type=int, default=PAIRS_PER_IMAGE,
                        help=f"每张大图包含的对数 (default: {PAIRS_PER_IMAGE})")
    parser.add_argument("--emb_dir", type=str, default=str(EMBEDDING_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no_viz", action="store_true",
                        help="跳过可视化，只保存 JSON")
    parser.add_argument("--viz_only", action="store_true",
                        help="仅重新生成可视化 (从已有 pairs_all.json 加载)")
    parser.add_argument("--natural_order", action="store_true",
                        help="按 EgoTel 原始顺序展示前 N 对 (默认按相似度降序取 top N)")
    parser.add_argument("--random_sample", action="store_true",
                        help="从全部配对中随机抽样 N 对展示 (均匀覆盖各分数段)")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.viz_only:
        # ── 仅可视化模式：从已有 JSON 加载 ──
        json_path = out_dir / "pairs_all.json"
        if not json_path.exists():
            print(f"  ❌ 找不到 {json_path}，请先运行配对生成 JSON")
            return
        print(f"  从 {json_path} 加载已有配对...")
        with open(json_path) as f:
            data = json.load(f)
        pairs_all = data["pairs"]
        print(f"  已加载 {len(pairs_all)} 对")
    else:
        # ── 1. 加载嵌入 ──
        udas_emb, tel_emb, udas_meta, tel_meta = load_data(emb_dir)

        # ── 2. 配对 ──
        pairs_all = per_egotel_sampled_pairing(
            udas_emb, tel_emb, udas_meta, tel_meta,
            n_sample=args.n_sample, seed=args.seed)

        # ── 3. 排序 (按相似度降序，或保持 EgoTel 原始顺序) ──
        if not args.natural_order:
            pairs_all = sorted(pairs_all, key=lambda x: -x["similarity"])

        # ── 4. 保存全部 pairs JSON ──
        json_path = out_dir / "pairs_all.json"
        with open(json_path, "w") as f:
            json.dump({
                "n_sample": args.n_sample,
                "seed": args.seed,
                "n_pairs": len(pairs_all),
                "pairs": pairs_all,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON (全部 {len(pairs_all)} 对) 已保存: {json_path}")

    # ── 5. 可视化前 N 对 ──
    if not args.no_viz:
        viz_top = min(args.viz_top, len(pairs_all))

        if args.random_sample:
            # 从全部配对中随机抽样 (均匀代表各分数段)
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(pairs_all), size=viz_top, replace=False)
            viz_pairs = [pairs_all[i] for i in indices]
            print(f"  随机抽样 {viz_top} 对 (seed={args.seed})")
        elif args.natural_order:
            # 按 EgoTel 原始顺序：若 JSON 已排序则按 pair_id 恢复
            viz_pairs = sorted(pairs_all, key=lambda x: x["pair_id"])[:viz_top]
            print(f"  按 EgoTel 原始顺序展示前 {viz_top} 对")
        else:
            viz_pairs = pairs_all[:viz_top]
            print(f"  按相似度降序展示 top {viz_top} 对")

        # 加载首帧图片
        viz_pairs = load_first_frames(viz_pairs, max_frames=viz_top)

        viz_dir = out_dir / "viz"
        # 清空旧图
        import shutil
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        visualize_pairs_with_frames(viz_pairs, viz_dir,
                                    n_per_image=args.pairs_per_image)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ 全部完成！总耗时 {total_time:.1f}s")
    print(f"  总配对: {len(pairs_all)} 对 (覆盖全部 EgoTel)")
    order_desc = "随机抽样" if args.random_sample else ("原始顺序" if args.natural_order else "相似度降序")
    print(f"  可视化: {order_desc} 前 {min(args.viz_top, len(pairs_all))} 对 → {out_dir}/viz/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
