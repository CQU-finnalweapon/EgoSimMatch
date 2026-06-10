"""
Batch 可视化 — 展示 batch 中 16 对 (udas×tel) 的配对效果。

功能:
  1. 相似度热力图 (16×16)
  2. 配对角线相似度分布
  3. 每对 segment 的 task 文本对比
  4. (可选) 加载第一帧图像显示

用法:
  python scripts/visualize_batch.py --batch_id 0                         # 显示第 0 个 batch
  python scripts/visualize_batch.py --batch_id 0 --output viz_output     # 保存到目录
  python scripts/visualize_batch.py --batch_id 0 --load_images           # 加载帧图像
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# ── 配置 ──────────────────────────────────────────────
BATCHES_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/batches")
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/batch_viz")
EMBEDDING_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/embeddings")

# 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_batch(batch_id: int, batches_dir: Path = BATCHES_DIR) -> dict:
    """加载 batch JSON 文件。"""
    path = batches_dir / f"batch_{batch_id:06d}.json"
    if not path.exists():
        # 也尝试 greedy 输出目录
        alt_path = Path(str(batches_dir).replace("batches", "batches_greedy")) / f"batch_{batch_id:06d}.json"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Batch 文件不存在: {path} 也不在 {alt_path}")
    with open(path) as f:
        return json.load(f)


def load_batches_info(batches_dir: Path = BATCHES_DIR) -> dict:
    """加载 batch 汇总信息。"""
    # 尝试多个可能的文件名
    candidates = [
        batches_dir / "egoudas_egotel_batches_info.json",
        batches_dir / "batches_info.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    return {}


def visualize_batch_standalone(batch: dict, output_path: str = None):
    """
    不加载图像，只显示相似度热力图和文本信息。
    生成一个 matplotlib figure 并保存/显示。
    """
    n = batch["n_udas"]  # 16
    sim_matrix = np.array(batch["sim_matrix"])

    # ── 创建 figure ──
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 2])

    # ── 标题 ──
    fig.suptitle(f"Batch #{batch['batch_id']:06d}  —  "
                 f"相似度: mean={batch['diag_sim_mean']:.4f}  "
                 f"min={batch['diag_sim_min']:.4f}  "
                 f"max={batch['diag_sim_max']:.4f}",
                 fontsize=16, fontweight="bold")

    # ── 热力图 ──
    ax_heat = fig.add_subplot(gs[0, :])
    im = ax_heat.imshow(sim_matrix, cmap="viridis", aspect="auto",
                        vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_title("16×16 相似度矩阵 (udas × tel)", fontsize=13)
    ax_heat.set_xlabel("Tel segment index", fontsize=10)
    ax_heat.set_ylabel("Udas segment index", fontsize=10)

    # 标注对角线数值
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6, color=color)

    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    # ── 配对明细 ──
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    # 构建配对信息表格
    table_data = []
    col_labels = ["Pair", "Sim", "Udas Task", "Tel Task", "U Frames", "T Frames"]
    for i in range(n):
        u = batch["udas"][i]
        t = batch["tel"][i]
        sim = sim_matrix[i, i]
        u_task = u.get("task", "")[:40]
        t_task = t.get("task", "")[:40]
        u_frames = len(u.get("frame_indices", []))
        t_frames = len(t.get("frame_indices", []))
        table_data.append([f"{i}", f"{sim:.4f}", u_task, t_task,
                          str(u_frames), str(t_frames)])

    table = ax_info.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="left",
                          colWidths=[0.05, 0.07, 0.28, 0.28, 0.06, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # 给最高和最低的 pair 标色
    diag_sims = [sim_matrix[i, i] for i in range(n)]
    best_idx = np.argmax(diag_sims)
    worst_idx = np.argmin(diag_sims)
    for i in range(n):
        color = "#d4edda" if i == best_idx else "#f8d7da" if i == worst_idx else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    # ── 相似度分布 ──
    ax_hist = fig.add_subplot(gs[2, 0])
    diag = [sim_matrix[i, i] for i in range(n)]
    all_vals = sim_matrix.flatten()
    ax_hist.hist(all_vals, bins=20, alpha=0.5, label="All pairs", color="gray")
    ax_hist.hist(diag, bins=10, alpha=0.7, label="Matched pairs", color="green")
    ax_hist.axvline(batch["diag_sim_mean"], color="red", linestyle="--",
                    label=f'Mean={batch["diag_sim_mean"]:.3f}')
    ax_hist.set_xlabel("Similarity")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("相似度分布")
    ax_hist.legend(fontsize=8)

    # ── Udas vs Tel 帧数分布 ──
    ax_frames = fig.add_subplot(gs[2, 1])
    u_frames = [len(b["frame_indices"]) for b in batch["udas"]]
    t_frames = [len(b["frame_indices"]) for b in batch["tel"]]
    x = np.arange(n)
    ax_frames.bar(x - 0.2, u_frames, 0.4, label="Udas", alpha=0.8)
    ax_frames.bar(x + 0.2, t_frames, 0.4, label="Tel", alpha=0.8)
    ax_frames.set_xlabel("Pair index")
    ax_frames.set_ylabel("Frame count")
    ax_frames.set_title("帧数对比 (每对)")
    ax_frames.legend(fontsize=8)

    # ── Task 文本相似度 vs 视觉相似度 ──
    ax_task = fig.add_subplot(gs[2, 2])
    # 简单统计: 相同 task 的数量
    same_task = sum(
        1 for i in range(n)
        if batch["udas"][i].get("task", "").strip() == batch["tel"][i].get("task", "").strip()
    )
    diff_task = n - same_task
    ax_task.pie([same_task, diff_task], labels=[f"Same task\n({same_task})", f"Diff task\n({diff_task})"],
                colors=["#4CAF50", "#FF9800"], autopct="%1.0f%%", startangle=90)
    ax_task.set_title("Pair: 相同 Task 文本?")

    # ── 汇总统计 ──
    ax_stats = fig.add_subplot(gs[2, 3])
    ax_stats.axis("off")
    summary = (
        f"Batch #{batch['batch_id']}\n"
        f"==============\n"
        f"Diag mean:  {batch['diag_sim_mean']:.4f}\n"
        f"Diag min:   {batch['diag_sim_min']:.4f}\n"
        f"Diag max:   {batch['diag_sim_max']:.4f}\n"
        f"\nUdas frames:\n"
        f"  total: {sum(u_frames)}\n"
        f"  avg:   {np.mean(u_frames):.1f}\n"
        f"Tel frames:\n"
        f"  total: {sum(t_frames)}\n"
        f"  avg:   {np.mean(t_frames):.1f}\n"
        f"\nSame task: {same_task}/{n}"
    )
    ax_stats.text(0.1, 0.5, summary, fontsize=10, family="monospace",
                  va="center", transform=ax_stats.transAxes)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ 已保存 → {output_path}")
    else:
        plt.show()
    plt.close()


def generate_html_report(batch: dict, output_path: str):
    """生成一个独立的 HTML 报告。"""
    n = batch["n_udas"]
    sim_matrix = np.array(batch["sim_matrix"])
    diag_sims = [sim_matrix[i, i] for i in range(n)]

    # 构建 HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Batch #{batch['batch_id']:06d} 可视化</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #4CAF50; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.heatmap {{ display: inline-block; margin: 20px 0; }}
.heatmap-row {{ display: flex; }}
.cell {{ width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;
         font-size: 9px; color: white; text-shadow: 0 0 2px rgba(0,0,0,0.8); }}
.cell-label {{ width: 60px; font-size: 10px; padding: 0 5px; display: flex; align-items: center; }}
.stats {{ margin: 20px 0; }}
.best {{ background: #d4edda !important; }}
.worst {{ background: #f8d7da !important; }}
</style></head><body>
<h1>Batch #{batch['batch_id']:06d}</h1>
<div class="stats">
  <b>对角线相似度:</b> mean={batch['diag_sim_mean']:.4f},
  min={batch['diag_sim_min']:.4f}, max={batch['diag_sim_max']:.4f}
</div>

<h2>16×16 相似度矩阵</h2>
<div class="heatmap">
"""
    # 生成热力图 HTML
    html += '<div class="heatmap-row"><div class="cell-label"></div>'
    for j in range(n):
        html += f'<div class="cell-label" style="justify-content:center;font-weight:bold">T{j}</div>'
    html += '</div>'

    for i in range(n):
        html += f'<div class="heatmap-row"><div class="cell-label" style="font-weight:bold">U{i}</div>'
        for j in range(n):
            val = sim_matrix[i, j]
            intensity = int(255 * (1 - val))
            r = intensity
            g = int(255 * (1 - val * 0.5))
            b = int(255 * (1 - val * 0.3))
            is_diag = "★ " if i == j else ""
            html += f'<div class="cell" style="background:rgb({r},{g},{b})">{is_diag}{val:.2f}</div>'
        html += '</div>'

    html += "</div>"

    # 配对明细表
    html += "<h2>配对明细</h2><table><tr><th>#</th><th>Sim</th><th>Udas Task</th><th>Tel Task</th>"
    html += "<th>U Frames</th><th>T Frames</th><th>U Epi</th><th>T Epi</th></tr>"

    best_idx = int(np.argmax(diag_sims))
    worst_idx = int(np.argmin(diag_sims))

    for i in range(n):
        u, t = batch["udas"][i], batch["tel"][i]
        cls = ' class="best"' if i == best_idx else ' class="worst"' if i == worst_idx else ""
        html += f"<tr{cls}>"
        html += f"<td>{i}</td><td>{diag_sims[i]:.4f}</td>"
        html += f"<td>{u.get('task','')[:60]}</td><td>{t.get('task','')[:60]}</td>"
        html += f"<td>{len(u.get('frame_indices',[]))}</td><td>{len(t.get('frame_indices',[]))}</td>"
        html += f"<td>{u.get('episode','')}</td><td>{t.get('episode','')}</td>"
        html += "</tr>"

    html += "</table></body></html>"

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  ✅ HTML 已保存 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch 可视化")
    parser.add_argument("--batch_id", type=int, required=True,
                        help="要可视化的 batch ID")
    parser.add_argument("--batches_dir", type=str, default=str(BATCHES_DIR),
                        help="batch 文件目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径 (默认: {output_dir}/batch_{id}.png)")
    parser.add_argument("--html", action="store_true",
                        help="同时生成 HTML 报告")
    parser.add_argument("--no_plot", action="store_true",
                        help="只生成 HTML，不生成 matplotlib 图")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用 batch")
    args = parser.parse_args()

    batches_dir = Path(args.batches_dir)

    # 列出所有 batch
    if args.list:
        info = load_batches_info(batches_dir)
        if info:
            print(f"\nBatch 汇总:")
            print(f"  总 batch 数: {info.get('n_batches', '?')}")
            print(f"  相似度均值: {info.get('diag_sim_mean_avg', '?'):.4f}")
            print(f"  Udas 使用: {info.get('total_udas_used', '?')}")
            print(f"  Tel 使用:  {info.get('total_tel_used', '?')}")
            print()
        # 列出 batch 文件
        batch_files = sorted(batches_dir.glob("batch_*.json"))
        n_info = info.get("n_batches", 0) if info else 0
        print(f"  找到 {len(batch_files)} 个 batch JSON 文件")
        if batch_files:
            print(f"  ID 范围: 0 ~ {len(batch_files) - 1}")
            # 显示前 5 和后 5 个 batch 的相似度
            for f in batch_files[:5]:
                with open(f) as fh:
                    b = json.load(fh)
                bid = b.get("batch_id", f.stem)
                print(f"    batch {bid:>6d}: sim_mean={b.get('diag_sim_mean',0):.4f}")
            if len(batch_files) > 10:
                print(f"    ... ({len(batch_files)-10} 个中间文件省略)")
                for f in batch_files[-5:]:
                    with open(f) as fh:
                        b = json.load(fh)
                    bid = b.get("batch_id", f.stem)
                    print(f"    batch {bid:>6d}: sim_mean={b.get('diag_sim_mean',0):.4f}")
        return

    # 加载 batch
    batch = load_batch(args.batch_id, batches_dir)
    print(f"\n加载 Batch #{batch['batch_id']:06d}")
    print(f"  相似度: mean={batch['diag_sim_mean']:.4f}, "
          f"min={batch['diag_sim_min']:.4f}, max={batch['diag_sim_max']:.4f}")

    # 设置输出路径
    viz_dir = OUTPUT_DIR
    if args.output:
        viz_dir = Path(args.output)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 生成 matplotlib 图
    if not args.no_plot:
        png_path = str(viz_dir / f"batch_{args.batch_id:06d}.png")
        visualize_batch_standalone(batch, png_path)

    # 生成 HTML
    if args.html or args.no_plot:
        html_path = str(viz_dir / f"batch_{args.batch_id:06d}.html")
        generate_html_report(batch, html_path)

    print(f"  ✅ Batch #{args.batch_id:06d} 可视化完成")


if __name__ == "__main__":
    main()
