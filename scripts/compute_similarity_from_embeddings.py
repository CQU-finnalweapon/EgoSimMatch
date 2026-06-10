"""
从已存储的 embedding 计算相似度 — 无需重新加载模型，秒级完成。

用法:
  # EgouDas × EgoTel 交叉相似度 (全量 N×M)
  python scripts/compute_similarity_from_embeddings.py --mode cross

  # EgouDas 内部自相似度
  python scripts/compute_similarity_from_embeddings.py --mode self --dataset egoudas

  # EgoTel 内部自相似度
  python scripts/compute_similarity_from_embeddings.py --mode self --dataset egotel

  # Top-K 检索: 对每个 EgouDas segment 找 EgoTel 中最相似的 K 个
  python scripts/compute_similarity_from_embeddings.py --mode topk --k 10 --output outputs/udas_tel_topk.json

  # 跨模态 pair 列表: 给定一个 pair 文件，直接查表输出相似度
  python scripts/compute_similarity_from_embeddings.py --mode pairs --pair_file pairs.json

存储格式说明:
  build_full_embeddings.py 生成:
    outputs/embeddings/egoudas_embeddings.npy   # [N, D] float32
    outputs/embeddings/egoudas_metadata.json     # N 条 metadata
    outputs/embeddings/egotel_embeddings.npy     # [M, D] float32
    outputs/embeddings/egotel_metadata.json      # M 条 metadata
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBEDDING_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/embeddings")
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/similarity")


def load_embeddings(dataset_name: str) -> tuple:
    """加载 embedding 矩阵和 metadata。返回 (embeddings, metadata)。"""
    emb_path = EMBEDDING_DIR / f"{dataset_name}_embeddings.npy"
    meta_path = EMBEDDING_DIR / f"{dataset_name}_metadata.json"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"未找到 {emb_path}，请先运行 build_full_embeddings.py"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"[{dataset_name}] 加载 {len(embeddings)} 个 embedding，维度 {embeddings.shape[1]}")
    return embeddings, metadata


def compute_cross_similarity(
    udas_embs: np.ndarray,
    tel_embs: np.ndarray,
    udas_meta: List[dict],
    tel_meta: List[dict],
    output_dir: Path,
    save_full: bool = True,
):
    """计算 EgouDas × EgoTel 全量交叉相似度矩阵。"""
    n, m = len(udas_embs), len(tel_embs)
    print(f"\n计算交叉相似度 {n}×{m} ...")
    t0 = time.time()

    # 分块计算，避免大矩阵 OOM（N×M 可能很大）
    chunk_size = 5000
    sim_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        sim_matrix[i:i_end] = udas_embs[i:i_end] @ tel_embs.T
        if i % (chunk_size * 5) == 0:
            print(f"  进度: {i_end}/{n}")

    elapsed = time.time() - t0
    print(f"相似度计算完成，耗时 {elapsed:.1f}s")

    # 统计
    print(f"\n相似度统计:")
    print(f"  Max:  {sim_matrix.max():.4f}")
    print(f"  Min:  {sim_matrix.min():.4f}")
    print(f"  Mean: {sim_matrix.mean():.4f}")
    print(f"  Std:  {sim_matrix.std():.4f}")

    # 相同 task 的相似度 vs 不同 task
    same_task_sims = []
    diff_task_sims = []
    for i in range(n):
        for j in range(m):
            if udas_meta[i]["task"] == tel_meta[j]["task"]:
                same_task_sims.append(sim_matrix[i, j])
            else:
                diff_task_sims.append(sim_matrix[i, j])

    if same_task_sims:
        same_arr = np.array(same_task_sims)
        print(f"\n相同 task 相似度: mean={same_arr.mean():.4f}, std={same_arr.std():.4f}")
    if diff_task_sims:
        diff_arr = np.array(diff_task_sims)
        print(f"不同 task 相似度: mean={diff_arr.mean():.4f}, std={diff_arr.std():.4f}")

    # 保存完整相似度矩阵
    if save_full:
        output_dir.mkdir(parents=True, exist_ok=True)
        sim_path = output_dir / "cross_similarity.npy"
        np.save(sim_path, sim_matrix)
        print(f"\n完整相似度矩阵已保存: {sim_path}")

    return sim_matrix


def compute_topk(
    udas_embs: np.ndarray,
    tel_embs: np.ndarray,
    udas_meta: List[dict],
    tel_meta: List[dict],
    k: int = 10,
    output_path: Optional[Path] = None,
) -> List[dict]:
    """对每个 EgouDas segment 找 EgoTel 中最相似的 K 个。"""
    n = len(udas_embs)
    print(f"\n计算 Top-{k} 检索 (每 EgouDas → EgoTel) ...")

    results = []
    chunk_size = 2000

    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        chunk_sim = udas_embs[i:i_end] @ tel_embs.T  # [chunk, M]

        # 对每个 query 找 top-k
        for local_i in range(chunk_sim.shape[0]):
            global_i = i + local_i
            scores = chunk_sim[local_i]
            top_indices = np.argsort(-scores)[:k]  # 降序

            matches = []
            for rank, j in enumerate(top_indices):
                matches.append({
                    "rank": int(rank + 1),
                    "similarity": float(scores[j]),
                    "egotel_meta": tel_meta[int(j)],
                })

            results.append({
                "query_idx": int(global_i),
                "query_meta": udas_meta[global_i],
                "top_k": matches,
            })

    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Top-{k} 结果已保存: {output_path}")

    # 简要输出
    print(f"\nTop-{k} 检索完成，共 {len(results)} 条 query")
    for r in results[:5]:
        top1 = r["top_k"][0]
        same = "✓" if r["query_meta"]["task"] == top1["egotel_meta"]["task"] else "✗"
        print(f"  {same} {r['query_meta']['task'][:50]} → "
              f"sim={top1['similarity']:.4f} | {top1['egotel_meta']['task'][:50]}")

    return results


def compute_self_similarity(
    embs: np.ndarray,
    metadata: List[dict],
    dataset_name: str,
    output_dir: Path,
    top_k_save: int = 1000,
):
    """计算数据集内部的自相似度矩阵。"""
    n = len(embs)
    print(f"\n计算 {dataset_name} 内部自相似度 {n}×{n} ...")

    if n > 20000:
        print(f"  ⚠️  {n}×{n} 矩阵过大，仅输出统计信息，不保存完整矩阵")
        # 分块统计
        chunk_size = 5000
        all_sims = []
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            block = embs[i:i_end] @ embs.T
            # 取上三角（不含对角线）
            for bi in range(block.shape[0]):
                gi = i + bi
                upper = block[bi, gi+1:]
                all_sims.extend(upper.tolist())
            print(f"  进度: {i_end}/{n}")
        sims = np.array(all_sims)
    else:
        sim_matrix = embs @ embs.T  # [N, N]
        # 掩码掉对角线
        mask = ~np.eye(n, dtype=bool)
        sims = sim_matrix[mask]

    print(f"  自相似度: mean={sims.mean():.4f}, std={sims.std():.4f}, "
          f"max={sims.max():.4f}, min={sims.min():.4f}")

    # 找最相似的 top_k 对（不含自身）
    if n <= 20000:
        sim_matrix = embs @ embs.T
        np.fill_diagonal(sim_matrix, -1)  # 排除自身
        flat_indices = np.argsort(-sim_matrix.ravel())
        print(f"\n  Top 10 最相似对（不含自身）:")
        count = 0
        for idx in flat_indices:
            i, j = divmod(idx, n)
            if count >= 10:
                break
            count += 1
            same = "✓" if metadata[i]["task"] == metadata[j]["task"] else "✗"
            print(f"    {same} sim={sim_matrix[i, j]:.4f} | "
                  f"[{i}] {metadata[i]['task'][:40]} ↔ [{j}] {metadata[j]['task'][:40]}")


def compute_from_pairs(
    udas_embs: np.ndarray,
    tel_embs: np.ndarray,
    udas_meta: List[dict],
    tel_meta: List[dict],
    pair_file: str,
    output_path: Optional[str] = None,
):
    """
    从给定的 pair 文件查表计算相似度。
    pair 文件格式: JSON 数组，每个元素含 "egoudas_idx" 和 "egotel_idx"（或 meta 字段用于匹配）。
    """
    with open(pair_file) as f:
        pairs = json.load(f)

    print(f"\n从 pair 文件加载 {len(pairs)} 对 ...")

    results = []
    for pair in pairs:
        if "egoudas_idx" in pair and "egotel_idx" in pair:
            i, j = pair["egoudas_idx"], pair["egotel_idx"]
        elif "query_meta" in pair and "egotel_meta" in pair:
            # 通过 session+episode+frame 匹配
            i = _find_index(udas_meta, pair["query_meta"])
            j = _find_index(tel_meta, pair["egotel_meta"])
        else:
            continue

        if i is not None and j is not None:
            sim = float(udas_embs[i] @ tel_embs[j].T)
            results.append({
                "egoudas_idx": i,
                "egotel_idx": j,
                "similarity": sim,
                "egoudas_task": udas_meta[i]["task"],
                "egotel_task": tel_meta[j]["task"],
            })

    # 按相似度排序
    results.sort(key=lambda x: -x["similarity"])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_path}")

    print(f"\n计算完成，共 {len(results)} 对有效结果")
    for r in results[:10]:
        same = "✓" if r["egoudas_task"] == r["egotel_task"] else "✗"
        print(f"  {same} sim={r['similarity']:.4f} | {r['egoudas_task'][:40]} ↔ {r['egotel_task'][:40]}")

    return results


def _find_index(metadata: List[dict], query_meta: dict) -> Optional[int]:
    """通过 metadata 字段匹配找到 index。"""
    for i, meta in enumerate(metadata):
        match = True
        for k in ("session", "episode", "task_index", "frame_start", "task_dir", "dataset_dir"):
            if k in query_meta and k in meta:
                if meta[k] != query_meta[k]:
                    match = False
                    break
        if match:
            return i
    return None


# ── 入口 ───────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从已存储 embedding 计算相似度")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["cross", "self", "topk", "pairs"])
    parser.add_argument("--dataset", type=str, default="egoudas",
                        help="self 模式下的数据集名")
    parser.add_argument("--k", type=int, default=10,
                        help="topk 模式下的 K 值")
    parser.add_argument("--pair_file", type=str, default=None,
                        help="pairs 模式下的 pair 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "cross":
        # EgouDas × EgoTel 交叉
        udas_embs, udas_meta = load_embeddings("egoudas")
        tel_embs, tel_meta = load_embeddings("egotel")
        compute_cross_similarity(
            udas_embs, tel_embs, udas_meta, tel_meta,
            output_dir=OUTPUT_DIR,
            save_full=(udas_embs.shape[0] * tel_embs.shape[0] < 500_000_000),
        )

    elif args.mode == "self":
        # 内部自相似度
        embs, meta = load_embeddings(args.dataset)
        compute_self_similarity(embs, meta, args.dataset, OUTPUT_DIR)

    elif args.mode == "topk":
        # Top-K 检索
        udas_embs, udas_meta = load_embeddings("egoudas")
        tel_embs, tel_meta = load_embeddings("egotel")
        output_path = args.output or str(OUTPUT_DIR / "topk_results.json")
        compute_topk(udas_embs, tel_embs, udas_meta, tel_meta,
                     k=args.k, output_path=Path(output_path))

    elif args.mode == "pairs":
        # 给定 pair 列表
        if not args.pair_file:
            print("❌ --pair_file 参数必填")
            sys.exit(1)
        udas_embs, udas_meta = load_embeddings("egoudas")
        tel_embs, tel_meta = load_embeddings("egotel")
        compute_from_pairs(udas_embs, tel_embs, udas_meta, tel_meta,
                           args.pair_file, args.output)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
