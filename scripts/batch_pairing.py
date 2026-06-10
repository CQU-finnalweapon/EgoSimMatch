"""
Batch Pairing — 将 EgouDas + EgoTel 的 embedding 配成 16:16 训练 batch。

流程:
  1. 加载两个数据集的 embedding 和 metadata
  2. FAISS k-means 聚类全部 820K 条 embedding
  3. 每个簇内: Hungarian 最大化 16×16 相似度配对
  4. 跨簇 leftovers 合并
  5. 保存 batch 到文件

用法:
  python scripts/batch_pairing.py                          # 默认 2048 簇
  python scripts/batch_pairing.py --n_clusters 4096        # 更多簇
  python scripts/batch_pairing.py --batch_size 32           # 更大 batch

输出:
  outputs/batches/{dataset}_batches_info.json              # 批次汇总
  outputs/batches/batch_000000~batch_N.json                # 每个 batch 明细
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── 配置 ──────────────────────────────────────────────
EMBEDDING_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/embeddings")
OUTPUT_DIR = Path("/mnt/vepfs01/output/klayzhou/EgoSimMatch/outputs/batches")

# 默认参数
DEFAULT_N_CLUSTERS = 2048
DEFAULT_BATCH_SIZE = 32


# ══════════════════════════════════════════════════════
#  1. 数据加载
# ══════════════════════════════════════════════════════

def load_data(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict]]:
    """加载两个数据集的 embedding (L2-normalized float32) 和 metadata。"""
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

    # 验证维度一致
    assert udas_emb.shape[1] == tel_emb.shape[1], \
        f"embedding 维度不一致: {udas_emb.shape[1]} vs {tel_emb.shape[1]}"
    assert udas_emb.shape[0] == len(udas_meta), \
        f"egoudas embedding 数与 metadata 数不一致"
    assert tel_emb.shape[0] == len(tel_meta), \
        f"egotel embedding 数与 metadata 数不一致"

    print(f"  总 segments: {len(udas_meta) + len(tel_meta)}")
    return udas_emb, tel_emb, udas_meta, tel_meta


# ══════════════════════════════════════════════════════
#  2. FAISS k-means 聚类
# ══════════════════════════════════════════════════════

def run_kmeans(
    udas_emb: np.ndarray,
    tel_emb: np.ndarray,
    n_clusters: int,
    niter: int = 300,
    seed: int = 42,
    gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对全部 820K embedding 做 k-means 聚类。

    Args:
        udas_emb: (N_udas, D) float32
        tel_emb: (N_tel, D) float32
        n_clusters: 簇数量
        niter: 迭代次数
        seed: 随机种子
        gpu: 是否使用 GPU

    Returns:
        (udas_labels, tel_labels): 每个 segment 所属的簇 ID
    """
    all_emb = np.vstack([udas_emb, tel_emb])
    n_udas = udas_emb.shape[0]

    print(f"\n{'=' * 60}")
    print(f"  k-means 聚类 ({n_clusters} 簇, {niter} 轮迭代)")
    print(f"  总数据: {all_emb.shape[0]} × {all_emb.shape[1]}")
    print(f"{'=' * 60}")

    t0 = time.time()

    try:
        udas_labels, tel_labels = _run_faiss_kmeans(
            all_emb, n_udas, n_clusters, niter, seed, gpu
        )
        # 检查是否 collapse — 非空簇太少说明高维 collapse
        n_nonempty_faiss = len(set(udas_labels) | set(tel_labels))
        if n_nonempty_faiss < max(10, n_clusters // 10):
            print(f"  ⚠️ FAISS k-means collapse: 仅 {n_nonempty_faiss}/{n_clusters} 非空簇")
            print(f"  → 回退到 sklearn PCA + MiniBatchKMeans ...")
            raise ValueError(f"FAISS clustering collapsed ({n_nonempty_faiss} non-empty)")
    except Exception as e:
        print(f"  FAISS k-means 失败 ({e})，回退到 sklearn MiniBatchKMeans ...")
        print(f"  (如果 sklearn 未安装，请在 Volcengine 上运行: pip install scikit-learn)")
        udas_labels, tel_labels = _run_sklearn_kmeans(
            all_emb, n_udas, n_clusters, niter, seed
        )

    # 统计
    n_nonempty = len(set(udas_labels) | set(tel_labels))
    from collections import Counter
    u_counts = Counter(udas_labels)
    t_counts = Counter(tel_labels)
    print(f"  非空簇: {n_nonempty}/{n_clusters}")
    print(f"  udas 分布: avg={len(udas_labels)/max(n_nonempty,1):.0f}/簇, "
          f"min={min(u_counts.values())}, max={max(u_counts.values())}")
    print(f"  tel 分布:  avg={len(tel_labels)/max(n_nonempty,1):.0f}/簇, "
          f"min={min(t_counts.values())}, max={max(t_counts.values())}")

    t = time.time() - t0
    print(f"  聚类完成，耗时 {t:.1f}s")
    return udas_labels, tel_labels


def _run_faiss_kmeans(
    all_emb: np.ndarray, n_udas: int,
    n_clusters: int, niter: int, seed: int, gpu: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """用 FAISS k-means 聚类（多重初始化 + 去除 spherical）。"""
    import faiss

    d = all_emb.shape[1]
    kmeans = faiss.Kmeans(
        d=d, k=n_clusters, niter=niter, seed=seed,
        gpu=gpu, verbose=True, spherical=False,
        nredo=3,   # collapse 检测会快速 fallback 到 sklearn，不需要太多重试
    )
    kmeans.train(all_emb)

    _, labels = kmeans.index.search(all_emb, 1)
    labels = labels.ravel()
    return labels[:n_udas], labels[n_udas:]


def _run_sklearn_kmeans(
    all_emb: np.ndarray, n_udas: int,
    n_clusters: int, niter: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 sklearn MiniBatchKMeans 聚类（更适合高维球面数据）。
    先 PCA 降维到 256D 加速。
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ❌ sklearn 未安装！请运行: pip install scikit-learn")
        raise

    print(f"  Step 1: PCA 4096D → 256D ...")
    pca = PCA(n_components=256, random_state=seed)
    emb_pca = pca.fit_transform(all_emb)
    print(f"  PCA 完成: 解释方差={pca.explained_variance_ratio_.sum():.3f}")

    print(f"  Step 2: MiniBatchKMeans {n_clusters} 簇 ...")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        batch_size=4096, max_iter=niter,
        n_init=5, verbose=True,
    )
    labels = km.fit_predict(emb_pca)
    print(f"  MiniBatchKMeans 完成, 耗时 {time.time()-t0:.1f}s")
    return labels[:n_udas], labels[n_udas:]


# ══════════════════════════════════════════════════════
#  3. 簇内 Hungarian 配对
# ══════════════════════════════════════════════════════

def hungarian_batch_in_cluster(
    udas_emb: np.ndarray,
    tel_emb: np.ndarray,
    cluster_u_idx: List[int],
    cluster_t_idx: List[int],
    batch_size: int = 16,
) -> Tuple[List[dict], List[int], List[int]]:
    """
    在一个簇内，用 Hungarian 算法生成 batch_size 个配对。

    Args:
        udas_emb: 全部 udas embedding
        tel_emb: 全部 tel embedding
        cluster_u_idx: 该簇内 udas 的全局下标
        cluster_t_idx: 该簇内 tel 的全局下标
        batch_size: 每 batch 每侧数量

    Returns:
        (batches, leftovers_u, leftovers_t)
        batches: [{"udas": [idx...], "tel": [idx...], "sim_matrix": [[...]]}, ...]
        leftovers_u: 未配对的 udas 全局下标
        leftovers_t: 未配对的 tel 全局下标
    """
    cluster_u_arr = np.array(cluster_u_idx, dtype=np.int64)
    cluster_t_arr = np.array(cluster_t_idx, dtype=np.int64)

    U = len(cluster_u_idx)
    T = len(cluster_t_idx)
    n_batches = min(U // batch_size, T // batch_size)

    if n_batches == 0:
        return [], list(cluster_u_idx), list(cluster_t_idx)

    batches = []
    # 用列表维护剩余下标（局部下标，非全局）
    remain_u = list(range(U))
    remain_t = list(range(T))

    for bi in range(n_batches):
        ru, rt = remain_u, remain_t

        # 相似度矩阵 (len(ru), len(rt))
        sim = udas_emb[cluster_u_arr[ru]] @ tel_emb[cluster_t_arr[rt]].T

        # Hungarian: 求最大化总和的一对一匹配
        # linear_sum_assignment 默认最小化，传入 -sim
        row_ind, col_ind = linear_sum_assignment(-sim)

        # 获取每对匹配的相似度，降序排列
        pair_scores = [(sim[row_ind[k], col_ind[k]], row_ind[k], col_ind[k])
                       for k in range(len(row_ind))]
        pair_scores.sort(key=lambda x: -x[0])

        # 取 top batch_size 对
        top_pairs = pair_scores[:batch_size]

        # 转为全局下标
        batch_u = [int(cluster_u_arr[ru[p[1]]]) for p in top_pairs]
        batch_t = [int(cluster_t_arr[rt[p[2]]]) for p in top_pairs]

        # 该 batch 的相似度子矩阵 (batch_size × batch_size)
        batch_rows = [p[1] for p in top_pairs]
        batch_cols = [p[2] for p in top_pairs]
        batch_sim = sim[np.ix_(batch_rows, batch_cols)].tolist()

        # 统计 batch 内相似度
        diag_sim = [p[0] for p in top_pairs]

        batches.append({
            "batch_idx_in_cluster": bi,
            "udas": batch_u,
            "tel": batch_t,
            "sim_matrix": batch_sim,
            "diag_sim_mean": float(np.mean(diag_sim)),
            "diag_sim_min": float(np.min(diag_sim)),
            "diag_sim_max": float(np.max(diag_sim)),
        })

        # 移除已使用的下标（local → 映射到实际值）
        used_u = set(ru[p[1]] for p in top_pairs)
        used_t = set(rt[p[2]] for p in top_pairs)
        remain_u = [i for i in ru if i not in used_u]
        remain_t = [j for j in rt if j not in used_t]

    leftovers_u = [int(cluster_u_arr[i]) for i in remain_u]
    leftovers_t = [int(cluster_t_arr[j]) for j in remain_t]

    return batches, leftovers_u, leftovers_t


# ══════════════════════════════════════════════════════
#  4. 跨簇 Leftover 合并
# ══════════════════════════════════════════════════════

def merge_leftovers_simple(
    udas_emb: np.ndarray,
    tel_emb: np.ndarray,
    all_leftovers_u: List[int],
    all_leftovers_t: List[int],
    batch_size: int = 16,
) -> Tuple[List[dict], List[int], List[int]]:
    """
    跨簇合并 leftovers。直接对全部 leftovers 做一次 Hungarian。

    Returns:
        (batches, remaining_u, remaining_t)
    """
    print(f"\n{'=' * 60}")
    print(f"  跨簇 Leftover 合并")
    print(f"  剩余 udas: {len(all_leftovers_u)}, tel: {len(all_leftovers_t)}")
    print(f"{'=' * 60}")

    leftover_u = list(all_leftovers_u)
    leftover_t = list(all_leftovers_t)

    batches, ru, rt = hungarian_batch_in_cluster(
        udas_emb, tel_emb,
        leftover_u, leftover_t,
        batch_size,
    )
    return batches, ru, rt


# ══════════════════════════════════════════════════════
#  5. 保存
# ══════════════════════════════════════════════════════

def save_batches(
    batches: List[dict],
    udas_meta: List[dict],
    tel_meta: List[dict],
    output_dir: Path,
    dataset_name: str = "egoudas_egotel",
):
    """
    保存 batch 到文件。

    输出:
      - {output_dir}/batches_info.json: 汇总信息
      - {output_dir}/batch_000000.json ~ batch_N.json: 每个 batch 明细
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  保存 {len(batches)} 个 batch 到 {output_dir}")
    print(f"{'=' * 60}")

    # 写入每个 batch
    all_diag_means = []
    for i, batch in enumerate(batches):
        # 注入 metadata 信息
        batch_with_meta = {
            "batch_id": i,
            "n_udas": len(batch["udas"]),
            "n_tel": len(batch["tel"]),
            "diag_sim_mean": batch["diag_sim_mean"],
            "diag_sim_min": batch["diag_sim_min"],
            "diag_sim_max": batch["diag_sim_max"],
            "udas": [
                {**udas_meta[idx],
                 "embedding_idx": int(idx)}
                for idx in batch["udas"]
            ],
            "tel": [
                {**tel_meta[idx],
                 "embedding_idx": int(idx)}
                for idx in batch["tel"]
            ],
            "sim_matrix": batch["sim_matrix"],
        }

        batch_path = output_dir / f"batch_{i:06d}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_with_meta, f, ensure_ascii=False, indent=2)

        all_diag_means.append(batch["diag_sim_mean"])

    # 汇总信息
    info = {
        "dataset": dataset_name,
        "n_batches": len(batches),
        "batch_size_per_side": len(batches[0]["udas"]) if batches else 0,
        "total_udas_used": sum(len(b["udas"]) for b in batches),
        "total_tel_used": sum(len(b["tel"]) for b in batches),
        "diag_sim_mean_avg": float(np.mean(all_diag_means)) if all_diag_means else 0,
        "diag_sim_mean_std": float(np.std(all_diag_means)) if all_diag_means else 0,
        "diag_sim_min_avg": float(np.mean([b["diag_sim_min"] for b in batches])) if batches else 0,
        "batch_sizes": [len(b["udas"]) for b in batches],
        "diag_sim_percentiles": {
            "p10": float(np.percentile(all_diag_means, 10)) if all_diag_means else 0,
            "p25": float(np.percentile(all_diag_means, 25)) if all_diag_means else 0,
            "p50": float(np.percentile(all_diag_means, 50)) if all_diag_means else 0,
            "p75": float(np.percentile(all_diag_means, 75)) if all_diag_means else 0,
            "p90": float(np.percentile(all_diag_means, 90)) if all_diag_means else 0,
        },
    }

    info_path = output_dir / f"{dataset_name}_batches_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"  共 {len(batches)} 个 batch")
    print(f"  相似度均值: {info['diag_sim_mean_avg']:.4f} ± {info['diag_sim_mean_std']:.4f}")
    print(f"  已保存汇总 → {info_path}")
    return info


# ══════════════════════════════════════════════════════
#  6. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Batch Pairing — EgouDas + EgoTel 配成 16:16 batch")
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help=f"k-means 簇数 (默认 {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每 batch 每侧数量 (默认 {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--niter", type=int, default=100,
                        help="k-means 迭代轮数 (默认 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    parser.add_argument("--gpu", action="store_true",
                        help="使用 GPU 加速 FAISS k-means")
    parser.add_argument("--skip_clustering", action="store_true",
                        help="跳过聚类，直接从已保存的 labels 加载")
    parser.add_argument("--emb_dir", type=str, default=str(EMBEDDING_DIR),
                        help=f"embedding 目录 (默认 {EMBEDDING_DIR})")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认 {OUTPUT_DIR})")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Step 1: 加载数据 ──
    udas_emb, tel_emb, udas_meta, tel_meta = load_data(emb_dir)

    # ── Step 2: 聚类 ──
    labels_path = emb_dir / "cluster_labels.npz"

    if args.skip_clustering and labels_path.exists():
        print(f"\n  跳过聚类，从 {labels_path} 加载")
        data = np.load(labels_path)
        udas_labels = data["udas_labels"]
        tel_labels = data["tel_labels"]
    else:
        udas_labels, tel_labels = run_kmeans(
            udas_emb, tel_emb,
            n_clusters=args.n_clusters,
            niter=args.niter,
            seed=args.seed,
            gpu=args.gpu,
        )
        # 缓存 labels 以便快速重跑
        np.savez(labels_path, udas_labels=udas_labels, tel_labels=tel_labels)
        print(f"  已缓存 labels → {labels_path}")

    # ── Step 3: 逐簇 Hungarian 配对 ──
    print(f"\n{'=' * 60}")
    print(f"  逐簇 Hungarian 配对 (batch_size={args.batch_size})")
    print(f"{'=' * 60}")

    all_batches = []
    all_leftovers_u = []
    all_leftovers_t = []
    total_clusters = max(udas_labels.max(), tel_labels.max()) + 1

    t0 = time.time()
    for c in range(total_clusters):
        c_u_idx = np.where(udas_labels == c)[0].tolist()
        c_t_idx = np.where(tel_labels == c)[0].tolist()

        if not c_u_idx or not c_t_idx:
            all_leftovers_u.extend(c_u_idx)
            all_leftovers_t.extend(c_t_idx)
            continue

        batches, leftover_u, leftover_t = hungarian_batch_in_cluster(
            udas_emb, tel_emb,
            c_u_idx, c_t_idx,
            batch_size=args.batch_size,
        )

        all_batches.extend(batches)
        all_leftovers_u.extend(leftover_u)
        all_leftovers_t.extend(leftover_t)

        if (c + 1) % 200 == 0 or c == total_clusters - 1:
            n_batches_sofar = len(all_batches)
            n_paired = n_batches_sofar * args.batch_size
            elapsed = time.time() - t0
            print(f"    簇 {c + 1}/{total_clusters}: "
                  f"已生成 {n_batches_sofar} 个 batch ({n_paired} 对), "
                  f"leftovers: U={len(all_leftovers_u)} T={len(all_leftovers_t)} "
                  f"耗时 {elapsed:.1f}s")

    cluster_time = time.time() - t0
    print(f"\n  簇内配对完成: {len(all_batches)} 个 batch ({cluster_time:.1f}s)")
    print(f"  leftovers: udas={len(all_leftovers_u)}, tel={len(all_leftovers_t)}")

    # ── Step 4: 跨簇 leftovers 合并 ──
    if all_leftovers_u and all_leftovers_t:
        merge_batches, remaining_u, remaining_t = merge_leftovers_simple(
            udas_emb, tel_emb,
            all_leftovers_u, all_leftovers_t,
            batch_size=args.batch_size,
        )
        all_batches.extend(merge_batches)
        print(f"  跨簇合并完成: 额外 {len(merge_batches)} 个 batch")
        print(f"  最终未配对: udas={len(remaining_u)}, tel={len(remaining_t)}")
    else:
        remaining_u, remaining_t = [], []

    # ── Step 5: 保存 ──
    info = save_batches(all_batches, udas_meta, tel_meta, out_dir)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ 完成！总耗时 {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  总 batch 数: {info['n_batches']}")
    print(f"  使用率: udas {info['total_udas_used']}/{len(udas_meta)} "
          f"({100 * info['total_udas_used'] / len(udas_meta):.1f}%), "
          f"tel {info['total_tel_used']}/{len(tel_meta)} "
          f"({100 * info['total_tel_used'] / len(tel_meta):.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
