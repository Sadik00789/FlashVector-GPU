"""Reproducible multi-engine GPU benchmark suite for FlashVector-GPU v1.0.0.

Measures Recall@10, batched QPS, and mean per-query latency across CAGRA,
Warp HNSW, IVF-PQ, IVF-SQ8, IVF-RaBitQ, FP16 Tensor Core (wmma), on-device
bitmask filtering, and GPU Hybrid CSR Sparse + RRF, then renders
`assets/benchmark_pareto.png`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gpu_vector_index
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"


def make_clustered_dataset(
    num_vectors: int, dim: int, num_clusters: int, num_queries: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, size=(num_clusters, dim)).astype(np.float32)
    cluster_ids = np.arange(num_vectors) % num_clusters
    noise = rng.normal(0.0, 0.22, size=(num_vectors, dim)).astype(np.float32)
    dataset = np.ascontiguousarray(centers[cluster_ids] + noise, dtype=np.float32)

    q_cids = rng.integers(0, num_clusters, size=num_queries)
    q_noise = rng.normal(0.0, 0.22, size=(num_queries, dim)).astype(np.float32)
    queries = np.ascontiguousarray(centers[q_cids] + q_noise, dtype=np.float32)
    return dataset, queries


def exact_topk_l2(dataset: np.ndarray, queries: np.ndarray, top_k: int) -> np.ndarray:
    d_norms = np.sum(dataset * dataset, axis=1, keepdims=True).T
    out = np.empty((queries.shape[0], top_k), dtype=np.uint32)
    chunk = 128
    for s in range(0, queries.shape[0], chunk):
        q = queries[s : s + chunk]
        q_norms = np.sum(q * q, axis=1, keepdims=True)
        dists = q_norms + d_norms - 2.0 * (q @ dataset.T)
        idx = np.argpartition(dists, top_k, axis=1)[:, :top_k]
        out[s : s + chunk] = idx.astype(np.uint32)
    return out


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    k = gt.shape[1]
    matched = sum(len(set(pred[i].tolist()) & set(gt[i].tolist())) for i in range(gt.shape[0]))
    return matched / (gt.shape[0] * k)


def best_qps(fn, num_queries: int, reps: int = 5) -> tuple[float, float, np.ndarray]:
    # Warmup
    labels, dists = fn()
    best_elapsed = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        labels, dists = fn()
        best_elapsed = min(best_elapsed, time.perf_counter() - t0)
    qps = num_queries / best_elapsed
    lat_us = (best_elapsed / num_queries) * 1e6
    return qps, lat_us, labels


def run() -> dict[str, object]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    dim = 128
    n_vecs = 50_000
    n_queries = 500
    top_k = 10

    print(f"Generating N={n_vecs:,} x {dim}-D clustered benchmark dataset (Q={n_queries})...")
    dataset, queries = make_clustered_dataset(n_vecs, dim, num_clusters=64, num_queries=n_queries)
    gt = exact_topk_l2(dataset, queries, top_k)

    index = gpu_vector_index.FlashVectorGPU(
        dim=dim, max_elements=n_vecs, m=32, ef_construction=128, nlist=128, m_pq=16, metric="l2"
    )
    t_build0 = time.perf_counter()
    index.build(dataset)
    build_sec = time.perf_counter() - t_build0
    print(f"Index built in {build_sec:.2f}s")

    rows: list[dict[str, object]] = []

    for ef in [32, 64, 128]:
        qps, lat, lbl = best_qps(
            lambda e=ef: index.search_cagra(queries, top_k=top_k, ef_search=e), n_queries
        )
        rows.append(
            {
                "engine": "CAGRA",
                "param": f"efSearch={ef}",
                "recall": round(compute_recall(lbl, gt), 4),
                "qps": round(qps, 1),
                "latency_us": round(lat, 2),
            }
        )

    for ef in [32, 64, 128, 256]:
        qps, lat, lbl = best_qps(
            lambda e=ef: index.search(queries, top_k=top_k, ef_search=e), n_queries
        )
        rows.append(
            {
                "engine": "Warp HNSW",
                "param": f"efSearch={ef}",
                "recall": round(compute_recall(lbl, gt), 4),
                "qps": round(qps, 1),
                "latency_us": round(lat, 2),
            }
        )

    for rk in [1, 32, 128, 512]:
        qps, lat, lbl = best_qps(
            lambda r=rk: index.search_ivf(queries, top_k=top_k, nprobe=16, rerank_k=r, mode="pq"),
            n_queries,
        )
        rows.append(
            {
                "engine": "IVF-PQ",
                "param": f"nprobe=16, rerank={rk}",
                "recall": round(compute_recall(lbl, gt), 4),
                "qps": round(qps, 1),
                "latency_us": round(lat, 2),
            }
        )

    for rk in [32, 128, 512]:
        qps, lat, lbl = best_qps(
            lambda r=rk: index.search_ivf(queries, top_k=top_k, nprobe=16, rerank_k=r, mode="sq8"),
            n_queries,
        )
        rows.append(
            {
                "engine": "IVF-SQ8",
                "param": f"nprobe=16, rerank={rk}",
                "recall": round(compute_recall(lbl, gt), 4),
                "qps": round(qps, 1),
                "latency_us": round(lat, 2),
            }
        )

    for rk in [32, 128, 512]:
        qps, lat, lbl = best_qps(
            lambda r=rk: index.search_ivf(
                queries, top_k=top_k, nprobe=16, rerank_k=r, mode="rabitq"
            ),
            n_queries,
        )
        rows.append(
            {
                "engine": "IVF-RaBitQ",
                "param": f"nprobe=16, rerank={rk}",
                "recall": round(compute_recall(lbl, gt), 4),
                "qps": round(qps, 1),
                "latency_us": round(lat, 2),
            }
        )

    qps_tc, lat_tc, lbl_tc = best_qps(lambda: index.search_fp16(queries, top_k=top_k), n_queries)
    rows.append(
        {
            "engine": "FP16 Tensor Core",
            "param": "wmma 16x16x16",
            "recall": round(compute_recall(lbl_tc, gt), 4),
            "qps": round(qps_tc, 1),
            "latency_us": round(lat_tc, 2),
        }
    )

    # On-device 25% selectivity bitmask filtering
    mask = (np.arange(n_vecs) % 4) == 0
    qps_f_cagra, lat_f_cagra, _ = best_qps(
        lambda: index.search_cagra(queries, top_k=top_k, ef_search=64, filter_mask=mask),
        n_queries,
    )
    qps_f_hnsw, lat_f_hnsw, _ = best_qps(
        lambda: index.search(queries, top_k=top_k, ef_search=64, filter_mask=mask),
        n_queries,
    )

    # Sparse CSR + Hybrid GPU RRF
    sparse_docs = [[(int(i % 1024), 1.0), (int((i * 13) % 4096), 0.5)] for i in range(n_vecs)]
    index.build_sparse(sparse_docs)
    sparse_qs = [[(int(q % 1024), 1.0), (int((q * 13) % 4096), 0.5)] for q in range(n_queries)]
    qps_hyb, lat_hyb, _ = best_qps(
        lambda: index.hybrid_search(queries, sparse_qs, top_k=top_k, candidate_k=64, ef_search=64),
        n_queries,
    )

    # High-dimensional (D=512) FP16 Tensor Core exact scan check
    dim_hi = 512
    n_hi = 10_000
    n_q_hi = 256
    ds_hi, qs_hi = make_clustered_dataset(
        n_hi, dim_hi, num_clusters=32, num_queries=n_q_hi, seed=99
    )
    gt_hi = exact_topk_l2(ds_hi, qs_hi, top_k)
    idx_hi = gpu_vector_index.FlashVectorGPU(
        dim=dim_hi, max_elements=n_hi, m=16, ef_construction=64, nlist=32, m_pq=16, metric="l2"
    )
    idx_hi.build(ds_hi)
    qps_hi_tc, lat_hi_tc, lbl_hi_tc = best_qps(
        lambda: idx_hi.search_fp16(qs_hi, top_k=top_k), n_q_hi
    )
    recall_hi_tc = round(compute_recall(lbl_hi_tc, gt_hi), 4)

    summary = {
        "n_vecs": n_vecs,
        "dim": dim,
        "n_queries": n_queries,
        "top_k": top_k,
        "build_sec": round(build_sec, 2),
        "rows": rows,
        "extras": {
            "cagra_filtered_25pct": {
                "qps": round(qps_f_cagra, 1),
                "latency_us": round(lat_f_cagra, 2),
            },
            "hnsw_filtered_25pct": {
                "qps": round(qps_f_hnsw, 1),
                "latency_us": round(lat_f_hnsw, 2),
            },
            "hybrid_sparse_rrf": {"qps": round(qps_hyb, 1), "latency_us": round(lat_hyb, 2)},
            "fp16_tc_512d_10k": {
                "recall": recall_hi_tc,
                "qps": round(qps_hi_tc, 1),
                "latency_us": round(lat_hi_tc, 2),
            },
        },
    }

    plot_pareto_chart(rows, summary["extras"], ASSETS_DIR / "benchmark_pareto.png")
    print(json.dumps(summary, indent=2))
    return summary


def plot_pareto_chart(rows: list[dict[str, object]], extras: dict, out_path: Path) -> None:
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)
    fig.patch.set_facecolor("#090b10")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f131d")
        ax.grid(True, linestyle="--", alpha=0.22, color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    styles = {
        "CAGRA": ("#00f0ff", "o-", 2.8, 7),
        "Warp HNSW": ("#38bdf8", "s--", 2.2, 6),
        "IVF-RaBitQ": ("#34d399", "^-", 2.2, 7),
        "IVF-SQ8": ("#f59e0b", "D:", 2.0, 6),
        "IVF-PQ": ("#a855f7", "v-.", 2.0, 6),
        "FP16 Tensor Core": ("#fb7185", "*", 0.0, 12),
    }

    for eng, (color, fmt, lw, ms) in styles.items():
        pts = [r for r in rows if r["engine"] == eng]
        if not pts:
            continue
        xs = [float(p["recall"]) for p in pts]
        ys = [float(p["qps"]) for p in pts]
        if lw > 0:
            ax1.plot(xs, ys, fmt, color=color, linewidth=lw, markersize=ms, label=eng)
        else:
            ax1.plot(xs, ys, fmt, color=color, markersize=ms, linestyle="None", label=eng)

    ax1.set_yscale("log")
    ax1.set_title(
        "Recall@10 vs. Batch QPS (50k × 128-D, RTX 3050 sm_86)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Recall@10", fontsize=10.5)
    ax1.set_ylabel("Queries Per Second (log scale)", fontsize=10.5)
    ax1.legend(
        loc="lower left", fontsize=8.5, frameon=True, facecolor="#090b10", edgecolor="#1e293b"
    )

    # Right panel: representative high-accuracy operating points latency comparison
    picks = [
        (
            "CAGRA\n(ef=64)",
            next(r for r in rows if r["engine"] == "CAGRA" and r["param"] == "efSearch=64"),
            "#00f0ff",
        ),
        (
            "Warp HNSW\n(ef=64)",
            next(r for r in rows if r["engine"] == "Warp HNSW" and r["param"] == "efSearch=64"),
            "#38bdf8",
        ),
        (
            "CAGRA\n25% Mask",
            {
                "latency_us": extras["cagra_filtered_25pct"]["latency_us"],
                "qps": extras["cagra_filtered_25pct"]["qps"],
            },
            "#22d3ee",
        ),
        (
            "IVF-RaBitQ\n(rk=128)",
            next(
                r for r in rows if r["engine"] == "IVF-RaBitQ" and "rerank=128" in str(r["param"])
            ),
            "#34d399",
        ),
        (
            "IVF-SQ8\n(rk=128)",
            next(r for r in rows if r["engine"] == "IVF-SQ8" and "rerank=128" in str(r["param"])),
            "#f59e0b",
        ),
        (
            "IVF-PQ\n(rk=128)",
            next(r for r in rows if r["engine"] == "IVF-PQ" and "rerank=128" in str(r["param"])),
            "#a855f7",
        ),
        (
            "Hybrid\nRRF",
            {
                "latency_us": extras["hybrid_sparse_rrf"]["latency_us"],
                "qps": extras["hybrid_sparse_rrf"]["qps"],
            },
            "#f472b6",
        ),
    ]
    labels_b = [p[0] for p in picks]
    lats = [float(p[1]["latency_us"]) for p in picks]
    cols = [p[2] for p in picks]

    bars = ax2.bar(labels_b, lats, color=cols, alpha=0.88, edgecolor="#090b10", linewidth=0.8)
    ax2.set_title(
        "Mean Per-Query Batch Latency (µs) — Lower is Faster",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("Latency (µs / query)", fontsize=10.5)
    ax2.tick_params(axis="x", labelsize=8.5)
    for bar, val in zip(bars, lats):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(lats) * 0.02,
            f"{val:.1f} µs",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#e2e8f0",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    run()
