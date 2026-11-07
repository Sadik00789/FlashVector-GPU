"""
Generate publication-quality QPS vs Recall@k Pareto Frontier plots
"""

import matplotlib.pyplot as plt

def plot_pareto_frontier(output_path="pareto_frontier.png"):
    print(f"Generating Pareto Frontier chart -> {output_path}...")

    # Data points: (Recall@10, QPS) measured on SIFT1M (1,000,000 x 128-D)
    # against the ann-benchmarks ground truth with 1,000 queries, on an
    # NVIDIA RTX 3050 Laptop (sm_86). See the README benchmark section.
    flashvector_hnsw = [
        (0.7568, 93686),
        (0.8767, 51229),
        (0.9511, 38338),
        (0.9848, 15719),
        (0.9958, 5629),
    ]

    flashvector_ivf_pq = [
        (0.3520, 83062),
        (0.4960, 34657),
        (0.5269, 22697),
        (0.5382, 14906),
        (0.5422, 9522),
        (0.5420, 5800),
    ]

    hnswlib_cpu = [
        (0.9800, 13413),
        (0.9948, 7427),
        (0.9981, 4342),
    ]

    plt.figure(figsize=(9, 6), dpi=300)

    # Plot FlashVector-GPU HNSW
    r_fv, q_fv = zip(*flashvector_hnsw)
    plt.plot(r_fv, q_fv, 'o-', color='#00f0ff', linewidth=3, markersize=8, label='FlashVector-GPU HNSW (Ampere sm_86)')

    # Plot FlashVector-GPU IVF-PQ
    r_pq, q_pq = zip(*flashvector_ivf_pq)
    plt.plot(r_pq, q_pq, 'D--', color='#a855f7', linewidth=2, markersize=6, label='FlashVector-GPU IVF-PQ (Ampere sm_86)')

    # Plot HNSWLib (CPU)
    r_hnsw, q_hnsw = zip(*hnswlib_cpu)
    plt.plot(r_hnsw, q_hnsw, '^:', color='#94a3b8', linewidth=2, markersize=6, label='HNSWLib (CPU, 12 threads)')

    plt.title('FlashVector-GPU: SIFT1M (128-D) QPS vs. Recall@10', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Recall@10 Accuracy', fontsize=12)
    plt.ylabel('Queries Per Second (QPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(frameon=True, facecolor='#0f1118', edgecolor='#1e2230', fontsize=10)

    plt.xlim(0.30, 1.00)
    plt.ylim(0, 100000)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved successfully to {output_path}")

if __name__ == "__main__":
    plot_pareto_frontier()
