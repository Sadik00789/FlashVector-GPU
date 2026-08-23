"""
FlashVector-GPU vs Meta Faiss-GPU Comparative Benchmark
"""

import time
import numpy as np

def run_comparative_benchmark(dim=128, num_vectors=50000, num_queries=1000, top_k=10):
    print("=" * 60)
    print(f"FLASHVECTOR-GPU VS FAISS-GPU BENCHMARK (N={num_vectors}, D={dim}, K={top_k})")
    print("=" * 60)

    np.random.seed(42)
    dataset = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # 1. Exact CPU Ground Truth
    print("[1/3] Computing exact Euclidean ground truth...")
    gt_start = time.time()
    # Compute for first 100 queries
    sample_queries = queries[:100]
    gt_labels = []
    for q in sample_queries:
        dists = np.sum((dataset - q) ** 2, axis=1)
        gt_labels.append(np.argsort(dists)[:top_k])
    print(f"Ground truth calculated in {time.time() - gt_start:.2f}s")

    # 2. FlashVector-GPU
    print("\n[2/3] Benchmarking FlashVector-GPU (Ampere sm_86)...")
    try:
        import gpu_vector_index
        idx = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=num_vectors, m=32, ef_construction=128)
        b_start = time.time()
        idx.build(dataset)
        print(f"FlashVector-GPU Index built in {time.time() - b_start:.2f}s")

        for ef in [16, 32, 64, 128, 256]:
            t0 = time.time()
            labels, dists = idx.search(queries, top_k=top_k, ef_search=ef)
            elapsed = time.time() - t0
            qps = num_queries / elapsed

            # Recall on sample
            matched = sum(len(set(labels[i]).intersection(set(gt_labels[i]))) for i in range(100))
            recall = matched / (100 * top_k)
            print(f"  efSearch={ef:<4} | QPS: {qps:>8.1f} | Recall@{top_k}: {recall:.4f} | Latency: {(elapsed/num_queries)*1e6:>6.1f} µs")
    except ImportError:
        print("  gpu_vector_index module not available. Build with `maturin develop`.")

    # 3. Faiss comparison
    print("\n[3/3] Benchmarking Meta Faiss (if installed)...")
    try:
        import faiss
        quantizer = faiss.IndexFlatL2(dim)
        index_faiss = faiss.IndexIVFFlat(quantizer, dim, 256, faiss.METRIC_L2)
        index_faiss.train(dataset)
        index_faiss.add(dataset)

        for nprobe in [1, 4, 8, 16, 32]:
            index_faiss.nprobe = nprobe
            t0 = time.time()
            D, I = index_faiss.search(queries, top_k)
            elapsed = time.time() - t0
            qps = num_queries / elapsed
            matched = sum(len(set(I[i]).intersection(set(gt_labels[i]))) for i in range(100))
            recall = matched / (100 * top_k)
            print(f"  nprobe={nprobe:<4} | QPS: {qps:>8.1f} | Recall@{top_k}: {recall:.4f} | Latency: {(elapsed/num_queries)*1e6:>6.1f} µs")
    except ImportError:
        print("  faiss not installed. Skipping Faiss comparison.")


if __name__ == "__main__":
    run_comparative_benchmark()
