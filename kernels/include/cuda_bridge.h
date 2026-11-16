#ifndef FLASHVECTOR_CUDA_BRIDGE_H
#define FLASHVECTOR_CUDA_BRIDGE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device Management & Multi-GPU P2P
int cuda_init_device(int device_id);
int cuda_set_device(int device_id);
int cuda_get_device_count(int* count);
int cuda_get_device_memory(size_t* free_bytes, size_t* total_bytes);
int cuda_device_synchronize(void);
int cuda_enable_peer_access(int device, int peer);
int cuda_memcpy_peer_async(
    void* dst,
    int dst_device,
    const void* src,
    int src_device,
    size_t bytes,
    void* stream
);

// Memory Management (Pinned, Device, Mapped Host)
int cuda_malloc_device(void** ptr, size_t bytes);
int cuda_free_device(void* ptr);
int cuda_malloc_host(void** ptr, size_t bytes);
int cuda_free_host(void* ptr);
int cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
int cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);
int cuda_memset_device_async(void* dst, int value, size_t bytes, void* stream);
int cuda_host_get_device_pointer(void** dev_ptr, void* host_ptr);

// Stream Management
int cuda_create_stream(void** stream);
int cuda_destroy_stream(void* stream);
int cuda_sync_stream(void* stream);

// HNSW Beam Search Kernel Dispatch (with optional device-side bitmask filter)
int cuda_hnsw_search_batch(
    const HnswGpuGraph* graph,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t dim,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    TraversalHop* d_out_hops,
    uint32_t* d_out_hop_counts,
    uint32_t max_hops_per_query,
    void* stream
);

// HNSW Debug Beam Search (records per-iteration candidate queue snapshots + hops)
int cuda_hnsw_search_batch_debug(
    const HnswGpuGraph* graph,
    const float* d_query,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    TraversalHop* d_out_hops,
    uint32_t* d_out_hop_count,
    uint32_t max_hops,
    BeamStepRecord* d_out_steps,
    uint32_t* d_out_step_count,
    uint32_t max_steps,
    void* stream
);

// CAGRA Flat Fixed-Out-Degree Graph Search
int cuda_cagra_search_batch(
    const CagraGpuGraph* graph,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    TraversalHop* d_out_hops,
    uint32_t* d_out_hop_counts,
    uint32_t max_hops_per_query,
    BeamStepRecord* d_out_steps,
    uint32_t* d_out_step_count,
    uint32_t max_steps,
    void* stream
);

// IVF-PQ ADC Search (supports device bitmask & two-stage rerank up to M=10240)
int cuda_ivf_pq_search_batch(
    const IvfPqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

// IVF-SQ8 Scalar Quantization Search
int cuda_ivf_sq_search_batch(
    const IvfSqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

// IVF-RaBitQ Randomized Binary Quantization (FWHT + popcount + exact rerank)
int cuda_ivf_rabitq_search_batch(
    const IvfRabitqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

// FP16 Tensor Core (wmma 16x16x16) Batched Search
int cuda_fp16_tc_search_batch(
    const Fp16GpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

// Flat Delta-Buffer Scan (for concurrent secondary stream search on live inserts)
int cuda_flat_scan_batch(
    const float* d_vectors,
    const uint32_t* d_vec_ids,
    uint32_t num_vectors,
    uint32_t dim,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

// Device-side Top-K Merge across Shards or Main + Delta Buffers
int cuda_merge_topk(
    const GpuSearchResult* d_shard_results,
    uint32_t num_shards,
    uint32_t num_queries,
    uint32_t shard_top_k,
    uint32_t out_top_k,
    GpuSearchResult* d_out_results,
    void* stream
);

// CSR Sparse Dot-Product + Top-K + On-Device Reciprocal Rank Fusion (RRF)
int cuda_sparse_search_batch(
    const SparseCsrGpuMatrix* matrix,
    const uint32_t* d_q_indptr,
    const uint32_t* d_q_indices,
    const float* d_q_values,
    uint32_t num_queries,
    uint32_t top_k,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
);

int cuda_rrf_fuse(
    const GpuSearchResult* d_dense_results,
    uint32_t dense_k,
    const GpuSearchResult* d_sparse_results,
    uint32_t sparse_k,
    uint32_t num_queries,
    uint32_t out_top_k,
    float rrf_k,
    float dense_weight,
    float sparse_weight,
    GpuSearchResult* d_out_results,
    void* stream
);

// Sanity & Verification Test Kernels
int cuda_compute_distances_warp(
    const float* d_queries,
    const float* d_dataset,
    uint32_t num_queries,
    uint32_t num_vectors,
    uint32_t dim,
    MetricType metric,
    float* d_out_distances,
    void* stream
);

int cuda_bitonic_sort_test(
    const float* d_keys_in,
    const uint32_t* d_vals_in,
    float* d_keys_out,
    uint32_t* d_vals_out,
    uint32_t n,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif // FLASHVECTOR_CUDA_BRIDGE_H
