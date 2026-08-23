#ifndef FLASHVECTOR_CUDA_BRIDGE_H
#define FLASHVECTOR_CUDA_BRIDGE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device Management
int cuda_init_device(int device_id);
int cuda_get_device_memory(size_t* free_bytes, size_t* total_bytes);
int cuda_device_synchronize(void);

// Memory Management (Pinned & Device)
int cuda_malloc_device(void** ptr, size_t bytes);
int cuda_free_device(void* ptr);
int cuda_malloc_host(void** ptr, size_t bytes);
int cuda_free_host(void* ptr);
int cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
int cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);
int cuda_memset_device_async(void* dst, int value, size_t bytes, void* stream);

// Stream Management
int cuda_create_stream(void** stream);
int cuda_destroy_stream(void* stream);
int cuda_sync_stream(void* stream);

// HNSW Beam Search Kernel Dispatch
int cuda_hnsw_search_batch(
    const HnswGpuGraph* graph,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t dim,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    GpuSearchResult* d_out_results,
    TraversalHop* d_out_hops,
    uint32_t* d_out_hop_counts,
    uint32_t max_hops_per_query,
    void* stream
);

// IVF-PQ Asymmetric Distance Computation (ADC) Kernel Dispatch
int cuda_ivf_pq_search_batch(
    const IvfPqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    MetricType metric,
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
