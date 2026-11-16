#ifndef FLASHVECTOR_SPARSE_HYBRID_CUH
#define FLASHVECTOR_SPARSE_HYBRID_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

void launch_flat_scan(
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
    cudaStream_t stream
);

void launch_merge_topk(
    const GpuSearchResult* d_shard_results,
    uint32_t num_shards,
    uint32_t num_queries,
    uint32_t shard_top_k,
    uint32_t out_top_k,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

void launch_sparse_search(
    const SparseCsrGpuMatrix* matrix,
    const uint32_t* d_q_indptr,
    const uint32_t* d_q_indices,
    const float* d_q_values,
    uint32_t num_queries,
    uint32_t top_k,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

void launch_rrf_fuse(
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
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_SPARSE_HYBRID_CUH
