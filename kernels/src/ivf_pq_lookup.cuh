#ifndef FLASHVECTOR_IVF_PQ_LOOKUP_CUH
#define FLASHVECTOR_IVF_PQ_LOOKUP_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

// Stride 257 eliminates 32-way shared-memory bank conflicts when threads in a
// warp read smem_lut[m * 257 + code] in parallel.
#define SMEM_PQ_STRIDE 257

void launch_ivf_pq_search(
    const IvfPqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_IVF_PQ_LOOKUP_CUH
