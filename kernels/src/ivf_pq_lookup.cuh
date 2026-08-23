#ifndef FLASHVECTOR_IVF_PQ_LOOKUP_CUH
#define FLASHVECTOR_IVF_PQ_LOOKUP_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

#define MAX_PQ_M 64
#define PQ_CENTROIDS 256
// Stride of 257 floats eliminates 32-way shared memory bank conflicts
#define SMEM_PQ_STRIDE 257

void launch_ivf_pq_search(
    const IvfPqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    MetricType metric,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_IVF_PQ_LOOKUP_CUH
