#ifndef FLASHVECTOR_IVF_QUANT_LOOKUP_CUH
#define FLASHVECTOR_IVF_QUANT_LOOKUP_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

void launch_ivf_sq_search(
    const IvfSqGpuTables* tables,
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

void launch_ivf_rabitq_search(
    const IvfRabitqGpuTables* tables,
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

void launch_fp16_tc_search(
    const Fp16GpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_IVF_QUANT_LOOKUP_CUH
