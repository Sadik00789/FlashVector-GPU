#ifndef FLASHVECTOR_CAGRA_SEARCH_CUH
#define FLASHVECTOR_CAGRA_SEARCH_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

#define CAGRA_MAX_BEAM 256
#define CAGRA_VISITED_SIZE 2048
#define CAGRA_TEAM_SIZE 8
#define CAGRA_TEAMS_PER_WARP (WARP_SIZE / CAGRA_TEAM_SIZE)

__device__ __forceinline__ bool cagra_claim_visited_atomic(uint32_t* table, uint32_t node_id) {
    uint32_t h = (node_id * 2654435761U) & (CAGRA_VISITED_SIZE - 1);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t idx = (h + p) & (CAGRA_VISITED_SIZE - 1);
        uint32_t prev = atomicCAS(&table[idx], 0xFFFFFFFFU, node_id);
        if (prev == node_id) return false;
        if (prev == 0xFFFFFFFFU) return true;
    }
    return true;
}

void launch_cagra_search(
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
    uint32_t* d_out_step_counts,
    uint32_t max_steps_per_query,
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_CAGRA_SEARCH_CUH
