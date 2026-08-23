#ifndef FLASHVECTOR_HNSW_TRAVERSE_CUH
#define FLASHVECTOR_HNSW_TRAVERSE_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

#define MAX_BEAM_WIDTH 256
#define MAX_NEIGHBORS_PER_NODE 64
#define VISITED_HASH_TABLE_SIZE 1024

// 4-Way open-addressing visited hash table with linear probing
__device__ __forceinline__ bool insert_visited(uint32_t* table, uint32_t node_id) {
    uint32_t h = (node_id * 2654435761U) % VISITED_HASH_TABLE_SIZE;
    for (int p = 0; p < 4; ++p) {
        uint32_t idx = (h + p) % VISITED_HASH_TABLE_SIZE;
        if (table[idx] == node_id) return false; // Already visited
        if (table[idx] == 0xFFFFFFFFU) {
            table[idx] = node_id;
            return true; // Successfully marked as newly visited
        }
    }
    return true; // Table bucket full, proceed conservatively
}

void launch_hnsw_search(
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
    cudaStream_t stream
);

} // namespace flashvector

#endif // FLASHVECTOR_HNSW_TRAVERSE_CUH
