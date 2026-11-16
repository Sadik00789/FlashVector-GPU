#include "cagra_search.cuh"

namespace flashvector {

#define CAGRA_WARPS_PER_BLOCK 4
#define CAGRA_THREADS_PER_BLOCK (CAGRA_WARPS_PER_BLOCK * WARP_SIZE)

#define CAGRA_DBG_WARPS_PER_BLOCK 2
#define CAGRA_DBG_THREADS_PER_BLOCK (CAGRA_DBG_WARPS_PER_BLOCK * WARP_SIZE)

// Fast unfiltered CAGRA flat fixed-out-degree graph search (4 warps/block).
// Uses parallel lock-free atomicCAS probing on the shared-memory visited hash
// table and 4 parallel 8-lane sub-warp teams to evaluate multiple neighbors
// concurrently per warp iteration.
__global__ void cagra_traverse_fast_kernel(
    const float* __restrict__ d_vectors,
    const uint32_t* __restrict__ d_adjacency,
    const uint32_t* __restrict__ d_seed_nodes,
    uint32_t num_nodes,
    uint32_t dim,
    uint32_t graph_degree,
    uint32_t entry_point,
    uint32_t num_seeds,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    GpuSearchResult* __restrict__ d_out_results
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (global_warp_id >= (int)num_queries) return;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int team_id = lane_id / CAGRA_TEAM_SIZE;
    int team_lane = lane_id % CAGRA_TEAM_SIZE;

    uint32_t query_idx = (uint32_t)global_warp_id;
    const float* query = &d_queries[query_idx * dim];

    __shared__ CandidateList<CAGRA_MAX_BEAM> s_candidates[CAGRA_WARPS_PER_BLOCK];
    __shared__ uint32_t s_visited_hash[CAGRA_WARPS_PER_BLOCK][CAGRA_VISITED_SIZE];
    __shared__ uint32_t s_pending_ids[CAGRA_WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ float s_pending_dists[CAGRA_WARPS_PER_BLOCK][WARP_SIZE];

    CandidateList<CAGRA_MAX_BEAM>& candidates = s_candidates[warp_in_block];
    uint32_t* visited_table = s_visited_hash[warp_in_block];
    uint32_t* pending_ids = s_pending_ids[warp_in_block];
    float* pending_dists = s_pending_dists[warp_in_block];

    uint32_t clamped_ef = ef_search;
    if (clamped_ef < top_k) clamped_ef = top_k;
    if (clamped_ef > CAGRA_MAX_BEAM) clamped_ef = CAGRA_MAX_BEAM;
    if (clamped_ef == 0) clamped_ef = 1;

    if (lane_id == 0) {
        candidates.init((int)clamped_ef);
    }
    for (int i = lane_id; i < CAGRA_VISITED_SIZE; i += WARP_SIZE) {
        visited_table[i] = 0xFFFFFFFFU;
    }
    __syncwarp();

    // Seed beam cooperatively using entry_point + optional flat seed nodes
    uint32_t total_seeds = (num_seeds > 0 && d_seed_nodes != nullptr) ? num_seeds : 1U;
    for (uint32_t base = 0; base < total_seeds; base += CAGRA_TEAMS_PER_WARP) {
        uint32_t s_idx = base + (uint32_t)team_id;
        uint32_t seed = 0xFFFFFFFFU;
        if (s_idx < total_seeds) {
            seed = (num_seeds > 0 && d_seed_nodes != nullptr) ? d_seed_nodes[s_idx] : entry_point;
            if (seed >= num_nodes) seed = entry_point;
        }
        bool active_seed = (seed < num_nodes);
        if (active_seed && team_lane == 0) {
            if (cagra_claim_visited_atomic(visited_table, seed)) {
                pending_ids[team_id] = seed;
            } else {
                pending_ids[team_id] = 0xFFFFFFFFU;
            }
        } else if (team_lane == 0) {
            pending_ids[team_id] = 0xFFFFFFFFU;
        }
        __syncwarp();

        uint32_t eval_seed = pending_ids[team_id];
        uint32_t safe_seed = (eval_seed < num_nodes) ? eval_seed : 0U;
        float d = subwarp_compute_distance(
            query, &d_vectors[safe_seed * dim], (int)dim, lane_id, CAGRA_TEAM_SIZE, metric
        );
        if (eval_seed < num_nodes && team_lane == 0) {
            pending_dists[team_id] = d;
        }
        __syncwarp();

        if (lane_id == 0) {
            for (int t = 0; t < CAGRA_TEAMS_PER_WARP; ++t) {
                if (pending_ids[t] < num_nodes) {
                    candidates.insert(pending_ids[t], pending_dists[t]);
                }
            }
        }
        __syncwarp();
    }

    int max_iterations = (int)clamped_ef * 2;
    for (int iter = 0; iter < max_iterations; ++iter) {
        int current_idx = -1;
        uint32_t current_node = 0;
        if (lane_id == 0) {
            current_idx = candidates.get_next_unvisited();
            if (current_idx >= 0) {
                current_node = candidates.ids[current_idx];
                candidates.visited[current_idx] = true;
            }
        }
        current_idx = __shfl_sync(FULL_WARP_MASK, current_idx, 0);
        if (current_idx < 0) break;
        current_node = __shfl_sync(FULL_WARP_MASK, current_node, 0);

        const uint32_t* neighbors = &d_adjacency[current_node * graph_degree];

        for (uint32_t n_offset = 0; n_offset < graph_degree; n_offset += WARP_SIZE) {
            uint32_t n_idx = n_offset + (uint32_t)lane_id;
            uint32_t neighbor_id = (n_idx < graph_degree) ? neighbors[n_idx] : 0xFFFFFFFFU;

            bool need_eval = false;
            if (neighbor_id < num_nodes) {
                need_eval = cagra_claim_visited_atomic(visited_table, neighbor_id);
            }

            unsigned int active_mask = __ballot_sync(FULL_WARP_MASK, need_eval);
            if (active_mask == 0) continue;

            int num_active = __popc(active_mask);
            int slot = __popc(active_mask & ((1U << lane_id) - 1U));
            if (need_eval) {
                pending_ids[slot] = neighbor_id;
            }
            __syncwarp();

            // Evaluate queued neighbors in parallel batches of CAGRA_TEAMS_PER_WARP (4)
            for (int batch = 0; batch < num_active; batch += CAGRA_TEAMS_PER_WARP) {
                int idx = batch + team_id;
                bool has_target = (idx < num_active);
                uint32_t target = has_target ? pending_ids[idx] : 0U;
                if (target >= num_nodes) target = 0U;
                float dist = subwarp_compute_distance(
                    query, &d_vectors[target * dim], (int)dim, lane_id, CAGRA_TEAM_SIZE, metric
                );
                if (has_target && team_lane == 0) {
                    pending_dists[idx] = dist;
                }
                __syncwarp();
            }

            if (lane_id == 0) {
                for (int i = 0; i < num_active; ++i) {
                    candidates.insert(pending_ids[i], pending_dists[i]);
                }
            }
            __syncwarp();
        }
    }

    if (lane_id == 0) {
        uint32_t out_base = query_idx * top_k;
        int count = candidates.size < (int)top_k ? candidates.size : (int)top_k;
        for (int k = 0; k < count; ++k) {
            d_out_results[out_base + k].id = candidates.ids[k];
            d_out_results[out_base + k].distance = candidates.distances[k];
        }
        for (int k = count; k < (int)top_k; ++k) {
            d_out_results[out_base + k].id = 0xFFFFFFFFU;
            d_out_results[out_base + k].distance = 1e30f;
        }
    }
}

// Filtered and/or debug CAGRA search kernel (2 warps/block).
template <bool ENABLE_DEBUG>
__global__ void cagra_traverse_filtered_kernel(
    const float* __restrict__ d_vectors,
    const uint32_t* __restrict__ d_adjacency,
    const uint32_t* __restrict__ d_seed_nodes,
    uint32_t num_nodes,
    uint32_t dim,
    uint32_t graph_degree,
    uint32_t entry_point,
    uint32_t num_seeds,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results,
    TraversalHop* __restrict__ d_out_hops,
    uint32_t* __restrict__ d_out_hop_counts,
    uint32_t max_hops_per_query,
    BeamStepRecord* __restrict__ d_out_steps,
    uint32_t* __restrict__ d_out_step_counts,
    uint32_t max_steps_per_query
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (global_warp_id >= (int)num_queries) return;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int team_id = lane_id / CAGRA_TEAM_SIZE;
    int team_lane = lane_id % CAGRA_TEAM_SIZE;

    uint32_t query_idx = (uint32_t)global_warp_id;
    const float* query = &d_queries[query_idx * dim];

    __shared__ CandidateList<CAGRA_MAX_BEAM> s_beam[CAGRA_DBG_WARPS_PER_BLOCK];
    __shared__ CandidateList<CAGRA_MAX_BEAM> s_valid_results[CAGRA_DBG_WARPS_PER_BLOCK];
    __shared__ uint32_t s_visited_hash[CAGRA_DBG_WARPS_PER_BLOCK][CAGRA_VISITED_SIZE];
    __shared__ uint32_t s_pending_ids[CAGRA_DBG_WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ float s_pending_dists[CAGRA_DBG_WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t s_pending_valid[CAGRA_DBG_WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t s_hop_count[CAGRA_DBG_WARPS_PER_BLOCK];
    __shared__ uint32_t s_step_count[CAGRA_DBG_WARPS_PER_BLOCK];

    CandidateList<CAGRA_MAX_BEAM>& beam = s_beam[warp_in_block];
    CandidateList<CAGRA_MAX_BEAM>& valid_results = s_valid_results[warp_in_block];
    uint32_t* visited_table = s_visited_hash[warp_in_block];
    uint32_t* pending_ids = s_pending_ids[warp_in_block];
    float* pending_dists = s_pending_dists[warp_in_block];
    uint8_t* pending_valid = s_pending_valid[warp_in_block];

    uint32_t clamped_ef = ef_search;
    if (clamped_ef < top_k) clamped_ef = top_k;
    if (clamped_ef > CAGRA_MAX_BEAM) clamped_ef = CAGRA_MAX_BEAM;
    if (clamped_ef == 0) clamped_ef = 1;

    if (lane_id == 0) {
        beam.init((int)clamped_ef);
        valid_results.init((int)(top_k > 0 ? top_k : 1));
        s_hop_count[warp_in_block] = 0;
        s_step_count[warp_in_block] = 0;
    }
    for (int i = lane_id; i < CAGRA_VISITED_SIZE; i += WARP_SIZE) {
        visited_table[i] = 0xFFFFFFFFU;
    }
    __syncwarp();

    uint32_t total_seeds = (num_seeds > 0 && d_seed_nodes != nullptr) ? num_seeds : 1U;
    for (uint32_t base = 0; base < total_seeds; base += CAGRA_TEAMS_PER_WARP) {
        uint32_t s_idx = base + (uint32_t)team_id;
        uint32_t seed = 0xFFFFFFFFU;
        if (s_idx < total_seeds) {
            seed = (num_seeds > 0 && d_seed_nodes != nullptr) ? d_seed_nodes[s_idx] : entry_point;
            if (seed >= num_nodes) seed = entry_point;
        }
        if (team_lane == 0) {
            if (seed < num_nodes && cagra_claim_visited_atomic(visited_table, seed)) {
                pending_ids[team_id] = seed;
                pending_valid[team_id] = bitmask_allows(d_valid_mask, seed) ? 1U : 0U;
            } else {
                pending_ids[team_id] = 0xFFFFFFFFU;
                pending_valid[team_id] = 0U;
            }
        }
        __syncwarp();

        uint32_t eval_seed = pending_ids[team_id];
        uint32_t safe_seed = (eval_seed < num_nodes) ? eval_seed : 0U;
        float d = subwarp_compute_distance(
            query, &d_vectors[safe_seed * dim], (int)dim, lane_id, CAGRA_TEAM_SIZE, metric
        );
        if (eval_seed < num_nodes && team_lane == 0) {
            pending_dists[team_id] = d;
        }
        __syncwarp();

        if (lane_id == 0) {
            for (int t = 0; t < CAGRA_TEAMS_PER_WARP; ++t) {
                if (pending_ids[t] < num_nodes) {
                    beam.insert(pending_ids[t], pending_dists[t]);
                    if (pending_valid[t]) {
                        valid_results.insert(pending_ids[t], pending_dists[t]);
                    }
                }
            }
        }
        __syncwarp();
    }

    int max_iterations = (int)clamped_ef * 2;
    for (int iter = 0; iter < max_iterations; ++iter) {
        int current_idx = -1;
        uint32_t current_node = 0;
        float current_dist = 0.0f;
        uint32_t hop_start = 0;

        if (lane_id == 0) {
            current_idx = beam.get_next_unvisited();
            if (current_idx >= 0) {
                current_node = beam.ids[current_idx];
                current_dist = beam.distances[current_idx];
                beam.visited[current_idx] = true;
                hop_start = s_hop_count[warp_in_block];
            }
        }
        current_idx = __shfl_sync(FULL_WARP_MASK, current_idx, 0);
        if (current_idx < 0) break;
        current_node = __shfl_sync(FULL_WARP_MASK, current_node, 0);
        current_dist = __shfl_sync(FULL_WARP_MASK, current_dist, 0);
        hop_start = __shfl_sync(FULL_WARP_MASK, hop_start, 0);

        int need_bridge = 0;
        if (lane_id == 0) {
            need_bridge = (valid_results.size < (int)top_k) ? 1 : 0;
        }
        need_bridge = __shfl_sync(FULL_WARP_MASK, need_bridge, 0);

        const uint32_t* neighbors = &d_adjacency[current_node * graph_degree];

        for (uint32_t n_offset = 0; n_offset < graph_degree; n_offset += WARP_SIZE) {
            uint32_t n_idx = n_offset + (uint32_t)lane_id;
            uint32_t neighbor_id = (n_idx < graph_degree) ? neighbors[n_idx] : 0xFFFFFFFFU;

            bool allowed = false;
            bool need_eval = false;
            if (neighbor_id < num_nodes) {
                allowed = bitmask_allows(d_valid_mask, neighbor_id);
                if (allowed || need_bridge) {
                    need_eval = cagra_claim_visited_atomic(visited_table, neighbor_id);
                }
            }

            unsigned int active_mask = __ballot_sync(FULL_WARP_MASK, need_eval);
            if (active_mask == 0) continue;

            int num_active = __popc(active_mask);
            int slot = __popc(active_mask & ((1U << lane_id) - 1U));
            if (need_eval) {
                pending_ids[slot] = neighbor_id;
                pending_valid[slot] = allowed ? 1U : 0U;
            }
            __syncwarp();

            for (int batch = 0; batch < num_active; batch += CAGRA_TEAMS_PER_WARP) {
                int idx = batch + team_id;
                bool has_target = (idx < num_active);
                uint32_t target = has_target ? pending_ids[idx] : 0U;
                if (target >= num_nodes) target = 0U;
                float dist = subwarp_compute_distance(
                    query, &d_vectors[target * dim], (int)dim, lane_id, CAGRA_TEAM_SIZE, metric
                );
                if (has_target && team_lane == 0) {
                    pending_dists[idx] = dist;
                }
                __syncwarp();
            }

            if (lane_id == 0) {
                for (int i = 0; i < num_active; ++i) {
                    uint32_t nid = pending_ids[i];
                    float ndist = pending_dists[i];
                    beam.insert(nid, ndist);
                    if (pending_valid[i]) {
                        valid_results.insert(nid, ndist);
                    }
                    if constexpr (ENABLE_DEBUG) {
                        if (d_out_hops != nullptr) {
                            uint32_t h_idx = s_hop_count[warp_in_block];
                            if (h_idx < max_hops_per_query) {
                                uint32_t pos = query_idx * max_hops_per_query + h_idx;
                                d_out_hops[pos].step = h_idx;
                                d_out_hops[pos].from_node = current_node;
                                d_out_hops[pos].to_node = nid;
                                d_out_hops[pos].distance = ndist;
                            }
                            s_hop_count[warp_in_block] = h_idx + 1U;
                        }
                    }
                }
            }
            __syncwarp();
        }

        if constexpr (ENABLE_DEBUG) {
            if (lane_id == 0 && d_out_steps != nullptr && d_out_step_counts != nullptr) {
                uint32_t step_idx = s_step_count[warp_in_block];
                if (step_idx < max_steps_per_query) {
                    BeamStepRecord& rec = d_out_steps[query_idx * max_steps_per_query + step_idx];
                    rec.iteration = (uint32_t)iter;
                    rec.expanded_node = current_node;
                    rec.expanded_distance = current_dist;
                    rec.eval_start = hop_start;
                    uint32_t hop_end = s_hop_count[warp_in_block];
                    rec.eval_count = (hop_end > hop_start) ? (hop_end - hop_start) : 0U;
                    int snap_n = beam.size < BEAM_DEBUG_MAX_CANDS ? beam.size : BEAM_DEBUG_MAX_CANDS;
                    rec.cand_count = (uint32_t)snap_n;
                    for (int c = 0; c < snap_n; ++c) {
                        rec.candidates[c].id = beam.ids[c];
                        rec.candidates[c].distance = beam.distances[c];
                        rec.candidates[c].visited = beam.visited[c] ? 1U : 0U;
                        rec.candidates[c]._pad = 0U;
                    }
                }
                s_step_count[warp_in_block] = step_idx + 1U;
            }
        }
        __syncwarp();
    }

    if (lane_id == 0) {
        uint32_t out_base = query_idx * top_k;
        int count = valid_results.size < (int)top_k ? valid_results.size : (int)top_k;
        for (int k = 0; k < count; ++k) {
            d_out_results[out_base + k].id = valid_results.ids[k];
            d_out_results[out_base + k].distance = valid_results.distances[k];
        }
        for (int k = count; k < (int)top_k; ++k) {
            d_out_results[out_base + k].id = 0xFFFFFFFFU;
            d_out_results[out_base + k].distance = 1e30f;
        }
        if (d_out_hop_counts != nullptr) {
            d_out_hop_counts[query_idx] = s_hop_count[warp_in_block];
        }
        if (d_out_step_counts != nullptr) {
            d_out_step_counts[query_idx] = s_step_count[warp_in_block];
        }
    }
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
) {
    if (num_queries == 0 || graph->num_nodes == 0 || graph->graph_degree == 0) return;

    bool enable_debug = (d_out_steps != nullptr && d_out_step_counts != nullptr && max_steps_per_query > 0) ||
                        (d_out_hops != nullptr && d_out_hop_counts != nullptr && max_hops_per_query > 0);

    if (enable_debug || d_valid_mask != nullptr) {
        uint32_t num_blocks = (num_queries + CAGRA_DBG_WARPS_PER_BLOCK - 1) / CAGRA_DBG_WARPS_PER_BLOCK;
        if (enable_debug) {
            cagra_traverse_filtered_kernel<true><<<num_blocks, CAGRA_DBG_THREADS_PER_BLOCK, 0, stream>>>(
                graph->d_vectors, graph->d_adjacency, graph->d_seed_nodes,
                graph->num_nodes, graph->dim, graph->graph_degree,
                graph->entry_point, graph->num_seeds,
                d_queries, num_queries, top_k, ef_search, metric,
                d_valid_mask, d_out_results, d_out_hops, d_out_hop_counts,
                max_hops_per_query, d_out_steps, d_out_step_counts, max_steps_per_query
            );
        } else {
            cagra_traverse_filtered_kernel<false><<<num_blocks, CAGRA_DBG_THREADS_PER_BLOCK, 0, stream>>>(
                graph->d_vectors, graph->d_adjacency, graph->d_seed_nodes,
                graph->num_nodes, graph->dim, graph->graph_degree,
                graph->entry_point, graph->num_seeds,
                d_queries, num_queries, top_k, ef_search, metric,
                d_valid_mask, d_out_results, nullptr, nullptr, 0, nullptr, nullptr, 0
            );
        }
        return;
    }

    uint32_t num_blocks = (num_queries + CAGRA_WARPS_PER_BLOCK - 1) / CAGRA_WARPS_PER_BLOCK;
    cagra_traverse_fast_kernel<<<num_blocks, CAGRA_THREADS_PER_BLOCK, 0, stream>>>(
        graph->d_vectors, graph->d_adjacency, graph->d_seed_nodes,
        graph->num_nodes, graph->dim, graph->graph_degree,
        graph->entry_point, graph->num_seeds,
        d_queries, num_queries, top_k, ef_search, metric, d_out_results
    );
}

} // namespace flashvector

extern "C" {

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
    uint32_t* d_out_step_counts,
    uint32_t max_steps_per_query,
    void* stream
) {
    if (!graph || !d_queries || !d_out_results) return -1;

    flashvector::launch_cagra_search(
        graph, d_queries, num_queries, top_k, ef_search, metric,
        d_valid_mask, d_out_results, d_out_hops, d_out_hop_counts,
        max_hops_per_query, d_out_steps, d_out_step_counts, max_steps_per_query,
        (cudaStream_t)stream
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
