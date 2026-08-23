#include "hnsw_traverse.cuh"
#include <stdio.h>

namespace flashvector {

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)

// Warp-cooperative HNSW beam-search graph traversal kernel
__global__ void hnsw_warp_traverse_kernel(
    const float* __restrict__ d_vectors,
    const uint32_t* __restrict__ d_adjacency,
    const uint32_t* __restrict__ d_degree,
    uint32_t num_nodes,
    uint32_t dim,
    uint32_t m_max,
    uint32_t entry_point,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t ef_search,
    MetricType metric,
    GpuSearchResult* __restrict__ d_out_results,
    TraversalHop* __restrict__ d_out_hops,
    uint32_t* __restrict__ d_out_hop_counts,
    uint32_t max_hops_per_query
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (global_warp_id >= num_queries) return;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    uint32_t query_idx = global_warp_id;
    const float* query = &d_queries[query_idx * dim];

    // Shared memory candidate list per warp
    __shared__ CandidateList<MAX_BEAM_WIDTH> s_candidates[WARPS_PER_BLOCK];
    __shared__ uint32_t s_hop_count[WARPS_PER_BLOCK];
    __shared__ uint32_t s_visited_hash[WARPS_PER_BLOCK][VISITED_HASH_TABLE_SIZE];

    CandidateList<MAX_BEAM_WIDTH>& candidates = s_candidates[warp_in_block];
    uint32_t* visited_table = s_visited_hash[warp_in_block];

    uint32_t clamped_ef = (ef_search > MAX_BEAM_WIDTH) ? MAX_BEAM_WIDTH : ef_search;

    // Initialize visited hash table and candidate queue in lane 0
    if (lane_id == 0) {
        candidates.init(clamped_ef);
        s_hop_count[warp_in_block] = 0;
    }
    
    // Clear 4-way visited table across warp
    for (int i = lane_id; i < VISITED_HASH_TABLE_SIZE; i += WARP_SIZE) {
        visited_table[i] = 0xFFFFFFFFU;
    }
    __syncwarp();

    // Compute distance from query to entry point
    float ep_dist = warp_compute_distance(query, &d_vectors[entry_point * dim], dim, lane_id, metric);

    if (lane_id == 0) {
        candidates.insert(entry_point, ep_dist);
        insert_visited(visited_table, entry_point);
    }
    __syncwarp();

    // Beam search main loop
    int max_iterations = (int)clamped_ef * 2;
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Step 1: Select closest unvisited candidate
        int current_idx = -1;
        uint32_t current_node = 0;
        float current_dist = 0.0f;

        if (lane_id == 0) {
            current_idx = candidates.get_next_unvisited();
            if (current_idx >= 0) {
                current_node = candidates.ids[current_idx];
                current_dist = candidates.distances[current_idx];
                candidates.visited[current_idx] = true;
            }
        }

        // Broadcast selected node to all warp lanes
        current_idx = __shfl_sync(FULL_WARP_MASK, current_idx, 0);
        if (current_idx < 0) {
            break; // All reachable candidates visited
        }
        current_node = __shfl_sync(FULL_WARP_MASK, current_node, 0);
        current_dist = __shfl_sync(FULL_WARP_MASK, current_dist, 0);

        // Fetch neighbor degree
        uint32_t degree = d_degree[current_node];
        if (degree > m_max) degree = m_max;

        const uint32_t* neighbors = &d_adjacency[current_node * m_max];

        // Step 2: Iterate over outgoing neighbor edges in coalesced batches of 32
        for (uint32_t n_offset = 0; n_offset < degree; n_offset += WARP_SIZE) {
            uint32_t n_idx = n_offset + lane_id;
            uint32_t neighbor_id = (n_idx < degree) ? neighbors[n_idx] : 0xFFFFFFFFU;

            // Check if valid neighbor and not already visited via 4-way hash table
            bool is_unvisited = false;
            if (neighbor_id != 0xFFFFFFFFU && neighbor_id < num_nodes) {
                is_unvisited = insert_visited(visited_table, neighbor_id);
            }

            // Ballot to find which lanes have valid unvisited neighbors
            unsigned int active_mask = __ballot_sync(FULL_WARP_MASK, is_unvisited);

            // Sequentially evaluate distances for each active neighbor using the entire warp
            while (active_mask != 0) {
                int leader_lane = __ffs(active_mask) - 1;
                uint32_t target_node = __shfl_sync(FULL_WARP_MASK, neighbor_id, leader_lane);

                // All 32 threads cooperatively compute distance to target_node
                float dist = warp_compute_distance(
                    query,
                    &d_vectors[target_node * dim],
                    dim,
                    lane_id,
                    metric
                );

                // Lane 0 inserts into candidate queue and records traversal hop
                if (lane_id == 0) {
                    candidates.insert(target_node, dist);

                    if (d_out_hops != nullptr && s_hop_count[warp_in_block] < max_hops_per_query) {
                        uint32_t hop_idx = s_hop_count[warp_in_block]++;
                        uint32_t out_hop_pos = query_idx * max_hops_per_query + hop_idx;
                        d_out_hops[out_hop_pos].step = hop_idx;
                        d_out_hops[out_hop_pos].from_node = current_node;
                        d_out_hops[out_hop_pos].to_node = target_node;
                        d_out_hops[out_hop_pos].distance = dist;
                    }
                }

                // Clear leader lane from active mask
                active_mask &= ~(1U << leader_lane);
            }
        }
        __syncwarp();
    }

    // Step 3: Write out top-k nearest neighbors
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

        if (d_out_hop_counts != nullptr) {
            d_out_hop_counts[query_idx] = s_hop_count[warp_in_block];
        }
    }
}

// Verification test kernel: computes distance matrix using warp reductions
__global__ void compute_distances_warp_kernel(
    const float* __restrict__ d_queries,
    const float* __restrict__ d_dataset,
    uint32_t num_queries,
    uint32_t num_vectors,
    uint32_t dim,
    MetricType metric,
    float* __restrict__ d_out_distances
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int total_elements = num_queries * num_vectors;
    if (global_warp_id >= total_elements) return;

    int lane_id = threadIdx.x % WARP_SIZE;

    uint32_t q_idx = global_warp_id / num_vectors;
    uint32_t v_idx = global_warp_id % num_vectors;

    const float* query = &d_queries[q_idx * dim];
    const float* vector = &d_dataset[v_idx * dim];

    float dist = warp_compute_distance(query, vector, dim, lane_id, metric);

    if (lane_id == 0) {
        d_out_distances[global_warp_id] = dist;
    }
}

// Verification test kernel: bitonic sort test in registers
__global__ void bitonic_sort_test_kernel(
    const float* __restrict__ d_keys_in,
    const uint32_t* __restrict__ d_vals_in,
    float* __restrict__ d_keys_out,
    uint32_t* __restrict__ d_vals_out,
    uint32_t n
) {
    int tid = threadIdx.x;
    if (tid >= WARP_SIZE) return;

    float key = (tid < n) ? d_keys_in[tid] : 1e30f;
    uint32_t val = (tid < n) ? d_vals_in[tid] : 0xFFFFFFFFU;

    warp_bitonic_sort_32(key, val, tid);

    if (tid < n) {
        d_keys_out[tid] = key;
        d_vals_out[tid] = val;
    }
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
) {
    if (num_queries == 0) return;

    uint32_t total_warps = num_queries;
    uint32_t num_blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    hnsw_warp_traverse_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        graph->d_vectors,
        graph->d_adjacency,
        graph->d_degree,
        graph->num_nodes,
        dim,
        graph->m_max,
        graph->entry_point,
        d_queries,
        num_queries,
        top_k,
        ef_search,
        metric,
        d_out_results,
        d_out_hops,
        d_out_hop_counts,
        max_hops_per_query
    );
}

} // namespace flashvector

// C FFI Bridge Implementation
extern "C" {

int cuda_init_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return (int)err;
    return (int)cudaFree(0); // Initialize context
}

int cuda_get_device_memory(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes) return -1;
    return (int)cudaMemGetInfo(free_bytes, total_bytes);
}

int cuda_device_synchronize(void) {
    return (int)cudaDeviceSynchronize();
}

int cuda_malloc_device(void** ptr, size_t bytes) {
    if (!ptr || bytes == 0) return -1;
    return (int)cudaMalloc(ptr, bytes);
}

int cuda_free_device(void* ptr) {
    if (!ptr) return 0;
    return (int)cudaFree(ptr);
}

int cuda_malloc_host(void** ptr, size_t bytes) {
    if (!ptr || bytes == 0) return -1;
    return (int)cudaHostAlloc(ptr, bytes, cudaHostAllocMapped);
}

int cuda_free_host(void* ptr) {
    if (!ptr) return 0;
    return (int)cudaFreeHost(ptr);
}

int cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    if (!dst || !src) return -1;
    if (bytes == 0) return 0;
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

int cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    if (!dst || !src) return -1;
    if (bytes == 0) return 0;
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

int cuda_create_stream(void** stream) {
    if (!stream) return -1;
    return (int)cudaStreamCreateWithFlags((cudaStream_t*)stream, cudaStreamNonBlocking);
}

int cuda_destroy_stream(void* stream) {
    if (!stream) return 0;
    return (int)cudaStreamDestroy((cudaStream_t)stream);
}

int cuda_sync_stream(void* stream) {
    if (!stream) return (int)cudaDeviceSynchronize();
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

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
) {
    if (!graph || !d_queries || !d_out_results) {
        return -1;
    }

    cudaStream_t s = (cudaStream_t)stream;
    flashvector::launch_hnsw_search(
        graph,
        d_queries,
        num_queries,
        dim,
        top_k,
        ef_search,
        metric,
        d_out_results,
        d_out_hops,
        d_out_hop_counts,
        max_hops_per_query,
        s
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_compute_distances_warp(
    const float* d_queries,
    const float* d_dataset,
    uint32_t num_queries,
    uint32_t num_vectors,
    uint32_t dim,
    MetricType metric,
    float* d_out_distances,
    void* stream
) {
    if (!d_queries || !d_dataset || !d_out_distances) return -1;

    uint32_t total_pairs = num_queries * num_vectors;
    uint32_t num_blocks = (total_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    flashvector::compute_distances_warp_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, (cudaStream_t)stream>>>(
        d_queries,
        d_dataset,
        num_queries,
        num_vectors,
        dim,
        metric,
        d_out_distances
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_bitonic_sort_test(
    const float* d_keys_in,
    const uint32_t* d_vals_in,
    float* d_keys_out,
    uint32_t* d_vals_out,
    uint32_t n,
    void* stream
) {
    if (!d_keys_in || !d_vals_in || !d_keys_out || !d_vals_out) return -1;

    flashvector::bitonic_sort_test_kernel<<<1, WARP_SIZE, 0, (cudaStream_t)stream>>>(
        d_keys_in,
        d_vals_in,
        d_keys_out,
        d_vals_out,
        n
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
