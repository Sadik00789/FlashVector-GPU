#ifndef FLASHVECTOR_BITONIC_TOPK_CUH
#define FLASHVECTOR_BITONIC_TOPK_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffffU

namespace flashvector {

// Device-side Bitonic Sort step using warp shuffle
__device__ __forceinline__ void bitonic_stage_warp(
    float& key,
    uint32_t& val,
    int stage,
    int step,
    int lane_id
) {
    int partner = lane_id ^ (1 << step);
    float partner_key = __shfl_xor_sync(FULL_WARP_MASK, key, 1 << step);
    uint32_t partner_val = __shfl_xor_sync(FULL_WARP_MASK, val, 1 << step);

    bool direction = (lane_id & (1 << stage)) == 0;
    bool should_swap = (key > partner_key) == direction;

    if (partner > lane_id && should_swap) {
        key = partner_key;
        val = partner_val;
    } else if (partner < lane_id && !should_swap) {
        key = partner_key;
        val = partner_val;
    }
}

// 32-element Bitonic Sort across a single warp in registers
__device__ __forceinline__ void warp_bitonic_sort_32(
    float& key,
    uint32_t& val,
    int lane_id
) {
    // Stage 0: 2-element sub-sequences
    bitonic_stage_warp(key, val, 1, 0, lane_id);

    // Stage 1: 4-element sub-sequences
    bitonic_stage_warp(key, val, 2, 1, lane_id);
    bitonic_stage_warp(key, val, 2, 0, lane_id);

    // Stage 2: 8-element sub-sequences
    bitonic_stage_warp(key, val, 3, 2, lane_id);
    bitonic_stage_warp(key, val, 3, 1, lane_id);
    bitonic_stage_warp(key, val, 3, 0, lane_id);

    // Stage 3: 16-element sub-sequences
    bitonic_stage_warp(key, val, 4, 3, lane_id);
    bitonic_stage_warp(key, val, 4, 2, lane_id);
    bitonic_stage_warp(key, val, 4, 1, lane_id);
    bitonic_stage_warp(key, val, 4, 0, lane_id);

    // Stage 4: 32-element sequence
    bitonic_stage_warp(key, val, 5, 4, lane_id);
    bitonic_stage_warp(key, val, 5, 3, lane_id);
    bitonic_stage_warp(key, val, 5, 2, lane_id);
    bitonic_stage_warp(key, val, 5, 1, lane_id);
    bitonic_stage_warp(key, val, 5, 0, lane_id);
}

// Fixed-capacity sorted candidate queue for HNSW & IVF beam search
template <int MAX_CAPACITY = 256>
struct CandidateList {
    uint32_t ids[MAX_CAPACITY];
    float distances[MAX_CAPACITY];
    bool visited[MAX_CAPACITY];
    int size;
    int capacity;

    __device__ __forceinline__ void init(int max_cap) {
        size = 0;
        capacity = (max_cap < MAX_CAPACITY) ? max_cap : MAX_CAPACITY;
    }

    __device__ __forceinline__ bool insert(uint32_t id, float dist) {
        if (size >= capacity && dist >= distances[size - 1]) {
            return false;
        }

        int insert_pos = size;
        for (int i = 0; i < size; ++i) {
            if (ids[i] == id) {
                return false;
            }
            if (dist < distances[i]) {
                insert_pos = i;
                for (int j = i; j < size; ++j) {
                    if (ids[j] == id) return false;
                }
                break;
            }
        }

        if (insert_pos >= capacity) {
            return false;
        }

        int end = (size < capacity) ? size : (capacity - 1);
        for (int i = end; i > insert_pos; --i) {
            ids[i] = ids[i - 1];
            distances[i] = distances[i - 1];
            visited[i] = visited[i - 1];
        }

        ids[insert_pos] = id;
        distances[insert_pos] = dist;
        visited[insert_pos] = false;

        if (size < capacity) {
            size++;
        }
        return true;
    }

    __device__ __forceinline__ int get_next_unvisited() const {
        for (int i = 0; i < size; ++i) {
            if (!visited[i]) {
                return i;
            }
        }
        return -1;
    }
};

// Block-wide parallel top-k merger across all threads in a thread block
template <int MAX_K = 64>
__device__ __inline__ void block_bitonic_merge_topk(
    const CandidateList<MAX_K>& local_list,
    GpuSearchResult* smem_merged,
    int top_k,
    int tid,
    int block_size
) {
    // 1. Thread 0 initializes merged queue in shared memory
    if (tid == 0) {
        for (int k = 0; k < top_k; ++k) {
            smem_merged[k].id = 0xFFFFFFFFU;
            smem_merged[k].distance = 1e30f;
        }
    }
    __syncthreads();

    // 2. Sequential/Reduction merge into shared memory top-k list
    // Each thread tries to insert its candidates into the shared memory top-k
    for (int t = 0; t < block_size; ++t) {
        if (tid == t) {
            for (int k = 0; k < local_list.size; ++k) {
                uint32_t id = local_list.ids[k];
                float dist = local_list.distances[k];

                if (dist < smem_merged[top_k - 1].distance) {
                    // Check duplicate
                    bool duplicate = false;
                    int insert_pos = -1;

                    for (int i = 0; i < top_k; ++i) {
                        if (smem_merged[i].id == id) {
                            duplicate = true;
                            break;
                        }
                        if (dist < smem_merged[i].distance && insert_pos == -1) {
                            insert_pos = i;
                        }
                    }

                    if (!duplicate && insert_pos >= 0 && insert_pos < top_k) {
                        for (int j = top_k - 1; j > insert_pos; --j) {
                            smem_merged[j] = smem_merged[j - 1];
                        }
                        smem_merged[insert_pos].id = id;
                        smem_merged[insert_pos].distance = dist;
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace flashvector

#endif // FLASHVECTOR_BITONIC_TOPK_CUH
