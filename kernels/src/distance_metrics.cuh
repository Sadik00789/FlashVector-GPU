#ifndef FLASHVECTOR_DISTANCE_METRICS_CUH
#define FLASHVECTOR_DISTANCE_METRICS_CUH

#include <cuda_runtime.h>
#ifdef FLASHVECTOR_ENABLE_FP16
#include <cuda_fp16.h>
#endif
#include <math.h>
#include "../include/types.h"

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffffU

namespace flashvector {

// Device-side bitmask predicate: nullptr means every ID is allowed.
__device__ __forceinline__ bool bitmask_allows(const uint32_t* __restrict__ d_mask, uint32_t id) {
    if (d_mask == nullptr) return true;
    return ((d_mask[id >> 5] >> (id & 31U)) & 1U) != 0U;
}

// Warp-level sum reduction using __shfl_down_sync across all 32 lanes
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Sub-warp sum reduction across `team_size` contiguous lanes (4, 8, 16, or 32)
__device__ __forceinline__ float subwarp_reduce_sum(float val, int team_size, unsigned int team_mask) {
    for (int offset = team_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(team_mask, val, offset, team_size);
    }
    return val;
}

// Warp-level min reduction returning pair of (min_val, min_idx)
__device__ __forceinline__ void warp_reduce_min(float& val, uint32_t& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(FULL_WARP_MASK, val, offset);
        uint32_t other_idx = __shfl_down_sync(FULL_WARP_MASK, idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Warp-level Euclidean (L2) squared distance between query vector and dataset vector
__device__ __forceinline__ float warp_l2_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id
) {
    float partial_sum = 0.0f;

    for (int d = lane_id; d < dim; d += WARP_SIZE) {
        float diff = query[d] - target[d];
        partial_sum = fmaf(diff, diff, partial_sum);
    }

    float total_dist = warp_reduce_sum(partial_sum);
    return __shfl_sync(FULL_WARP_MASK, total_dist, 0);
}

// Warp-level Cosine distance: 1.0f - (dot / (norm_a * norm_b))
__device__ __forceinline__ float warp_cosine_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id
) {
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_t = 0.0f;

    for (int d = lane_id; d < dim; d += WARP_SIZE) {
        float q_val = query[d];
        float t_val = target[d];
        dot = fmaf(q_val, t_val, dot);
        norm_q = fmaf(q_val, q_val, norm_q);
        norm_t = fmaf(t_val, t_val, norm_t);
    }

    dot = warp_reduce_sum(dot);
    norm_q = warp_reduce_sum(norm_q);
    norm_t = warp_reduce_sum(norm_t);

    dot = __shfl_sync(FULL_WARP_MASK, dot, 0);
    norm_q = __shfl_sync(FULL_WARP_MASK, norm_q, 0);
    norm_t = __shfl_sync(FULL_WARP_MASK, norm_t, 0);

    float denom = sqrtf(norm_q) * sqrtf(norm_t) + 1e-8f;
    float similarity = dot / denom;
    return 1.0f - similarity;
}

// Warp-level Inner Product distance: -dot (for minimization in top-k priority queue)
__device__ __forceinline__ float warp_ip_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id
) {
    float dot = 0.0f;

    for (int d = lane_id; d < dim; d += WARP_SIZE) {
        dot = fmaf(query[d], target[d], dot);
    }

    dot = warp_reduce_sum(dot);
    dot = __shfl_sync(FULL_WARP_MASK, dot, 0);
    return -dot;
}

// Dispatcher for any supported metric
__device__ __forceinline__ float warp_compute_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id,
    MetricType metric
) {
    switch (metric) {
        case METRIC_L2:
            return warp_l2_distance(query, target, dim, lane_id);
        case METRIC_COSINE:
            return warp_cosine_distance(query, target, dim, lane_id);
        case METRIC_INNER_PRODUCT:
            return warp_ip_distance(query, target, dim, lane_id);
        default:
            return warp_l2_distance(query, target, dim, lane_id);
    }
}

// Sub-warp distance computation across `team_size` threads (for CAGRA parallel teams).
// Returns the reduced distance in all lanes of the sub-warp team.
__device__ __forceinline__ float subwarp_compute_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id,
    int team_size,
    MetricType metric
) {
    int sub_lane = lane_id & (team_size - 1);
    int team_base = lane_id & ~(team_size - 1);
    unsigned int team_mask = (team_size >= WARP_SIZE)
        ? FULL_WARP_MASK
        : (((1U << team_size) - 1U) << team_base);

    if (metric == METRIC_L2) {
        float acc = 0.0f;
        for (int d = sub_lane; d < dim; d += team_size) {
            float diff = query[d] - target[d];
            acc = fmaf(diff, diff, acc);
        }
        float total = subwarp_reduce_sum(acc, team_size, team_mask);
        return __shfl_sync(team_mask, total, 0, team_size);
    }
    if (metric == METRIC_INNER_PRODUCT) {
        float dot = 0.0f;
        for (int d = sub_lane; d < dim; d += team_size) {
            dot = fmaf(query[d], target[d], dot);
        }
        float total = subwarp_reduce_sum(dot, team_size, team_mask);
        return -__shfl_sync(team_mask, total, 0, team_size);
    }

    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_t = 0.0f;
    for (int d = sub_lane; d < dim; d += team_size) {
        float q_v = query[d];
        float t_v = target[d];
        dot = fmaf(q_v, t_v, dot);
        norm_q = fmaf(q_v, q_v, norm_q);
        norm_t = fmaf(t_v, t_v, norm_t);
    }
    dot = __shfl_sync(team_mask, subwarp_reduce_sum(dot, team_size, team_mask), 0, team_size);
    norm_q = __shfl_sync(team_mask, subwarp_reduce_sum(norm_q, team_size, team_mask), 0, team_size);
    norm_t = __shfl_sync(team_mask, subwarp_reduce_sum(norm_t, team_size, team_mask), 0, team_size);
    float denom = sqrtf(norm_q) * sqrtf(norm_t) + 1e-8f;
    return 1.0f - (dot / denom);
}

#ifdef FLASHVECTOR_ENABLE_FP16
// Warp-level FP16 target distance against FP32 query
__device__ __forceinline__ float warp_compute_distance_fp16(
    const float* __restrict__ query,
    const __half* __restrict__ target,
    int dim,
    int lane_id,
    MetricType metric
) {
    if (metric == METRIC_L2) {
        float acc = 0.0f;
        for (int d = lane_id; d < dim; d += WARP_SIZE) {
            float diff = query[d] - __half2float(target[d]);
            acc = fmaf(diff, diff, acc);
        }
        return __shfl_sync(FULL_WARP_MASK, warp_reduce_sum(acc), 0);
    }
    if (metric == METRIC_INNER_PRODUCT) {
        float dot = 0.0f;
        for (int d = lane_id; d < dim; d += WARP_SIZE) {
            dot = fmaf(query[d], __half2float(target[d]), dot);
        }
        return -__shfl_sync(FULL_WARP_MASK, warp_reduce_sum(dot), 0);
    }
    float dot = 0.0f, norm_q = 0.0f, norm_t = 0.0f;
    for (int d = lane_id; d < dim; d += WARP_SIZE) {
        float q_v = query[d];
        float t_v = __half2float(target[d]);
        dot = fmaf(q_v, t_v, dot);
        norm_q = fmaf(q_v, q_v, norm_q);
        norm_t = fmaf(t_v, t_v, norm_t);
    }
    dot = __shfl_sync(FULL_WARP_MASK, warp_reduce_sum(dot), 0);
    norm_q = __shfl_sync(FULL_WARP_MASK, warp_reduce_sum(norm_q), 0);
    norm_t = __shfl_sync(FULL_WARP_MASK, warp_reduce_sum(norm_t), 0);
    return 1.0f - (dot / (sqrtf(norm_q) * sqrtf(norm_t) + 1e-8f));
}
#endif // FLASHVECTOR_ENABLE_FP16

} // namespace flashvector

#endif // FLASHVECTOR_DISTANCE_METRICS_CUH
