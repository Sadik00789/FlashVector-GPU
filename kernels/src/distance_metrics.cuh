#ifndef FLASHVECTOR_DISTANCE_METRICS_CUH
#define FLASHVECTOR_DISTANCE_METRICS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "../include/types.h"

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffffU

namespace flashvector {

// Warp-level sum reduction using __shfl_down_sync across all 32 lanes
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
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
// 32 threads in the warp cooperatively compute the dot product of the difference
__device__ __forceinline__ float warp_l2_distance(
    const float* __restrict__ query,
    const float* __restrict__ target,
    int dim,
    int lane_id
) {
    float partial_sum = 0.0f;
    
    // Process in strides of WARP_SIZE
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
    return -dot; // Negative for min-heap top-k
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

} // namespace flashvector

#endif // FLASHVECTOR_DISTANCE_METRICS_CUH
