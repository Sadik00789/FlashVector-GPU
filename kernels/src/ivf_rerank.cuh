#ifndef FLASHVECTOR_IVF_RERANK_CUH
#define FLASHVECTOR_IVF_RERANK_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/types.h"
#include "distance_metrics.cuh"
#include "bitonic_topk.cuh"

namespace flashvector {

#define MAX_SHORTLIST_RERANK 10240u
#define IVF_STAGE2_THREADS 256

__host__ __device__ __forceinline__ uint32_t next_pow2_u32(uint32_t v) {
    if (v <= 1u) return 1u;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1u;
}

__host__ inline void ensure_dynamic_smem(const void* kernel_fn, size_t smem_bytes) {
    if (smem_bytes > 48u * 1024u) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
}

void launch_ivf_exact_rerank(
    const float* d_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t shortlist_m,
    uint32_t m_pow2,
    uint32_t top_k,
    MetricType metric,
    GpuSearchResult* d_shortlist,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
);

// Block-wide coarse quantizer: selects top-`nprobe` nearest Voronoi centroids into
// `smem_probed_clusters[0..nprobe)` and `smem_cluster_dists[0..nprobe)`.
__device__ __inline__ void ivf_find_nprobe_centroids(
    const float* __restrict__ query,
    const float* __restrict__ d_centroids,
    uint32_t dim,
    uint32_t nlist,
    uint32_t nprobe,
    uint32_t* smem_probed_clusters,
    float* smem_cluster_dists,
    int tid,
    int block_size
) {
    if (tid == 0) {
        for (uint32_t p = 0; p < nprobe; ++p) {
            smem_probed_clusters[p] = 0xFFFFFFFFU;
            smem_cluster_dists[p] = 1e30f;
        }
    }
    __syncthreads();

    uint32_t local_probed_c[32];
    float local_probed_d[32];
    int local_probed_count = 0;
    int max_local = (nprobe < 32u) ? (int)nprobe : 32;

    for (int i = 0; i < max_local; ++i) {
        local_probed_d[i] = 1e30f;
        local_probed_c[i] = 0xFFFFFFFFU;
    }

    for (uint32_t c = (uint32_t)tid; c < nlist; c += (uint32_t)block_size) {
        const float* centroid = &d_centroids[c * dim];
        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float diff = query[d] - centroid[d];
            dist = fmaf(diff, diff, dist);
        }

        if (dist < local_probed_d[max_local - 1]) {
            int insert_pos = max_local - 1;
            for (int j = 0; j < max_local; ++j) {
                if (dist < local_probed_d[j]) {
                    insert_pos = j;
                    break;
                }
            }
            for (int j = max_local - 1; j > insert_pos; --j) {
                local_probed_d[j] = local_probed_d[j - 1];
                local_probed_c[j] = local_probed_c[j - 1];
            }
            local_probed_d[insert_pos] = dist;
            local_probed_c[insert_pos] = c;
            if (local_probed_count < max_local) local_probed_count++;
        }
    }
    __syncthreads();

    for (int t = 0; t < block_size; ++t) {
        if (tid == t) {
            for (int i = 0; i < local_probed_count; ++i) {
                uint32_t c = local_probed_c[i];
                float dist = local_probed_d[i];

                if (dist < smem_cluster_dists[nprobe - 1]) {
                    int insert_pos = (int)nprobe - 1;
                    for (uint32_t j = 0; j < nprobe; ++j) {
                        if (dist < smem_cluster_dists[j]) {
                            insert_pos = (int)j;
                            break;
                        }
                    }
                    for (int j = (int)nprobe - 1; j > insert_pos; --j) {
                        smem_cluster_dists[j] = smem_cluster_dists[j - 1];
                        smem_probed_clusters[j] = smem_probed_clusters[j - 1];
                    }
                    smem_cluster_dists[insert_pos] = dist;
                    smem_probed_clusters[insert_pos] = c;
                }
            }
        }
        __syncthreads();
    }
}

// Helper to append one approximate candidate to a global bounded work buffer
// during Stage 1 and periodically prune down via parallel bitonic sort.
__device__ __inline__ void stage1_flush_if_full(
    GpuSearchResult* work_buf,
    uint32_t work_pow2,
    uint32_t shortlist_m,
    uint32_t* smem_count,
    float* smem_cutoff,
    int tid,
    int block_size
) {
    __syncthreads();
    uint32_t cnt = *smem_count;
    if (cnt + (uint32_t)block_size < work_pow2) return;

    for (uint32_t i = cnt + (uint32_t)tid; i < work_pow2; i += (uint32_t)block_size) {
        work_buf[i].id = 0xFFFFFFFFU;
        work_buf[i].distance = 1e30f;
    }
    __syncthreads();

    block_bitonic_sort_inplace(work_buf, work_pow2, tid, block_size);

    if (tid == 0) {
        *smem_count = shortlist_m;
        *smem_cutoff = work_buf[shortlist_m - 1].distance;
    }
    __syncthreads();
}

} // namespace flashvector

#endif // FLASHVECTOR_IVF_RERANK_CUH
