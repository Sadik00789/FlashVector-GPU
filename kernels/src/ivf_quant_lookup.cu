#define FLASHVECTOR_ENABLE_FP16 1
#include "ivf_quant_lookup.cuh"
#include "ivf_rerank.cuh"
#include <mma.h>
#include <float.h>

using namespace nvcuda;

namespace flashvector {

#define QUANT_BLOCK_SIZE 128
#define MAX_SMALL_RERANK 32u

// ============================================================================
// 1. IVF-SQ8 (8-bit Scalar Quantization over Voronoi residuals)
// ============================================================================

__global__ void ivf_sq_small_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_sq_mins,
    const float* __restrict__ d_sq_scales,
    const uint8_t* __restrict__ d_sq_codes,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    const float* __restrict__ d_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t nlist,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t shortlist,
    bool do_rerank,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0 || top_k == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_sq[];
    float* smem_q_res = smem_sq; // [dim]
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_q_res[dim];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];
    size_t raw_off = ((size_t)dim * sizeof(float)) +
                     ((size_t)nprobe * (sizeof(uint32_t) + sizeof(float)));
    size_t aligned_off = (raw_off + 7u) & ~7u;
    GpuSearchResult* smem_merged_topk = (GpuSearchResult*)((char*)smem_sq + aligned_off);

    ivf_find_nprobe_centroids(
        query, d_centroids, dim, nlist, nprobe,
        smem_probed_clusters, smem_cluster_dists, tid, block_size
    );

    CandidateList<MAX_SMALL_RERANK> local_candidates;
    local_candidates.init((int)shortlist);

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        const float* centroid = &d_centroids[cluster_id * dim];
        for (uint32_t d = (uint32_t)tid; d < dim; d += (uint32_t)block_size) {
            smem_q_res[d] = query[d] - centroid[d];
        }
        __syncthreads();

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t i = (uint32_t)tid; i < list_len; i += (uint32_t)block_size) {
            uint32_t vec_id = d_ivf_vec_ids[list_start + i];
            if (!bitmask_allows(d_valid_mask, vec_id)) continue;

            const uint8_t* codes = &d_sq_codes[(size_t)vec_id * dim];
            float dist = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                float rec_res = fmaf((float)codes[d], __ldg(&d_sq_scales[d]), __ldg(&d_sq_mins[d]));
                float diff = smem_q_res[d] - rec_res;
                dist = fmaf(diff, diff, dist);
            }
            local_candidates.insert(vec_id, dist);
        }
        __syncthreads();
    }

    block_bitonic_merge_topk<MAX_SMALL_RERANK>(
        local_candidates, smem_merged_topk, (int)shortlist, tid, block_size
    );

    if (do_rerank && d_vectors != nullptr) {
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warps_per_block = block_size / WARP_SIZE;
        for (uint32_t k = (uint32_t)warp_id; k < shortlist; k += (uint32_t)warps_per_block) {
            uint32_t vec_id = smem_merged_topk[k].id;
            if (vec_id >= num_vectors) continue;
            float exact = warp_compute_distance(
                query, &d_vectors[(size_t)vec_id * dim], (int)dim, lane_id, metric
            );
            if (lane_id == 0) smem_merged_topk[k].distance = exact;
        }
        __syncthreads();
    }

    uint32_t sort_pow2 = next_pow2_u32(shortlist);
    for (uint32_t k = shortlist + (uint32_t)tid; k < sort_pow2; k += (uint32_t)block_size) {
        smem_merged_topk[k] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();
    block_bitonic_sort_inplace(smem_merged_topk, sort_pow2, tid, block_size);

    uint32_t out_base = query_idx * top_k;
    for (uint32_t k = (uint32_t)tid; k < top_k; k += (uint32_t)block_size) {
        d_out_results[out_base + k] = (k < shortlist)
            ? smem_merged_topk[k]
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
}

__global__ void ivf_sq_stage1_shortlist_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_sq_mins,
    const float* __restrict__ d_sq_scales,
    const uint8_t* __restrict__ d_sq_codes,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    uint32_t dim,
    uint32_t nlist,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t nprobe,
    uint32_t shortlist_m,
    uint32_t work_pow2,
    uint32_t out_m_pow2,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_work,
    GpuSearchResult* __restrict__ d_shortlist
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_sq[];
    float* smem_q_res = smem_sq;
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_q_res[dim];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];
    uint32_t* smem_count = (uint32_t*)&smem_cluster_dists[nprobe];
    float* smem_cutoff = (float*)&smem_count[1];

    if (tid == 0) {
        *smem_count = 0u;
        *smem_cutoff = 1e30f;
    }

    ivf_find_nprobe_centroids(
        query, d_centroids, dim, nlist, nprobe,
        smem_probed_clusters, smem_cluster_dists, tid, block_size
    );

    GpuSearchResult* work_buf = &d_work[(size_t)query_idx * work_pow2];

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        const float* centroid = &d_centroids[cluster_id * dim];
        for (uint32_t d = (uint32_t)tid; d < dim; d += (uint32_t)block_size) {
            smem_q_res[d] = query[d] - centroid[d];
        }
        __syncthreads();

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t base = 0; base < list_len; base += (uint32_t)block_size) {
            uint32_t i = base + (uint32_t)tid;
            if (i < list_len) {
                uint32_t vec_id = d_ivf_vec_ids[list_start + i];
                if (bitmask_allows(d_valid_mask, vec_id)) {
                    const uint8_t* codes = &d_sq_codes[(size_t)vec_id * dim];
                    float dist = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        float rec_res = fmaf(
                            (float)codes[d], __ldg(&d_sq_scales[d]), __ldg(&d_sq_mins[d])
                        );
                        float diff = smem_q_res[d] - rec_res;
                        dist = fmaf(diff, diff, dist);
                    }
                    if (dist < *smem_cutoff) {
                        uint32_t pos = atomicAdd(smem_count, 1u);
                        if (pos < work_pow2) {
                            work_buf[pos].id = vec_id;
                            work_buf[pos].distance = dist;
                        }
                    }
                }
            }
            stage1_flush_if_full(
                work_buf, work_pow2, shortlist_m, smem_count, smem_cutoff, tid, block_size
            );
        }
    }

    __syncthreads();
    uint32_t final_cnt = *smem_count;
    if (final_cnt > work_pow2) final_cnt = work_pow2;
    for (uint32_t i = final_cnt + (uint32_t)tid; i < work_pow2; i += (uint32_t)block_size) {
        work_buf[i] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();
    block_bitonic_sort_inplace(work_buf, work_pow2, tid, block_size);

    GpuSearchResult* out_buf = &d_shortlist[(size_t)query_idx * out_m_pow2];
    for (uint32_t i = (uint32_t)tid; i < out_m_pow2; i += (uint32_t)block_size) {
        out_buf[i] = (i < shortlist_m && i < final_cnt)
            ? work_buf[i]
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
}

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
) {
    if (num_queries == 0 || top_k == 0 || tables->nlist == 0 || tables->dim == 0) return;
    if (top_k > 256u) top_k = 256u;

    uint32_t clamped_nprobe = nprobe == 0u ? 1u : nprobe;
    if (clamped_nprobe > tables->nlist) clamped_nprobe = tables->nlist;

    uint32_t shortlist = rerank_k == 0u ? top_k : rerank_k;
    if (shortlist < top_k) shortlist = top_k;
    if (shortlist > MAX_SHORTLIST_RERANK) shortlist = MAX_SHORTLIST_RERANK;
    bool do_rerank = (rerank_k != 1u) && (tables->d_vectors != nullptr);

    if (shortlist <= MAX_SMALL_RERANK) {
        uint32_t sort_pow2 = next_pow2_u32(shortlist);
        size_t raw_smem = ((size_t)tables->dim * sizeof(float)) +
                          ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float)));
        size_t aligned_smem = (raw_smem + 7u) & ~7u;
        size_t smem_bytes = aligned_smem + ((size_t)sort_pow2 * sizeof(GpuSearchResult));
        ensure_dynamic_smem((const void*)ivf_sq_small_kernel, smem_bytes);

        ivf_sq_small_kernel<<<num_queries, QUANT_BLOCK_SIZE, smem_bytes, stream>>>(
            tables->d_centroids, tables->d_sq_mins, tables->d_sq_scales, tables->d_sq_codes,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids, tables->d_vectors,
            tables->num_vectors, tables->dim, tables->nlist,
            d_queries, num_queries, top_k, clamped_nprobe, shortlist,
            do_rerank, metric, d_valid_mask, d_out_results
        );
        return;
    }

    uint32_t m_pow2 = next_pow2_u32(shortlist);
    uint32_t work_pow2 = m_pow2 * 2u;
    uint32_t chunk_queries = num_queries < 32u ? num_queries : 32u;

    GpuSearchResult* d_work = nullptr;
    GpuSearchResult* d_shortlist = nullptr;
    if (cudaMallocAsync(&d_work, (size_t)chunk_queries * work_pow2 * sizeof(GpuSearchResult), stream) != cudaSuccess) {
        return;
    }
    if (cudaMallocAsync(&d_shortlist, (size_t)chunk_queries * m_pow2 * sizeof(GpuSearchResult), stream) != cudaSuccess) {
        cudaFreeAsync(d_work, stream);
        return;
    }

    size_t smem_stage1 = ((size_t)tables->dim * sizeof(float)) +
                         ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float))) +
                         sizeof(uint32_t) + sizeof(float);
    ensure_dynamic_smem((const void*)ivf_sq_stage1_shortlist_kernel, smem_stage1);

    for (uint32_t q_off = 0; q_off < num_queries; q_off += chunk_queries) {
        uint32_t q_cnt = num_queries - q_off;
        if (q_cnt > chunk_queries) q_cnt = chunk_queries;

        const float* q_ptr = &d_queries[(size_t)q_off * tables->dim];
        GpuSearchResult* out_ptr = &d_out_results[(size_t)q_off * top_k];

        ivf_sq_stage1_shortlist_kernel<<<q_cnt, IVF_STAGE2_THREADS, smem_stage1, stream>>>(
            tables->d_centroids, tables->d_sq_mins, tables->d_sq_scales, tables->d_sq_codes,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids,
            tables->dim, tables->nlist, q_ptr, q_cnt, clamped_nprobe,
            shortlist, work_pow2, m_pow2, d_valid_mask, d_work, d_shortlist
        );

        if (do_rerank) {
            launch_ivf_exact_rerank(
                tables->d_vectors, tables->num_vectors, tables->dim,
                q_ptr, q_cnt, shortlist, m_pow2, top_k, metric,
                d_shortlist, out_ptr, stream
            );
        } else {
            for (uint32_t qi = 0; qi < q_cnt; ++qi) {
                cudaMemcpyAsync(
                    &out_ptr[(size_t)qi * top_k],
                    &d_shortlist[(size_t)qi * m_pow2],
                    (size_t)top_k * sizeof(GpuSearchResult),
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
        }
    }

    cudaFreeAsync(d_work, stream);
    cudaFreeAsync(d_shortlist, stream);
}

// ============================================================================
// 2. IVF-RaBitQ (3-stage FWHT Orthogonal Rotation + 4-bit Popcount + Rerank)
// ============================================================================

__device__ __inline__ void fwht_rotate_query_shared(
    float* smem_rot,
    const float* __restrict__ query,
    const float* __restrict__ d_rot_signs,
    uint32_t dim,
    uint32_t padded_dim,
    int tid,
    int block_size
) {
    for (uint32_t d = (uint32_t)tid; d < padded_dim; d += (uint32_t)block_size) {
        smem_rot[d] = (d < dim) ? query[d] : 0.0f;
    }
    __syncthreads();

    float inv_sqrt_p = rsqrtf((float)padded_dim);
    uint32_t half_p = padded_dim >> 1;

    for (uint32_t stage = 0; stage < 3u; ++stage) {
        const float* signs = &d_rot_signs[stage * padded_dim];
        for (uint32_t d = (uint32_t)tid; d < padded_dim; d += (uint32_t)block_size) {
            smem_rot[d] *= signs[d];
        }
        __syncthreads();

        for (uint32_t len = 1u; len < padded_dim; len <<= 1) {
            for (uint32_t i = (uint32_t)tid; i < half_p; i += (uint32_t)block_size) {
                uint32_t blk = (i / len) * (len << 1);
                uint32_t off = i & (len - 1u);
                uint32_t a = blk + off;
                uint32_t b = a + len;
                float x = smem_rot[a];
                float y = smem_rot[b];
                smem_rot[a] = x + y;
                smem_rot[b] = x - y;
            }
            __syncthreads();
        }

        for (uint32_t d = (uint32_t)tid; d < padded_dim; d += (uint32_t)block_size) {
            smem_rot[d] *= inv_sqrt_p;
        }
        __syncthreads();
    }
}

// Quantizes rotated query residual (smem_rot_q - rot_centroid) into 4 packed
// bitplanes `smem_q_planes[4 * words_per_vec]` via warp __ballot_sync and
// writes `(q_min, delta, sum_u)` to shared memory.
__device__ __inline__ void rabitq_quantize_query_residual(
    const float* smem_rot_q,
    const float* __restrict__ rot_centroid,
    uint32_t padded_dim,
    uint32_t words_per_vec,
    uint32_t* smem_q_planes,
    float* smem_reduce_min,
    float* smem_reduce_max,
    float* smem_q_params, // [0]=q_min, [1]=delta, [2]=sum_u
    int tid,
    int block_size
) {
    float lmin = 1e30f;
    float lmax = -1e30f;
    for (uint32_t d = (uint32_t)tid; d < padded_dim; d += (uint32_t)block_size) {
        float val = smem_rot_q[d] - rot_centroid[d];
        if (val < lmin) lmin = val;
        if (val > lmax) lmax = val;
    }
    smem_reduce_min[tid] = lmin;
    smem_reduce_max[tid] = lmax;
    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem_reduce_min[tid + s] < smem_reduce_min[tid]) {
                smem_reduce_min[tid] = smem_reduce_min[tid + s];
            }
            if (smem_reduce_max[tid + s] > smem_reduce_max[tid]) {
                smem_reduce_max[tid] = smem_reduce_max[tid + s];
            }
        }
        __syncthreads();
    }

    float q_min = smem_reduce_min[0];
    float q_max = smem_reduce_max[0];
    float range = q_max - q_min;
    float delta = (range > 1e-12f) ? (range / 15.0f) : 0.0f;
    float inv_delta = (range > 1e-12f) ? (15.0f / range) : 0.0f;

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warps_per_block = block_size / WARP_SIZE;

    uint32_t local_sum_u = 0u;
    for (uint32_t w = (uint32_t)warp_id; w < words_per_vec; w += (uint32_t)warps_per_block) {
        uint32_t d = w * 32u + (uint32_t)lane_id;
        uint32_t u = 0u;
        if (d < padded_dim && delta > 0.0f) {
            float val = smem_rot_q[d] - rot_centroid[d];
            int qi = __float2int_rn((val - q_min) * inv_delta);
            if (qi < 0) qi = 0;
            if (qi > 15) qi = 15;
            u = (uint32_t)qi;
        }
        local_sum_u += u;

        uint32_t p0 = __ballot_sync(FULL_WARP_MASK, (u & 1u) != 0u);
        uint32_t p1 = __ballot_sync(FULL_WARP_MASK, (u & 2u) != 0u);
        uint32_t p2 = __ballot_sync(FULL_WARP_MASK, (u & 4u) != 0u);
        uint32_t p3 = __ballot_sync(FULL_WARP_MASK, (u & 8u) != 0u);
        if (lane_id == 0) {
            smem_q_planes[0u * words_per_vec + w] = p0;
            smem_q_planes[1u * words_per_vec + w] = p1;
            smem_q_planes[2u * words_per_vec + w] = p2;
            smem_q_planes[3u * words_per_vec + w] = p3;
        }
    }

    // Reuse smem_reduce_min as float sum_u accumulator across threads
    smem_reduce_min[tid] = (float)local_sum_u;
    __syncthreads();
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_reduce_min[tid] += smem_reduce_min[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        smem_q_params[0] = q_min;
        smem_q_params[1] = delta;
        smem_q_params[2] = smem_reduce_min[0];
    }
    __syncthreads();
}

__device__ __forceinline__ float rabitq_popc_distance(
    const uint32_t* __restrict__ vec_bits,
    const float* __restrict__ meta, // [res_norm_sq, rescale, popcnt_b, vec_norm_sq]
    const uint32_t* smem_q_planes,
    uint32_t words_per_vec,
    uint32_t padded_dim,
    float q_min,
    float delta,
    float sum_u,
    float q_c_dist_sq,
    float q_norm_sq,
    MetricType metric
) {
    uint32_t acc0 = 0u, acc1 = 0u, acc2 = 0u, acc3 = 0u;
    for (uint32_t w = 0; w < words_per_vec; ++w) {
        uint32_t b = vec_bits[w];
        acc0 += (uint32_t)__popc(b & smem_q_planes[0u * words_per_vec + w]);
        acc1 += (uint32_t)__popc(b & smem_q_planes[1u * words_per_vec + w]);
        acc2 += (uint32_t)__popc(b & smem_q_planes[2u * words_per_vec + w]);
        acc3 += (uint32_t)__popc(b & smem_q_planes[3u * words_per_vec + w]);
    }
    float s_bu = (float)acc0 + 2.0f * (float)acc1 + 4.0f * (float)acc2 + 8.0f * (float)acc3;

    float res_norm_sq = meta[0];
    float rescale = meta[1];
    float popcnt_b = meta[2];
    float vec_norm_sq = meta[3];

    float inner_est = rescale * (
        q_min * (2.0f * popcnt_b - (float)padded_dim) +
        delta * (2.0f * s_bu - sum_u)
    );
    float est_l2 = q_c_dist_sq + res_norm_sq - 2.0f * inner_est;

    if (metric == METRIC_INNER_PRODUCT) {
        return -0.5f * (q_norm_sq + vec_norm_sq - est_l2);
    }
    if (metric == METRIC_COSINE) {
        return 0.5f * est_l2;
    }
    return est_l2;
}

__global__ void ivf_rabitq_small_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_rot_centroids,
    const uint32_t* __restrict__ d_rabitq_bits,
    const float* __restrict__ d_rabitq_meta,
    const float* __restrict__ d_rot_signs,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    const float* __restrict__ d_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t padded_dim,
    uint32_t words_per_vec,
    uint32_t nlist,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t shortlist,
    bool do_rerank,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0 || top_k == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_rq[];
    float* smem_rot_q = smem_rq; // [padded_dim]
    float* smem_reduce_min = &smem_rot_q[padded_dim]; // [block_size]
    float* smem_reduce_max = &smem_reduce_min[block_size]; // [block_size]
    float* smem_q_params = &smem_reduce_max[block_size]; // [4]: q_min, delta, sum_u, q_norm_sq
    uint32_t* smem_q_planes = (uint32_t*)&smem_q_params[4]; // [4 * words_per_vec]
    uint32_t* smem_probed_clusters = &smem_q_planes[4u * words_per_vec];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];
    size_t raw_off = ((size_t)padded_dim + (size_t)block_size * 2u + 4u) * sizeof(float) +
                     ((size_t)words_per_vec * 4u * sizeof(uint32_t)) +
                     ((size_t)nprobe * (sizeof(uint32_t) + sizeof(float)));
    size_t aligned_off = (raw_off + 7u) & ~7u;
    GpuSearchResult* smem_merged_topk = (GpuSearchResult*)((char*)smem_rq + aligned_off);

    // Compute query ||q||^2 for inner product support
    float local_q_norm = 0.0f;
    for (uint32_t d = (uint32_t)tid; d < dim; d += (uint32_t)block_size) {
        float v = query[d];
        local_q_norm = fmaf(v, v, local_q_norm);
    }
    smem_reduce_min[tid] = local_q_norm;
    __syncthreads();
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_reduce_min[tid] += smem_reduce_min[tid + s];
        __syncthreads();
    }
    if (tid == 0) smem_q_params[3] = smem_reduce_min[0];
    __syncthreads();
    float q_norm_sq = smem_q_params[3];

    // Rotate query into orthogonal space via 3-stage FWHT
    fwht_rotate_query_shared(smem_rot_q, query, d_rot_signs, dim, padded_dim, tid, block_size);

    ivf_find_nprobe_centroids(
        query, d_centroids, dim, nlist, nprobe,
        smem_probed_clusters, smem_cluster_dists, tid, block_size
    );

    CandidateList<MAX_SMALL_RERANK> local_candidates;
    local_candidates.init((int)shortlist);

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        float q_c_dist_sq = smem_cluster_dists[p];
        rabitq_quantize_query_residual(
            smem_rot_q, &d_rot_centroids[cluster_id * padded_dim],
            padded_dim, words_per_vec, smem_q_planes,
            smem_reduce_min, smem_reduce_max, smem_q_params, tid, block_size
        );

        float q_min = smem_q_params[0];
        float delta = smem_q_params[1];
        float sum_u = smem_q_params[2];

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t i = (uint32_t)tid; i < list_len; i += (uint32_t)block_size) {
            uint32_t vec_id = d_ivf_vec_ids[list_start + i];
            if (!bitmask_allows(d_valid_mask, vec_id)) continue;

            float est_dist = rabitq_popc_distance(
                &d_rabitq_bits[(size_t)vec_id * words_per_vec],
                &d_rabitq_meta[(size_t)vec_id * 4u],
                smem_q_planes, words_per_vec, padded_dim,
                q_min, delta, sum_u, q_c_dist_sq, q_norm_sq, metric
            );
            local_candidates.insert(vec_id, est_dist);
        }
        __syncthreads();
    }

    block_bitonic_merge_topk<MAX_SMALL_RERANK>(
        local_candidates, smem_merged_topk, (int)shortlist, tid, block_size
    );

    if (do_rerank && d_vectors != nullptr) {
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warps_per_block = block_size / WARP_SIZE;
        for (uint32_t k = (uint32_t)warp_id; k < shortlist; k += (uint32_t)warps_per_block) {
            uint32_t vec_id = smem_merged_topk[k].id;
            if (vec_id >= num_vectors) continue;
            float exact = warp_compute_distance(
                query, &d_vectors[(size_t)vec_id * dim], (int)dim, lane_id, metric
            );
            if (lane_id == 0) smem_merged_topk[k].distance = exact;
        }
        __syncthreads();
    }

    uint32_t sort_pow2 = next_pow2_u32(shortlist);
    for (uint32_t k = shortlist + (uint32_t)tid; k < sort_pow2; k += (uint32_t)block_size) {
        smem_merged_topk[k] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();
    block_bitonic_sort_inplace(smem_merged_topk, sort_pow2, tid, block_size);

    uint32_t out_base = query_idx * top_k;
    for (uint32_t k = (uint32_t)tid; k < top_k; k += (uint32_t)block_size) {
        d_out_results[out_base + k] = (k < shortlist)
            ? smem_merged_topk[k]
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
}

__global__ void ivf_rabitq_stage1_shortlist_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_rot_centroids,
    const uint32_t* __restrict__ d_rabitq_bits,
    const float* __restrict__ d_rabitq_meta,
    const float* __restrict__ d_rot_signs,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    uint32_t dim,
    uint32_t padded_dim,
    uint32_t words_per_vec,
    uint32_t nlist,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t nprobe,
    uint32_t shortlist_m,
    uint32_t work_pow2,
    uint32_t out_m_pow2,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_work,
    GpuSearchResult* __restrict__ d_shortlist
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_rq[];
    float* smem_rot_q = smem_rq;
    float* smem_reduce_min = &smem_rot_q[padded_dim];
    float* smem_reduce_max = &smem_reduce_min[block_size];
    float* smem_q_params = &smem_reduce_max[block_size];
    uint32_t* smem_q_planes = (uint32_t*)&smem_q_params[4];
    uint32_t* smem_probed_clusters = &smem_q_planes[4u * words_per_vec];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];
    uint32_t* smem_count = (uint32_t*)&smem_cluster_dists[nprobe];
    float* smem_cutoff = (float*)&smem_count[1];

    if (tid == 0) {
        *smem_count = 0u;
        *smem_cutoff = 1e30f;
    }

    float local_q_norm = 0.0f;
    for (uint32_t d = (uint32_t)tid; d < dim; d += (uint32_t)block_size) {
        float v = query[d];
        local_q_norm = fmaf(v, v, local_q_norm);
    }
    smem_reduce_min[tid] = local_q_norm;
    __syncthreads();
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_reduce_min[tid] += smem_reduce_min[tid + s];
        __syncthreads();
    }
    if (tid == 0) smem_q_params[3] = smem_reduce_min[0];
    __syncthreads();
    float q_norm_sq = smem_q_params[3];

    fwht_rotate_query_shared(smem_rot_q, query, d_rot_signs, dim, padded_dim, tid, block_size);

    ivf_find_nprobe_centroids(
        query, d_centroids, dim, nlist, nprobe,
        smem_probed_clusters, smem_cluster_dists, tid, block_size
    );

    GpuSearchResult* work_buf = &d_work[(size_t)query_idx * work_pow2];

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        float q_c_dist_sq = smem_cluster_dists[p];
        rabitq_quantize_query_residual(
            smem_rot_q, &d_rot_centroids[cluster_id * padded_dim],
            padded_dim, words_per_vec, smem_q_planes,
            smem_reduce_min, smem_reduce_max, smem_q_params, tid, block_size
        );

        float q_min = smem_q_params[0];
        float delta = smem_q_params[1];
        float sum_u = smem_q_params[2];

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t base = 0; base < list_len; base += (uint32_t)block_size) {
            uint32_t i = base + (uint32_t)tid;
            if (i < list_len) {
                uint32_t vec_id = d_ivf_vec_ids[list_start + i];
                if (bitmask_allows(d_valid_mask, vec_id)) {
                    float est_dist = rabitq_popc_distance(
                        &d_rabitq_bits[(size_t)vec_id * words_per_vec],
                        &d_rabitq_meta[(size_t)vec_id * 4u],
                        smem_q_planes, words_per_vec, padded_dim,
                        q_min, delta, sum_u, q_c_dist_sq, q_norm_sq, metric
                    );
                    if (est_dist < *smem_cutoff) {
                        uint32_t pos = atomicAdd(smem_count, 1u);
                        if (pos < work_pow2) {
                            work_buf[pos].id = vec_id;
                            work_buf[pos].distance = est_dist;
                        }
                    }
                }
            }
            stage1_flush_if_full(
                work_buf, work_pow2, shortlist_m, smem_count, smem_cutoff, tid, block_size
            );
        }
    }

    __syncthreads();
    uint32_t final_cnt = *smem_count;
    if (final_cnt > work_pow2) final_cnt = work_pow2;
    for (uint32_t i = final_cnt + (uint32_t)tid; i < work_pow2; i += (uint32_t)block_size) {
        work_buf[i] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();
    block_bitonic_sort_inplace(work_buf, work_pow2, tid, block_size);

    GpuSearchResult* out_buf = &d_shortlist[(size_t)query_idx * out_m_pow2];
    for (uint32_t i = (uint32_t)tid; i < out_m_pow2; i += (uint32_t)block_size) {
        out_buf[i] = (i < shortlist_m && i < final_cnt)
            ? work_buf[i]
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
}

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
) {
    if (num_queries == 0 || top_k == 0 || tables->nlist == 0 || tables->padded_dim == 0) return;
    if (top_k > 256u) top_k = 256u;

    uint32_t clamped_nprobe = nprobe == 0u ? 1u : nprobe;
    if (clamped_nprobe > tables->nlist) clamped_nprobe = tables->nlist;

    uint32_t shortlist = rerank_k == 0u ? top_k : rerank_k;
    if (shortlist < top_k) shortlist = top_k;
    if (shortlist > MAX_SHORTLIST_RERANK) shortlist = MAX_SHORTLIST_RERANK;
    bool do_rerank = (rerank_k != 1u) && (tables->d_vectors != nullptr);

    if (shortlist <= MAX_SMALL_RERANK) {
        uint32_t sort_pow2 = next_pow2_u32(shortlist);
        size_t raw_smem = ((size_t)tables->padded_dim + (size_t)QUANT_BLOCK_SIZE * 2u + 4u) * sizeof(float) +
                          ((size_t)tables->words_per_vec * 4u * sizeof(uint32_t)) +
                          ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float)));
        size_t aligned_smem = (raw_smem + 7u) & ~7u;
        size_t smem_bytes = aligned_smem + ((size_t)sort_pow2 * sizeof(GpuSearchResult));
        ensure_dynamic_smem((const void*)ivf_rabitq_small_kernel, smem_bytes);

        ivf_rabitq_small_kernel<<<num_queries, QUANT_BLOCK_SIZE, smem_bytes, stream>>>(
            tables->d_centroids, tables->d_rot_centroids, tables->d_rabitq_bits,
            tables->d_rabitq_meta, tables->d_rot_signs,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids, tables->d_vectors,
            tables->num_vectors, tables->dim, tables->padded_dim, tables->words_per_vec,
            tables->nlist, d_queries, num_queries, top_k, clamped_nprobe,
            shortlist, do_rerank, metric, d_valid_mask, d_out_results
        );
        return;
    }

    uint32_t m_pow2 = next_pow2_u32(shortlist);
    uint32_t work_pow2 = m_pow2 * 2u;
    uint32_t chunk_queries = num_queries < 32u ? num_queries : 32u;

    GpuSearchResult* d_work = nullptr;
    GpuSearchResult* d_shortlist = nullptr;
    if (cudaMallocAsync(&d_work, (size_t)chunk_queries * work_pow2 * sizeof(GpuSearchResult), stream) != cudaSuccess) {
        return;
    }
    if (cudaMallocAsync(&d_shortlist, (size_t)chunk_queries * m_pow2 * sizeof(GpuSearchResult), stream) != cudaSuccess) {
        cudaFreeAsync(d_work, stream);
        return;
    }

    size_t smem_stage1 = ((size_t)tables->padded_dim + (size_t)IVF_STAGE2_THREADS * 2u + 4u) * sizeof(float) +
                         ((size_t)tables->words_per_vec * 4u * sizeof(uint32_t)) +
                         ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float))) +
                         sizeof(uint32_t) + sizeof(float);
    ensure_dynamic_smem((const void*)ivf_rabitq_stage1_shortlist_kernel, smem_stage1);

    for (uint32_t q_off = 0; q_off < num_queries; q_off += chunk_queries) {
        uint32_t q_cnt = num_queries - q_off;
        if (q_cnt > chunk_queries) q_cnt = chunk_queries;

        const float* q_ptr = &d_queries[(size_t)q_off * tables->dim];
        GpuSearchResult* out_ptr = &d_out_results[(size_t)q_off * top_k];

        ivf_rabitq_stage1_shortlist_kernel<<<q_cnt, IVF_STAGE2_THREADS, smem_stage1, stream>>>(
            tables->d_centroids, tables->d_rot_centroids, tables->d_rabitq_bits,
            tables->d_rabitq_meta, tables->d_rot_signs,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids,
            tables->dim, tables->padded_dim, tables->words_per_vec, tables->nlist,
            q_ptr, q_cnt, clamped_nprobe, shortlist, work_pow2, m_pow2,
            metric, d_valid_mask, d_work, d_shortlist
        );

        if (do_rerank) {
            launch_ivf_exact_rerank(
                tables->d_vectors, tables->num_vectors, tables->dim,
                q_ptr, q_cnt, shortlist, m_pow2, top_k, metric,
                d_shortlist, out_ptr, stream
            );
        } else {
            for (uint32_t qi = 0; qi < q_cnt; ++qi) {
                cudaMemcpyAsync(
                    &out_ptr[(size_t)qi * top_k],
                    &d_shortlist[(size_t)qi * m_pow2],
                    (size_t)top_k * sizeof(GpuSearchResult),
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
        }
    }

    cudaFreeAsync(d_work, stream);
    cudaFreeAsync(d_shortlist, stream);
}

// ============================================================================
// 3. FP16 Tensor Core (nvcuda::wmma 16x16x16) Batched Similarity Search
// ============================================================================

#define TC_WARPS_PER_BLOCK 4
#define TC_THREADS_PER_BLOCK (TC_WARPS_PER_BLOCK * WARP_SIZE)
#define TC_TILE_M 16
#define TC_TILE_N 16
#define TC_TILE_K 16
#define TC_VEC_STEP (TC_WARPS_PER_BLOCK * TC_TILE_N) // 64 vectors per tile step

__global__ void fp16_tc_wmma_search_kernel(
    const uint16_t* __restrict__ d_vectors_fp16, // [num_vectors * padded_dim]
    const float* __restrict__ d_vec_norms_sq,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t padded_dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t q_base = blockIdx.x * TC_TILE_M;
    if (q_base >= num_queries || top_k == 0) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;

    __shared__ __half smem_q[TC_TILE_M * TC_TILE_K];
    __shared__ __half smem_x[TC_WARPS_PER_BLOCK][TC_TILE_N * TC_TILE_K];
    __shared__ float smem_dots[TC_WARPS_PER_BLOCK][TC_TILE_M * TC_TILE_N];
    __shared__ float smem_q_norms[TC_TILE_M];
    __shared__ GpuSearchResult smem_topk[TC_TILE_M][256];

    // Initialize per-query top-k buffers and query ||q||^2
    if (tid < TC_TILE_M) {
        uint32_t q_idx = q_base + (uint32_t)tid;
        float qn = 0.0f;
        if (q_idx < num_queries) {
            const float* q_ptr = &d_queries[q_idx * dim];
            for (uint32_t d = 0; d < dim; ++d) {
                qn = fmaf(q_ptr[d], q_ptr[d], qn);
            }
        }
        smem_q_norms[tid] = qn;
    }
    for (int i = tid; i < TC_TILE_M * (int)top_k; i += TC_THREADS_PER_BLOCK) {
        int r = i / (int)top_k;
        int c = i % (int)top_k;
        smem_topk[r][c] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();

    // Stream dataset in chunks of 64 vectors (4 x 16-vector Tensor Core tiles)
    for (uint32_t n_base = 0; n_base < num_vectors; n_base += TC_VEC_STEP) {
        wmma::fragment<wmma::accumulator, TC_TILE_M, TC_TILE_N, TC_TILE_K, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (uint32_t k_off = 0; k_off < padded_dim; k_off += TC_TILE_K) {
            // Cooperatively load [16, 16] query tile (convert FP32 -> FP16)
            for (int idx = tid; idx < TC_TILE_M * TC_TILE_K; idx += TC_THREADS_PER_BLOCK) {
                int q_local = idx / TC_TILE_K;
                int k_local = idx % TC_TILE_K;
                uint32_t q_global = q_base + (uint32_t)q_local;
                uint32_t d_global = k_off + (uint32_t)k_local;
                float val = (q_global < num_queries && d_global < dim)
                    ? d_queries[q_global * dim + d_global]
                    : 0.0f;
                smem_q[idx] = __float2half(val);
            }

            // Cooperatively load 4 [16, 16] dataset tiles (one per warp)
            for (int w = 0; w < TC_WARPS_PER_BLOCK; ++w) {
                uint32_t tile_n_base = n_base + (uint32_t)w * TC_TILE_N;
                for (int idx = tid; idx < TC_TILE_N * TC_TILE_K; idx += TC_THREADS_PER_BLOCK) {
                    int v_local = idx / TC_TILE_K;
                    int k_local = idx % TC_TILE_K;
                    uint32_t v_global = tile_n_base + (uint32_t)v_local;
                    uint32_t d_global = k_off + (uint32_t)k_local;
                    uint16_t bits = (v_global < num_vectors && d_global < padded_dim)
                        ? d_vectors_fp16[(size_t)v_global * padded_dim + d_global]
                        : 0u;
                    smem_x[w][idx] = __ushort_as_half(bits);
                }
            }
            __syncthreads();

            // Tensor Core wmma::mma_sync executed warp-uniformly
            wmma::fragment<wmma::matrix_a, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, wmma::col_major> frag_b;
            wmma::load_matrix_sync(frag_a, smem_q, TC_TILE_K);
            wmma::load_matrix_sync(frag_b, smem_x[warp_id], TC_TILE_K);
            wmma::mma_sync(acc, frag_a, frag_b, acc);
            __syncthreads();
        }

        wmma::store_matrix_sync(smem_dots[warp_id], acc, TC_TILE_N, wmma::mem_row_major);
        __syncthreads();

        // Threads 0..15 update each query row's sorted top-k list in parallel
        if (tid < TC_TILE_M) {
            uint32_t q_global = q_base + (uint32_t)tid;
            if (q_global < num_queries) {
                float qn = smem_q_norms[tid];
                GpuSearchResult* row_topk = smem_topk[tid];
                float worst = row_topk[top_k - 1].distance;

                for (int w = 0; w < TC_WARPS_PER_BLOCK; ++w) {
                    uint32_t tile_n_base = n_base + (uint32_t)w * TC_TILE_N;
                    for (int j = 0; j < TC_TILE_N; ++j) {
                        uint32_t doc_id = tile_n_base + (uint32_t)j;
                        if (doc_id >= num_vectors) break;
                        if (!bitmask_allows(d_valid_mask, doc_id)) continue;

                        float dot = smem_dots[w][tid * TC_TILE_N + j];
                        float dist;
                        if (metric == METRIC_INNER_PRODUCT) {
                            dist = -dot;
                        } else if (metric == METRIC_COSINE) {
                            float denom = sqrtf(qn * d_vec_norms_sq[doc_id]) + 1e-8f;
                            dist = 1.0f - (dot / denom);
                        } else {
                            dist = fmaxf(0.0f, qn + d_vec_norms_sq[doc_id] - 2.0f * dot);
                        }

                        if (dist < worst) {
                            int pos = (int)top_k - 1;
                            while (pos > 0 && dist < row_topk[pos - 1].distance) {
                                row_topk[pos] = row_topk[pos - 1];
                                pos--;
                            }
                            row_topk[pos].id = doc_id;
                            row_topk[pos].distance = dist;
                            worst = row_topk[top_k - 1].distance;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TC_TILE_M * (int)top_k; i += TC_THREADS_PER_BLOCK) {
        uint32_t q_local = (uint32_t)(i / (int)top_k);
        uint32_t k_idx = (uint32_t)(i % (int)top_k);
        uint32_t q_global = q_base + q_local;
        if (q_global < num_queries) {
            d_out_results[(size_t)q_global * top_k + k_idx] = smem_topk[q_local][k_idx];
        }
    }
}

void launch_fp16_tc_search(
    const Fp16GpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0 || top_k == 0 || tables->num_vectors == 0 || tables->padded_dim == 0) {
        return;
    }
    if (top_k > 256u) top_k = 256u;
    uint32_t num_blocks = (num_queries + TC_TILE_M - 1u) / TC_TILE_M;

    fp16_tc_wmma_search_kernel<<<num_blocks, TC_THREADS_PER_BLOCK, 0, stream>>>(
        tables->d_vectors_fp16, tables->d_vec_norms_sq,
        tables->num_vectors, tables->dim, tables->padded_dim,
        d_queries, num_queries, top_k, metric, d_valid_mask, d_out_results
    );
}

} // namespace flashvector

extern "C" {

int cuda_ivf_sq_search_batch(
    const IvfSqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!tables || !d_queries || !d_out_results) return -1;
    flashvector::launch_ivf_sq_search(
        tables, d_queries, num_queries, top_k, nprobe, rerank_k,
        metric, d_valid_mask, d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_ivf_rabitq_search_batch(
    const IvfRabitqGpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!tables || !d_queries || !d_out_results) return -1;
    flashvector::launch_ivf_rabitq_search(
        tables, d_queries, num_queries, top_k, nprobe, rerank_k,
        metric, d_valid_mask, d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_fp16_tc_search_batch(
    const Fp16GpuTables* tables,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!tables || !d_queries || !d_out_results) return -1;
    flashvector::launch_fp16_tc_search(
        tables, d_queries, num_queries, top_k, metric, d_valid_mask,
        d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
