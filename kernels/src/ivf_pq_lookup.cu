#include "ivf_pq_lookup.cuh"
#include "ivf_rerank.cuh"
#include <stdio.h>

namespace flashvector {

#define BLOCK_SIZE_IVF 128
#define MAX_RERANK_CANDIDATES 32

// Stage-2 follow-up kernel for large shortlists (M up to 10,240):
// Rescores top-M approximate candidates in `d_shortlist` against full-precision
// `d_vectors` (resident in VRAM or mapped pinned host memory) across warps,
// runs `block_bitonic_sort_inplace`, and emits sorted top-k.
__global__ void ivf_exact_rerank_and_bitonic_kernel(
    const float* __restrict__ d_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t shortlist_m,
    uint32_t m_pow2,
    uint32_t top_k,
    MetricType metric,
    GpuSearchResult* __restrict__ d_shortlist,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warps_per_block = block_size / WARP_SIZE;

    const float* query = &d_queries[query_idx * dim];
    GpuSearchResult* shortlist = &d_shortlist[(size_t)query_idx * m_pow2];

    for (uint32_t idx = (uint32_t)warp_id; idx < shortlist_m; idx += (uint32_t)warps_per_block) {
        uint32_t vec_id = shortlist[idx].id;
        if (vec_id >= num_vectors) {
            if (lane_id == 0) {
                shortlist[idx] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
            }
            continue;
        }
        float exact = warp_compute_distance(
            query,
            &d_vectors[(size_t)vec_id * dim],
            (int)dim,
            lane_id,
            metric
        );
        if (lane_id == 0) {
            shortlist[idx].distance = exact;
        }
    }
    for (uint32_t idx = shortlist_m + (uint32_t)tid; idx < m_pow2; idx += (uint32_t)block_size) {
        shortlist[idx] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();

    block_bitonic_sort_inplace(shortlist, m_pow2, tid, block_size);

    uint32_t out_base = query_idx * top_k;
    for (uint32_t k = (uint32_t)tid; k < top_k; k += (uint32_t)block_size) {
        d_out_results[out_base + k] = (k < shortlist_m)
            ? shortlist[k]
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
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
) {
    ivf_exact_rerank_and_bitonic_kernel<<<num_queries, IVF_STAGE2_THREADS, 0, stream>>>(
        d_vectors, num_vectors, dim, d_queries, num_queries,
        shortlist_m, m_pow2, top_k, metric, d_shortlist, d_out_results
    );
}

// Fast single-pass IVF-PQ ADC + exact rerank kernel for shortlist <= 32.
// Skips vectors masked out by `d_valid_mask` directly before reading PQ codes.
__global__ void ivf_pq_adc_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_pq_codebooks,
    const uint8_t* __restrict__ d_pq_codes,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    const float* __restrict__ d_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t nlist,
    uint32_t m_pq,
    uint32_t sub_dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    uint32_t rerank_k,
    bool do_rerank,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0 || top_k == 0 || m_pq == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_pool[];
    float* smem_lut = smem_pool;
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_lut[m_pq * SMEM_PQ_STRIDE];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];

    size_t lut_bytes = (size_t)m_pq * SMEM_PQ_STRIDE * sizeof(float);
    size_t raw_offset = lut_bytes + (size_t)nprobe * (sizeof(uint32_t) + sizeof(float));
    size_t aligned_offset = (raw_offset + 7u) & ~7u;
    GpuSearchResult* smem_merged_topk =
        (GpuSearchResult*)((char*)smem_pool + aligned_offset);

    ivf_find_nprobe_centroids(
        query, d_centroids, dim, nlist, nprobe,
        smem_probed_clusters, smem_cluster_dists, tid, block_size
    );

    CandidateList<MAX_RERANK_CANDIDATES> local_candidates;
    uint32_t shortlist = rerank_k == 0 ? 1u : rerank_k;
    if (shortlist > MAX_RERANK_CANDIDATES) shortlist = MAX_RERANK_CANDIDATES;
    if (shortlist < top_k) shortlist = top_k;
    local_candidates.init((int)shortlist);

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        for (uint32_t m = 0; m < m_pq; ++m) {
            for (uint32_t c = (uint32_t)tid; c < 256u; c += (uint32_t)block_size) {
                const float* cb_sub = &d_pq_codebooks[(m * 256u + c) * sub_dim];
                const float* centroid_sub = &d_centroids[cluster_id * dim + m * sub_dim];

                float dist = 0.0f;
                for (uint32_t sd = 0; sd < sub_dim; ++sd) {
                    float diff = (query[m * sub_dim + sd] - centroid_sub[sd]) - cb_sub[sd];
                    dist = fmaf(diff, diff, dist);
                }
                smem_lut[m * SMEM_PQ_STRIDE + c] = dist;
            }
        }
        __syncthreads();

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t i = (uint32_t)tid; i < list_len; i += (uint32_t)block_size) {
            uint32_t vec_id = d_ivf_vec_ids[list_start + i];
            if (!bitmask_allows(d_valid_mask, vec_id)) continue;

            const uint8_t* codes = &d_pq_codes[(size_t)vec_id * m_pq];
            float adc_dist = 0.0f;
            #pragma unroll 8
            for (uint32_t m = 0; m < m_pq; ++m) {
                adc_dist += smem_lut[m * SMEM_PQ_STRIDE + codes[m]];
            }
            local_candidates.insert(vec_id, adc_dist);
        }
        __syncthreads();
    }

    block_bitonic_merge_topk<MAX_RERANK_CANDIDATES>(
        local_candidates, smem_merged_topk, (int)shortlist, tid, block_size
    );

    if (do_rerank && d_vectors != nullptr && shortlist > 0) {
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warps_per_block = block_size / WARP_SIZE;

        for (uint32_t k = (uint32_t)warp_id; k < shortlist; k += (uint32_t)warps_per_block) {
            uint32_t vec_id = smem_merged_topk[k].id;
            if (vec_id >= num_vectors) continue;
            float dist = warp_compute_distance(
                query, &d_vectors[(size_t)vec_id * dim], (int)dim, lane_id, metric
            );
            if (lane_id == 0) {
                smem_merged_topk[k].distance = dist;
            }
        }
        __syncthreads();
    }

    uint32_t sort_pow2 = next_pow2_u32(shortlist);
    for (uint32_t k = shortlist + (uint32_t)tid; k < sort_pow2; k += (uint32_t)block_size) {
        smem_merged_topk[k].id = 0xFFFFFFFFU;
        smem_merged_topk[k].distance = 1e30f;
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

// Stage-1 kernel for large rerank shortlists (M > 32, up to 10,240):
// Emits top-M approximate ADC candidates per query into `d_shortlist` for
// Pass-2 exact rescoring + `bitonic_topk.cuh` reduction.
__global__ void ivf_pq_stage1_shortlist_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_pq_codebooks,
    const uint8_t* __restrict__ d_pq_codes,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    uint32_t dim,
    uint32_t nlist,
    uint32_t m_pq,
    uint32_t sub_dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t nprobe,
    uint32_t shortlist_m,
    uint32_t work_pow2,
    uint32_t out_m_pow2,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_work,      // [num_queries * work_pow2]
    GpuSearchResult* __restrict__ d_shortlist  // [num_queries * out_m_pow2]
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || nprobe == 0 || m_pq == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    extern __shared__ float smem_pool[];
    float* smem_lut = smem_pool;
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_lut[m_pq * SMEM_PQ_STRIDE];
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

        for (uint32_t m = 0; m < m_pq; ++m) {
            for (uint32_t c = (uint32_t)tid; c < 256u; c += (uint32_t)block_size) {
                const float* cb_sub = &d_pq_codebooks[(m * 256u + c) * sub_dim];
                const float* centroid_sub = &d_centroids[cluster_id * dim + m * sub_dim];
                float dist = 0.0f;
                for (uint32_t sd = 0; sd < sub_dim; ++sd) {
                    float diff = (query[m * sub_dim + sd] - centroid_sub[sd]) - cb_sub[sd];
                    dist = fmaf(diff, diff, dist);
                }
                smem_lut[m * SMEM_PQ_STRIDE + c] = dist;
            }
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
                    const uint8_t* codes = &d_pq_codes[(size_t)vec_id * m_pq];
                    float adc_dist = 0.0f;
                    #pragma unroll 8
                    for (uint32_t m = 0; m < m_pq; ++m) {
                        adc_dist += smem_lut[m * SMEM_PQ_STRIDE + codes[m]];
                    }
                    if (adc_dist < *smem_cutoff) {
                        uint32_t pos = atomicAdd(smem_count, 1u);
                        if (pos < work_pow2) {
                            work_buf[pos].id = vec_id;
                            work_buf[pos].distance = adc_dist;
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
        work_buf[i].id = 0xFFFFFFFFU;
        work_buf[i].distance = 1e30f;
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

void launch_ivf_pq_search(
    const IvfPqGpuTables* tables,
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
    if (num_queries == 0 || top_k == 0 || tables->m_pq == 0 || tables->nlist == 0) return;

    if (top_k > 256u) top_k = 256u;

    uint32_t clamped_nprobe = nprobe;
    if (clamped_nprobe == 0u) clamped_nprobe = 1u;
    if (clamped_nprobe > tables->nlist) clamped_nprobe = tables->nlist;

    uint32_t shortlist = rerank_k == 0u ? top_k : rerank_k;
    if (shortlist < top_k) shortlist = top_k;
    if (shortlist > MAX_SHORTLIST_RERANK) shortlist = MAX_SHORTLIST_RERANK;
    bool do_rerank = (rerank_k != 1u) && (tables->d_vectors != nullptr);

    // Fast single-pass shared-memory path when shortlist <= 32
    if (shortlist <= MAX_RERANK_CANDIDATES) {
        uint32_t sort_pow2 = next_pow2_u32(shortlist);
        dim3 grid(num_queries);
        dim3 block(BLOCK_SIZE_IVF);

        size_t raw_smem = ((size_t)tables->m_pq * SMEM_PQ_STRIDE * sizeof(float)) +
                          ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float)));
        size_t aligned_smem = (raw_smem + 7u) & ~7u;
        size_t smem_bytes = aligned_smem + ((size_t)sort_pow2 * sizeof(GpuSearchResult));
        ensure_dynamic_smem((const void*)ivf_pq_adc_kernel, smem_bytes);

        ivf_pq_adc_kernel<<<grid, block, smem_bytes, stream>>>(
            tables->d_centroids, tables->d_pq_codebooks, tables->d_pq_codes,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids, tables->d_vectors,
            tables->num_vectors, tables->dim, tables->nlist, tables->m_pq, tables->sub_dim,
            d_queries, num_queries, top_k, clamped_nprobe, shortlist,
            do_rerank, metric, d_valid_mask, d_out_results
        );
        return;
    }

    // Two-stage pass for large shortlists (32 < M <= 10,240): Stage-1 ADC shortlist
    // followed by warp-cooperative Stage-2 exact rescoring + bitonic top-k.
    uint32_t m_pow2 = next_pow2_u32(shortlist);
    uint32_t work_pow2 = m_pow2 * 2u;
    uint32_t chunk_queries = num_queries < 32u ? num_queries : 32u;

    GpuSearchResult* d_work = nullptr;
    GpuSearchResult* d_shortlist = nullptr;
    size_t work_bytes = (size_t)chunk_queries * work_pow2 * sizeof(GpuSearchResult);
    size_t shortlist_bytes = (size_t)chunk_queries * m_pow2 * sizeof(GpuSearchResult);

    if (cudaMallocAsync(&d_work, work_bytes, stream) != cudaSuccess) return;
    if (cudaMallocAsync(&d_shortlist, shortlist_bytes, stream) != cudaSuccess) {
        cudaFreeAsync(d_work, stream);
        return;
    }

    size_t smem_stage1 = ((size_t)tables->m_pq * SMEM_PQ_STRIDE * sizeof(float)) +
                         ((size_t)clamped_nprobe * (sizeof(uint32_t) + sizeof(float))) +
                         sizeof(uint32_t) + sizeof(float);
    ensure_dynamic_smem((const void*)ivf_pq_stage1_shortlist_kernel, smem_stage1);

    for (uint32_t q_off = 0; q_off < num_queries; q_off += chunk_queries) {
        uint32_t q_cnt = num_queries - q_off;
        if (q_cnt > chunk_queries) q_cnt = chunk_queries;

        const float* q_ptr = &d_queries[(size_t)q_off * tables->dim];
        GpuSearchResult* out_ptr = &d_out_results[(size_t)q_off * top_k];

        ivf_pq_stage1_shortlist_kernel<<<q_cnt, IVF_STAGE2_THREADS, smem_stage1, stream>>>(
            tables->d_centroids, tables->d_pq_codebooks, tables->d_pq_codes,
            tables->d_ivf_offsets, tables->d_ivf_vec_ids,
            tables->dim, tables->nlist, tables->m_pq, tables->sub_dim,
            q_ptr, q_cnt, clamped_nprobe, shortlist, work_pow2, m_pow2,
            d_valid_mask, d_work, d_shortlist
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

} // namespace flashvector

extern "C" {

int cuda_ivf_pq_search_batch(
    const IvfPqGpuTables* tables,
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
    if (!tables || !d_queries || !d_out_results) {
        return -1;
    }

    flashvector::launch_ivf_pq_search(
        tables, d_queries, num_queries, top_k, nprobe, rerank_k,
        metric, d_valid_mask, d_out_results, (cudaStream_t)stream
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
