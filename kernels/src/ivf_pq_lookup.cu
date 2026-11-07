#include "ivf_pq_lookup.cuh"
#include <stdio.h>

namespace flashvector {

#define BLOCK_SIZE_IVF 128

// Shortlist size for exact re-ranking, capped by the block-local collector.
#define MAX_RERANK_CANDIDATES 32

// Batched IVF-PQ ADC search kernel with residual codes and exact re-ranking.
//
// Each block handles one query. Stage 1 probes the nprobe nearest Voronoi
// cells and ranks their PQ codes with asymmetric distance computation against
// a per-cell residual lookup table. Stage 2 re-scores the shortlist with exact
// distances against the full-precision vectors, which removes the quantization
// bias that otherwise caps recall.
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
    MetricType metric,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;
    if (nprobe == 0 || top_k == 0 || m_pq == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    // Dynamic shared memory layout:
    // [0 .. m_pq * SMEM_PQ_STRIDE]: Residual Distance Lookup Table (LUT)
    // [offset .. offset + nprobe]: Probed centroid IDs
    // [offset .. offset + nprobe]: Probed centroid distances
    // [aligned offset .. aligned offset + shortlist]: Merged results
    extern __shared__ float smem_pool[];
    float* smem_lut = smem_pool; // size: m_pq * 257
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_lut[m_pq * SMEM_PQ_STRIDE];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];

    // GpuSearchResult is 8-byte aligned; the raw float/uint offsets above are
    // not necessarily 8-byte aligned when m_pq is odd, so round the byte offset
    // up before reinterpreting the tail of the shared-memory pool.
    size_t lut_bytes = (size_t)m_pq * SMEM_PQ_STRIDE * sizeof(float);
    size_t probed_bytes = (size_t)nprobe * sizeof(uint32_t);
    size_t dist_bytes = (size_t)nprobe * sizeof(float);
    size_t raw_offset = lut_bytes + probed_bytes + dist_bytes;
    size_t aligned_offset = (raw_offset + 7u) & ~7u;
    GpuSearchResult* smem_merged_topk =
        (GpuSearchResult*)((char*)smem_pool + aligned_offset);

    // Initialize probed clusters list in thread 0
    if (tid == 0) {
        for (uint32_t p = 0; p < nprobe; ++p) {
            smem_probed_clusters[p] = 0xFFFFFFFFU;
            smem_cluster_dists[p] = 1e30f;
        }
    }
    __syncthreads();

    // 1. Race-Free Coarse Quantization: Find closest nprobe Voronoi centroids
    uint32_t local_probed_c[16];
    float local_probed_d[16];
    int local_probed_count = 0;
    int max_local = (nprobe < 16) ? (int)nprobe : 16;

    for (int i = 0; i < max_local; ++i) {
        local_probed_d[i] = 1e30f;
        local_probed_c[i] = 0xFFFFFFFFU;
    }

    for (uint32_t c = tid; c < nlist; c += block_size) {
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

    // Sequentially merge local top centroids into shared memory block-wide list
    for (int t = 0; t < block_size; ++t) {
        if (tid == t) {
            for (int i = 0; i < local_probed_count; ++i) {
                uint32_t c = local_probed_c[i];
                float dist = local_probed_d[i];

                if (dist < smem_cluster_dists[nprobe - 1]) {
                    int insert_pos = nprobe - 1;
                    for (uint32_t j = 0; j < nprobe; ++j) {
                        if (dist < smem_cluster_dists[j]) {
                            insert_pos = j;
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

    // 2. Scan the probed inverted lists.
    // Keep the thread-local collector compact: a 256-entry list would spill
    // several kilobytes per thread to local memory and destroy throughput.
    CandidateList<MAX_RERANK_CANDIDATES> local_candidates;
    uint32_t shortlist = rerank_k == 0 ? 1u : rerank_k;
    if (shortlist > MAX_RERANK_CANDIDATES) shortlist = MAX_RERANK_CANDIDATES;
    if (shortlist < top_k) shortlist = top_k;
    local_candidates.init((int)shortlist);

    // Codes are trained on x - centroid(x), so each cell needs its own lookup
    // table built against the residual q - centroid. One table per probed cell
    // is rebuilt in place; a single table holding the minimum across cells
    // would score every candidate against the wrong cell offset.
    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        // Build the residual LUT for this cell with 257-float stride, which
        // keeps all 32 lanes on distinct shared-memory banks during the scan.
        for (uint32_t m = 0; m < m_pq; ++m) {
            for (uint32_t c = tid; c < 256; c += block_size) {
                const float* cb_sub = &d_pq_codebooks[(m * 256 + c) * sub_dim];
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

        for (uint32_t i = tid; i < list_len; i += block_size) {
            uint32_t vec_pos = list_start + i;
            uint32_t vec_id = d_ivf_vec_ids[vec_pos];
            const uint8_t* codes = &d_pq_codes[vec_id * m_pq];

            // ADC Distance accumulation using the 257-strided residual LUT
            float adc_dist = 0.0f;
            #pragma unroll 8
            for (uint32_t m = 0; m < m_pq; ++m) {
                uint8_t code = codes[m];
                adc_dist += smem_lut[m * SMEM_PQ_STRIDE + code];
            }

            local_candidates.insert(vec_id, adc_dist);
        }
        __syncthreads();
    }

    // 3. Block-wide parallel top-k reduction/merger across all 128 threads
    block_bitonic_merge_topk<MAX_RERANK_CANDIDATES>(
        local_candidates,
        smem_merged_topk,
        (int)shortlist,
        tid,
        block_size
    );

    // 4. Exact re-ranking of the shortlist against the full-precision vectors.
    //
    // ADC ranks candidates on quantized codes; the error it introduces is what
    // limits recall. Re-scoring the full shortlist with true distances lets a
    // candidate that ADC ranked below top-k still be promoted into the result.
    //
    // Each warp owns one candidate so its reads are coalesced across 32 lanes;
    // a thread-per-candidate mapping would issue scattered 512-byte reads and
    // dominate the query latency.
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warps_per_block = block_size / WARP_SIZE;

    for (uint32_t k = (uint32_t)warp_id; k < shortlist; k += (uint32_t)warps_per_block) {
        uint32_t vec_id = smem_merged_topk[k].id;
        if (vec_id >= num_vectors) continue;

        float dist = warp_compute_distance(
            query,
            &d_vectors[vec_id * dim],
            (int)dim,
            lane_id,
            metric
        );
        if (lane_id == 0) {
            smem_merged_topk[k].distance = dist;
        }
    }
    __syncthreads();

    // 5. Re-sort the re-scored shortlist and emit the top-k.
    // The shortlist is at most 32 entries, so a single pass of selection sort
    // is cheaper than another bitonic network.
    if (tid == 0) {
        for (uint32_t i = 0; i < shortlist; ++i) {
            uint32_t best = i;
            for (uint32_t j = i + 1; j < shortlist; ++j) {
                if (smem_merged_topk[j].distance < smem_merged_topk[best].distance) {
                    best = j;
                }
            }
            if (best != i) {
                GpuSearchResult tmp = smem_merged_topk[i];
                smem_merged_topk[i] = smem_merged_topk[best];
                smem_merged_topk[best] = tmp;
            }
        }

        uint32_t out_base = query_idx * top_k;
        for (uint32_t k = 0; k < top_k; ++k) {
            d_out_results[out_base + k] = smem_merged_topk[k];
        }
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
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0 || top_k == 0 || tables->m_pq == 0 || tables->nlist == 0) return;

    // The host enforces top_k <= 256; clamp again here so an out-of-contract
    // caller can never overflow the dynamic shared-memory merge buffer.
    if (top_k > 256) top_k = 256;

    uint32_t clamped_nprobe = nprobe;
    if (clamped_nprobe == 0) clamped_nprobe = 1;
    if (clamped_nprobe > tables->nlist) clamped_nprobe = tables->nlist;

    uint32_t shortlist = rerank_k;
    if (shortlist == 0) shortlist = top_k;
    if (shortlist > MAX_RERANK_CANDIDATES) shortlist = MAX_RERANK_CANDIDATES;
    if (shortlist < top_k) shortlist = top_k;
    if (shortlist > 256) shortlist = 256;

    dim3 grid(num_queries);
    dim3 block(BLOCK_SIZE_IVF);

    // Shared memory: residual LUT (m_pq * 257 floats) + probed list
    // (nprobe u32 + nprobe f32) + 8-byte-aligned shortlist.
    size_t raw_smem = ((size_t)tables->m_pq * SMEM_PQ_STRIDE * sizeof(float)) +
                      ((size_t)clamped_nprobe * sizeof(uint32_t)) +
                      ((size_t)clamped_nprobe * sizeof(float));
    size_t aligned_smem = (raw_smem + 7u) & ~7u;
    size_t smem_bytes = aligned_smem + ((size_t)shortlist * sizeof(GpuSearchResult));

    ivf_pq_adc_kernel<<<grid, block, smem_bytes, stream>>>(
        tables->d_centroids,
        tables->d_pq_codebooks,
        tables->d_pq_codes,
        tables->d_ivf_offsets,
        tables->d_ivf_vec_ids,
        tables->d_vectors,
        tables->num_vectors,
        tables->dim,
        tables->nlist,
        tables->m_pq,
        tables->sub_dim,
        d_queries,
        num_queries,
        top_k,
        clamped_nprobe,
        shortlist,
        metric,
        d_out_results
    );
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
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!tables || !d_queries || !d_out_results) {
        return -1;
    }

    cudaStream_t s = (cudaStream_t)stream;
    flashvector::launch_ivf_pq_search(
        tables,
        d_queries,
        num_queries,
        top_k,
        nprobe,
        rerank_k,
        metric,
        d_out_results,
        s
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
