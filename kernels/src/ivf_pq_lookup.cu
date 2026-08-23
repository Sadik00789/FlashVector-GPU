#include "ivf_pq_lookup.cuh"
#include <stdio.h>

namespace flashvector {

#define BLOCK_SIZE_IVF 128

// Batched IVF-PQ ADC search kernel
// Each thread block processes one query vector
__global__ void ivf_pq_adc_kernel(
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_pq_codebooks,
    const uint8_t* __restrict__ d_pq_codes,
    const uint32_t* __restrict__ d_ivf_offsets,
    const uint32_t* __restrict__ d_ivf_vec_ids,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t nlist,
    uint32_t m_pq,
    uint32_t sub_dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    uint32_t nprobe,
    MetricType metric,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* query = &d_queries[query_idx * dim];

    // Dynamic shared memory layout:
    // [0 .. m_pq * SMEM_PQ_STRIDE]: Distance Lookup Table (LUT)
    // [offset .. offset + nprobe]: Probed centroid IDs
    // [offset .. offset + nprobe]: Probed centroid distances
    // [offset .. offset + top_k]: Merged top-k results
    extern __shared__ float smem_pool[];
    float* smem_lut = smem_pool; // size: m_pq * 257
    uint32_t* smem_probed_clusters = (uint32_t*)&smem_lut[m_pq * SMEM_PQ_STRIDE];
    float* smem_cluster_dists = (float*)&smem_probed_clusters[nprobe];
    GpuSearchResult* smem_merged_topk = (GpuSearchResult*)&smem_cluster_dists[nprobe];

    // Initialize probed clusters list in thread 0
    if (tid == 0) {
        for (uint32_t p = 0; p < nprobe; ++p) {
            smem_probed_clusters[p] = 0xFFFFFFFFU;
            smem_cluster_dists[p] = 1e30f;
        }
    }
    __syncthreads();

    // 1. Race-Free Coarse Quantization: Find closest nprobe Voronoi centroids
    // Each thread tracks its local top candidates
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

    // 2. Build Asymmetric Distance Lookup Table (LUT) in Shared Memory with STRIDE 257
    for (uint32_t m = 0; m < m_pq; ++m) {
        for (uint32_t c = tid; c < 256; c += block_size) {
            const float* q_sub = &query[m * sub_dim];
            const float* cb_sub = &d_pq_codebooks[(m * 256 + c) * sub_dim];

            float dist = 0.0f;
            for (uint32_t sd = 0; sd < sub_dim; ++sd) {
                float diff = q_sub[sd] - cb_sub[sd];
                dist = fmaf(diff, diff, dist);
            }
            smem_lut[m * SMEM_PQ_STRIDE + c] = dist;
        }
    }
    __syncthreads();

    // 3. Scan assigned inverted lists and accumulate ADC distances
    CandidateList<64> local_candidates;
    local_candidates.init(top_k < 64 ? top_k : 64);

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = smem_probed_clusters[p];
        if (cluster_id >= nlist) continue;

        uint32_t list_start = d_ivf_offsets[cluster_id];
        uint32_t list_end = d_ivf_offsets[cluster_id + 1];
        uint32_t list_len = list_end - list_start;

        for (uint32_t i = tid; i < list_len; i += block_size) {
            uint32_t vec_pos = list_start + i;
            uint32_t vec_id = d_ivf_vec_ids[vec_pos];
            const uint8_t* codes = &d_pq_codes[vec_id * m_pq];

            // ADC Distance accumulation using 257-strided bank-conflict-free LUT
            float adc_dist = 0.0f;
            #pragma unroll 8
            for (uint32_t m = 0; m < m_pq; ++m) {
                uint8_t code = codes[m];
                adc_dist += smem_lut[m * SMEM_PQ_STRIDE + code];
            }

            local_candidates.insert(vec_id, adc_dist);
        }
    }
    __syncthreads();

    // 4. Block-wide parallel top-k reduction/merger across all 128 threads
    block_bitonic_merge_topk<64>(
        local_candidates,
        smem_merged_topk,
        (int)top_k,
        tid,
        block_size
    );

    // Thread 0 writes merged top-k results to global memory
    if (tid == 0) {
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
    MetricType metric,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0) return;

    dim3 grid(num_queries);
    dim3 block(BLOCK_SIZE_IVF);

    // Shared memory: LUT (m_pq * 257 floats) + probed list (nprobe u32 + nprobe f32) + top-k merged results
    size_t smem_bytes = (tables->m_pq * SMEM_PQ_STRIDE * sizeof(float)) + 
                        (nprobe * sizeof(uint32_t)) + 
                        (nprobe * sizeof(float)) + 
                        (top_k * sizeof(GpuSearchResult));

    ivf_pq_adc_kernel<<<grid, block, smem_bytes, stream>>>(
        tables->d_centroids,
        tables->d_pq_codebooks,
        tables->d_pq_codes,
        tables->d_ivf_offsets,
        tables->d_ivf_vec_ids,
        tables->num_vectors,
        tables->dim,
        tables->nlist,
        tables->m_pq,
        tables->sub_dim,
        d_queries,
        num_queries,
        top_k,
        nprobe,
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
        metric,
        d_out_results,
        s
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
