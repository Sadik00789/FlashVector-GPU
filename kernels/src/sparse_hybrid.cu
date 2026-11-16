#include "sparse_hybrid.cuh"
#include "ivf_rerank.cuh"

namespace flashvector {

#define FLAT_WARPS_PER_BLOCK 4
#define FLAT_THREADS_PER_BLOCK (FLAT_WARPS_PER_BLOCK * WARP_SIZE)

// ============================================================================
// 1. Concurrent Delta-Buffer Flat Scan Kernel (secondary CUDA stream)
// ============================================================================

__global__ void flat_scan_kernel(
    const float* __restrict__ d_vectors,
    const uint32_t* __restrict__ d_vec_ids,
    uint32_t num_vectors,
    uint32_t dim,
    const float* __restrict__ d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || top_k == 0) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    const float* query = &d_queries[query_idx * dim];

    __shared__ GpuSearchResult smem_topk[256];
    __shared__ GpuSearchResult smem_warp_cand[FLAT_WARPS_PER_BLOCK];

    for (uint32_t k = (uint32_t)tid; k < top_k; k += FLAT_THREADS_PER_BLOCK) {
        smem_topk[k] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();

    for (uint32_t base = 0; base < num_vectors; base += FLAT_WARPS_PER_BLOCK) {
        uint32_t idx = base + (uint32_t)warp_id;
        GpuSearchResult cand{0xFFFFFFFFU, 1e30f};

        if (idx < num_vectors && bitmask_allows(d_valid_mask, idx)) {
            float dist = warp_compute_distance(
                query, &d_vectors[(size_t)idx * dim], (int)dim, lane_id, metric
            );
            if (lane_id == 0) {
                cand.id = (d_vec_ids != nullptr) ? d_vec_ids[idx] : idx;
                cand.distance = dist;
            }
        }
        if (lane_id == 0) {
            smem_warp_cand[warp_id] = cand;
        }
        __syncthreads();

        if (tid == 0) {
            for (int w = 0; w < FLAT_WARPS_PER_BLOCK; ++w) {
                GpuSearchResult item = smem_warp_cand[w];
                if (item.id == 0xFFFFFFFFU) continue;
                if (item.distance < smem_topk[top_k - 1].distance) {
                    int pos = (int)top_k - 1;
                    while (pos > 0 && item.distance < smem_topk[pos - 1].distance) {
                        smem_topk[pos] = smem_topk[pos - 1];
                        pos--;
                    }
                    smem_topk[pos] = item;
                }
            }
        }
        __syncthreads();
    }

    uint32_t out_base = query_idx * top_k;
    for (uint32_t k = (uint32_t)tid; k < top_k; k += FLAT_THREADS_PER_BLOCK) {
        d_out_results[out_base + k] = smem_topk[k];
    }
}

void launch_flat_scan(
    const float* d_vectors,
    const uint32_t* d_vec_ids,
    uint32_t num_vectors,
    uint32_t dim,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0 || top_k == 0) return;
    if (top_k > 256u) top_k = 256u;
    if (num_vectors == 0) {
        cudaMemsetAsync(d_out_results, 0xFF, (size_t)num_queries * top_k * sizeof(GpuSearchResult), stream);
        return;
    }

    flat_scan_kernel<<<num_queries, FLAT_THREADS_PER_BLOCK, 0, stream>>>(
        d_vectors, d_vec_ids, num_vectors, dim,
        d_queries, num_queries, top_k, metric, d_valid_mask, d_out_results
    );
}

// ============================================================================
// 2. Device-Side Multi-Shard / Main+Delta Top-K Reduction Kernel
// ============================================================================

#define MERGE_BLOCK_SIZE 256
#define MERGE_HASH_SIZE 2048u

__device__ __forceinline__ bool merge_insert_unique_min(
    uint32_t* hash_ids,
    float* hash_dists,
    uint32_t id,
    float dist
) {
    uint32_t h = (id * 2654435761U) & (MERGE_HASH_SIZE - 1u);
    for (uint32_t p = 0; p < 16u; ++p) {
        uint32_t idx = (h + p) & (MERGE_HASH_SIZE - 1u);
        uint32_t prev = atomicCAS(&hash_ids[idx], 0xFFFFFFFFU, id);
        if (prev == 0xFFFFFFFFU) {
            hash_dists[idx] = dist;
            return true;
        }
        if (prev == id) {
            // Keep minimum distance across shards if duplicate ID appears
            if (dist < hash_dists[idx]) hash_dists[idx] = dist;
            return false;
        }
    }
    return false;
}

__global__ void merge_topk_kernel(
    const GpuSearchResult* __restrict__ d_shard_results, // [num_shards * num_queries * shard_top_k]
    uint32_t num_shards,
    uint32_t num_queries,
    uint32_t shard_top_k,
    uint32_t out_top_k,
    uint32_t buf_pow2,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || out_top_k == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ char smem_merge_raw[];
    GpuSearchResult* smem_items = (GpuSearchResult*)smem_merge_raw; // [buf_pow2]

    uint32_t total = num_shards * shard_top_k;
    for (uint32_t i = (uint32_t)tid; i < buf_pow2; i += (uint32_t)block_size) {
        if (i < total) {
            uint32_t shard = i / shard_top_k;
            uint32_t k = i % shard_top_k;
            size_t src = ((size_t)shard * num_queries + query_idx) * shard_top_k + k;
            GpuSearchResult r = d_shard_results[src];
            smem_items[i] = (r.id != 0xFFFFFFFFU) ? r : GpuSearchResult{0xFFFFFFFFU, 1e30f};
        } else {
            smem_items[i] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
        }
    }
    __syncthreads();

    block_bitonic_sort_inplace(smem_items, buf_pow2, tid, block_size);

    // Deduplicate sorted stream in thread 0 while keeping best out_top_k entries
    if (tid == 0) {
        uint32_t out_base = query_idx * out_top_k;
        uint32_t written = 0u;
        for (uint32_t i = 0u; i < total && written < out_top_k; ++i) {
            GpuSearchResult cur = smem_items[i];
            if (cur.id == 0xFFFFFFFFU) break;
            bool dup = false;
            for (uint32_t j = 0u; j < written; ++j) {
                if (d_out_results[out_base + j].id == cur.id) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                d_out_results[out_base + written] = cur;
                written++;
            }
        }
        for (uint32_t k = written; k < out_top_k; ++k) {
            d_out_results[out_base + k] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
        }
    }
}

void launch_merge_topk(
    const GpuSearchResult* d_shard_results,
    uint32_t num_shards,
    uint32_t num_queries,
    uint32_t shard_top_k,
    uint32_t out_top_k,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_shards == 0 || num_queries == 0 || out_top_k == 0) return;
    if (out_top_k > 256u) out_top_k = 256u;

    uint32_t total = num_shards * shard_top_k;
    uint32_t buf_pow2 = next_pow2_u32(total > 0 ? total : 1u);
    if (buf_pow2 > 4096u) buf_pow2 = 4096u;
    size_t smem_bytes = (size_t)buf_pow2 * sizeof(GpuSearchResult);

    merge_topk_kernel<<<num_queries, MERGE_BLOCK_SIZE, smem_bytes, stream>>>(
        d_shard_results, num_shards, num_queries, shard_top_k, out_top_k, buf_pow2, d_out_results
    );
}

// ============================================================================
// 3. CSR Sparse Dot-Product Kernel (SPLADE / BM25) + Block Bitonic Top-K
// ============================================================================

#define SPARSE_BLOCK_SIZE 256
#define SPARSE_HASH_SIZE 2048u
#define SPARSE_HASH_MAX_NNZ 1024u

__device__ __forceinline__ float sparse_hash_lookup(
    const uint32_t* keys,
    const float* vals,
    uint32_t term_id
) {
    uint32_t h = (term_id * 2654435761U) & (SPARSE_HASH_SIZE - 1u);
    #pragma unroll 8
    for (uint32_t p = 0u; p < 16u; ++p) {
        uint32_t idx = (h + p) & (SPARSE_HASH_SIZE - 1u);
        uint32_t k = keys[idx];
        if (k == term_id) return vals[idx];
        if (k == 0xFFFFFFFFU) return 0.0f;
    }
    return 0.0f;
}

__device__ __forceinline__ float sparse_binary_search(
    const uint32_t* __restrict__ q_idx,
    const float* __restrict__ q_val,
    uint32_t q_nnz,
    uint32_t term_id
) {
    uint32_t lo = 0u;
    uint32_t hi = q_nnz;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t k = q_idx[mid];
        if (k == term_id) return q_val[mid];
        if (k < term_id) lo = mid + 1u;
        else hi = mid;
    }
    return 0.0f;
}

__global__ void sparse_csr_search_kernel(
    const uint32_t* __restrict__ d_doc_indptr,
    const uint32_t* __restrict__ d_doc_indices,
    const float* __restrict__ d_doc_values,
    uint32_t num_docs,
    const uint32_t* __restrict__ d_q_indptr,
    const uint32_t* __restrict__ d_q_indices,
    const float* __restrict__ d_q_values,
    uint32_t num_queries,
    uint32_t top_k,
    const uint32_t* __restrict__ d_valid_mask,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || top_k == 0) return;

    int tid = threadIdx.x;
    uint32_t q_start = d_q_indptr[query_idx];
    uint32_t q_end = d_q_indptr[query_idx + 1];
    uint32_t q_nnz = q_end - q_start;

    __shared__ uint32_t smem_q_keys[SPARSE_HASH_SIZE];
    __shared__ float smem_q_vals[SPARSE_HASH_SIZE];
    __shared__ GpuSearchResult smem_buf[512]; // [0..256)=best so far, [256..512)=current batch

    for (uint32_t i = (uint32_t)tid; i < SPARSE_HASH_SIZE; i += SPARSE_BLOCK_SIZE) {
        smem_q_keys[i] = 0xFFFFFFFFU;
        smem_q_vals[i] = 0.0f;
    }
    for (int i = tid; i < 512; i += SPARSE_BLOCK_SIZE) {
        smem_buf[i] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();

    if (q_nnz == 0u) {
        uint32_t out_base = query_idx * top_k;
        for (uint32_t k = (uint32_t)tid; k < top_k; k += SPARSE_BLOCK_SIZE) {
            d_out_results[out_base + k] = GpuSearchResult{0xFFFFFFFFU, 1e30f};
        }
        return;
    }

    bool use_hash = (q_nnz <= SPARSE_HASH_MAX_NNZ);
    if (use_hash) {
        for (uint32_t i = (uint32_t)tid; i < q_nnz; i += SPARSE_BLOCK_SIZE) {
            uint32_t term = d_q_indices[q_start + i];
            float weight = d_q_values[q_start + i];
            uint32_t h = (term * 2654435761U) & (SPARSE_HASH_SIZE - 1u);
            for (uint32_t p = 0u; p < 32u; ++p) {
                uint32_t slot = (h + p) & (SPARSE_HASH_SIZE - 1u);
                uint32_t prev = atomicCAS(&smem_q_keys[slot], 0xFFFFFFFFU, term);
                if (prev == 0xFFFFFFFFU || prev == term) {
                    smem_q_vals[slot] = weight;
                    break;
                }
            }
        }
        __syncthreads();
    }

    // Stream documents in chunks of 256 and maintain top-k via parallel 512-entry bitonic merge
    for (uint32_t doc_base = 0u; doc_base < num_docs; doc_base += SPARSE_BLOCK_SIZE) {
        uint32_t doc_id = doc_base + (uint32_t)tid;
        GpuSearchResult cand{0xFFFFFFFFU, 1e30f};

        if (doc_id < num_docs && bitmask_allows(d_valid_mask, doc_id)) {
            uint32_t d_start = d_doc_indptr[doc_id];
            uint32_t d_end = d_doc_indptr[doc_id + 1];
            float dot = 0.0f;

            if (use_hash) {
                for (uint32_t p = d_start; p < d_end; ++p) {
                    float q_w = sparse_hash_lookup(smem_q_keys, smem_q_vals, d_doc_indices[p]);
                    if (q_w != 0.0f) {
                        dot = fmaf(d_doc_values[p], q_w, dot);
                    }
                }
            } else {
                const uint32_t* q_idx_ptr = &d_q_indices[q_start];
                const float* q_val_ptr = &d_q_values[q_start];
                for (uint32_t p = d_start; p < d_end; ++p) {
                    float q_w = sparse_binary_search(q_idx_ptr, q_val_ptr, q_nnz, d_doc_indices[p]);
                    if (q_w != 0.0f) {
                        dot = fmaf(d_doc_values[p], q_w, dot);
                    }
                }
            }

            if (dot > 0.0f) {
                cand.id = doc_id;
                cand.distance = -dot; // Negative score so ascending sort places highest score first
            }
        }

        smem_buf[256 + tid] = cand;
        __syncthreads();

        block_bitonic_sort_inplace(smem_buf, 512u, tid, SPARSE_BLOCK_SIZE);
    }

    uint32_t out_base = query_idx * top_k;
    for (uint32_t k = (uint32_t)tid; k < top_k; k += SPARSE_BLOCK_SIZE) {
        d_out_results[out_base + k] = smem_buf[k];
    }
}

void launch_sparse_search(
    const SparseCsrGpuMatrix* matrix,
    const uint32_t* d_q_indptr,
    const uint32_t* d_q_indices,
    const float* d_q_values,
    uint32_t num_queries,
    uint32_t top_k,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0 || top_k == 0) return;
    if (top_k > 256u) top_k = 256u;
    if (matrix->num_docs == 0) {
        cudaMemsetAsync(d_out_results, 0xFF, (size_t)num_queries * top_k * sizeof(GpuSearchResult), stream);
        return;
    }

    sparse_csr_search_kernel<<<num_queries, SPARSE_BLOCK_SIZE, 0, stream>>>(
        matrix->d_indptr, matrix->d_indices, matrix->d_values, matrix->num_docs,
        d_q_indptr, d_q_indices, d_q_values, num_queries, top_k, d_valid_mask, d_out_results
    );
}

// ============================================================================
// 4. On-Device Reciprocal Rank Fusion (RRF) Kernel
// ============================================================================

#define RRF_HASH_SIZE 2048u

__device__ __inline__ void rrf_accumulate(
    uint32_t* hash_ids,
    float* hash_scores,
    uint32_t doc_id,
    float score
) {
    uint32_t h = (doc_id * 2654435761U) & (RRF_HASH_SIZE - 1u);
    for (uint32_t p = 0u; p < 32u; ++p) {
        uint32_t slot = (h + p) & (RRF_HASH_SIZE - 1u);
        uint32_t prev = atomicCAS(&hash_ids[slot], 0xFFFFFFFFU, doc_id);
        if (prev == 0xFFFFFFFFU || prev == doc_id) {
            atomicAdd(&hash_scores[slot], score);
            return;
        }
    }
}

__global__ void rrf_fuse_kernel(
    const GpuSearchResult* __restrict__ d_dense_results,
    uint32_t dense_k,
    const GpuSearchResult* __restrict__ d_sparse_results,
    uint32_t sparse_k,
    uint32_t num_queries,
    uint32_t out_top_k,
    float rrf_k,
    float dense_weight,
    float sparse_weight,
    GpuSearchResult* __restrict__ d_out_results
) {
    uint32_t query_idx = blockIdx.x;
    if (query_idx >= num_queries || out_top_k == 0) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ uint32_t smem_ids[RRF_HASH_SIZE];
    __shared__ float smem_scores[RRF_HASH_SIZE];
    __shared__ GpuSearchResult smem_sorted[RRF_HASH_SIZE];

    for (uint32_t i = (uint32_t)tid; i < RRF_HASH_SIZE; i += (uint32_t)block_size) {
        smem_ids[i] = 0xFFFFFFFFU;
        smem_scores[i] = 0.0f;
    }
    __syncthreads();

    const GpuSearchResult* dense_row = &d_dense_results[(size_t)query_idx * dense_k];
    const GpuSearchResult* sparse_row = &d_sparse_results[(size_t)query_idx * sparse_k];

    for (uint32_t r = (uint32_t)tid; r < dense_k; r += (uint32_t)block_size) {
        uint32_t doc_id = dense_row[r].id;
        if (doc_id != 0xFFFFFFFFU) {
            float contrib = dense_weight / (rrf_k + (float)(r + 1u));
            rrf_accumulate(smem_ids, smem_scores, doc_id, contrib);
        }
    }
    for (uint32_t r = (uint32_t)tid; r < sparse_k; r += (uint32_t)block_size) {
        uint32_t doc_id = sparse_row[r].id;
        if (doc_id != 0xFFFFFFFFU) {
            float contrib = sparse_weight / (rrf_k + (float)(r + 1u));
            rrf_accumulate(smem_ids, smem_scores, doc_id, contrib);
        }
    }
    __syncthreads();

    for (uint32_t i = (uint32_t)tid; i < RRF_HASH_SIZE; i += (uint32_t)block_size) {
        uint32_t id = smem_ids[i];
        smem_sorted[i] = (id != 0xFFFFFFFFU)
            ? GpuSearchResult{id, -smem_scores[i]} // Negative RRF score for ascending bitonic sort
            : GpuSearchResult{0xFFFFFFFFU, 1e30f};
    }
    __syncthreads();

    block_bitonic_sort_inplace(smem_sorted, RRF_HASH_SIZE, tid, block_size);

    uint32_t out_base = query_idx * out_top_k;
    for (uint32_t k = (uint32_t)tid; k < out_top_k; k += (uint32_t)block_size) {
        d_out_results[out_base + k] = smem_sorted[k];
    }
}

void launch_rrf_fuse(
    const GpuSearchResult* d_dense_results,
    uint32_t dense_k,
    const GpuSearchResult* d_sparse_results,
    uint32_t sparse_k,
    uint32_t num_queries,
    uint32_t out_top_k,
    float rrf_k,
    float dense_weight,
    float sparse_weight,
    GpuSearchResult* d_out_results,
    cudaStream_t stream
) {
    if (num_queries == 0 || out_top_k == 0) return;
    if (out_top_k > 256u) out_top_k = 256u;

    rrf_fuse_kernel<<<num_queries, MERGE_BLOCK_SIZE, 0, stream>>>(
        d_dense_results, dense_k, d_sparse_results, sparse_k,
        num_queries, out_top_k, rrf_k, dense_weight, sparse_weight, d_out_results
    );
}

} // namespace flashvector

extern "C" {

int cuda_flat_scan_batch(
    const float* d_vectors,
    const uint32_t* d_vec_ids,
    uint32_t num_vectors,
    uint32_t dim,
    const float* d_queries,
    uint32_t num_queries,
    uint32_t top_k,
    MetricType metric,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!d_queries || !d_out_results) return -1;
    if (num_vectors > 0 && !d_vectors) return -1;
    flashvector::launch_flat_scan(
        d_vectors, d_vec_ids, num_vectors, dim, d_queries, num_queries,
        top_k, metric, d_valid_mask, d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_merge_topk(
    const GpuSearchResult* d_shard_results,
    uint32_t num_shards,
    uint32_t num_queries,
    uint32_t shard_top_k,
    uint32_t out_top_k,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!d_shard_results || !d_out_results) return -1;
    flashvector::launch_merge_topk(
        d_shard_results, num_shards, num_queries, shard_top_k, out_top_k,
        d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_sparse_search_batch(
    const SparseCsrGpuMatrix* matrix,
    const uint32_t* d_q_indptr,
    const uint32_t* d_q_indices,
    const float* d_q_values,
    uint32_t num_queries,
    uint32_t top_k,
    const uint32_t* d_valid_mask,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!matrix || !d_q_indptr || !d_out_results) return -1;
    flashvector::launch_sparse_search(
        matrix, d_q_indptr, d_q_indices, d_q_values, num_queries,
        top_k, d_valid_mask, d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

int cuda_rrf_fuse(
    const GpuSearchResult* d_dense_results,
    uint32_t dense_k,
    const GpuSearchResult* d_sparse_results,
    uint32_t sparse_k,
    uint32_t num_queries,
    uint32_t out_top_k,
    float rrf_k,
    float dense_weight,
    float sparse_weight,
    GpuSearchResult* d_out_results,
    void* stream
) {
    if (!d_dense_results || !d_sparse_results || !d_out_results) return -1;
    flashvector::launch_rrf_fuse(
        d_dense_results, dense_k, d_sparse_results, sparse_k, num_queries,
        out_top_k, rrf_k, dense_weight, sparse_weight, d_out_results, (cudaStream_t)stream
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

} // extern "C"
