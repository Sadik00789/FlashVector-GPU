#ifndef FLASHVECTOR_TYPES_H
#define FLASHVECTOR_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#define ALIGNAS_8 alignas(8)
extern "C" {
#else
#define ALIGNAS_8 _Alignas(8)
#endif

// Distance Metric Types
typedef enum {
    METRIC_L2 = 0,
    METRIC_COSINE = 1,
    METRIC_INNER_PRODUCT = 2
} MetricType;

// 3D Cartesian coordinates for PCA / UMAP / Three.js rendering
typedef struct {
    float x;
    float y;
    float z;
} Vector3D;

// Single search query descriptor
typedef struct {
    uint32_t query_id;
    const float* data;
    uint32_t dim;
    uint32_t top_k;
    uint32_t ef_search;
    uint32_t nprobe;
} GpuQuery;

// Search result candidate with explicit 64-bit alignment
typedef struct ALIGNAS_8 {
    uint32_t id;
    float distance;
} GpuSearchResult;

// Graph traversal hop record with explicit 64-bit alignment
typedef struct ALIGNAS_8 {
    uint32_t step;
    uint32_t from_node;
    uint32_t to_node;
    float distance;
} TraversalHop;

// Snapshot of a single candidate queue entry for hop-by-hop beam debugging
typedef struct ALIGNAS_8 {
    uint32_t id;
    float distance;
    uint32_t visited;
    uint32_t _pad;
} BeamCandidateRecord;

#define BEAM_DEBUG_MAX_CANDS 16

// Per-iteration beam state emitted by debug graph traversals
typedef struct ALIGNAS_8 {
    uint32_t iteration;
    uint32_t expanded_node;
    float expanded_distance;
    uint32_t eval_start;
    uint32_t eval_count;
    uint32_t cand_count;
    BeamCandidateRecord candidates[BEAM_DEBUG_MAX_CANDS];
} BeamStepRecord;

// Index configuration hyperparameters
typedef struct {
    uint32_t dim;
    uint32_t max_elements;
    uint32_t m;                // HNSW max outgoing edges per node (e.g. 16, 32, 64)
    uint32_t ef_construction;  // Construction beam width (e.g. 100, 200)
    uint32_t nlist;            // IVF Voronoi centroids (e.g. 256, 1024)
    uint32_t m_pq;             // Product quantization sub-vector partitions (e.g. 8, 16, 32)
    uint32_t metric;           // 0: L2, 1: Cosine, 2: Inner Product
} IndexConfig;

// Device-side IVF-PQ Codebook & Inverted List tables
typedef struct {
    const float* d_centroids;       // [nlist * dim] Voronoi centroids
    const float* d_pq_codebooks;    // [m_pq * 256 * sub_dim] Centroids per subspace
    const uint8_t* d_pq_codes;      // [num_vectors * m_pq] Quantized subvector codes
    const uint32_t* d_ivf_offsets;  // [nlist + 1] Prefix sum offsets for IVF lists
    const uint32_t* d_ivf_vec_ids;  // [num_vectors] Vector IDs in IVF posting lists
    const float* d_vectors;         // [num_vectors * dim] Full-precision vectors (device or mapped)
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t nlist;
    uint32_t m_pq;
    uint32_t sub_dim;
} IvfPqGpuTables;

// Device-side IVF-SQ8 Scalar Quantization tables
typedef struct {
    const float* d_centroids;       // [nlist * dim]
    const float* d_sq_mins;         // [dim] Per-dimension residual lower bound
    const float* d_sq_scales;       // [dim] Per-dimension quantization step (range / 255)
    const uint8_t* d_sq_codes;      // [num_vectors * dim] 8-bit quantized residuals
    const uint32_t* d_ivf_offsets;  // [nlist + 1]
    const uint32_t* d_ivf_vec_ids;  // [num_vectors]
    const float* d_vectors;         // [num_vectors * dim] Full-precision vectors for reranking
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t nlist;
} IvfSqGpuTables;

// Device-side IVF-RaBitQ Binary Quantization tables (FWHT orthogonal rotation + popcount)
typedef struct {
    const float* d_centroids;       // [nlist * dim] Original centroids for coarse probing
    const float* d_rot_centroids;   // [nlist * padded_dim] Rotated centroids
    const uint32_t* d_rabitq_bits;  // [num_vectors * words_per_vec] Packed sign bits
    const float* d_rabitq_meta;     // [num_vectors * 4] (res_norm_sq, rescale, popcnt_b, vec_norm_sq)
    const float* d_rot_signs;       // [3 * padded_dim] +-1 signs for 3-stage FWHT orthogonal rotation
    const uint32_t* d_ivf_offsets;  // [nlist + 1]
    const uint32_t* d_ivf_vec_ids;  // [num_vectors]
    const float* d_vectors;         // [num_vectors * dim] Full-precision vectors for reranking
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t padded_dim;            // Next power of 2 >= dim
    uint32_t words_per_vec;         // padded_dim / 32
    uint32_t nlist;
} IvfRabitqGpuTables;

// Device-side FP16 Tensor Core dataset descriptor
typedef struct {
    const uint16_t* d_vectors_fp16; // [num_vectors * padded_dim] IEEE-754 half vectors
    const float* d_vec_norms_sq;    // [num_vectors] Precomputed ||x||^2
    const float* d_vectors;         // Optional FP32 vectors for exact rerank
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t padded_dim;            // Multiple of 16 for wmma 16x16x16 tiles
} Fp16GpuTables;

// Device-side HNSW Graph Structure
typedef struct {
    const float* d_vectors;         // [num_nodes * dim] Raw or normalized vector data
    const uint32_t* d_adjacency;    // [num_nodes * m_max] Flattened neighbor adjacency array
    const uint32_t* d_degree;       // [num_nodes] Number of active neighbors per node
    uint32_t num_nodes;
    uint32_t dim;
    uint32_t m_max;
    uint32_t entry_point;
} HnswGpuGraph;

// Device-side CAGRA Fixed-Out-Degree Flat Graph Structure
typedef struct {
    const float* d_vectors;         // [num_nodes * dim]
    const uint32_t* d_adjacency;    // [num_nodes * graph_degree] Fixed-degree adjacency (INVALID=0xFFFFFFFF)
    const uint32_t* d_seed_nodes;   // Optional multi-start seed nodes [num_seeds]
    uint32_t num_nodes;
    uint32_t dim;
    uint32_t graph_degree;          // Fixed out-degree (typically 32 or 64)
    uint32_t entry_point;
    uint32_t num_seeds;
} CagraGpuGraph;

// Device-side CSR Sparse Matrix Descriptor (SPLADE / BM25)
typedef struct {
    const uint32_t* d_indptr;       // [num_docs + 1]
    const uint32_t* d_indices;      // [nnz] Sorted term IDs per document
    const float* d_values;          // [nnz] Nonzero term weights
    uint32_t num_docs;
    uint32_t nnz;
} SparseCsrGpuMatrix;

#undef ALIGNAS_8

#ifdef __cplusplus
}
#endif

#endif // FLASHVECTOR_TYPES_H
