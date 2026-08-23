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

// 3D Cartesian coordinates for PCA / Three.js rendering
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

// Index configuration hyperparameters
typedef struct {
    uint32_t dim;
    uint32_t max_elements;
    uint32_t m;                // HNSW max outgoing edges per node (e.g. 16, 32, 64)
    uint32_t ef_construction; // Construction beam width (e.g. 100, 200)
    uint32_t nlist;           // IVF Voronoi centroids (e.g. 256, 1024)
    uint32_t m_pq;            // Product quantization sub-vector partitions (e.g. 8, 16, 32)
    uint32_t nbits_pq;        // Bits per sub-quantizer (typically 8 for 256 centroids)
    uint32_t metric;          // 0: L2, 1: Cosine, 2: Inner Product
} IndexConfig;

// Device-side IVF-PQ Codebook & Inverted List tables
typedef struct {
    const float* d_centroids;       // [nlist * dim] Voronoi centroids
    const float* d_pq_codebooks;    // [m_pq * 256 * sub_dim] Centroids per subspace
    const uint8_t* d_pq_codes;      // [num_vectors * m_pq] Quantized subvector codes
    const uint32_t* d_ivf_offsets;  // [nlist + 1] Prefix sum offsets for IVF lists
    const uint32_t* d_ivf_vec_ids;  // [num_vectors] Vector IDs in IVF posting lists
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t nlist;
    uint32_t m_pq;
    uint32_t sub_dim;
} IvfPqGpuTables;

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

#undef ALIGNAS_8

#ifdef __cplusplus
}
#endif

#endif // FLASHVECTOR_TYPES_H
