use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use crate::cluster::{KMeans, ProductQuantizer};
use crate::ffi::{
    gpu_cagra_search_batch, gpu_device_synchronize, gpu_flat_scan_batch, gpu_fp16_tc_search_batch,
    gpu_hnsw_search_batch, gpu_hnsw_search_batch_debug, gpu_host_get_device_pointer,
    gpu_ivf_pq_search_batch, gpu_ivf_rabitq_search_batch, gpu_ivf_sq_search_batch, gpu_rrf_fuse,
    gpu_sparse_search_batch, BeamStepRecord, CagraGpuGraph, CudaError, Fp16GpuTables,
    GpuSearchResult, HnswGpuGraph, IndexConfig, IvfPqGpuTables, IvfRabitqGpuTables, IvfSqGpuTables,
    MetricType, SparseCsrGpuMatrix, TraversalHop,
};
use crate::filter::RoaringBitset;
use crate::memory::{DeviceBuffer, PinnedBuffer};
use crate::metrics::MetricsTracker;
use crate::persistence::{write_aligned_snapshot, MmapSnapshot, SnapshotManifest};
use crate::quant::{encode_fp16_padded, RabitqEncoder, ScalarQuantizer8};
use crate::sparse::{pack_sparse_queries, HybridHit, SparseCsrData, SparseHit};
use crate::streams::{
    CudaStreamPool, MAX_BATCH_SIZE, MAX_BEAM_STEPS, MAX_HOPS, MAX_MASK_WORDS, MAX_SUPPORTED_DIM,
    MAX_TOP_K,
};

#[inline]
fn normalize_slice_in_place(values: &mut [f32], dim: usize) {
    for vec in values.chunks_exact_mut(dim) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let inv = 1.0 / norm;
        for x in vec.iter_mut() {
            *x *= inv;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvfQuantMode {
    Pq,
    Sq8,
    Rabitq,
}

#[derive(Debug, Clone)]
pub struct BeamDebugTrace {
    pub results: Vec<GpuSearchResult>,
    pub hops: Vec<TraversalHop>,
    pub steps: Vec<BeamStepRecord>,
}

#[derive(Debug, Clone)]
pub struct IvfProbeTelemetry {
    pub rank: usize,
    pub cluster_id: u32,
    pub centroid_distance: f32,
    pub list_len: usize,
    pub best_distance: Option<f32>,
    pub result_ids: Vec<u32>,
}

/// Per-Voronoi-cell PQ reconstruction statistics used by the Studio heatmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterPqDistortion {
    pub cluster_id: usize,
    /// Mean squared reconstruction error normalized by the original vector norm.
    pub avg_reconstruction_error: f32,
    pub vector_count: usize,
}

struct DeltaBuffer {
    vectors: Vec<f32>,
    ids: Vec<u32>,
    d_vectors: DeviceBuffer<f32>,
    d_ids: DeviceBuffer<u32>,
    d_valid_words: DeviceBuffer<u32>,
    capacity: usize,
}

impl DeltaBuffer {
    fn empty() -> Result<Self, CudaError> {
        Ok(Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            d_vectors: DeviceBuffer::new(0)?,
            d_ids: DeviceBuffer::new(0)?,
            d_valid_words: DeviceBuffer::new(0)?,
            capacity: 0,
        })
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

struct DeltaScanJob<'a> {
    ds: &'a crate::streams::CudaStream,
    _q_guard: Option<parking_lot::MutexGuard<'a, DeviceBuffer<f32>>>,
    _res_guard: parking_lot::MutexGuard<'a, DeviceBuffer<GpuSearchResult>>,
    _mask_guard: parking_lot::MutexGuard<'a, DeviceBuffer<u32>>,
    delta_host: Vec<GpuSearchResult>,
    synced: bool,
}

impl DeltaScanJob<'_> {
    fn sync_and_take(&mut self) -> Result<Vec<GpuSearchResult>, CudaError> {
        if !self.synced {
            self.ds.sync()?;
            self.synced = true;
        }
        Ok(std::mem::take(&mut self.delta_host))
    }
}

impl Drop for DeltaScanJob<'_> {
    fn drop(&mut self) {
        if !self.synced {
            let _ = self.ds.sync();
            self.synced = true;
        }
    }
}

pub struct GpuVectorIndex {
    config: IndexConfig,
    metrics: Arc<MetricsTracker>,
    streams: Arc<CudaStreamPool>,

    // Raw vectors (main index slots 0..num_vectors)
    num_vectors: usize,
    raw_vectors: PinnedBuffer<f32>,
    d_vectors: DeviceBuffer<f32>,
    use_mapped_rerank: bool,

    // Logical ID mapping + tombstones + live delta buffer
    logical_ids: Vec<u32>,
    id_remapped: bool,
    next_logical_id: u32,
    tombstones: RoaringBitset,
    main_valid_words: Vec<u32>,
    d_main_valid_words: DeviceBuffer<u32>,
    delta: DeltaBuffer,

    // HNSW Graph
    entry_point: u32,
    h_adjacency: PinnedBuffer<u32>,
    d_adjacency: DeviceBuffer<u32>,
    h_degrees: PinnedBuffer<u32>,
    d_degrees: DeviceBuffer<u32>,

    // CAGRA Flat Fixed-Degree Graph
    cagra_degree: usize,
    cagra_seeds: Vec<u32>,
    h_cagra_adj: PinnedBuffer<u32>,
    d_cagra_adj: DeviceBuffer<u32>,
    d_cagra_seeds: DeviceBuffer<u32>,

    // IVF-PQ Tables
    h_centroids: PinnedBuffer<f32>,
    d_centroids: DeviceBuffer<f32>,
    h_codebooks: PinnedBuffer<f32>,
    d_codebooks: DeviceBuffer<f32>,
    h_codes: PinnedBuffer<u8>,
    d_codes: DeviceBuffer<u8>,
    h_ivf_offsets: PinnedBuffer<u32>,
    d_ivf_offsets: DeviceBuffer<u32>,
    h_ivf_vec_ids: PinnedBuffer<u32>,
    d_ivf_vec_ids: DeviceBuffer<u32>,

    // IVF-SQ8 Tables
    h_sq_mins: Vec<f32>,
    d_sq_mins: DeviceBuffer<f32>,
    h_sq_scales: Vec<f32>,
    d_sq_scales: DeviceBuffer<f32>,
    h_sq_codes: Vec<u8>,
    d_sq_codes: DeviceBuffer<u8>,

    // IVF-RaBitQ Tables
    rabitq_padded_dim: usize,
    h_rabitq_rot_signs: Vec<f32>,
    d_rabitq_rot_signs: DeviceBuffer<f32>,
    h_rabitq_rot_centroids: Vec<f32>,
    d_rabitq_rot_centroids: DeviceBuffer<f32>,
    h_rabitq_bits: Vec<u32>,
    d_rabitq_bits: DeviceBuffer<u32>,
    h_rabitq_meta: Vec<f32>,
    d_rabitq_meta: DeviceBuffer<f32>,

    // FP16 Tensor Core Buffers
    fp16_padded_dim: usize,
    h_vectors_fp16: Vec<u16>,
    d_vectors_fp16: DeviceBuffer<u16>,
    h_vec_norms_sq: Vec<f32>,
    d_vec_norms_sq: DeviceBuffer<f32>,

    // CSR Sparse Matrix (SPLADE / BM25)
    sparse_data: SparseCsrData,
    d_sparse_indptr: DeviceBuffer<u32>,
    d_sparse_indices: DeviceBuffer<u32>,
    d_sparse_values: DeviceBuffer<f32>,

    is_built: bool,
}

impl GpuVectorIndex {
    pub fn new(config: IndexConfig) -> Result<Self, CudaError> {
        Self::new_with_streams(config, 4)
    }

    pub fn new_with_streams(config: IndexConfig, pool_size: usize) -> Result<Self, CudaError> {
        let streams = Arc::new(CudaStreamPool::new(pool_size)?);
        let metrics = Arc::new(MetricsTracker::new());

        Ok(Self {
            config,
            metrics,
            streams,
            num_vectors: 0,
            raw_vectors: PinnedBuffer::new(0)?,
            d_vectors: DeviceBuffer::new(0)?,
            use_mapped_rerank: false,
            logical_ids: Vec::new(),
            id_remapped: false,
            next_logical_id: 0,
            tombstones: RoaringBitset::new(),
            main_valid_words: Vec::new(),
            d_main_valid_words: DeviceBuffer::new(0)?,
            delta: DeltaBuffer::empty()?,
            entry_point: 0,
            h_adjacency: PinnedBuffer::new(0)?,
            d_adjacency: DeviceBuffer::new(0)?,
            h_degrees: PinnedBuffer::new(0)?,
            d_degrees: DeviceBuffer::new(0)?,
            cagra_degree: 0,
            cagra_seeds: Vec::new(),
            h_cagra_adj: PinnedBuffer::new(0)?,
            d_cagra_adj: DeviceBuffer::new(0)?,
            d_cagra_seeds: DeviceBuffer::new(0)?,
            h_centroids: PinnedBuffer::new(0)?,
            d_centroids: DeviceBuffer::new(0)?,
            h_codebooks: PinnedBuffer::new(0)?,
            d_codebooks: DeviceBuffer::new(0)?,
            h_codes: PinnedBuffer::new(0)?,
            d_codes: DeviceBuffer::new(0)?,
            h_ivf_offsets: PinnedBuffer::new(0)?,
            d_ivf_offsets: DeviceBuffer::new(0)?,
            h_ivf_vec_ids: PinnedBuffer::new(0)?,
            d_ivf_vec_ids: DeviceBuffer::new(0)?,
            h_sq_mins: Vec::new(),
            d_sq_mins: DeviceBuffer::new(0)?,
            h_sq_scales: Vec::new(),
            d_sq_scales: DeviceBuffer::new(0)?,
            h_sq_codes: Vec::new(),
            d_sq_codes: DeviceBuffer::new(0)?,
            rabitq_padded_dim: 0,
            h_rabitq_rot_signs: Vec::new(),
            d_rabitq_rot_signs: DeviceBuffer::new(0)?,
            h_rabitq_rot_centroids: Vec::new(),
            d_rabitq_rot_centroids: DeviceBuffer::new(0)?,
            h_rabitq_bits: Vec::new(),
            d_rabitq_bits: DeviceBuffer::new(0)?,
            h_rabitq_meta: Vec::new(),
            d_rabitq_meta: DeviceBuffer::new(0)?,
            fp16_padded_dim: 0,
            h_vectors_fp16: Vec::new(),
            d_vectors_fp16: DeviceBuffer::new(0)?,
            h_vec_norms_sq: Vec::new(),
            d_vec_norms_sq: DeviceBuffer::new(0)?,
            sparse_data: SparseCsrData::default(),
            d_sparse_indptr: DeviceBuffer::new(0)?,
            d_sparse_indices: DeviceBuffer::new(0)?,
            d_sparse_values: DeviceBuffer::new(0)?,
            is_built: false,
        })
    }

    #[inline]
    pub fn metric(&self) -> MetricType {
        match self.config.metric {
            0 => MetricType::L2,
            1 => MetricType::Cosine,
            2 => MetricType::InnerProduct,
            _ => MetricType::L2,
        }
    }

    /// Control whether IVF exact reranking streams vectors directly from pinned
    /// host memory via zero-copy `cudaHostGetDevicePointer` instead of VRAM.
    pub fn set_mapped_rerank(&mut self, enabled: bool) {
        self.use_mapped_rerank = enabled;
    }

    fn rerank_vectors_device_ptr(&self) -> *const f32 {
        if self.use_mapped_rerank && !self.raw_vectors.is_empty() {
            if let Ok(ptr) = gpu_host_get_device_pointer(self.raw_vectors.as_ptr() as *mut f32) {
                return ptr;
            }
        }
        self.d_vectors.as_ptr()
    }

    fn validate_search_shape(
        &self,
        num_queries: usize,
        top_k: usize,
        query_floats: usize,
    ) -> Result<(), CudaError> {
        let dim = self.config.dim as usize;
        if dim == 0 || dim > MAX_SUPPORTED_DIM {
            return Err(CudaError::InvalidArgument);
        }
        if num_queries > MAX_BATCH_SIZE || query_floats != num_queries * dim {
            return Err(CudaError::InvalidArgument);
        }
        if top_k > MAX_TOP_K {
            return Err(CudaError::InvalidArgument);
        }
        Ok(())
    }

    pub fn build(&mut self, dataset: &[f32]) -> Result<(), CudaError> {
        let dim = self.config.dim as usize;
        assert!(dim > 0, "Dimension must be greater than zero");
        assert!(
            dim <= MAX_SUPPORTED_DIM,
            "Dimension exceeds maximum supported"
        );
        assert!(!dataset.is_empty(), "Dataset cannot be empty");

        let normalized_storage: Vec<f32>;
        let dataset: &[f32] = if self.config.metric == MetricType::Cosine as u32 {
            let mut normalized = dataset.to_vec();
            normalize_slice_in_place(&mut normalized, dim);
            normalized_storage = normalized;
            &normalized_storage
        } else {
            dataset
        };

        let num_vectors = dataset.len() / dim;
        assert!(num_vectors > 0, "Dataset cannot be empty");
        let logical_ids: Vec<u32> = (0..num_vectors as u32).collect();
        self.build_internal_with_ids(dataset, logical_ids, num_vectors as u32, false)
    }

    fn build_internal_with_ids(
        &mut self,
        dataset: &[f32],
        logical_ids: Vec<u32>,
        next_logical_id: u32,
        id_remapped: bool,
    ) -> Result<(), CudaError> {
        let dim = self.config.dim as usize;
        let num_vectors = dataset.len() / dim;
        self.num_vectors = num_vectors;

        info!(
            "Building FlashVector-GPU Index for {} vectors of dimension {}",
            num_vectors, dim
        );

        let stream = self.streams.get_stream();

        // 1. Store Raw Vectors in Pinned & Device memory
        let mut raw_pinned = PinnedBuffer::new(dataset.len())?;
        raw_pinned.copy_from_slice(dataset);
        let mut d_vecs = DeviceBuffer::new(dataset.len())?;
        d_vecs.copy_from_pinned_async(&raw_pinned, stream.raw())?;

        // 2. Build HNSW Graph
        let m = self.config.m.max(4) as usize;
        let ef_construction = self.config.ef_construction as usize;
        let (adjacency, degrees, best_node) =
            crate::graph::build_hnsw(dataset, num_vectors, dim, m, ef_construction);

        let h_adj = PinnedBuffer::from_slice(&adjacency)?;
        let h_deg = PinnedBuffer::from_slice(&degrees)?;
        let mut d_adj = DeviceBuffer::new(adjacency.len())?;
        let mut d_deg = DeviceBuffer::new(degrees.len())?;
        d_adj.copy_from_pinned_async(&h_adj, stream.raw())?;
        d_deg.copy_from_pinned_async(&h_deg, stream.raw())?;

        // 3. Build CAGRA flat fixed-out-degree graph seeded from HNSW + NN-descent
        let cagra_degree = m.next_power_of_two().clamp(16, 64);
        let (cagra_adj, actual_cagra_deg, _, cagra_seeds) = crate::graph::build_cagra(
            dataset,
            num_vectors,
            dim,
            cagra_degree,
            Some((&adjacency, &degrees, best_node)),
        );
        let h_cagra = PinnedBuffer::from_slice(&cagra_adj)?;
        let mut d_cagra = DeviceBuffer::new(cagra_adj.len())?;
        d_cagra.copy_from_pinned_async(&h_cagra, stream.raw())?;
        let mut d_seeds = DeviceBuffer::new(cagra_seeds.len())?;
        d_seeds.copy_from_host_async(&cagra_seeds, stream.raw())?;

        // 4. Train IVF Voronoi Centroids + PQ + SQ8 + RaBitQ + FP16
        let nlist = (self.config.nlist as usize).min(num_vectors).max(1);
        let m_pq = self.config.m_pq as usize;
        assert!(m_pq > 0, "m_pq must be greater than zero");
        assert!(
            dim.is_multiple_of(m_pq),
            "Dimension must be divisible by m_pq"
        );

        let kmeans_res = KMeans::fit(dataset, dim, nlist, 25);

        let mut residuals = vec![0.0f32; dataset.len()];
        residuals
            .par_chunks_exact_mut(dim)
            .enumerate()
            .for_each(|(vec_id, dst)| {
                let cluster = kmeans_res.assignments[vec_id] as usize;
                let centroid = &kmeans_res.centroids[cluster * dim..(cluster + 1) * dim];
                let vector = &dataset[vec_id * dim..(vec_id + 1) * dim];
                for d in 0..dim {
                    dst[d] = vector[d] - centroid[d];
                }
            });

        let pq = ProductQuantizer::train(&residuals, dim, m_pq, 25);
        let codes = pq.encode(&residuals);

        let sq8 = ScalarQuantizer8::fit(&residuals, dim);
        let sq_codes = sq8.encode(&residuals);

        let rabitq = RabitqEncoder::encode(
            dataset,
            &kmeans_res.centroids,
            &kmeans_res.assignments,
            dim,
            0x5F37_59DF_CAFE_BABEu64,
        );

        let (fp16_vecs, vec_norms_sq, fp16_padded_dim) = encode_fp16_padded(dataset, dim);

        // Inverted Posting Lists
        let mut cluster_lists: Vec<Vec<u32>> = vec![Vec::new(); nlist];
        for (vec_id, &c) in kmeans_res.assignments.iter().enumerate() {
            cluster_lists[c as usize].push(vec_id as u32);
        }
        let mut ivf_offsets = Vec::with_capacity(nlist + 1);
        let mut ivf_vec_ids = Vec::with_capacity(num_vectors);
        let mut offset = 0u32;
        for list in &cluster_lists {
            ivf_offsets.push(offset);
            ivf_vec_ids.extend_from_slice(list);
            offset += list.len() as u32;
        }
        ivf_offsets.push(offset);

        // Upload IVF-PQ
        let h_cent = PinnedBuffer::from_slice(&kmeans_res.centroids)?;
        let mut d_cent = DeviceBuffer::new(kmeans_res.centroids.len())?;
        d_cent.copy_from_pinned_async(&h_cent, stream.raw())?;

        let h_cb = PinnedBuffer::from_slice(&pq.codebooks)?;
        let mut d_cb = DeviceBuffer::new(pq.codebooks.len())?;
        d_cb.copy_from_pinned_async(&h_cb, stream.raw())?;

        let h_cd = PinnedBuffer::from_slice(&codes)?;
        let mut d_cd = DeviceBuffer::new(codes.len())?;
        d_cd.copy_from_pinned_async(&h_cd, stream.raw())?;

        let h_off = PinnedBuffer::from_slice(&ivf_offsets)?;
        let mut d_off = DeviceBuffer::new(ivf_offsets.len())?;
        d_off.copy_from_pinned_async(&h_off, stream.raw())?;

        let h_vids = PinnedBuffer::from_slice(&ivf_vec_ids)?;
        let mut d_vids = DeviceBuffer::new(ivf_vec_ids.len())?;
        d_vids.copy_from_pinned_async(&h_vids, stream.raw())?;

        // Upload IVF-SQ8
        let mut d_sq_mins = DeviceBuffer::new(sq8.mins.len())?;
        let mut d_sq_scales = DeviceBuffer::new(sq8.scales.len())?;
        let mut d_sq_codes = DeviceBuffer::new(sq_codes.len())?;
        d_sq_mins.copy_from_host_async(&sq8.mins, stream.raw())?;
        d_sq_scales.copy_from_host_async(&sq8.scales, stream.raw())?;
        d_sq_codes.copy_from_host_async(&sq_codes, stream.raw())?;

        // Upload IVF-RaBitQ
        let mut d_rq_signs = DeviceBuffer::new(rabitq.rot_signs.len())?;
        let mut d_rq_cent = DeviceBuffer::new(rabitq.rot_centroids.len())?;
        let mut d_rq_bits = DeviceBuffer::new(rabitq.bits.len())?;
        let mut d_rq_meta = DeviceBuffer::new(rabitq.meta.len())?;
        d_rq_signs.copy_from_host_async(&rabitq.rot_signs, stream.raw())?;
        d_rq_cent.copy_from_host_async(&rabitq.rot_centroids, stream.raw())?;
        d_rq_bits.copy_from_host_async(&rabitq.bits, stream.raw())?;
        d_rq_meta.copy_from_host_async(&rabitq.meta, stream.raw())?;

        // Upload FP16 Tensor Core buffers
        let mut d_fp16 = DeviceBuffer::new(fp16_vecs.len())?;
        let mut d_norms = DeviceBuffer::new(vec_norms_sq.len())?;
        d_fp16.copy_from_host_async(&fp16_vecs, stream.raw())?;
        d_norms.copy_from_host_async(&vec_norms_sq, stream.raw())?;

        // Initialize valid bitmask (all 1s for 0..num_vectors)
        let mask_words = num_vectors.div_ceil(32);
        let mut main_valid_words = vec![u32::MAX; mask_words];
        if !num_vectors.is_multiple_of(32) && !main_valid_words.is_empty() {
            let rem = num_vectors % 32;
            *main_valid_words.last_mut().unwrap() = (1u32 << rem) - 1;
        }
        let mut d_main_valid = DeviceBuffer::new(mask_words)?;
        d_main_valid.copy_from_host_async(&main_valid_words, stream.raw())?;

        stream.sync()?;

        self.raw_vectors = raw_pinned;
        self.d_vectors = d_vecs;
        self.logical_ids = logical_ids;
        self.id_remapped = id_remapped;
        self.next_logical_id = next_logical_id;
        self.tombstones = RoaringBitset::new();
        self.main_valid_words = main_valid_words;
        self.d_main_valid_words = d_main_valid;
        self.delta = DeltaBuffer::empty()?;

        self.h_adjacency = h_adj;
        self.d_adjacency = d_adj;
        self.h_degrees = h_deg;
        self.d_degrees = d_deg;
        self.entry_point = best_node;

        self.cagra_degree = actual_cagra_deg;
        self.cagra_seeds = cagra_seeds;
        self.h_cagra_adj = h_cagra;
        self.d_cagra_adj = d_cagra;
        self.d_cagra_seeds = d_seeds;

        self.h_centroids = h_cent;
        self.d_centroids = d_cent;
        self.h_codebooks = h_cb;
        self.d_codebooks = d_cb;
        self.h_codes = h_cd;
        self.d_codes = d_cd;
        self.h_ivf_offsets = h_off;
        self.d_ivf_offsets = d_off;
        self.h_ivf_vec_ids = h_vids;
        self.d_ivf_vec_ids = d_vids;

        self.h_sq_mins = sq8.mins;
        self.d_sq_mins = d_sq_mins;
        self.h_sq_scales = sq8.scales;
        self.d_sq_scales = d_sq_scales;
        self.h_sq_codes = sq_codes;
        self.d_sq_codes = d_sq_codes;

        self.rabitq_padded_dim = rabitq.padded_dim;
        self.h_rabitq_rot_signs = rabitq.rot_signs;
        self.d_rabitq_rot_signs = d_rq_signs;
        self.h_rabitq_rot_centroids = rabitq.rot_centroids;
        self.d_rabitq_rot_centroids = d_rq_cent;
        self.h_rabitq_bits = rabitq.bits;
        self.d_rabitq_bits = d_rq_bits;
        self.h_rabitq_meta = rabitq.meta;
        self.d_rabitq_meta = d_rq_meta;

        self.fp16_padded_dim = fp16_padded_dim;
        self.h_vectors_fp16 = fp16_vecs;
        self.d_vectors_fp16 = d_fp16;
        self.h_vec_norms_sq = vec_norms_sq;
        self.d_vec_norms_sq = d_norms;

        self.is_built = true;
        info!("FlashVector-GPU Index built and synchronized to VRAM");
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Tombstone Deletes, Live Delta Inserts, and Compaction
    // ------------------------------------------------------------------------

    /// Non-blocking delete via GPU tombstone bitset.
    pub fn delete_ids(&mut self, ids: &[u32]) -> Result<usize, CudaError> {
        if !self.is_built || ids.is_empty() {
            return Ok(0);
        }
        let mut deleted_count = 0usize;
        let id_lookup: HashMap<u32, usize> = if self.id_remapped {
            self.logical_ids
                .iter()
                .enumerate()
                .map(|(s, &lid)| (lid, s))
                .collect()
        } else {
            HashMap::new()
        };

        let mut delta_dirty = false;
        let delta_lookup: HashMap<u32, usize> = self
            .delta
            .ids
            .iter()
            .enumerate()
            .map(|(i, &lid)| (lid, i))
            .collect();

        for &lid in ids {
            if self.tombstones.contains(lid) {
                continue;
            }
            if let Some(&d_slot) = delta_lookup.get(&lid) {
                self.tombstones.insert(lid);
                deleted_count += 1;
                delta_dirty = true;
                let _ = d_slot;
                continue;
            }
            let main_slot = if self.id_remapped {
                id_lookup.get(&lid).copied()
            } else if (lid as usize) < self.num_vectors {
                Some(lid as usize)
            } else {
                None
            };
            if let Some(slot) = main_slot {
                self.tombstones.insert(lid);
                self.main_valid_words[slot >> 5] &= !(1u32 << (slot & 31));
                deleted_count += 1;
            }
        }

        if deleted_count > 0 {
            let stream = self.streams.get_stream();
            self.d_main_valid_words
                .copy_from_host_async(&self.main_valid_words, stream.raw())?;
            stream.sync()?;
            if delta_dirty {
                self.upload_delta_valid_mask()?;
            }
        }

        Ok(deleted_count)
    }

    /// Ingest live inserts into the flat GPU delta buffer searched concurrently
    /// on the secondary CUDA stream (`streams.delta_stream()`).
    pub fn add_vectors(&mut self, vectors: &[f32]) -> Result<Vec<u32>, CudaError> {
        let dim = self.config.dim as usize;
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        if !vectors.len().is_multiple_of(dim) {
            return Err(CudaError::InvalidArgument);
        }
        let count = vectors.len() / dim;

        if !self.is_built {
            self.build(vectors)?;
            return Ok((0..count as u32).collect());
        }

        let mut staged = vectors.to_vec();
        if self.config.metric == MetricType::Cosine as u32 {
            normalize_slice_in_place(&mut staged, dim);
        }

        let start_id = self.next_logical_id;
        let assigned: Vec<u32> = (0..count as u32).map(|i| start_id + i).collect();
        self.next_logical_id = self.next_logical_id.saturating_add(count as u32);

        self.delta.vectors.extend_from_slice(&staged);
        self.delta.ids.extend_from_slice(&assigned);

        let new_len = self.delta.len();
        if new_len > self.delta.capacity {
            let cap = new_len.next_power_of_two().max(256);
            self.delta.d_vectors = DeviceBuffer::new(cap * dim)?;
            self.delta.d_ids = DeviceBuffer::new(cap)?;
            self.delta.d_valid_words = DeviceBuffer::new(cap.div_ceil(32))?;
            self.delta.capacity = cap;
        }

        let delta_stream = self.streams.delta_stream();
        self.delta
            .d_vectors
            .copy_from_host_async(&self.delta.vectors, delta_stream.raw())?;
        self.delta
            .d_ids
            .copy_from_host_async(&self.delta.ids, delta_stream.raw())?;
        delta_stream.sync()?;
        self.upload_delta_valid_mask()?;

        Ok(assigned)
    }

    fn upload_delta_valid_mask(&mut self) -> Result<(), CudaError> {
        let len = self.delta.len();
        if len == 0 {
            return Ok(());
        }
        let num_words = len.div_ceil(32);
        let mut words = vec![0u32; num_words];
        for (i, &lid) in self.delta.ids.iter().enumerate() {
            if !self.tombstones.contains(lid) {
                words[i >> 5] |= 1u32 << (i & 31);
            }
        }
        let delta_stream = self.streams.delta_stream();
        self.delta
            .d_valid_words
            .copy_from_host_async(&words, delta_stream.raw())?;
        delta_stream.sync()?;
        Ok(())
    }

    /// Synchronously compact live main + delta vectors (removing tombstones)
    /// into a fresh HNSW + CAGRA + IVF index while preserving logical IDs.
    pub fn compact(&mut self) -> Result<(), CudaError> {
        if !self.is_built || (self.tombstones.is_empty() && self.delta.is_empty()) {
            return Ok(());
        }
        let dim = self.config.dim as usize;
        let (live_vecs, live_ids) = self.collect_live_vectors(dim);
        if live_vecs.is_empty() {
            self.num_vectors = 0;
            self.logical_ids.clear();
            self.delta = DeltaBuffer::empty()?;
            self.is_built = false;
            return Ok(());
        }
        let next_lid = self.next_logical_id;
        self.build_internal_with_ids(&live_vecs, live_ids, next_lid, true)
    }

    fn collect_live_vectors(&self, dim: usize) -> (Vec<f32>, Vec<u32>) {
        let mut live_vecs = Vec::new();
        let mut live_ids = Vec::new();
        let main_slice = self.raw_vectors.as_slice();
        for (slot, &lid) in self.logical_ids.iter().enumerate() {
            if !self.tombstones.contains(lid) {
                live_vecs.extend_from_slice(&main_slice[slot * dim..(slot + 1) * dim]);
                live_ids.push(lid);
            }
        }
        for (i, &lid) in self.delta.ids.iter().enumerate() {
            if !self.tombstones.contains(lid) {
                live_vecs.extend_from_slice(&self.delta.vectors[i * dim..(i + 1) * dim]);
                live_ids.push(lid);
            }
        }
        (live_vecs, live_ids)
    }

    /// Compact tombstones + delta buffer asynchronously on a background thread
    /// without blocking concurrent search readers during the graph/IVF rebuild.
    pub fn compact_in_background(
        index: Arc<RwLock<GpuVectorIndex>>,
    ) -> std::thread::JoinHandle<Result<(), CudaError>> {
        std::thread::spawn(move || {
            let (config, live_vecs, live_ids, watermark, sparse_opt) = {
                let guard = index.read();
                if !guard.is_built || (guard.tombstones.is_empty() && guard.delta.is_empty()) {
                    return Ok(());
                }
                let dim = guard.config.dim as usize;
                let (vecs, ids) = guard.collect_live_vectors(dim);
                let sparse_opt = if guard.sparse_data.num_docs > 0 {
                    Some(guard.sparse_data.clone())
                } else {
                    None
                };
                (guard.config, vecs, ids, guard.next_logical_id, sparse_opt)
            };

            if live_vecs.is_empty() {
                return Ok(());
            }

            let mut rebuilt = GpuVectorIndex::new(config)?;
            rebuilt.build_internal_with_ids(&live_vecs, live_ids, watermark, true)?;
            if let Some(sp) = sparse_opt {
                rebuilt.upload_sparse_data(sp)?;
            }

            let mut guard = index.write();
            // Reconcile any live inserts/deletes that landed while rebuilding
            let tail_start = guard
                .delta
                .ids
                .iter()
                .position(|&lid| lid >= watermark)
                .unwrap_or(guard.delta.ids.len());
            let tail_vecs = guard.delta.vectors[tail_start * config.dim as usize..].to_vec();
            let tail_ids = guard.delta.ids[tail_start..].to_vec();
            let post_tombstones = guard.tombstones.clone();
            let final_next_lid = guard.next_logical_id.max(watermark);

            *guard = rebuilt;
            guard.next_logical_id = final_next_lid;

            // Apply deletes that happened concurrently during background build
            let mut late_deletes = Vec::new();
            for &lid in &guard.logical_ids {
                if post_tombstones.contains(lid) {
                    late_deletes.push(lid);
                }
            }
            for &lid in &tail_ids {
                if post_tombstones.contains(lid) {
                    late_deletes.push(lid);
                }
            }

            if !tail_vecs.is_empty() {
                let dim = config.dim as usize;
                let ds = guard.streams.delta_stream();
                {
                    let delta = &mut guard.delta;
                    delta.vectors = tail_vecs;
                    delta.ids = tail_ids;
                    let cap = delta.len().next_power_of_two().max(256);
                    delta.d_vectors = DeviceBuffer::new(cap * dim)?;
                    delta.d_ids = DeviceBuffer::new(cap)?;
                    delta.d_valid_words = DeviceBuffer::new(cap.div_ceil(32))?;
                    delta.capacity = cap;
                    delta
                        .d_vectors
                        .copy_from_host_async(&delta.vectors, ds.raw())?;
                    delta.d_ids.copy_from_host_async(&delta.ids, ds.raw())?;
                }
                ds.sync()?;
                guard.upload_delta_valid_mask()?;
            }
            if !late_deletes.is_empty() {
                guard.delete_ids(&late_deletes)?;
            }
            Ok(())
        })
    }

    // ------------------------------------------------------------------------
    // Zero-Copy Index Persistence (mmap + 64-byte aligned snapshots)
    // ------------------------------------------------------------------------

    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> Result<(), CudaError> {
        if !self.is_built {
            return Err(CudaError::InvalidArgument);
        }
        use crate::persistence::as_byte_slice;

        let manifest = SnapshotManifest {
            version: 1,
            config: self.config,
            num_vectors: self.num_vectors,
            entry_point: self.entry_point,
            cagra_degree: self.cagra_degree,
            cagra_seeds: self.cagra_seeds.clone(),
            rabitq_padded_dim: self.rabitq_padded_dim,
            fp16_padded_dim: self.fp16_padded_dim,
            sparse_num_docs: self.sparse_data.num_docs,
            sparse_nnz: self.sparse_data.nnz(),
            next_logical_id: self.next_logical_id,
            sections: BTreeMap::new(),
        };

        let mut sections: Vec<(&str, &str, usize, &[u8])> = vec![
            (
                "vectors",
                "f32",
                self.raw_vectors.len(),
                as_byte_slice(self.raw_vectors.as_slice()),
            ),
            (
                "logical_ids",
                "u32",
                self.logical_ids.len(),
                as_byte_slice(&self.logical_ids),
            ),
            (
                "main_valid_words",
                "u32",
                self.main_valid_words.len(),
                as_byte_slice(&self.main_valid_words),
            ),
            (
                "hnsw_adjacency",
                "u32",
                self.h_adjacency.len(),
                as_byte_slice(self.h_adjacency.as_slice()),
            ),
            (
                "hnsw_degrees",
                "u32",
                self.h_degrees.len(),
                as_byte_slice(self.h_degrees.as_slice()),
            ),
            (
                "cagra_adjacency",
                "u32",
                self.h_cagra_adj.len(),
                as_byte_slice(self.h_cagra_adj.as_slice()),
            ),
            (
                "ivf_centroids",
                "f32",
                self.h_centroids.len(),
                as_byte_slice(self.h_centroids.as_slice()),
            ),
            (
                "pq_codebooks",
                "f32",
                self.h_codebooks.len(),
                as_byte_slice(self.h_codebooks.as_slice()),
            ),
            (
                "pq_codes",
                "u8",
                self.h_codes.len(),
                self.h_codes.as_slice(),
            ),
            (
                "ivf_offsets",
                "u32",
                self.h_ivf_offsets.len(),
                as_byte_slice(self.h_ivf_offsets.as_slice()),
            ),
            (
                "ivf_vec_ids",
                "u32",
                self.h_ivf_vec_ids.len(),
                as_byte_slice(self.h_ivf_vec_ids.as_slice()),
            ),
            (
                "sq_mins",
                "f32",
                self.h_sq_mins.len(),
                as_byte_slice(&self.h_sq_mins),
            ),
            (
                "sq_scales",
                "f32",
                self.h_sq_scales.len(),
                as_byte_slice(&self.h_sq_scales),
            ),
            ("sq_codes", "u8", self.h_sq_codes.len(), &self.h_sq_codes),
            (
                "rabitq_signs",
                "f32",
                self.h_rabitq_rot_signs.len(),
                as_byte_slice(&self.h_rabitq_rot_signs),
            ),
            (
                "rabitq_centroids",
                "f32",
                self.h_rabitq_rot_centroids.len(),
                as_byte_slice(&self.h_rabitq_rot_centroids),
            ),
            (
                "rabitq_bits",
                "u32",
                self.h_rabitq_bits.len(),
                as_byte_slice(&self.h_rabitq_bits),
            ),
            (
                "rabitq_meta",
                "f32",
                self.h_rabitq_meta.len(),
                as_byte_slice(&self.h_rabitq_meta),
            ),
            (
                "vectors_fp16",
                "u16",
                self.h_vectors_fp16.len(),
                as_byte_slice(&self.h_vectors_fp16),
            ),
            (
                "vec_norms_sq",
                "f32",
                self.h_vec_norms_sq.len(),
                as_byte_slice(&self.h_vec_norms_sq),
            ),
        ];

        let tombstones_vec = self.tombstones.to_vec();
        if !tombstones_vec.is_empty() {
            sections.push((
                "tombstones",
                "u32",
                tombstones_vec.len(),
                as_byte_slice(&tombstones_vec),
            ));
        }

        if !self.delta.is_empty() {
            sections.push((
                "delta_vectors",
                "f32",
                self.delta.vectors.len(),
                as_byte_slice(&self.delta.vectors),
            ));
            sections.push((
                "delta_ids",
                "u32",
                self.delta.ids.len(),
                as_byte_slice(&self.delta.ids),
            ));
        }

        if self.sparse_data.num_docs > 0 {
            sections.push((
                "sparse_indptr",
                "u32",
                self.sparse_data.indptr.len(),
                as_byte_slice(&self.sparse_data.indptr),
            ));
            sections.push((
                "sparse_indices",
                "u32",
                self.sparse_data.indices.len(),
                as_byte_slice(&self.sparse_data.indices),
            ));
            sections.push((
                "sparse_values",
                "f32",
                self.sparse_data.values.len(),
                as_byte_slice(&self.sparse_data.values),
            ));
        }

        write_aligned_snapshot(path.as_ref(), manifest, sections)
    }

    pub fn load_snapshot(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let snap = MmapSnapshot::open(path.as_ref())?;
        let m = &snap.manifest;
        let mut idx = Self::new(m.config)?;
        let stream = idx.streams.get_stream();

        let vecs = snap
            .section_typed::<f32>("vectors")?
            .ok_or_else(|| CudaError::IoError("missing vectors section".into()))?;
        let logical_ids = snap
            .section_typed::<u32>("logical_ids")?
            .map(|s| s.to_vec())
            .unwrap_or_else(|| (0..m.num_vectors as u32).collect());
        let valid_words = snap
            .section_typed::<u32>("main_valid_words")?
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![u32::MAX; m.num_vectors.div_ceil(32)]);

        let hnsw_adj = snap
            .section_typed::<u32>("hnsw_adjacency")?
            .ok_or_else(|| CudaError::IoError("missing hnsw_adjacency section".into()))?;
        let hnsw_deg = snap
            .section_typed::<u32>("hnsw_degrees")?
            .ok_or_else(|| CudaError::IoError("missing hnsw_degrees section".into()))?;
        let cagra_adj = snap
            .section_typed::<u32>("cagra_adjacency")?
            .ok_or_else(|| CudaError::IoError("missing cagra_adjacency section".into()))?;
        let centroids = snap
            .section_typed::<f32>("ivf_centroids")?
            .ok_or_else(|| CudaError::IoError("missing ivf_centroids section".into()))?;
        let cb = snap
            .section_typed::<f32>("pq_codebooks")?
            .ok_or_else(|| CudaError::IoError("missing pq_codebooks section".into()))?;
        let codes = snap
            .section_typed::<u8>("pq_codes")?
            .ok_or_else(|| CudaError::IoError("missing pq_codes section".into()))?;
        let off = snap
            .section_typed::<u32>("ivf_offsets")?
            .ok_or_else(|| CudaError::IoError("missing ivf_offsets section".into()))?;
        let vids = snap
            .section_typed::<u32>("ivf_vec_ids")?
            .ok_or_else(|| CudaError::IoError("missing ivf_vec_ids section".into()))?;
        let sq_mins = snap
            .section_typed::<f32>("sq_mins")?
            .ok_or_else(|| CudaError::IoError("missing sq_mins section".into()))?;
        let sq_scales = snap
            .section_typed::<f32>("sq_scales")?
            .ok_or_else(|| CudaError::IoError("missing sq_scales section".into()))?;
        let sq_codes = snap
            .section_typed::<u8>("sq_codes")?
            .ok_or_else(|| CudaError::IoError("missing sq_codes section".into()))?;
        let rq_signs = snap
            .section_typed::<f32>("rabitq_signs")?
            .ok_or_else(|| CudaError::IoError("missing rabitq_signs section".into()))?;
        let rq_cent = snap
            .section_typed::<f32>("rabitq_centroids")?
            .ok_or_else(|| CudaError::IoError("missing rabitq_centroids section".into()))?;
        let rq_bits = snap
            .section_typed::<u32>("rabitq_bits")?
            .ok_or_else(|| CudaError::IoError("missing rabitq_bits section".into()))?;
        let rq_meta = snap
            .section_typed::<f32>("rabitq_meta")?
            .ok_or_else(|| CudaError::IoError("missing rabitq_meta section".into()))?;
        let fp16_vecs = snap
            .section_typed::<u16>("vectors_fp16")?
            .ok_or_else(|| CudaError::IoError("missing vectors_fp16 section".into()))?;
        let norms_sq = snap
            .section_typed::<f32>("vec_norms_sq")?
            .ok_or_else(|| CudaError::IoError("missing vec_norms_sq section".into()))?;

        // DMA-transfer directly from the 64-byte-aligned mmap slices to VRAM
        idx.d_vectors = DeviceBuffer::new(vecs.len())?;
        idx.d_vectors.copy_from_host_async(vecs, stream.raw())?;
        idx.raw_vectors = PinnedBuffer::from_slice(vecs)?;

        idx.d_main_valid_words = DeviceBuffer::new(valid_words.len())?;
        idx.d_main_valid_words
            .copy_from_host_async(&valid_words, stream.raw())?;

        idx.d_adjacency = DeviceBuffer::new(hnsw_adj.len())?;
        idx.d_adjacency
            .copy_from_host_async(hnsw_adj, stream.raw())?;
        idx.h_adjacency = PinnedBuffer::from_slice(hnsw_adj)?;

        idx.d_degrees = DeviceBuffer::new(hnsw_deg.len())?;
        idx.d_degrees.copy_from_host_async(hnsw_deg, stream.raw())?;
        idx.h_degrees = PinnedBuffer::from_slice(hnsw_deg)?;

        idx.d_cagra_adj = DeviceBuffer::new(cagra_adj.len())?;
        idx.d_cagra_adj
            .copy_from_host_async(cagra_adj, stream.raw())?;
        idx.h_cagra_adj = PinnedBuffer::from_slice(cagra_adj)?;
        idx.d_cagra_seeds = DeviceBuffer::new(m.cagra_seeds.len())?;
        idx.d_cagra_seeds
            .copy_from_host_async(&m.cagra_seeds, stream.raw())?;

        idx.d_centroids = DeviceBuffer::new(centroids.len())?;
        idx.d_centroids
            .copy_from_host_async(centroids, stream.raw())?;
        idx.h_centroids = PinnedBuffer::from_slice(centroids)?;

        idx.d_codebooks = DeviceBuffer::new(cb.len())?;
        idx.d_codebooks.copy_from_host_async(cb, stream.raw())?;
        idx.h_codebooks = PinnedBuffer::from_slice(cb)?;

        idx.d_codes = DeviceBuffer::new(codes.len())?;
        idx.d_codes.copy_from_host_async(codes, stream.raw())?;
        idx.h_codes = PinnedBuffer::from_slice(codes)?;

        idx.d_ivf_offsets = DeviceBuffer::new(off.len())?;
        idx.d_ivf_offsets.copy_from_host_async(off, stream.raw())?;
        idx.h_ivf_offsets = PinnedBuffer::from_slice(off)?;

        idx.d_ivf_vec_ids = DeviceBuffer::new(vids.len())?;
        idx.d_ivf_vec_ids.copy_from_host_async(vids, stream.raw())?;
        idx.h_ivf_vec_ids = PinnedBuffer::from_slice(vids)?;

        idx.d_sq_mins = DeviceBuffer::new(sq_mins.len())?;
        idx.d_sq_mins.copy_from_host_async(sq_mins, stream.raw())?;
        idx.d_sq_scales = DeviceBuffer::new(sq_scales.len())?;
        idx.d_sq_scales
            .copy_from_host_async(sq_scales, stream.raw())?;
        idx.d_sq_codes = DeviceBuffer::new(sq_codes.len())?;
        idx.d_sq_codes
            .copy_from_host_async(sq_codes, stream.raw())?;

        idx.d_rabitq_rot_signs = DeviceBuffer::new(rq_signs.len())?;
        idx.d_rabitq_rot_signs
            .copy_from_host_async(rq_signs, stream.raw())?;
        idx.d_rabitq_rot_centroids = DeviceBuffer::new(rq_cent.len())?;
        idx.d_rabitq_rot_centroids
            .copy_from_host_async(rq_cent, stream.raw())?;
        idx.d_rabitq_bits = DeviceBuffer::new(rq_bits.len())?;
        idx.d_rabitq_bits
            .copy_from_host_async(rq_bits, stream.raw())?;
        idx.d_rabitq_meta = DeviceBuffer::new(rq_meta.len())?;
        idx.d_rabitq_meta
            .copy_from_host_async(rq_meta, stream.raw())?;

        idx.d_vectors_fp16 = DeviceBuffer::new(fp16_vecs.len())?;
        idx.d_vectors_fp16
            .copy_from_host_async(fp16_vecs, stream.raw())?;
        idx.d_vec_norms_sq = DeviceBuffer::new(norms_sq.len())?;
        idx.d_vec_norms_sq
            .copy_from_host_async(norms_sq, stream.raw())?;

        stream.sync()?;

        idx.num_vectors = m.num_vectors;
        idx.entry_point = m.entry_point;
        idx.cagra_degree = m.cagra_degree;
        idx.cagra_seeds = m.cagra_seeds.clone();
        idx.rabitq_padded_dim = m.rabitq_padded_dim;
        idx.fp16_padded_dim = m.fp16_padded_dim;
        idx.next_logical_id = m.next_logical_id;
        idx.id_remapped = logical_ids
            .iter()
            .enumerate()
            .any(|(i, &lid)| lid as usize != i);
        idx.logical_ids = logical_ids;
        idx.main_valid_words = valid_words;

        if let Some(ts) = snap.section_typed::<u32>("tombstones")? {
            for &id in ts {
                idx.tombstones.insert(id);
            }
        } else {
            for (slot, &lid) in idx.logical_ids.iter().enumerate() {
                if (idx.main_valid_words[slot >> 5] & (1u32 << (slot & 31))) == 0 {
                    idx.tombstones.insert(lid);
                }
            }
        }

        if let (Some(d_vecs), Some(d_ids)) = (
            snap.section_typed::<f32>("delta_vectors")?,
            snap.section_typed::<u32>("delta_ids")?,
        ) {
            if !d_ids.is_empty() {
                let dim = m.config.dim as usize;
                let ds = idx.streams.delta_stream();
                {
                    let delta = &mut idx.delta;
                    delta.vectors = d_vecs.to_vec();
                    delta.ids = d_ids.to_vec();
                    let cap = delta.len().next_power_of_two().max(256);
                    delta.d_vectors = DeviceBuffer::new(cap * dim)?;
                    delta.d_ids = DeviceBuffer::new(cap)?;
                    delta.d_valid_words = DeviceBuffer::new(cap.div_ceil(32))?;
                    delta.capacity = cap;
                    delta
                        .d_vectors
                        .copy_from_host_async(&delta.vectors, ds.raw())?;
                    delta.d_ids.copy_from_host_async(&delta.ids, ds.raw())?;
                }
                ds.sync()?;
                idx.upload_delta_valid_mask()?;
            }
        }

        idx.h_sq_mins = sq_mins.to_vec();
        idx.h_sq_scales = sq_scales.to_vec();
        idx.h_sq_codes = sq_codes.to_vec();
        idx.h_rabitq_rot_signs = rq_signs.to_vec();
        idx.h_rabitq_rot_centroids = rq_cent.to_vec();
        idx.h_rabitq_bits = rq_bits.to_vec();
        idx.h_rabitq_meta = rq_meta.to_vec();
        idx.h_vectors_fp16 = fp16_vecs.to_vec();
        idx.h_vec_norms_sq = norms_sq.to_vec();

        if let (Some(sp_ptr), Some(sp_idx), Some(sp_val)) = (
            snap.section_typed::<u32>("sparse_indptr")?,
            snap.section_typed::<u32>("sparse_indices")?,
            snap.section_typed::<f32>("sparse_values")?,
        ) {
            idx.upload_sparse_data(SparseCsrData {
                indptr: sp_ptr.to_vec(),
                indices: sp_idx.to_vec(),
                values: sp_val.to_vec(),
                num_docs: m.sparse_num_docs,
            })?;
        }

        idx.is_built = true;
        Ok(idx)
    }

    // ------------------------------------------------------------------------
    // Filter Mask Helper (Combines User Filter + GPU Tombstone Bitset)
    // ------------------------------------------------------------------------

    fn build_slot_mask_words(&self, user_logical_mask: Option<&[u32]>) -> Option<Vec<u32>> {
        if user_logical_mask.is_none() && self.tombstones.is_empty() {
            return None;
        }
        let num_words = self.num_vectors.div_ceil(32);
        let mut words = vec![0u32; num_words];
        if !self.id_remapped {
            if let Some(umask) = user_logical_mask {
                for w in 0..num_words {
                    let uw = umask.get(w).copied().unwrap_or(0);
                    words[w] = self.main_valid_words[w] & uw;
                }
            } else {
                words.copy_from_slice(&self.main_valid_words);
            }
        } else if let Some(umask) = user_logical_mask {
            for (slot, &lid) in self.logical_ids.iter().enumerate() {
                let alive = !self.tombstones.contains(lid);
                let uw = umask.get((lid as usize) >> 5).copied().unwrap_or(0);
                let uallow = ((uw >> (lid & 31)) & 1) != 0;
                if alive && uallow {
                    words[slot >> 5] |= 1u32 << (slot & 31);
                }
            }
        } else {
            words.copy_from_slice(&self.main_valid_words);
        }
        Some(words)
    }

    fn build_delta_mask_words(&self, user_logical_mask: Option<&[u32]>) -> Option<Vec<u32>> {
        if self.delta.is_empty() || (user_logical_mask.is_none() && self.tombstones.is_empty()) {
            return None;
        }
        let num_words = self.delta.len().div_ceil(32);
        let mut words = vec![0u32; num_words];
        for (i, &lid) in self.delta.ids.iter().enumerate() {
            if self.tombstones.contains(lid) {
                continue;
            }
            let allow = match user_logical_mask {
                Some(umask) => {
                    let w = umask.get((lid as usize) >> 5).copied().unwrap_or(0);
                    ((w >> (lid & 31)) & 1) != 0
                }
                None => true,
            };
            if allow {
                words[i >> 5] |= 1u32 << (i & 31);
            }
        }
        Some(words)
    }

    fn remap_debug_trace(&self, hops: &mut [TraversalHop], steps: &mut [BeamStepRecord]) {
        if !self.id_remapped {
            return;
        }
        for hop in hops.iter_mut() {
            if (hop.from_node as usize) < self.logical_ids.len() {
                hop.from_node = self.logical_ids[hop.from_node as usize];
            }
            if (hop.to_node as usize) < self.logical_ids.len() {
                hop.to_node = self.logical_ids[hop.to_node as usize];
            }
        }
        for step in steps.iter_mut() {
            if (step.expanded_node as usize) < self.logical_ids.len() {
                step.expanded_node = self.logical_ids[step.expanded_node as usize];
            }
            for c in &mut step.candidates[..step.cand_count as usize] {
                if (c.id as usize) < self.logical_ids.len() {
                    c.id = self.logical_ids[c.id as usize];
                }
            }
        }
    }

    fn remap_slot_results(&self, results: &mut [GpuSearchResult]) {
        if !self.id_remapped {
            return;
        }
        for r in results.iter_mut() {
            if r.id != u32::MAX && (r.id as usize) < self.logical_ids.len() {
                r.id = self.logical_ids[r.id as usize];
            }
        }
    }

    fn finish_delta_merge(
        &self,
        delta_job: Option<DeltaScanJob<'_>>,
        num_queries: usize,
        top_k: usize,
        main_results: &mut [GpuSearchResult],
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.remap_slot_results(main_results);
        let Some(mut job) = delta_job else {
            return Ok(main_results.to_vec());
        };
        let delta_host = job.sync_and_take()?;
        let mut merged = Vec::with_capacity(num_queries * top_k);
        for q in 0..num_queries {
            let main_q = &main_results[q * top_k..(q + 1) * top_k];
            let delta_q = &delta_host[q * top_k..(q + 1) * top_k];
            let mut combined: Vec<GpuSearchResult> = main_q
                .iter()
                .chain(delta_q.iter())
                .copied()
                .filter(|r| r.id != u32::MAX)
                .collect();
            combined.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            combined.truncate(top_k);
            while combined.len() < top_k {
                combined.push(GpuSearchResult::default());
            }
            merged.extend_from_slice(&combined);
        }
        Ok(merged)
    }

    /// Launch the live delta-buffer flat scan asynchronously on `streams.delta_stream()`
    /// so it executes concurrently with main HNSW/CAGRA/IVF traversal on the primary stream.
    fn launch_delta_scan_host(
        &self,
        prepared_queries: &[f32],
        num_queries: usize,
        top_k: usize,
        user_logical_mask: Option<&[u32]>,
    ) -> Result<Option<DeltaScanJob<'_>>, CudaError> {
        if self.delta.is_empty() {
            return Ok(None);
        }
        let ds = self.streams.delta_stream_ref();
        let mut d_q = ds.scratch_queries.lock();
        d_q.copy_from_host_async(prepared_queries, ds.raw())?;
        let ptr = d_q.as_ptr();
        self.launch_delta_scan_inner(ds, Some(d_q), ptr, num_queries, top_k, user_logical_mask)
            .map(Some)
    }

    fn launch_delta_scan_device(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        user_logical_mask: Option<&[u32]>,
    ) -> Result<Option<DeltaScanJob<'_>>, CudaError> {
        if self.delta.is_empty() {
            return Ok(None);
        }
        let ds = self.streams.delta_stream_ref();
        self.launch_delta_scan_inner(ds, None, d_queries, num_queries, top_k, user_logical_mask)
            .map(Some)
    }

    fn launch_delta_scan_inner<'a>(
        &'a self,
        ds: &'a crate::streams::CudaStream,
        q_guard: Option<parking_lot::MutexGuard<'a, DeviceBuffer<f32>>>,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        user_logical_mask: Option<&[u32]>,
    ) -> Result<DeltaScanJob<'a>, CudaError> {
        let mut d_delta_res = ds.scratch_delta_results.lock();
        let mut mask_guard = ds.scratch_mask.lock();
        let delta_mask_words = self.build_delta_mask_words(user_logical_mask);
        let d_delta_mask_ptr = if let Some(ref words) = delta_mask_words {
            if words.len() > MAX_MASK_WORDS {
                return Err(CudaError::InvalidArgument);
            }
            mask_guard.copy_from_host_async(words, ds.raw())?;
            mask_guard.as_ptr()
        } else {
            std::ptr::null()
        };

        gpu_flat_scan_batch(
            self.delta.d_vectors.as_ptr(),
            self.delta.d_ids.as_ptr(),
            self.delta.len() as u32,
            self.config.dim,
            d_queries,
            num_queries as u32,
            top_k as u32,
            self.metric(),
            d_delta_mask_ptr,
            d_delta_res.as_mut_ptr(),
            ds.raw(),
        )?;

        let mut delta_host = vec![GpuSearchResult::default(); num_queries * top_k];
        d_delta_res.copy_to_host_async(&mut delta_host, ds.raw())?;

        Ok(DeltaScanJob {
            ds,
            _q_guard: q_guard,
            _res_guard: d_delta_res,
            _mask_guard: mask_guard,
            delta_host,
            synced: false,
        })
    }

    fn prepare_queries(&self, queries: &[f32], num_queries: usize) -> Vec<f32> {
        let dim = self.config.dim as usize;
        let total = num_queries * dim;
        if self.config.metric == MetricType::Cosine as u32 {
            let mut normalized = queries[..total].to_vec();
            normalize_slice_in_place(&mut normalized, dim);
            normalized
        } else {
            queries[..total].to_vec()
        }
    }

    // ------------------------------------------------------------------------
    // HNSW Search + Debug Beam Scrubbing
    // ------------------------------------------------------------------------

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        let (results, _) = self.search_with_trajectory(query, top_k, ef_search)?;
        Ok(results)
    }

    pub fn search_with_trajectory(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<(Vec<GpuSearchResult>, Vec<TraversalHop>), CudaError> {
        let dbg = self.search_with_debug(query, top_k, ef_search, None)?;
        Ok((dbg.results, dbg.hops))
    }

    pub fn search_with_debug(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<BeamDebugTrace, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        let dim = self.config.dim as usize;
        assert!(query.len() >= dim, "Query shorter than index dimension");
        self.validate_search_shape(1, top_k, dim)?;
        if top_k == 0 {
            return Ok(BeamDebugTrace {
                results: Vec::new(),
                hops: Vec::new(),
                steps: Vec::new(),
            });
        }

        let start = Instant::now();
        let max_hops = (ef_search.saturating_mul(8)).clamp(64, MAX_HOPS);
        let max_steps = (ef_search.saturating_mul(2)).clamp(16, MAX_BEAM_STEPS);

        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_hops = stream.scratch_hops.lock();
        let mut d_hop_count = stream.scratch_hop_counts.lock();
        let mut d_steps = stream.scratch_steps.lock();
        let mut d_step_count = stream.scratch_step_counts.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        let prepared = self.prepare_queries(query, 1);
        d_q.copy_from_host_async(&prepared, stream.raw())?;

        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            if w.len() > MAX_MASK_WORDS {
                return Err(CudaError::InvalidArgument);
            }
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let graph = HnswGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_adjacency.as_ptr(),
            d_degree: self.d_degrees.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            m_max: self.config.m,
            entry_point: self.entry_point,
        };

        gpu_hnsw_search_batch_debug(
            &graph,
            d_q.as_ptr(),
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            d_hops.as_mut_ptr(),
            d_hop_count.as_mut_ptr(),
            max_hops as u32,
            d_steps.as_mut_ptr(),
            d_step_count.as_mut_ptr(),
            max_steps as u32,
            stream.raw(),
        )?;

        let mut h_results = vec![GpuSearchResult::default(); top_k];
        let mut h_hop_count = vec![0u32; 1];
        let mut h_step_count = vec![0u32; 1];
        let mut h_hops = vec![
            TraversalHop {
                step: 0,
                from_node: 0,
                to_node: 0,
                distance: 0.0,
            };
            max_hops
        ];
        let mut h_steps = vec![BeamStepRecord::default(); max_steps];

        let delta_job = self.launch_delta_scan_host(&prepared, 1, top_k, filter_mask)?;

        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        d_hop_count.copy_to_host_async(&mut h_hop_count, stream.raw())?;
        d_step_count.copy_to_host_async(&mut h_step_count, stream.raw())?;
        d_hops.copy_to_host_async(&mut h_hops, stream.raw())?;
        d_steps.copy_to_host_async(&mut h_steps, stream.raw())?;
        stream.sync()?;

        h_hops.truncate((h_hop_count[0] as usize).min(max_hops));
        h_steps.truncate((h_step_count[0] as usize).min(max_steps));

        let final_results = self.finish_delta_merge(delta_job, 1, top_k, &mut h_results)?;
        self.remap_debug_trace(&mut h_hops, &mut h_steps);

        self.metrics.record_query(start.elapsed(), 1);
        Ok(BeamDebugTrace {
            results: final_results,
            hops: h_hops,
            steps: h_steps,
        })
    }

    pub fn batch_search(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.batch_search_filtered(queries, num_queries, top_k, ef_search, None)
    }

    pub fn batch_search_filtered(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();

        let prepared = self.prepare_queries(queries, num_queries);
        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        d_q.copy_from_host_async(&prepared, stream.raw())?;
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            if w.len() > MAX_MASK_WORDS {
                return Err(CudaError::InvalidArgument);
            }
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let graph = HnswGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_adjacency.as_ptr(),
            d_degree: self.d_degrees.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            m_max: self.config.m,
            entry_point: self.entry_point,
        };

        gpu_hnsw_search_batch(
            &graph,
            d_q.as_ptr(),
            num_queries as u32,
            self.config.dim,
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            stream.raw(),
        )?;

        let delta_job = self.launch_delta_scan_host(&prepared, num_queries, top_k, filter_mask)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    pub fn batch_search_device_ptr(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.batch_search_device_ptr_filtered(d_queries, num_queries, top_k, ef_search, None)
    }

    pub fn batch_search_device_ptr_filtered(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if d_queries.is_null() {
            return Err(CudaError::InvalidArgument);
        }
        gpu_device_synchronize()?;
        self.validate_search_shape(num_queries, top_k, num_queries * self.config.dim as usize)?;

        let start = Instant::now();
        let stream = self.streams.get_stream();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let graph = HnswGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_adjacency.as_ptr(),
            d_degree: self.d_degrees.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            m_max: self.config.m,
            entry_point: self.entry_point,
        };

        gpu_hnsw_search_batch(
            &graph,
            d_queries,
            num_queries as u32,
            self.config.dim,
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            stream.raw(),
        )?;

        let delta_job =
            self.launch_delta_scan_device(d_queries, num_queries, top_k, filter_mask)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    // ------------------------------------------------------------------------
    // CAGRA Flat Fixed-Out-Degree Graph Search
    // ------------------------------------------------------------------------

    pub fn batch_search_cagra(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();

        let prepared = self.prepare_queries(queries, num_queries);
        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        d_q.copy_from_host_async(&prepared, stream.raw())?;
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let graph = CagraGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_cagra_adj.as_ptr(),
            d_seed_nodes: self.d_cagra_seeds.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            graph_degree: self.cagra_degree as u32,
            entry_point: self.entry_point,
            num_seeds: self.cagra_seeds.len() as u32,
        };

        gpu_cagra_search_batch(
            &graph,
            d_q.as_ptr(),
            num_queries as u32,
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            stream.raw(),
        )?;

        let delta_job = self.launch_delta_scan_host(&prepared, num_queries, top_k, filter_mask)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    pub fn search_cagra_with_debug(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<BeamDebugTrace, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        let dim = self.config.dim as usize;
        self.validate_search_shape(1, top_k, dim)?;
        if top_k == 0 {
            return Ok(BeamDebugTrace {
                results: Vec::new(),
                hops: Vec::new(),
                steps: Vec::new(),
            });
        }

        let start = Instant::now();
        let max_hops = (ef_search.saturating_mul(16)).clamp(64, MAX_HOPS);
        let max_steps = (ef_search.saturating_mul(2)).clamp(16, MAX_BEAM_STEPS);

        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_hops = stream.scratch_hops.lock();
        let mut d_hop_count = stream.scratch_hop_counts.lock();
        let mut d_steps = stream.scratch_steps.lock();
        let mut d_step_count = stream.scratch_step_counts.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        let prepared = self.prepare_queries(query, 1);
        d_q.copy_from_host_async(&prepared, stream.raw())?;
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let graph = CagraGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_cagra_adj.as_ptr(),
            d_seed_nodes: self.d_cagra_seeds.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            graph_degree: self.cagra_degree as u32,
            entry_point: self.entry_point,
            num_seeds: self.cagra_seeds.len() as u32,
        };

        gpu_cagra_search_batch(
            &graph,
            d_q.as_ptr(),
            1,
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            d_hops.as_mut_ptr(),
            d_hop_count.as_mut_ptr(),
            max_hops as u32,
            d_steps.as_mut_ptr(),
            d_step_count.as_mut_ptr(),
            max_steps as u32,
            stream.raw(),
        )?;

        let mut h_results = vec![GpuSearchResult::default(); top_k];
        let mut h_hop_count = vec![0u32; 1];
        let mut h_step_count = vec![0u32; 1];
        let mut h_hops = vec![
            TraversalHop {
                step: 0,
                from_node: 0,
                to_node: 0,
                distance: 0.0,
            };
            max_hops
        ];
        let mut h_steps = vec![BeamStepRecord::default(); max_steps];

        let delta_job = self.launch_delta_scan_host(&prepared, 1, top_k, filter_mask)?;

        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        d_hop_count.copy_to_host_async(&mut h_hop_count, stream.raw())?;
        d_step_count.copy_to_host_async(&mut h_step_count, stream.raw())?;
        d_hops.copy_to_host_async(&mut h_hops, stream.raw())?;
        d_steps.copy_to_host_async(&mut h_steps, stream.raw())?;
        stream.sync()?;

        h_hops.truncate((h_hop_count[0] as usize).min(max_hops));
        h_steps.truncate((h_step_count[0] as usize).min(max_steps));

        let final_results = self.finish_delta_merge(delta_job, 1, top_k, &mut h_results)?;
        self.remap_debug_trace(&mut h_hops, &mut h_steps);
        self.metrics.record_query(start.elapsed(), 1);
        Ok(BeamDebugTrace {
            results: final_results,
            hops: h_hops,
            steps: h_steps,
        })
    }

    // ------------------------------------------------------------------------
    // IVF-PQ, IVF-SQ8, IVF-RaBitQ, and FP16 Tensor Core Search
    // ------------------------------------------------------------------------

    pub fn batch_search_ivf(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.batch_search_ivf_mode(
            queries,
            num_queries,
            top_k,
            nprobe,
            rerank_k,
            IvfQuantMode::Pq,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_search_ivf_mode(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
        mode: IvfQuantMode,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();
        let dim = self.config.dim as usize;
        let sub_dim = dim / (self.config.m_pq as usize);
        let nlist = (self.h_centroids.len() / dim) as u32;

        let prepared = self.prepare_queries(queries, num_queries);
        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        d_q.copy_from_host_async(&prepared, stream.raw())?;
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let rerank_ptr = self.rerank_vectors_device_ptr();
        match mode {
            IvfQuantMode::Pq => {
                let tables = IvfPqGpuTables {
                    d_centroids: self.d_centroids.as_ptr(),
                    d_pq_codebooks: self.d_codebooks.as_ptr(),
                    d_pq_codes: self.d_codes.as_ptr(),
                    d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
                    d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
                    d_vectors: rerank_ptr,
                    num_vectors: self.num_vectors as u32,
                    dim: self.config.dim,
                    nlist,
                    m_pq: self.config.m_pq,
                    sub_dim: sub_dim as u32,
                };
                gpu_ivf_pq_search_batch(
                    &tables,
                    d_q.as_ptr(),
                    num_queries as u32,
                    top_k as u32,
                    nprobe as u32,
                    rerank_k as u32,
                    self.metric(),
                    d_mask_ptr,
                    d_results.as_mut_ptr(),
                    stream.raw(),
                )?;
            }
            IvfQuantMode::Sq8 => {
                let tables = IvfSqGpuTables {
                    d_centroids: self.d_centroids.as_ptr(),
                    d_sq_mins: self.d_sq_mins.as_ptr(),
                    d_sq_scales: self.d_sq_scales.as_ptr(),
                    d_sq_codes: self.d_sq_codes.as_ptr(),
                    d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
                    d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
                    d_vectors: rerank_ptr,
                    num_vectors: self.num_vectors as u32,
                    dim: self.config.dim,
                    nlist,
                };
                gpu_ivf_sq_search_batch(
                    &tables,
                    d_q.as_ptr(),
                    num_queries as u32,
                    top_k as u32,
                    nprobe as u32,
                    rerank_k as u32,
                    self.metric(),
                    d_mask_ptr,
                    d_results.as_mut_ptr(),
                    stream.raw(),
                )?;
            }
            IvfQuantMode::Rabitq => {
                let tables = IvfRabitqGpuTables {
                    d_centroids: self.d_centroids.as_ptr(),
                    d_rot_centroids: self.d_rabitq_rot_centroids.as_ptr(),
                    d_rabitq_bits: self.d_rabitq_bits.as_ptr(),
                    d_rabitq_meta: self.d_rabitq_meta.as_ptr(),
                    d_rot_signs: self.d_rabitq_rot_signs.as_ptr(),
                    d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
                    d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
                    d_vectors: rerank_ptr,
                    num_vectors: self.num_vectors as u32,
                    dim: self.config.dim,
                    padded_dim: self.rabitq_padded_dim as u32,
                    words_per_vec: (self.rabitq_padded_dim / 32) as u32,
                    nlist,
                };
                gpu_ivf_rabitq_search_batch(
                    &tables,
                    d_q.as_ptr(),
                    num_queries as u32,
                    top_k as u32,
                    nprobe as u32,
                    rerank_k as u32,
                    self.metric(),
                    d_mask_ptr,
                    d_results.as_mut_ptr(),
                    stream.raw(),
                )?;
            }
        }

        let delta_job = self.launch_delta_scan_host(&prepared, num_queries, top_k, filter_mask)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    pub fn batch_search_ivf_device_ptr(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if d_queries.is_null() {
            return Err(CudaError::InvalidArgument);
        }
        gpu_device_synchronize()?;
        self.validate_search_shape(num_queries, top_k, num_queries * self.config.dim as usize)?;

        let start = Instant::now();
        let dim = self.config.dim as usize;
        let sub_dim = dim / (self.config.m_pq as usize);

        let stream = self.streams.get_stream();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();
        let mask_words = self.build_slot_mask_words(None);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let tables = IvfPqGpuTables {
            d_centroids: self.d_centroids.as_ptr(),
            d_pq_codebooks: self.d_codebooks.as_ptr(),
            d_pq_codes: self.d_codes.as_ptr(),
            d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
            d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
            d_vectors: self.rerank_vectors_device_ptr(),
            num_vectors: self.num_vectors as u32,
            dim: self.config.dim,
            nlist: (self.h_centroids.len() / dim) as u32,
            m_pq: self.config.m_pq,
            sub_dim: sub_dim as u32,
        };

        gpu_ivf_pq_search_batch(
            &tables,
            d_queries,
            num_queries as u32,
            top_k as u32,
            nprobe as u32,
            rerank_k as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            stream.raw(),
        )?;

        let delta_job = self.launch_delta_scan_device(d_queries, num_queries, top_k, None)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    pub fn batch_search_fp16_tc(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();

        let prepared = self.prepare_queries(queries, num_queries);
        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();

        d_q.copy_from_host_async(&prepared, stream.raw())?;
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let tables = Fp16GpuTables {
            d_vectors_fp16: self.d_vectors_fp16.as_ptr(),
            d_vec_norms_sq: self.d_vec_norms_sq.as_ptr(),
            d_vectors: self.d_vectors.as_ptr(),
            num_vectors: self.num_vectors as u32,
            dim: self.config.dim,
            padded_dim: self.fp16_padded_dim as u32,
        };

        gpu_fp16_tc_search_batch(
            &tables,
            d_q.as_ptr(),
            num_queries as u32,
            top_k as u32,
            self.metric(),
            d_mask_ptr,
            d_results.as_mut_ptr(),
            stream.raw(),
        )?;

        let delta_job = self.launch_delta_scan_host(&prepared, num_queries, top_k, filter_mask)?;
        let mut h_results = vec![GpuSearchResult::default(); num_queries * top_k];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        let merged = self.finish_delta_merge(delta_job, num_queries, top_k, &mut h_results)?;
        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(merged)
    }

    pub fn search_ivf_pq(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.batch_search_ivf(query, 1, top_k, nprobe, 0)
    }

    /// IVF search returning per-probed-centroid telemetry for Studio probe scrubbing.
    pub fn search_ivf_with_debug(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
        mode: IvfQuantMode,
    ) -> Result<(Vec<GpuSearchResult>, Vec<IvfProbeTelemetry>), CudaError> {
        let results = self.batch_search_ivf_mode(query, 1, top_k, nprobe, rerank_k, mode, None)?;

        let dim = self.config.dim as usize;
        let prepared = self.prepare_queries(query, 1);
        let centroids = self.h_centroids.as_slice();
        let nlist = centroids.len() / dim;
        let offsets = self.h_ivf_offsets.as_slice();
        let vec_ids = self.h_ivf_vec_ids.as_slice();

        let mut centroid_dists: Vec<(u32, f32)> = (0..nlist)
            .map(|c| {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let d: f32 = prepared
                    .iter()
                    .zip(cent.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c as u32, d)
            })
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        centroid_dists.truncate(nprobe.max(1).min(nlist));

        let result_map: HashMap<u32, f32> = results
            .iter()
            .filter(|r| r.id != u32::MAX)
            .map(|r| (r.id, r.distance))
            .collect();

        let probes = centroid_dists
            .into_iter()
            .enumerate()
            .map(|(rank, (cluster_id, centroid_distance))| {
                let start = offsets[cluster_id as usize] as usize;
                let end = offsets[cluster_id as usize + 1] as usize;
                let mut hits_in_list = Vec::new();
                let mut best: Option<f32> = None;
                for &slot in &vec_ids[start..end] {
                    let lid = if self.id_remapped {
                        self.logical_ids[slot as usize]
                    } else {
                        slot
                    };
                    if let Some(&dist) = result_map.get(&lid) {
                        hits_in_list.push(lid);
                        best = Some(match best {
                            Some(cur) => cur.min(dist),
                            None => dist,
                        });
                    }
                }
                IvfProbeTelemetry {
                    rank,
                    cluster_id,
                    centroid_distance,
                    list_len: end - start,
                    best_distance: best,
                    result_ids: hits_in_list,
                }
            })
            .collect();

        Ok((results, probes))
    }

    // ------------------------------------------------------------------------
    // CSR Sparse & On-Device Hybrid Reciprocal Rank Fusion (RRF)
    // ------------------------------------------------------------------------

    pub fn build_sparse(&mut self, rows: &[Vec<(u32, f32)>]) -> Result<(), CudaError> {
        let data = SparseCsrData::from_rows(rows);
        self.upload_sparse_data(data)
    }

    fn upload_sparse_data(&mut self, data: SparseCsrData) -> Result<(), CudaError> {
        let stream = self.streams.get_stream();
        let mut d_indptr = DeviceBuffer::new(data.indptr.len())?;
        let mut d_indices = DeviceBuffer::new(data.indices.len())?;
        let mut d_values = DeviceBuffer::new(data.values.len())?;

        d_indptr.copy_from_host_async(&data.indptr, stream.raw())?;
        d_indices.copy_from_host_async(&data.indices, stream.raw())?;
        d_values.copy_from_host_async(&data.values, stream.raw())?;
        stream.sync()?;

        self.sparse_data = data;
        self.d_sparse_indptr = d_indptr;
        self.d_sparse_indices = d_indices;
        self.d_sparse_values = d_values;
        Ok(())
    }

    pub fn search_sparse(
        &self,
        queries: &[&[(u32, f32)]],
        top_k: usize,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<Vec<SparseHit>>, CudaError> {
        if self.sparse_data.num_docs == 0 || queries.is_empty() || top_k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }
        if queries.len() > MAX_BATCH_SIZE || top_k > MAX_TOP_K {
            return Err(CudaError::InvalidArgument);
        }

        let (q_indptr, q_indices, q_values) = pack_sparse_queries(queries);
        let stream = self.streams.get_stream();
        let mut d_q_indptr = DeviceBuffer::new(q_indptr.len())?;
        let mut d_q_indices = DeviceBuffer::new(q_indices.len())?;
        let mut d_q_values = DeviceBuffer::new(q_values.len())?;
        d_q_indptr.copy_from_host_async(&q_indptr, stream.raw())?;
        d_q_indices.copy_from_host_async(&q_indices, stream.raw())?;
        d_q_values.copy_from_host_async(&q_values, stream.raw())?;

        let mut d_results = stream.scratch_results.lock();
        let mut d_mask_buf = stream.scratch_mask.lock();
        let mask_words = self.build_slot_mask_words(filter_mask);
        let d_mask_ptr = if let Some(ref w) = mask_words {
            d_mask_buf.copy_from_host_async(w, stream.raw())?;
            d_mask_buf.as_ptr()
        } else {
            std::ptr::null()
        };

        let matrix = SparseCsrGpuMatrix {
            d_indptr: self.d_sparse_indptr.as_ptr(),
            d_indices: self.d_sparse_indices.as_ptr(),
            d_values: self.d_sparse_values.as_ptr(),
            num_docs: self.sparse_data.num_docs as u32,
            nnz: self.sparse_data.nnz() as u32,
        };

        gpu_sparse_search_batch(
            &matrix,
            d_q_indptr.as_ptr(),
            d_q_indices.as_ptr(),
            d_q_values.as_ptr(),
            queries.len() as u32,
            top_k as u32,
            d_mask_ptr,
            d_results.as_mut_ptr(),
            stream.raw(),
        )?;

        let mut h_res = vec![GpuSearchResult::default(); queries.len() * top_k];
        d_results.copy_to_host_async(&mut h_res, stream.raw())?;
        stream.sync()?;

        Ok(h_res
            .chunks_exact(top_k)
            .map(|row| {
                row.iter()
                    .filter(|r| r.id != u32::MAX)
                    .map(|r| SparseHit {
                        id: r.id,
                        score: -r.distance,
                    })
                    .collect()
            })
            .collect())
    }

    /// Hybrid Dense + CSR Sparse retrieval fused directly on GPU via Reciprocal Rank Fusion.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search(
        &self,
        dense_queries: &[f32],
        sparse_queries: &[&[(u32, f32)]],
        top_k: usize,
        candidate_k: usize,
        ef_search: usize,
        rrf_k: f32,
        dense_weight: f32,
        sparse_weight: f32,
        filter_mask: Option<&[u32]>,
    ) -> Result<Vec<Vec<HybridHit>>, CudaError> {
        let num_queries = sparse_queries.len();
        if num_queries == 0 || top_k == 0 {
            return Ok(vec![Vec::new(); num_queries]);
        }
        let cand_k = candidate_k.max(top_k).min(MAX_TOP_K);

        let dense = self.batch_search_filtered(
            dense_queries,
            num_queries,
            cand_k,
            ef_search.max(cand_k),
            filter_mask,
        )?;
        let sparse_hits = self.search_sparse(sparse_queries, cand_k, filter_mask)?;
        let mut sparse_flat = Vec::with_capacity(num_queries * cand_k);
        for row in &sparse_hits {
            for i in 0..cand_k {
                if let Some(hit) = row.get(i) {
                    sparse_flat.push(GpuSearchResult {
                        id: hit.id,
                        distance: -hit.score,
                    });
                } else {
                    sparse_flat.push(GpuSearchResult::default());
                }
            }
        }

        let stream = self.streams.get_stream();
        let mut d_dense = DeviceBuffer::new(dense.len())?;
        let mut d_sparse = DeviceBuffer::new(sparse_flat.len())?;
        let mut d_out = stream.scratch_results.lock();

        d_dense.copy_from_host_async(&dense, stream.raw())?;
        d_sparse.copy_from_host_async(&sparse_flat, stream.raw())?;

        gpu_rrf_fuse(
            d_dense.as_ptr(),
            cand_k as u32,
            d_sparse.as_ptr(),
            cand_k as u32,
            num_queries as u32,
            top_k as u32,
            rrf_k,
            dense_weight,
            sparse_weight,
            d_out.as_mut_ptr(),
            stream.raw(),
        )?;

        let mut h_fused = vec![GpuSearchResult::default(); num_queries * top_k];
        d_out.copy_to_host_async(&mut h_fused, stream.raw())?;
        stream.sync()?;

        Ok(h_fused
            .chunks_exact(top_k)
            .map(|row| {
                row.iter()
                    .filter(|r| r.id != u32::MAX)
                    .map(|r| HybridHit {
                        id: r.id,
                        rrf_score: -r.distance,
                    })
                    .collect()
            })
            .collect())
    }

    // ------------------------------------------------------------------------
    // Accessors & Distortion Telemetry
    // ------------------------------------------------------------------------

    pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> {
        let dim = self.config.dim as usize;
        if !self.id_remapped && id < self.num_vectors {
            if self.tombstones.contains(id as u32) {
                return None;
            }
            let start = id * dim;
            return Some(self.raw_vectors.as_slice()[start..start + dim].to_vec());
        }
        if let Some(pos) = self.logical_ids.iter().position(|&lid| lid as usize == id) {
            if self.tombstones.contains(id as u32) {
                return None;
            }
            let start = pos * dim;
            return Some(self.raw_vectors.as_slice()[start..start + dim].to_vec());
        }
        if let Some(pos) = self.delta.ids.iter().position(|&lid| lid as usize == id) {
            if self.tombstones.contains(id as u32) {
                return None;
            }
            let start = pos * dim;
            return Some(self.delta.vectors[start..start + dim].to_vec());
        }
        None
    }

    /// Total active (non-deleted) vectors across main index and delta buffer.
    pub fn num_vectors(&self) -> usize {
        let total_slots = self.num_vectors + self.delta.len();
        total_slots.saturating_sub(self.tombstones.len())
    }

    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    pub fn next_logical_id(&self) -> u32 {
        self.next_logical_id
    }

    pub fn dim(&self) -> usize {
        self.config.dim as usize
    }

    pub fn centroids(&self) -> &[f32] {
        self.h_centroids.as_slice()
    }

    pub fn metrics(&self) -> Arc<MetricsTracker> {
        Arc::clone(&self.metrics)
    }

    pub fn config(&self) -> IndexConfig {
        self.config
    }

    /// Compute per-cluster PQ reconstruction error from the trained codebooks.
    pub fn pq_distortion(&self, max_samples_per_cluster: usize) -> Vec<ClusterPqDistortion> {
        if !self.is_built || self.num_vectors == 0 {
            return Vec::new();
        }

        let dim = self.config.dim as usize;
        let m_pq = self.config.m_pq as usize;
        let sub_dim = dim / m_pq;
        let nlist = self.h_centroids.len() / dim;
        let sample_limit = max_samples_per_cluster.max(1);

        let vectors = self.raw_vectors.as_slice();
        let centroids = self.h_centroids.as_slice();
        let codebooks = self.h_codebooks.as_slice();
        let codes = self.h_codes.as_slice();
        let offsets = self.h_ivf_offsets.as_slice();
        let vec_ids = self.h_ivf_vec_ids.as_slice();

        (0..nlist)
            .map(|cluster_id| {
                let start = offsets[cluster_id] as usize;
                let end = offsets[cluster_id + 1] as usize;
                let vector_count = end - start;

                if vector_count == 0 {
                    return ClusterPqDistortion {
                        cluster_id,
                        avg_reconstruction_error: 0.0,
                        vector_count: 0,
                    };
                }

                let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
                let stride = vector_count.div_ceil(sample_limit).max(1);
                let mut error_sum = 0.0f64;
                let mut sampled = 0usize;

                let mut pos = start;
                while pos < end {
                    let vec_id = vec_ids[pos] as usize;
                    let vector = &vectors[vec_id * dim..(vec_id + 1) * dim];
                    let norm_sq: f32 = vector.iter().map(|x| x * x).sum();

                    let mut reconstruction_error = 0.0f32;
                    for m in 0..m_pq {
                        let code = codes[vec_id * m_pq + m] as usize;
                        let centroid_offset = (m * 256 + code) * sub_dim;
                        let cb = &codebooks[centroid_offset..centroid_offset + sub_dim];
                        for d in 0..sub_dim {
                            let residual = vector[m * sub_dim + d] - centroid[m * sub_dim + d];
                            let diff = residual - cb[d];
                            reconstruction_error += diff * diff;
                        }
                    }

                    error_sum += (reconstruction_error / norm_sq.max(1e-12)) as f64;
                    sampled += 1;
                    pos += stride;
                }

                ClusterPqDistortion {
                    cluster_id,
                    avg_reconstruction_error: (error_sum / sampled as f64) as f32,
                    vector_count,
                }
            })
            .collect()
    }
}

/// Deduplicate and merge search results across multiple shards or batches on host.
pub fn host_merge_topk(
    shard_results: &[&[GpuSearchResult]],
    num_queries: usize,
    shard_top_k: usize,
    out_top_k: usize,
) -> Vec<GpuSearchResult> {
    let mut out = Vec::with_capacity(num_queries * out_top_k);
    for q in 0..num_queries {
        let mut combined: Vec<GpuSearchResult> = Vec::new();
        for shard in shard_results {
            let start = q * shard_top_k;
            let end = (start + shard_top_k).min(shard.len());
            combined.extend(
                shard[start..end]
                    .iter()
                    .copied()
                    .filter(|r| r.id != u32::MAX),
            );
        }
        combined.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = HashSet::new();
        let mut kept = 0usize;
        for r in combined {
            if seen.insert(r.id) {
                out.push(r);
                kept += 1;
                if kept == out_top_k {
                    break;
                }
            }
        }
        while kept < out_top_k {
            out.push(GpuSearchResult::default());
            kept += 1;
        }
    }
    out
}
