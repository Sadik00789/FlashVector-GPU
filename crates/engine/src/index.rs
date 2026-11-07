use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use crate::cluster::{KMeans, ProductQuantizer};
use crate::ffi::{
    gpu_device_synchronize, gpu_hnsw_search_batch, gpu_ivf_pq_search_batch, CudaError,
    GpuSearchResult, HnswGpuGraph, IndexConfig, IvfPqGpuTables, MetricType, TraversalHop,
};
use crate::memory::{DeviceBuffer, PinnedBuffer};
use crate::metrics::MetricsTracker;
use crate::streams::{CudaStreamPool, MAX_BATCH_SIZE, MAX_HOPS, MAX_SUPPORTED_DIM, MAX_TOP_K};

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

/// Per-Voronoi-cell PQ reconstruction statistics used by the Studio heatmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterPqDistortion {
    pub cluster_id: usize,
    /// Mean squared reconstruction error normalized by the original vector norm.
    pub avg_reconstruction_error: f32,
    pub vector_count: usize,
}

pub struct GpuVectorIndex {
    config: IndexConfig,
    metrics: Arc<MetricsTracker>,
    streams: Arc<CudaStreamPool>,

    // Raw vectors
    num_vectors: usize,
    raw_vectors: PinnedBuffer<f32>,
    d_vectors: DeviceBuffer<f32>,

    // HNSW Graph
    entry_point: u32,
    h_adjacency: PinnedBuffer<u32>,
    d_adjacency: DeviceBuffer<u32>,
    h_degrees: PinnedBuffer<u32>,
    d_degrees: DeviceBuffer<u32>,

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

    is_built: bool,
}

impl GpuVectorIndex {
    pub fn new(config: IndexConfig) -> Result<Self, CudaError> {
        let streams = Arc::new(CudaStreamPool::new(4)?);
        let metrics = Arc::new(MetricsTracker::new());

        Ok(Self {
            config,
            metrics,
            streams,
            num_vectors: 0,
            raw_vectors: PinnedBuffer::new(0)?,
            d_vectors: DeviceBuffer::new(0)?,
            entry_point: 0,
            h_adjacency: PinnedBuffer::new(0)?,
            d_adjacency: DeviceBuffer::new(0)?,
            h_degrees: PinnedBuffer::new(0)?,
            d_degrees: DeviceBuffer::new(0)?,
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
            is_built: false,
        })
    }

    /// Map the raw configuration metric code onto the CUDA metric enum.
    #[inline]
    pub fn metric(&self) -> MetricType {
        match self.config.metric {
            0 => MetricType::L2,
            1 => MetricType::Cosine,
            2 => MetricType::InnerProduct,
            _ => MetricType::L2,
        }
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

        // Normalize the dataset once for cosine: graph construction and IVF-PQ
        // training must all operate on the same unit-length representation.
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
        self.num_vectors = num_vectors;

        info!(
            "Building FlashVector-GPU Index for {} vectors of dimension {}",
            num_vectors, dim
        );

        // 1. Store Raw Vectors in Pinned & Device memory
        let mut raw_pinned = PinnedBuffer::new(dataset.len())?;
        raw_pinned.copy_from_slice(dataset);
        let mut d_vecs = DeviceBuffer::new(dataset.len())?;

        let stream = self.streams.get_stream();
        d_vecs.copy_from_pinned_async(&raw_pinned, stream.raw())?;

        // 2. Build the HNSW graph by incremental insertion with beam search and
        // Malkov-Yashunin diverse neighbor selection.
        let m = self.config.m as usize;
        let ef_construction = self.config.ef_construction as usize;
        let (adjacency, degrees, best_node) =
            crate::graph::build_hnsw(dataset, num_vectors, dim, m, ef_construction);

        let mut h_adj = PinnedBuffer::new(adjacency.len())?;
        h_adj.copy_from_slice(&adjacency);
        let mut h_deg = PinnedBuffer::new(degrees.len())?;
        h_deg.copy_from_slice(&degrees);

        let mut d_adj = DeviceBuffer::new(num_vectors * m)?;
        let mut d_deg = DeviceBuffer::new(num_vectors)?;
        d_adj.copy_from_pinned_async(&h_adj, stream.raw())?;
        d_deg.copy_from_pinned_async(&h_deg, stream.raw())?;

        // 3. Train IVF Voronoi Centroids and Product Quantization Codebooks
        let nlist = (self.config.nlist as usize).min(num_vectors);
        let m_pq = self.config.m_pq as usize;
        assert!(m_pq > 0, "m_pq must be greater than zero");
        assert!(dim % m_pq == 0, "Dimension must be divisible by m_pq");

        let kmeans_res = KMeans::fit(dataset, dim, nlist, 10);
        let pq = ProductQuantizer::train(dataset, dim, m_pq, 8);
        let codes = pq.encode(dataset);

        // Build Inverted Posting Lists
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

        // Upload IVF-PQ to GPU
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

        stream.sync()?;

        self.raw_vectors = raw_pinned;
        self.d_vectors = d_vecs;
        self.h_adjacency = h_adj;
        self.d_adjacency = d_adj;
        self.h_degrees = h_deg;
        self.d_degrees = d_deg;
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
        self.entry_point = best_node;
        self.is_built = true;

        info!("FlashVector-GPU Index successfully built and synchronized to VRAM");
        Ok(())
    }

    /// Single query search with HNSW warp beam search
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        let (results, _) = self.search_with_trajectory(query, top_k, ef_search)?;
        Ok(results)
    }

    /// Optimized single-dispatch query search with trajectory hop recording
    pub fn search_with_trajectory(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<(Vec<GpuSearchResult>, Vec<TraversalHop>), CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        let dim = self.config.dim as usize;
        assert!(
            dim <= MAX_SUPPORTED_DIM,
            "Dimension {} exceeds maximum supported {}",
            dim,
            MAX_SUPPORTED_DIM
        );
        assert!(
            query.len() >= dim,
            "Query is shorter than the index dimension"
        );
        self.validate_search_shape(1, top_k, dim)?;
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let start = Instant::now();
        let max_hops = (ef_search.saturating_mul(4)).min(MAX_HOPS);

        let stream = self.streams.get_stream();

        // 1. Use pre-allocated device scratchpads from CudaStream
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();
        let mut d_hops = stream.scratch_hops.lock();
        let mut d_hop_count = stream.scratch_hop_counts.lock();

        // Normalize the query for cosine so the uploaded representation and the
        // normalized dataset live in the same unit-sphere space.
        let normalized_query: Vec<f32>;
        let query_slice = if self.config.metric == MetricType::Cosine as u32 {
            let mut q = query[..dim].to_vec();
            normalize_slice_in_place(&mut q, dim);
            normalized_query = q;
            &normalized_query
        } else {
            &query[..dim]
        };

        d_q.copy_from_host_async(query_slice, stream.raw())?;

        let graph = HnswGpuGraph {
            d_vectors: self.d_vectors.as_ptr(),
            d_adjacency: self.d_adjacency.as_ptr(),
            d_degree: self.d_degrees.as_ptr(),
            num_nodes: self.num_vectors as u32,
            dim: self.config.dim,
            m_max: self.config.m,
            entry_point: self.entry_point,
        };

        // 2. Single GPU Kernel Launch
        gpu_hnsw_search_batch(
            &graph,
            d_q.as_ptr(),
            1,
            self.config.dim,
            top_k as u32,
            ef_search as u32,
            self.metric(),
            d_results.as_mut_ptr(),
            d_hops.as_mut_ptr(),
            d_hop_count.as_mut_ptr(),
            max_hops as u32,
            stream.raw(),
        )?;

        // 3. Download results and recorded hop trajectory
        let mut h_results = vec![GpuSearchResult::default(); top_k];
        let mut h_hop_count = vec![0u32; 1];
        let mut h_all_hops = vec![
            TraversalHop {
                step: 0,
                from_node: 0,
                to_node: 0,
                distance: 0.0
            };
            max_hops
        ];

        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        d_hop_count.copy_to_host_async(&mut h_hop_count, stream.raw())?;
        d_hops.copy_to_host_async(&mut h_all_hops, stream.raw())?;

        stream.sync()?;

        let actual_hops = (h_hop_count[0] as usize).min(max_hops);
        h_all_hops.truncate(actual_hops);

        self.metrics.record_query(start.elapsed(), 1);
        Ok((h_results, h_all_hops))
    }

    /// Batched multi-query HNSW search in a single CUDA grid launch
    pub fn batch_search(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        let dim = self.config.dim as usize;
        assert!(
            dim <= MAX_SUPPORTED_DIM,
            "Dimension {} exceeds maximum supported {}",
            dim,
            MAX_SUPPORTED_DIM
        );
        let total_floats = num_queries * dim;
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();

        let normalized_queries: Vec<f32>;
        let queries_ref: &[f32] = if self.config.metric == MetricType::Cosine as u32 {
            let mut normalized = queries[..total_floats].to_vec();
            normalize_slice_in_place(&mut normalized, dim);
            normalized_queries = normalized;
            &normalized_queries
        } else {
            &queries[..total_floats]
        };

        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();

        d_q.copy_from_host_async(queries_ref, stream.raw())?;

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
            d_results.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            stream.raw(),
        )?;

        let total_results = num_queries * top_k;
        let mut h_results = vec![GpuSearchResult::default(); total_results];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(h_results)
    }

    /// Zero-copy search for queries already resident in GPU device memory.
    pub fn batch_search_device_ptr(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if d_queries.is_null() {
            return Err(CudaError::InvalidArgument);
        }
        // The caller's device buffer may have been produced on another CUDA
        // stream; drain it before the non-blocking engine stream consumes it.
        gpu_device_synchronize()?;
        self.validate_search_shape(num_queries, top_k, num_queries * self.config.dim as usize)?;
        if top_k > MAX_TOP_K || num_queries > MAX_BATCH_SIZE {
            return Err(CudaError::InvalidArgument);
        }

        let start = Instant::now();
        let stream = self.streams.get_stream();
        let mut d_results = stream.scratch_results.lock();

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
            d_results.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            stream.raw(),
        )?;

        let total_results = num_queries * top_k;
        let mut h_results = vec![GpuSearchResult::default(); total_results];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(h_results)
    }

    /// Batched IVF-PQ ADC search in a single CUDA grid launch
    pub fn batch_search_ivf(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        let dim = self.config.dim as usize;
        assert!(
            dim <= MAX_SUPPORTED_DIM,
            "Dimension {} exceeds maximum supported {}",
            dim,
            MAX_SUPPORTED_DIM
        );
        let total_floats = num_queries * dim;
        self.validate_search_shape(num_queries, top_k, queries.len())?;
        let start = Instant::now();
        let sub_dim = dim / (self.config.m_pq as usize);

        let normalized_queries: Vec<f32>;
        let queries_ref: &[f32] = if self.config.metric == MetricType::Cosine as u32 {
            let mut normalized = queries[..total_floats].to_vec();
            normalize_slice_in_place(&mut normalized, dim);
            normalized_queries = normalized;
            &normalized_queries
        } else {
            &queries[..total_floats]
        };

        let stream = self.streams.get_stream();
        let mut d_q = stream.scratch_queries.lock();
        let mut d_results = stream.scratch_results.lock();

        d_q.copy_from_host_async(queries_ref, stream.raw())?;

        let tables = IvfPqGpuTables {
            d_centroids: self.d_centroids.as_ptr(),
            d_pq_codebooks: self.d_codebooks.as_ptr(),
            d_pq_codes: self.d_codes.as_ptr(),
            d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
            d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
            num_vectors: self.num_vectors as u32,
            dim: self.config.dim,
            nlist: (self.h_centroids.len() / dim) as u32,
            m_pq: self.config.m_pq,
            sub_dim: sub_dim as u32,
        };

        gpu_ivf_pq_search_batch(
            &tables,
            d_q.as_ptr(),
            num_queries as u32,
            top_k as u32,
            nprobe as u32,
            self.metric(),
            d_results.as_mut_ptr(),
            stream.raw(),
        )?;

        let total_results = num_queries * top_k;
        let mut h_results = vec![GpuSearchResult::default(); total_results];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(h_results)
    }

    /// Zero-copy IVF-PQ search for queries already resident in GPU device memory.
    pub fn batch_search_ivf_device_ptr(
        &self,
        d_queries: *const f32,
        num_queries: usize,
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        assert!(self.is_built, "Index must be built before searching");
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if d_queries.is_null() {
            return Err(CudaError::InvalidArgument);
        }
        // The caller's device buffer may have been produced on another CUDA
        // stream; drain it before the non-blocking engine stream consumes it.
        gpu_device_synchronize()?;
        self.validate_search_shape(num_queries, top_k, num_queries * self.config.dim as usize)?;
        if top_k > MAX_TOP_K || num_queries > MAX_BATCH_SIZE {
            return Err(CudaError::InvalidArgument);
        }

        let start = Instant::now();
        let dim = self.config.dim as usize;
        let sub_dim = dim / (self.config.m_pq as usize);

        let stream = self.streams.get_stream();
        let mut d_results = stream.scratch_results.lock();

        let tables = IvfPqGpuTables {
            d_centroids: self.d_centroids.as_ptr(),
            d_pq_codebooks: self.d_codebooks.as_ptr(),
            d_pq_codes: self.d_codes.as_ptr(),
            d_ivf_offsets: self.d_ivf_offsets.as_ptr(),
            d_ivf_vec_ids: self.d_ivf_vec_ids.as_ptr(),
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
            self.metric(),
            d_results.as_mut_ptr(),
            stream.raw(),
        )?;

        let total_results = num_queries * top_k;
        let mut h_results = vec![GpuSearchResult::default(); total_results];
        d_results.copy_to_host_async(&mut h_results, stream.raw())?;
        stream.sync()?;

        self.metrics.record_query(start.elapsed(), num_queries);
        Ok(h_results)
    }

    /// Single query IVF-PQ search
    pub fn search_ivf_pq(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        self.batch_search_ivf(query, 1, top_k, nprobe)
    }

    pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> {
        if id >= self.num_vectors {
            return None;
        }
        let dim = self.config.dim as usize;
        let start = id * dim;
        Some(self.raw_vectors.as_slice()[start..start + dim].to_vec())
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn dim(&self) -> usize {
        self.config.dim as usize
    }

    pub fn metrics(&self) -> Arc<MetricsTracker> {
        Arc::clone(&self.metrics)
    }

    pub fn config(&self) -> IndexConfig {
        self.config
    }

    /// Compute per-cluster PQ reconstruction error from the trained codebooks.
    ///
    /// Each vector is rebuilt from its `m_pq` codebook centroids and compared
    /// against the stored representation. The reported error is the mean of
    /// `||x - x_hat||^2 / ||x||^2`, sampled evenly across each inverted list.
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

                let stride = ((vector_count + sample_limit - 1) / sample_limit).max(1);
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
                        let centroid = &codebooks[centroid_offset..centroid_offset + sub_dim];
                        for d in 0..sub_dim {
                            let diff = vector[m * sub_dim + d] - centroid[d];
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
