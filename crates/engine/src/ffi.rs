#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]
// Safe wrapper layer over the CUDA C FFI (`kernels/include/cuda_bridge.h`).

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    #[error("CUDA driver/runtime error code: {0}")]
    DriverError(i32),
    #[error("Invalid argument or null pointer passed to CUDA FFI")]
    InvalidArgument,
    #[error("Device out of memory: required {0} bytes")]
    OutOfMemory(usize),
    #[error("CUDA initialization failed on device {0}")]
    InitializationFailed(i32),
    #[error("I/O or persistence error: {0}")]
    IoError(String),
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MetricType {
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vector3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuQuery {
    pub query_id: u32,
    pub data: *const f32,
    pub dim: u32,
    pub top_k: u32,
    pub ef_search: u32,
    pub nprobe: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GpuSearchResult {
    pub id: u32,
    pub distance: f32,
}

impl Default for GpuSearchResult {
    fn default() -> Self {
        Self {
            id: u32::MAX,
            distance: f32::INFINITY,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TraversalHop {
    pub step: u32,
    pub from_node: u32,
    pub to_node: u32,
    pub distance: f32,
}

pub const BEAM_DEBUG_MAX_CANDS: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BeamCandidateRecord {
    pub id: u32,
    pub distance: f32,
    pub visited: u32,
    pub _pad: u32,
}

impl Default for BeamCandidateRecord {
    fn default() -> Self {
        Self {
            id: u32::MAX,
            distance: f32::INFINITY,
            visited: 0,
            _pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BeamStepRecord {
    pub iteration: u32,
    pub expanded_node: u32,
    pub expanded_distance: f32,
    pub eval_start: u32,
    pub eval_count: u32,
    pub cand_count: u32,
    pub candidates: [BeamCandidateRecord; BEAM_DEBUG_MAX_CANDS],
}

impl Default for BeamStepRecord {
    fn default() -> Self {
        Self {
            iteration: 0,
            expanded_node: u32::MAX,
            expanded_distance: f32::INFINITY,
            eval_start: 0,
            eval_count: 0,
            cand_count: 0,
            candidates: [BeamCandidateRecord::default(); BEAM_DEBUG_MAX_CANDS],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexConfig {
    pub dim: u32,
    pub max_elements: u32,
    pub m: u32,
    pub ef_construction: u32,
    pub nlist: u32,
    pub m_pq: u32,
    pub metric: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_elements: 100_000,
            m: 32,
            ef_construction: 128,
            nlist: 256,
            m_pq: 16,
            metric: MetricType::L2 as u32,
        }
    }
}

#[repr(C)]
pub struct IvfPqGpuTables {
    pub d_centroids: *const f32,
    pub d_pq_codebooks: *const f32,
    pub d_pq_codes: *const u8,
    pub d_ivf_offsets: *const u32,
    pub d_ivf_vec_ids: *const u32,
    pub d_vectors: *const f32,
    pub num_vectors: u32,
    pub dim: u32,
    pub nlist: u32,
    pub m_pq: u32,
    pub sub_dim: u32,
}

#[repr(C)]
pub struct IvfSqGpuTables {
    pub d_centroids: *const f32,
    pub d_sq_mins: *const f32,
    pub d_sq_scales: *const f32,
    pub d_sq_codes: *const u8,
    pub d_ivf_offsets: *const u32,
    pub d_ivf_vec_ids: *const u32,
    pub d_vectors: *const f32,
    pub num_vectors: u32,
    pub dim: u32,
    pub nlist: u32,
}

#[repr(C)]
pub struct IvfRabitqGpuTables {
    pub d_centroids: *const f32,
    pub d_rot_centroids: *const f32,
    pub d_rabitq_bits: *const u32,
    pub d_rabitq_meta: *const f32,
    pub d_rot_signs: *const f32,
    pub d_ivf_offsets: *const u32,
    pub d_ivf_vec_ids: *const u32,
    pub d_vectors: *const f32,
    pub num_vectors: u32,
    pub dim: u32,
    pub padded_dim: u32,
    pub words_per_vec: u32,
    pub nlist: u32,
}

#[repr(C)]
pub struct Fp16GpuTables {
    pub d_vectors_fp16: *const u16,
    pub d_vec_norms_sq: *const f32,
    pub d_vectors: *const f32,
    pub num_vectors: u32,
    pub dim: u32,
    pub padded_dim: u32,
}

#[repr(C)]
pub struct HnswGpuGraph {
    pub d_vectors: *const f32,
    pub d_adjacency: *const u32,
    pub d_degree: *const u32,
    pub num_nodes: u32,
    pub dim: u32,
    pub m_max: u32,
    pub entry_point: u32,
}

#[repr(C)]
pub struct CagraGpuGraph {
    pub d_vectors: *const f32,
    pub d_adjacency: *const u32,
    pub d_seed_nodes: *const u32,
    pub num_nodes: u32,
    pub dim: u32,
    pub graph_degree: u32,
    pub entry_point: u32,
    pub num_seeds: u32,
}

#[repr(C)]
pub struct SparseCsrGpuMatrix {
    pub d_indptr: *const u32,
    pub d_indices: *const u32,
    pub d_values: *const f32,
    pub num_docs: u32,
    pub nnz: u32,
}

extern "C" {
    fn cuda_init_device(device_id: libc::c_int) -> libc::c_int;
    fn cuda_set_device(device_id: libc::c_int) -> libc::c_int;
    fn cuda_get_device_count(count: *mut libc::c_int) -> libc::c_int;
    fn cuda_get_device_memory(
        free_bytes: *mut libc::size_t,
        total_bytes: *mut libc::size_t,
    ) -> libc::c_int;
    fn cuda_device_synchronize() -> libc::c_int;
    fn cuda_enable_peer_access(device: libc::c_int, peer: libc::c_int) -> libc::c_int;
    fn cuda_memcpy_peer_async(
        dst: *mut libc::c_void,
        dst_device: libc::c_int,
        src: *const libc::c_void,
        src_device: libc::c_int,
        bytes: libc::size_t,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_malloc_device(ptr: *mut *mut libc::c_void, bytes: libc::size_t) -> libc::c_int;
    fn cuda_free_device(ptr: *mut libc::c_void) -> libc::c_int;
    fn cuda_malloc_host(ptr: *mut *mut libc::c_void, bytes: libc::size_t) -> libc::c_int;
    fn cuda_free_host(ptr: *mut libc::c_void) -> libc::c_int;
    fn cuda_host_get_device_pointer(
        dev_ptr: *mut *mut libc::c_void,
        host_ptr: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_memcpy_h2d_async(
        dst: *mut libc::c_void,
        src: *const libc::c_void,
        bytes: libc::size_t,
        stream: *mut libc::c_void,
    ) -> libc::c_int;
    fn cuda_memcpy_d2h_async(
        dst: *mut libc::c_void,
        src: *const libc::c_void,
        bytes: libc::size_t,
        stream: *mut libc::c_void,
    ) -> libc::c_int;
    fn cuda_memset_device_async(
        dst: *mut libc::c_void,
        value: libc::c_int,
        bytes: libc::size_t,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_create_stream(stream: *mut *mut libc::c_void) -> libc::c_int;
    fn cuda_destroy_stream(stream: *mut libc::c_void) -> libc::c_int;
    fn cuda_sync_stream(stream: *mut libc::c_void) -> libc::c_int;

    fn cuda_hnsw_search_batch(
        graph: *const HnswGpuGraph,
        d_queries: *const f32,
        num_queries: u32,
        dim: u32,
        top_k: u32,
        ef_search: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        d_out_hops: *mut TraversalHop,
        d_out_hop_counts: *mut u32,
        max_hops_per_query: u32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_hnsw_search_batch_debug(
        graph: *const HnswGpuGraph,
        d_query: *const f32,
        top_k: u32,
        ef_search: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        d_out_hops: *mut TraversalHop,
        d_out_hop_count: *mut u32,
        max_hops: u32,
        d_out_steps: *mut BeamStepRecord,
        d_out_step_count: *mut u32,
        max_steps: u32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_cagra_search_batch(
        graph: *const CagraGpuGraph,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        ef_search: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        d_out_hops: *mut TraversalHop,
        d_out_hop_counts: *mut u32,
        max_hops_per_query: u32,
        d_out_steps: *mut BeamStepRecord,
        d_out_step_count: *mut u32,
        max_steps: u32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_ivf_pq_search_batch(
        tables: *const IvfPqGpuTables,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        nprobe: u32,
        rerank_k: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_ivf_sq_search_batch(
        tables: *const IvfSqGpuTables,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        nprobe: u32,
        rerank_k: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_ivf_rabitq_search_batch(
        tables: *const IvfRabitqGpuTables,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        nprobe: u32,
        rerank_k: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_fp16_tc_search_batch(
        tables: *const Fp16GpuTables,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_flat_scan_batch(
        d_vectors: *const f32,
        d_vec_ids: *const u32,
        num_vectors: u32,
        dim: u32,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        metric: MetricType,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_merge_topk(
        d_shard_results: *const GpuSearchResult,
        num_shards: u32,
        num_queries: u32,
        shard_top_k: u32,
        out_top_k: u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_sparse_search_batch(
        matrix: *const SparseCsrGpuMatrix,
        d_q_indptr: *const u32,
        d_q_indices: *const u32,
        d_q_values: *const f32,
        num_queries: u32,
        top_k: u32,
        d_valid_mask: *const u32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_rrf_fuse(
        d_dense_results: *const GpuSearchResult,
        dense_k: u32,
        d_sparse_results: *const GpuSearchResult,
        sparse_k: u32,
        num_queries: u32,
        out_top_k: u32,
        rrf_k: f32,
        dense_weight: f32,
        sparse_weight: f32,
        d_out_results: *mut GpuSearchResult,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_compute_distances_warp(
        d_queries: *const f32,
        d_dataset: *const f32,
        num_queries: u32,
        num_vectors: u32,
        dim: u32,
        metric: MetricType,
        d_out_distances: *mut f32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_bitonic_sort_test(
        d_keys_in: *const f32,
        d_vals_in: *const u32,
        d_keys_out: *mut f32,
        d_vals_out: *mut u32,
        n: u32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;
}

pub fn gpu_init(device_id: i32) -> Result<(), CudaError> {
    let ret = unsafe { cuda_init_device(device_id as libc::c_int) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::InitializationFailed(device_id))
    }
}

pub fn gpu_set_device(device_id: i32) -> Result<(), CudaError> {
    let ret = unsafe { cuda_set_device(device_id as libc::c_int) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_device_count() -> Result<usize, CudaError> {
    let mut count: libc::c_int = 0;
    let ret = unsafe { cuda_get_device_count(&mut count) };
    if ret == 0 {
        Ok(count.max(0) as usize)
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_get_memory() -> Result<(usize, usize), CudaError> {
    let mut free_bytes: libc::size_t = 0;
    let mut total_bytes: libc::size_t = 0;
    let ret = unsafe { cuda_get_device_memory(&mut free_bytes, &mut total_bytes) };
    if ret == 0 {
        Ok((free_bytes, total_bytes))
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_device_synchronize() -> Result<(), CudaError> {
    let ret = unsafe { cuda_device_synchronize() };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_enable_peer_access(device: i32, peer: i32) -> Result<(), CudaError> {
    let ret = unsafe { cuda_enable_peer_access(device as libc::c_int, peer as libc::c_int) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_memcpy_peer_async<T>(
    dst: *mut T,
    dst_device: i32,
    src: *const T,
    src_device: i32,
    count: usize,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    if count == 0 {
        return Ok(());
    }
    let bytes = count * std::mem::size_of::<T>();
    let ret = unsafe {
        cuda_memcpy_peer_async(
            dst as *mut libc::c_void,
            dst_device as libc::c_int,
            src as *const libc::c_void,
            src_device as libc::c_int,
            bytes,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_malloc_device<T>(count: usize) -> Result<*mut T, CudaError> {
    if count == 0 {
        return Ok(std::ptr::null_mut());
    }
    let bytes = count * std::mem::size_of::<T>();
    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { cuda_malloc_device(&mut ptr, bytes) };
    if ret == 0 && !ptr.is_null() {
        Ok(ptr as *mut T)
    } else {
        Err(CudaError::OutOfMemory(bytes))
    }
}

pub fn gpu_free_device<T>(ptr: *mut T) -> Result<(), CudaError> {
    if ptr.is_null() {
        return Ok(());
    }
    let ret = unsafe { cuda_free_device(ptr as *mut libc::c_void) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_malloc_host<T>(count: usize) -> Result<*mut T, CudaError> {
    if count == 0 {
        return Ok(std::ptr::null_mut());
    }
    let bytes = count * std::mem::size_of::<T>();
    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { cuda_malloc_host(&mut ptr, bytes) };
    if ret == 0 && !ptr.is_null() {
        Ok(ptr as *mut T)
    } else {
        Err(CudaError::OutOfMemory(bytes))
    }
}

pub fn gpu_free_host<T>(ptr: *mut T) -> Result<(), CudaError> {
    if ptr.is_null() {
        return Ok(());
    }
    let ret = unsafe { cuda_free_host(ptr as *mut libc::c_void) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_host_get_device_pointer<T>(host_ptr: *mut T) -> Result<*const T, CudaError> {
    if host_ptr.is_null() {
        return Ok(std::ptr::null());
    }
    let mut dev_ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { cuda_host_get_device_pointer(&mut dev_ptr, host_ptr as *mut libc::c_void) };
    if ret == 0 && !dev_ptr.is_null() {
        Ok(dev_ptr as *const T)
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_memcpy_h2d_async<T>(
    dst: *mut T,
    src: *const T,
    count: usize,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    if count == 0 {
        return Ok(());
    }
    let bytes = count * std::mem::size_of::<T>();
    let ret = unsafe {
        cuda_memcpy_h2d_async(
            dst as *mut libc::c_void,
            src as *const libc::c_void,
            bytes,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_memcpy_d2h_async<T>(
    dst: *mut T,
    src: *const T,
    count: usize,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    if count == 0 {
        return Ok(());
    }
    let bytes = count * std::mem::size_of::<T>();
    let ret = unsafe {
        cuda_memcpy_d2h_async(
            dst as *mut libc::c_void,
            src as *const libc::c_void,
            bytes,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_memset_device_async<T>(
    dst: *mut T,
    value: i32,
    count: usize,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    if count == 0 {
        return Ok(());
    }
    let bytes = count * std::mem::size_of::<T>();
    let ret = unsafe { cuda_memset_device_async(dst as *mut libc::c_void, value, bytes, stream) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_create_stream() -> Result<*mut libc::c_void, CudaError> {
    let mut stream: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { cuda_create_stream(&mut stream) };
    if ret == 0 && !stream.is_null() {
        Ok(stream)
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_destroy_stream(stream: *mut libc::c_void) -> Result<(), CudaError> {
    if stream.is_null() {
        return Ok(());
    }
    let ret = unsafe { cuda_destroy_stream(stream) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_sync_stream(stream: *mut libc::c_void) -> Result<(), CudaError> {
    let ret = unsafe { cuda_sync_stream(stream) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_hnsw_search_batch(
    graph: &HnswGpuGraph,
    d_queries: *const f32,
    num_queries: u32,
    dim: u32,
    top_k: u32,
    ef_search: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    d_out_hops: *mut TraversalHop,
    d_out_hop_counts: *mut u32,
    max_hops_per_query: u32,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_hnsw_search_batch(
            graph as *const HnswGpuGraph,
            d_queries,
            num_queries,
            dim,
            top_k,
            ef_search,
            metric,
            d_valid_mask,
            d_out_results,
            d_out_hops,
            d_out_hop_counts,
            max_hops_per_query,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_hnsw_search_batch_debug(
    graph: &HnswGpuGraph,
    d_query: *const f32,
    top_k: u32,
    ef_search: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    d_out_hops: *mut TraversalHop,
    d_out_hop_count: *mut u32,
    max_hops: u32,
    d_out_steps: *mut BeamStepRecord,
    d_out_step_count: *mut u32,
    max_steps: u32,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_hnsw_search_batch_debug(
            graph as *const HnswGpuGraph,
            d_query,
            top_k,
            ef_search,
            metric,
            d_valid_mask,
            d_out_results,
            d_out_hops,
            d_out_hop_count,
            max_hops,
            d_out_steps,
            d_out_step_count,
            max_steps,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_cagra_search_batch(
    graph: &CagraGpuGraph,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    ef_search: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    d_out_hops: *mut TraversalHop,
    d_out_hop_counts: *mut u32,
    max_hops_per_query: u32,
    d_out_steps: *mut BeamStepRecord,
    d_out_step_count: *mut u32,
    max_steps: u32,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_cagra_search_batch(
            graph as *const CagraGpuGraph,
            d_queries,
            num_queries,
            top_k,
            ef_search,
            metric,
            d_valid_mask,
            d_out_results,
            d_out_hops,
            d_out_hop_counts,
            max_hops_per_query,
            d_out_steps,
            d_out_step_count,
            max_steps,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_ivf_pq_search_batch(
    tables: &IvfPqGpuTables,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    nprobe: u32,
    rerank_k: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_ivf_pq_search_batch(
            tables as *const IvfPqGpuTables,
            d_queries,
            num_queries,
            top_k,
            nprobe,
            rerank_k,
            metric,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_ivf_sq_search_batch(
    tables: &IvfSqGpuTables,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    nprobe: u32,
    rerank_k: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_ivf_sq_search_batch(
            tables as *const IvfSqGpuTables,
            d_queries,
            num_queries,
            top_k,
            nprobe,
            rerank_k,
            metric,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_ivf_rabitq_search_batch(
    tables: &IvfRabitqGpuTables,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    nprobe: u32,
    rerank_k: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_ivf_rabitq_search_batch(
            tables as *const IvfRabitqGpuTables,
            d_queries,
            num_queries,
            top_k,
            nprobe,
            rerank_k,
            metric,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_fp16_tc_search_batch(
    tables: &Fp16GpuTables,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_fp16_tc_search_batch(
            tables as *const Fp16GpuTables,
            d_queries,
            num_queries,
            top_k,
            metric,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_flat_scan_batch(
    d_vectors: *const f32,
    d_vec_ids: *const u32,
    num_vectors: u32,
    dim: u32,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    metric: MetricType,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_flat_scan_batch(
            d_vectors,
            d_vec_ids,
            num_vectors,
            dim,
            d_queries,
            num_queries,
            top_k,
            metric,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_merge_topk(
    d_shard_results: *const GpuSearchResult,
    num_shards: u32,
    num_queries: u32,
    shard_top_k: u32,
    out_top_k: u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_merge_topk(
            d_shard_results,
            num_shards,
            num_queries,
            shard_top_k,
            out_top_k,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_sparse_search_batch(
    matrix: &SparseCsrGpuMatrix,
    d_q_indptr: *const u32,
    d_q_indices: *const u32,
    d_q_values: *const f32,
    num_queries: u32,
    top_k: u32,
    d_valid_mask: *const u32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_sparse_search_batch(
            matrix as *const SparseCsrGpuMatrix,
            d_q_indptr,
            d_q_indices,
            d_q_values,
            num_queries,
            top_k,
            d_valid_mask,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_rrf_fuse(
    d_dense_results: *const GpuSearchResult,
    dense_k: u32,
    d_sparse_results: *const GpuSearchResult,
    sparse_k: u32,
    num_queries: u32,
    out_top_k: u32,
    rrf_k: f32,
    dense_weight: f32,
    sparse_weight: f32,
    d_out_results: *mut GpuSearchResult,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_rrf_fuse(
            d_dense_results,
            dense_k,
            d_sparse_results,
            sparse_k,
            num_queries,
            out_top_k,
            rrf_k,
            dense_weight,
            sparse_weight,
            d_out_results,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_compute_distances_warp(
    d_queries: *const f32,
    d_dataset: *const f32,
    num_queries: u32,
    num_vectors: u32,
    dim: u32,
    metric: MetricType,
    d_out_distances: *mut f32,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret = unsafe {
        cuda_compute_distances_warp(
            d_queries,
            d_dataset,
            num_queries,
            num_vectors,
            dim,
            metric,
            d_out_distances,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}

pub fn gpu_bitonic_sort_test(
    d_keys_in: *const f32,
    d_vals_in: *const u32,
    d_keys_out: *mut f32,
    d_vals_out: *mut u32,
    n: u32,
    stream: *mut libc::c_void,
) -> Result<(), CudaError> {
    let ret =
        unsafe { cuda_bitonic_sort_test(d_keys_in, d_vals_in, d_keys_out, d_vals_out, n, stream) };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}
