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

#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct IndexConfig {
    pub dim: u32,
    pub max_elements: u32,
    pub m: u32,
    pub ef_construction: u32,
    pub nlist: u32,
    pub m_pq: u32,
    pub nbits_pq: u32,
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
            nbits_pq: 8,
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
    pub num_vectors: u32,
    pub dim: u32,
    pub nlist: u32,
    pub m_pq: u32,
    pub sub_dim: u32,
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

extern "C" {
    fn cuda_init_device(device_id: libc::c_int) -> libc::c_int;
    fn cuda_get_device_memory(free_bytes: *mut libc::size_t, total_bytes: *mut libc::size_t) -> libc::c_int;
    fn cuda_device_synchronize() -> libc::c_int;

    fn cuda_malloc_device(ptr: *mut *mut libc::c_void, bytes: libc::size_t) -> libc::c_int;
    fn cuda_free_device(ptr: *mut libc::c_void) -> libc::c_int;
    fn cuda_malloc_host(ptr: *mut *mut libc::c_void, bytes: libc::size_t) -> libc::c_int;
    fn cuda_free_host(ptr: *mut libc::c_void) -> libc::c_int;

    fn cuda_memcpy_h2d_async(dst: *mut libc::c_void, src: *const libc::c_void, bytes: libc::size_t, stream: *mut libc::c_void) -> libc::c_int;
    fn cuda_memcpy_d2h_async(dst: *mut libc::c_void, src: *const libc::c_void, bytes: libc::size_t, stream: *mut libc::c_void) -> libc::c_int;

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
        d_out_results: *mut GpuSearchResult,
        d_out_hops: *mut TraversalHop,
        d_out_hop_counts: *mut u32,
        max_hops_per_query: u32,
        stream: *mut libc::c_void,
    ) -> libc::c_int;

    fn cuda_ivf_pq_search_batch(
        tables: *const IvfPqGpuTables,
        d_queries: *const f32,
        num_queries: u32,
        top_k: u32,
        nprobe: u32,
        metric: MetricType,
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

pub fn gpu_get_memory() -> Result<(usize, usize), CudaError> {
    let mut free_bytes: libc::size_t = 0;
    let mut total_bytes: libc::size_t = 0;
    let ret = unsafe { cuda_get_device_memory(&mut free_bytes, &mut total_bytes) };
    if ret == 0 {
        Ok((free_bytes as usize, total_bytes as usize))
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

pub fn gpu_ivf_pq_search_batch(
    tables: &IvfPqGpuTables,
    d_queries: *const f32,
    num_queries: u32,
    top_k: u32,
    nprobe: u32,
    metric: MetricType,
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
            metric,
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
    let ret = unsafe {
        cuda_bitonic_sort_test(
            d_keys_in,
            d_vals_in,
            d_keys_out,
            d_vals_out,
            n,
            stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(ret))
    }
}
