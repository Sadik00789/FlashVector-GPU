use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::ffi::{
    gpu_create_stream, gpu_destroy_stream, gpu_sync_stream, CudaError, GpuSearchResult,
    TraversalHop,
};
use crate::memory::DeviceBuffer;

pub struct CudaStream {
    raw: *mut libc::c_void,
    id: usize,
    pub scratch_queries: parking_lot::Mutex<DeviceBuffer<f32>>,
    pub scratch_results: parking_lot::Mutex<DeviceBuffer<GpuSearchResult>>,
    pub scratch_hops: parking_lot::Mutex<DeviceBuffer<TraversalHop>>,
    pub scratch_hop_counts: parking_lot::Mutex<DeviceBuffer<u32>>,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new(id: usize) -> Result<Self, CudaError> {
        let raw = gpu_create_stream()?;

        // Pre-allocate device scratchpads (Max batch 1024 queries * 1536 dim, Top-K 100, Max Hops 2048)
        let scratch_queries = DeviceBuffer::new(1024 * 1536)?;
        let scratch_results = DeviceBuffer::new(1024 * 100)?;
        let scratch_hops = DeviceBuffer::new(2048)?;
        let scratch_hop_counts = DeviceBuffer::new(1024)?;

        Ok(Self {
            raw,
            id,
            scratch_queries: parking_lot::Mutex::new(scratch_queries),
            scratch_results: parking_lot::Mutex::new(scratch_results),
            scratch_hops: parking_lot::Mutex::new(scratch_hops),
            scratch_hop_counts: parking_lot::Mutex::new(scratch_hop_counts),
        })
    }

    #[inline]
    pub fn raw(&self) -> *mut libc::c_void {
        self.raw
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn sync(&self) -> Result<(), CudaError> {
        gpu_sync_stream(self.raw)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = gpu_destroy_stream(self.raw);
        }
    }
}

pub struct CudaStreamPool {
    streams: Vec<Arc<CudaStream>>,
    counter: AtomicUsize,
}

impl CudaStreamPool {
    pub fn new(pool_size: usize) -> Result<Self, CudaError> {
        let size = if pool_size == 0 { 4 } else { pool_size };
        let mut streams = Vec::with_capacity(size);

        for i in 0..size {
            streams.push(Arc::new(CudaStream::new(i)?));
        }

        Ok(Self {
            streams,
            counter: AtomicUsize::new(0),
        })
    }

    pub fn get_stream(&self) -> Arc<CudaStream> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        Arc::clone(&self.streams[idx])
    }

    pub fn sync_all(&self) -> Result<(), CudaError> {
        for stream in &self.streams {
            stream.sync()?;
        }
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.streams.len()
    }
}
