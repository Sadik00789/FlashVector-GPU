use crate::ffi::{
    gpu_create_stream, gpu_destroy_stream, gpu_sync_stream, BeamStepRecord, CudaError,
    GpuSearchResult, TraversalHop,
};
use crate::memory::DeviceBuffer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum vector dimension supported by the pre-allocated query scratchpad.
pub const MAX_SUPPORTED_DIM: usize = 4096;
/// Maximum number of queries in one batched dispatch.
pub const MAX_BATCH_SIZE: usize = 1024;
/// Maximum top-k supported by the beam-search candidate queues and merge buffer.
pub const MAX_TOP_K: usize = 256;
/// Maximum trajectory hops recorded per query.
pub const MAX_HOPS: usize = 4096;
/// Maximum per-iteration beam snapshots recorded in debug mode.
pub const MAX_BEAM_STEPS: usize = 512;
/// Maximum u32 bitmask words pre-allocated on each stream (covers up to 2M vectors).
pub const MAX_MASK_WORDS: usize = 65_536;

pub struct CudaStream {
    raw: *mut libc::c_void,
    id: usize,
    pub scratch_queries: parking_lot::Mutex<DeviceBuffer<f32>>,
    pub scratch_results: parking_lot::Mutex<DeviceBuffer<GpuSearchResult>>,
    pub scratch_hops: parking_lot::Mutex<DeviceBuffer<TraversalHop>>,
    pub scratch_hop_counts: parking_lot::Mutex<DeviceBuffer<u32>>,
    pub scratch_steps: parking_lot::Mutex<DeviceBuffer<BeamStepRecord>>,
    pub scratch_step_counts: parking_lot::Mutex<DeviceBuffer<u32>>,
    pub scratch_mask: parking_lot::Mutex<DeviceBuffer<u32>>,
    pub scratch_delta_results: parking_lot::Mutex<DeviceBuffer<GpuSearchResult>>,
    pub scratch_merge_results: parking_lot::Mutex<DeviceBuffer<GpuSearchResult>>,
}

// SAFETY: CudaStream owns a CUDA stream handle whose operations are serialized
// through the per-stream scratchpad mutexes; CUDA stream handles are designed
// to be shared across host threads.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new(id: usize) -> Result<Self, CudaError> {
        let raw = gpu_create_stream()?;

        let scratch_queries = DeviceBuffer::new(MAX_BATCH_SIZE * MAX_SUPPORTED_DIM)?;
        let scratch_results = DeviceBuffer::new(MAX_BATCH_SIZE * MAX_TOP_K)?;
        let scratch_hops = DeviceBuffer::new(MAX_HOPS)?;
        let scratch_hop_counts = DeviceBuffer::new(MAX_BATCH_SIZE)?;
        let scratch_steps = DeviceBuffer::new(MAX_BEAM_STEPS)?;
        let scratch_step_counts = DeviceBuffer::new(MAX_BATCH_SIZE)?;
        let scratch_mask = DeviceBuffer::new(MAX_MASK_WORDS)?;
        let scratch_delta_results = DeviceBuffer::new(MAX_BATCH_SIZE * MAX_TOP_K)?;
        let scratch_merge_results = DeviceBuffer::new(2 * MAX_BATCH_SIZE * MAX_TOP_K)?;

        Ok(Self {
            raw,
            id,
            scratch_queries: parking_lot::Mutex::new(scratch_queries),
            scratch_results: parking_lot::Mutex::new(scratch_results),
            scratch_hops: parking_lot::Mutex::new(scratch_hops),
            scratch_hop_counts: parking_lot::Mutex::new(scratch_hop_counts),
            scratch_steps: parking_lot::Mutex::new(scratch_steps),
            scratch_step_counts: parking_lot::Mutex::new(scratch_step_counts),
            scratch_mask: parking_lot::Mutex::new(scratch_mask),
            scratch_delta_results: parking_lot::Mutex::new(scratch_delta_results),
            scratch_merge_results: parking_lot::Mutex::new(scratch_merge_results),
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
    delta_stream: Arc<CudaStream>,
    counter: AtomicUsize,
}

impl CudaStreamPool {
    pub fn new(pool_size: usize) -> Result<Self, CudaError> {
        let size = if pool_size == 0 { 4 } else { pool_size };
        let mut streams = Vec::with_capacity(size);

        for i in 0..size {
            streams.push(Arc::new(CudaStream::new(i)?));
        }
        let delta_stream = Arc::new(CudaStream::new(size)?);

        Ok(Self {
            streams,
            delta_stream,
            counter: AtomicUsize::new(0),
        })
    }

    pub fn get_stream(&self) -> Arc<CudaStream> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        Arc::clone(&self.streams[idx])
    }

    /// Dedicated secondary CUDA stream used for concurrent live delta-buffer scans.
    pub fn delta_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.delta_stream)
    }

    /// Borrowed reference to the dedicated secondary CUDA delta stream.
    pub fn delta_stream_ref(&self) -> &CudaStream {
        &self.delta_stream
    }

    pub fn sync_all(&self) -> Result<(), CudaError> {
        for stream in &self.streams {
            stream.sync()?;
        }
        self.delta_stream.sync()?;
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.streams.len()
    }
}
