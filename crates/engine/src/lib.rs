pub mod ffi;
pub mod memory;
pub mod streams;
pub mod cluster;
pub mod metrics;
pub mod index;

pub use ffi::{
    CudaError, GpuSearchResult, IndexConfig, MetricType, TraversalHop, Vector3D,
    gpu_init, gpu_get_memory, gpu_device_synchronize,
};
pub use memory::{DeviceBuffer, PinnedBuffer};
pub use streams::{CudaStream, CudaStreamPool};
pub use cluster::{KMeans, KMeansResult, ProductQuantizer};
pub use metrics::{LatencyStats, MetricsTracker, RecallEvaluator};
pub use index::GpuVectorIndex;

/// Initialize GPU runtime and context for the current process
pub fn init_gpu(device_id: i32) -> Result<(), CudaError> {
    gpu_init(device_id)
}
