pub mod cluster;
pub mod ffi;
pub mod graph;
pub mod index;
pub mod memory;
pub mod metrics;
pub mod streams;

pub use cluster::{KMeans, KMeansResult, ProductQuantizer};
pub use ffi::{
    gpu_device_synchronize, gpu_get_memory, gpu_init, CudaError, GpuSearchResult, IndexConfig,
    MetricType, TraversalHop, Vector3D,
};
pub use index::{ClusterPqDistortion, GpuVectorIndex};
pub use memory::{DeviceBuffer, PinnedBuffer};
pub use metrics::{LatencyStats, MetricsTracker, RecallEvaluator};
pub use streams::{CudaStream, CudaStreamPool};

/// Initialize GPU runtime and context for the current process
pub fn init_gpu(device_id: i32) -> Result<(), CudaError> {
    gpu_init(device_id)
}
