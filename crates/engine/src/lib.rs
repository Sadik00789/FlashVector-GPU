// Numeric kernels index several parallel buffers by the same loop variable, so
// index loops are clearer than zipped iterator chains here.
#![allow(clippy::needless_range_loop)]

pub mod cluster;
pub mod ffi;
pub mod filter;
pub mod graph;
pub mod index;
pub mod memory;
pub mod metrics;
pub mod persistence;
pub mod quant;
pub mod sharded;
pub mod sparse;
pub mod streams;

pub use cluster::{KMeans, ProductQuantizer};
pub use ffi::{
    gpu_device_count, gpu_device_synchronize, gpu_get_memory, gpu_init, gpu_set_device,
    BeamCandidateRecord, BeamStepRecord, CudaError, GpuSearchResult, IndexConfig, MetricType,
    TraversalHop, Vector3D,
};
pub use filter::{MetadataFilter, MetadataIndex, RoaringBitset, VectorMetadata};
pub use index::{
    host_merge_topk, BeamDebugTrace, ClusterPqDistortion, GpuVectorIndex, IvfProbeTelemetry,
    IvfQuantMode,
};
pub use memory::{DeviceBuffer, PinnedBuffer};
pub use metrics::{LatencyStats, MetricsTracker, RecallEvaluator};
pub use persistence::{MmapSnapshot, SnapshotManifest};
pub use quant::{encode_fp16_padded, RabitqEncoder, ScalarQuantizer8};
pub use sharded::{ShardAlgorithm, ShardedGpuIndex};
pub use sparse::{cpu_rrf_fuse, HybridHit, SparseCsrData, SparseHit};
pub use streams::{CudaStream, CudaStreamPool};

/// Initialize GPU runtime and context for the current process
pub fn init_gpu(device_id: i32) -> Result<(), CudaError> {
    gpu_init(device_id)
}

/// `true` when built with `FLASHVECTOR_CPU_ONLY=1`: the CUDA bridge is a host-memory
/// stub, so device discovery and every kernel dispatch return an error.
pub const CPU_ONLY: bool = cfg!(flashvector_cpu_only);
