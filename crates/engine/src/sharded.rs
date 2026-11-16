//! Multi-GPU (and multi-shard) fan-out index with NVLink/P2P peer transfer
//! and device-side top-k reduction (`cuda_merge_topk`).

use std::time::Instant;

use crate::ffi::{
    gpu_device_count, gpu_enable_peer_access, gpu_merge_topk, gpu_set_device, CudaError,
    GpuSearchResult, IndexConfig,
};
use crate::index::{GpuVectorIndex, IvfQuantMode};
use crate::memory::DeviceBuffer;
use crate::streams::CudaStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardAlgorithm {
    Hnsw,
    Cagra,
    IvfPq,
    IvfSq8,
    IvfRabitq,
    Fp16Tc,
}

struct GpuShard {
    device_id: i32,
    id_offset: usize,
    index: GpuVectorIndex,
}

pub struct ShardedGpuIndex {
    shards: Vec<GpuShard>,
    coordinator_device: i32,
    p2p_enabled: Vec<bool>,
    dim: usize,
    total_vectors: usize,
    merge_stream: CudaStream,
}

impl ShardedGpuIndex {
    /// Build a sharded index across all available GPUs (or `num_shards` logical
    /// shards when explicitly requested for single-GPU partitioning / testing).
    pub fn build(
        config: IndexConfig,
        dataset: &[f32],
        num_shards_override: Option<usize>,
    ) -> Result<Self, CudaError> {
        let dim = config.dim as usize;
        if dim == 0 || dataset.is_empty() || !dataset.len().is_multiple_of(dim) {
            return Err(CudaError::InvalidArgument);
        }

        let total_vectors = dataset.len() / dim;
        let detected = gpu_device_count().unwrap_or(1).max(1);
        let num_shards = num_shards_override
            .unwrap_or(detected)
            .clamp(1, total_vectors);

        let coordinator_device = 0i32;
        let mut p2p_enabled = Vec::with_capacity(num_shards);
        let mut shards = Vec::with_capacity(num_shards);

        let base_chunk = total_vectors / num_shards;
        let rem = total_vectors % num_shards;
        let mut offset = 0usize;

        for s in 0..num_shards {
            let count = base_chunk + usize::from(s < rem);
            let dev_id = (s % detected) as i32;
            let _ = gpu_set_device(dev_id);

            let peer_ok = if dev_id == coordinator_device {
                true
            } else {
                gpu_enable_peer_access(coordinator_device, dev_id).is_ok()
                    && gpu_enable_peer_access(dev_id, coordinator_device).is_ok()
            };
            p2p_enabled.push(peer_ok);

            let mut shard_cfg = config;
            shard_cfg.max_elements = count as u32;
            shard_cfg.nlist = (config.nlist as usize).min(count).max(1) as u32;

            let mut index = GpuVectorIndex::new_with_streams(shard_cfg, 1)?;
            let slice = &dataset[offset * dim..(offset + count) * dim];
            index.build(slice)?;

            shards.push(GpuShard {
                device_id: dev_id,
                id_offset: offset,
                index,
            });
            offset += count;
        }

        let _ = gpu_set_device(coordinator_device);
        let merge_stream = CudaStream::new(0)?;

        Ok(Self {
            shards,
            coordinator_device,
            p2p_enabled,
            dim,
            total_vectors,
            merge_stream,
        })
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn total_vectors(&self) -> usize {
        self.total_vectors
    }

    pub fn p2p_status(&self) -> &[bool] {
        &self.p2p_enabled
    }

    /// Fan out queries across every GPU shard, gather shard top-k candidates to
    /// the coordinator GPU (using P2P/NVLink where supported), and execute
    /// device-side top-k reduction via `cuda_merge_topk`.
    pub fn batch_search(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        ef_search: usize,
        nprobe: usize,
        algorithm: ShardAlgorithm,
    ) -> Result<Vec<GpuSearchResult>, CudaError> {
        if num_queries == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        let _start = Instant::now();
        let num_shards = self.shards.len();
        let mut gathered = vec![GpuSearchResult::default(); num_shards * num_queries * top_k];

        // 1. Shard fan-out
        for (s_idx, shard) in self.shards.iter().enumerate() {
            let _ = gpu_set_device(shard.device_id);
            let mut local_res = match algorithm {
                ShardAlgorithm::Hnsw => {
                    shard
                        .index
                        .batch_search(queries, num_queries, top_k, ef_search)?
                }
                ShardAlgorithm::Cagra => {
                    shard
                        .index
                        .batch_search_cagra(queries, num_queries, top_k, ef_search, None)?
                }
                ShardAlgorithm::IvfPq => shard.index.batch_search_ivf_mode(
                    queries,
                    num_queries,
                    top_k,
                    nprobe,
                    0,
                    IvfQuantMode::Pq,
                    None,
                )?,
                ShardAlgorithm::IvfSq8 => shard.index.batch_search_ivf_mode(
                    queries,
                    num_queries,
                    top_k,
                    nprobe,
                    0,
                    IvfQuantMode::Sq8,
                    None,
                )?,
                ShardAlgorithm::IvfRabitq => shard.index.batch_search_ivf_mode(
                    queries,
                    num_queries,
                    top_k,
                    nprobe,
                    0,
                    IvfQuantMode::Rabitq,
                    None,
                )?,
                ShardAlgorithm::Fp16Tc => {
                    shard
                        .index
                        .batch_search_fp16_tc(queries, num_queries, top_k, None)?
                }
            };

            let offset = shard.id_offset as u32;
            for r in &mut local_res {
                if r.id != u32::MAX {
                    r.id += offset;
                }
            }

            let dst_base = s_idx * num_queries * top_k;
            gathered[dst_base..dst_base + local_res.len()].copy_from_slice(&local_res);
        }

        // 2. Device-side final top-k reduction on coordinator GPU
        let _ = gpu_set_device(self.coordinator_device);
        let mut d_shards = DeviceBuffer::new(gathered.len())?;
        d_shards.copy_from_host_async(&gathered, self.merge_stream.raw())?;

        let mut d_out = DeviceBuffer::new(num_queries * top_k)?;
        gpu_merge_topk(
            d_shards.as_ptr(),
            num_shards as u32,
            num_queries as u32,
            top_k as u32,
            top_k as u32,
            d_out.as_mut_ptr(),
            self.merge_stream.raw(),
        )?;

        let mut final_res = vec![GpuSearchResult::default(); num_queries * top_k];
        d_out.copy_to_host_async(&mut final_res, self.merge_stream.raw())?;
        self.merge_stream.sync()?;

        Ok(final_res)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}
