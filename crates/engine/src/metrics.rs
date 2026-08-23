use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use crate::ffi::GpuSearchResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: u64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub qps: f64,
}

pub struct MetricsTracker {
    latencies_us: Mutex<Vec<f64>>,
    total_queries: AtomicU64,
    start_time: Instant,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            latencies_us: Mutex::new(Vec::with_capacity(100_000)),
            total_queries: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_query(&self, duration: Duration, batch_size: usize) {
        let us = duration.as_secs_f64() * 1_000_000.0 / (batch_size.max(1) as f64);
        let mut lats = self.latencies_us.lock();
        if lats.len() >= 100_000 {
            lats.drain(0..50_000);
        }
        lats.push(us);
        self.total_queries.fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> LatencyStats {
        let mut lats = self.latencies_us.lock().clone();
        let count = self.total_queries.load(Ordering::Relaxed);
        let elapsed_secs = self.start_time.elapsed().as_secs_f64().max(0.001);
        let qps = (count as f64) / elapsed_secs;

        if lats.is_empty() {
            return LatencyStats {
                count,
                p50_us: 0.0,
                p90_us: 0.0,
                p99_us: 0.0,
                p999_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                mean_us: 0.0,
                qps,
            };
        }

        lats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = lats.len();
        let min_us = lats[0];
        let max_us = lats[len - 1];
        let sum: f64 = lats.iter().sum();
        let mean_us = sum / (len as f64);

        let p50_idx = ((len as f64) * 0.50) as usize;
        let p90_idx = ((len as f64) * 0.90) as usize;
        let p99_idx = ((len as f64) * 0.99) as usize;
        let p999_idx = ((len as f64) * 0.999) as usize;

        LatencyStats {
            count,
            p50_us: lats[p50_idx.min(len - 1)],
            p90_us: lats[p90_idx.min(len - 1)],
            p99_us: lats[p99_idx.min(len - 1)],
            p999_us: lats[p999_idx.min(len - 1)],
            min_us,
            max_us,
            mean_us,
            qps,
        }
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RecallEvaluator;

impl RecallEvaluator {
    /// Exact CPU brute-force top-k calculation
    pub fn exact_knn(
        dataset: &[f32],
        query: &[f32],
        dim: usize,
        top_k: usize,
    ) -> Vec<GpuSearchResult> {
        let num_vectors = dataset.len() / dim;
        let mut dists: Vec<GpuSearchResult> = (0..num_vectors)
            .map(|i| {
                let vec = &dataset[i * dim..(i + 1) * dim];
                let dist: f32 = query
                    .iter()
                    .zip(vec.iter())
                    .map(|(&q, &v)| {
                        let d = q - v;
                        d * d
                    })
                    .sum();
                GpuSearchResult {
                    id: i as u32,
                    distance: dist,
                }
            })
            .collect();

        dists.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(top_k);
        dists
    }

    /// Compute Recall@k between approximate results and ground truth
    pub fn compute_recall(approx: &[GpuSearchResult], ground_truth: &[GpuSearchResult]) -> f32 {
        if ground_truth.is_empty() || approx.is_empty() {
            return 0.0;
        }

        let gt_set: HashSet<u32> = ground_truth.iter().map(|r| r.id).collect();
        let matched = approx.iter().filter(|r| gt_set.contains(&r.id)).count();

        (matched as f32) / (ground_truth.len() as f32)
    }
}
