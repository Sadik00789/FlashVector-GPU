use engine::{GpuSearchResult, GpuVectorIndex, IvfQuantMode};
use parking_lot::RwLock;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchAlgorithm {
    Hnsw,
    Cagra,
    IvfPq,
    IvfSq8,
    IvfRabitq,
}

impl SearchAlgorithm {
    pub fn from_request(algorithm: Option<&str>, use_ivf: Option<bool>) -> Self {
        if let Some(raw) = algorithm {
            match raw.trim().to_ascii_lowercase().as_str() {
                "cagra" => return Self::Cagra,
                "ivf_pq" | "ivf-pq" | "ivf" | "pq" => return Self::IvfPq,
                "ivf_sq8" | "ivf-sq8" | "sq8" => return Self::IvfSq8,
                "ivf_rabitq" | "ivf-rabitq" | "rabitq" => return Self::IvfRabitq,
                _ => return Self::Hnsw,
            }
        }
        if use_ivf.unwrap_or(false) {
            Self::IvfPq
        } else {
            Self::Hnsw
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::Cagra => "cagra",
            Self::IvfPq => "ivf_pq",
            Self::IvfSq8 => "ivf_sq8",
            Self::IvfRabitq => "ivf_rabitq",
        }
    }

    pub fn ivf_mode(self) -> Option<IvfQuantMode> {
        match self {
            Self::IvfPq => Some(IvfQuantMode::Pq),
            Self::IvfSq8 => Some(IvfQuantMode::Sq8),
            Self::IvfRabitq => Some(IvfQuantMode::Rabitq),
            Self::Hnsw | Self::Cagra => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchKey {
    pub algorithm: SearchAlgorithm,
    pub top_k: usize,
    pub ef_or_nprobe: usize,
    pub rerank_k: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BatcherStats {
    pub total_requests: u64,
    pub total_batches: u64,
    pub avg_batch_size: f64,
}

struct QueuedQuery {
    key: BatchKey,
    query: Vec<f32>,
    reply: oneshot::Sender<Result<(Vec<GpuSearchResult>, f64), String>>,
}

type BatchExecutor =
    dyn Fn(BatchKey, &[f32], usize) -> Result<Vec<GpuSearchResult>, String> + Send + Sync;

/// Sub-millisecond dynamic micro-batching queue that coalesces concurrent HTTP and
/// WebSocket queries sharing search parameters into single batched GPU kernel launches.
pub struct SearchBatcher {
    tx: mpsc::UnboundedSender<QueuedQuery>,
    total_requests: Arc<AtomicU64>,
    total_batches: Arc<AtomicU64>,
}

impl SearchBatcher {
    pub fn for_index(index: Arc<RwLock<GpuVectorIndex>>) -> Self {
        Self::with_executor(
            Duration::from_micros(350),
            64,
            Arc::new(move |key, queries, num_queries| {
                let guard = index.read();
                let res = match key.algorithm {
                    SearchAlgorithm::Hnsw => {
                        guard.batch_search(queries, num_queries, key.top_k, key.ef_or_nprobe)
                    }
                    SearchAlgorithm::Cagra => guard.batch_search_cagra(
                        queries,
                        num_queries,
                        key.top_k,
                        key.ef_or_nprobe,
                        None,
                    ),
                    SearchAlgorithm::IvfPq
                    | SearchAlgorithm::IvfSq8
                    | SearchAlgorithm::IvfRabitq => {
                        let mode = key.algorithm.ivf_mode().unwrap_or(IvfQuantMode::Pq);
                        guard.batch_search_ivf_mode(
                            queries,
                            num_queries,
                            key.top_k,
                            key.ef_or_nprobe,
                            key.rerank_k,
                            mode,
                            None,
                        )
                    }
                };
                res.map_err(|e| format!("{e:?}"))
            }),
        )
    }

    pub fn with_executor(
        max_wait: Duration,
        max_batch_size: usize,
        executor: Arc<BatchExecutor>,
    ) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<QueuedQuery>();
        let total_requests = Arc::new(AtomicU64::new(0));
        let total_batches = Arc::new(AtomicU64::new(0));
        let req_ctr = Arc::clone(&total_requests);
        let batch_ctr = Arc::clone(&total_batches);
        let max_bs = max_batch_size.max(1);

        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                let mut items = Vec::with_capacity(max_bs);
                items.push(first);
                let deadline = tokio::time::Instant::now() + max_wait;

                while items.len() < max_bs {
                    match rx.try_recv() {
                        Ok(next) => items.push(next),
                        Err(_) => {
                            let rem =
                                deadline.saturating_duration_since(tokio::time::Instant::now());
                            if rem.is_zero() {
                                break;
                            }
                            match tokio::time::timeout(rem, rx.recv()).await {
                                Ok(Some(next)) => items.push(next),
                                _ => break,
                            }
                        }
                    }
                }

                // Partition by compatible kernel launch parameters
                let mut groups: Vec<(BatchKey, Vec<QueuedQuery>)> = Vec::new();
                for item in items {
                    if let Some((_, bucket)) = groups.iter_mut().find(|(k, _)| *k == item.key) {
                        bucket.push(item);
                    } else {
                        let key = item.key;
                        groups.push((key, vec![item]));
                    }
                }

                for (key, bucket) in groups {
                    let num_queries = bucket.len();
                    let dim = bucket[0].query.len();
                    let mut flat = Vec::with_capacity(num_queries * dim);
                    let mut senders = Vec::with_capacity(num_queries);
                    for q in bucket {
                        flat.extend_from_slice(&q.query);
                        senders.push(q.reply);
                    }

                    req_ctr.fetch_add(num_queries as u64, Ordering::Relaxed);
                    batch_ctr.fetch_add(1, Ordering::Relaxed);

                    let exec = Arc::clone(&executor);
                    let exec_res = tokio::task::spawn_blocking(move || {
                        let t0 = Instant::now();
                        let res = exec(key, &flat, num_queries);
                        let latency_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                        (res, latency_us)
                    })
                    .await;

                    match exec_res {
                        Ok((Ok(flat_results), latency_us)) => {
                            let k = key.top_k;
                            for (qi, tx_reply) in senders.into_iter().enumerate() {
                                let start = qi * k;
                                let end = ((qi + 1) * k).min(flat_results.len());
                                let slice = if start < flat_results.len() {
                                    flat_results[start..end].to_vec()
                                } else {
                                    Vec::new()
                                };
                                let _ = tx_reply.send(Ok((slice, latency_us)));
                            }
                        }
                        Ok((Err(err), _)) => {
                            for tx_reply in senders {
                                let _ = tx_reply.send(Err(err.clone()));
                            }
                        }
                        Err(join_err) => {
                            let msg = format!("batch worker join error: {join_err}");
                            for tx_reply in senders {
                                let _ = tx_reply.send(Err(msg.clone()));
                            }
                        }
                    }
                }
            }
        });

        Self {
            tx,
            total_requests,
            total_batches,
        }
    }

    pub async fn submit(
        &self,
        key: BatchKey,
        query: Vec<f32>,
    ) -> Result<(Vec<GpuSearchResult>, f64), String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(QueuedQuery {
                key,
                query,
                reply: reply_tx,
            })
            .map_err(|_| "search batcher shut down".to_string())?;
        reply_rx
            .await
            .map_err(|_| "search batcher dropped reply".to_string())?
    }

    pub fn stats(&self) -> BatcherStats {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_batches = self.total_batches.load(Ordering::Relaxed);
        let avg_batch_size = if total_batches == 0 {
            0.0
        } else {
            total_requests as f64 / total_batches as f64
        };
        BatcherStats {
            total_requests,
            total_batches,
            avg_batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[tokio::test]
    async fn test_micro_batcher_coalesces_concurrent_queries() {
        let batch_calls = Arc::new(AtomicUsize::new(0));
        let calls_ref = Arc::clone(&batch_calls);
        let batcher = Arc::new(SearchBatcher::with_executor(
            Duration::from_millis(5),
            32,
            Arc::new(move |key, queries, num_queries| {
                calls_ref.fetch_add(1, Ordering::Relaxed);
                let dim = queries.len() / num_queries;
                let mut out = Vec::with_capacity(num_queries * key.top_k);
                for qi in 0..num_queries {
                    let tag = queries[qi * dim] as u32;
                    for r in 0..key.top_k {
                        out.push(GpuSearchResult {
                            id: tag * 100 + r as u32,
                            distance: r as f32,
                        });
                    }
                }
                Ok(out)
            }),
        ));

        let key = BatchKey {
            algorithm: SearchAlgorithm::Cagra,
            top_k: 4,
            ef_or_nprobe: 32,
            rerank_k: 0,
        };

        let mut handles = Vec::new();
        for qid in 0..8u32 {
            let b = Arc::clone(&batcher);
            handles.push(tokio::spawn(async move {
                let (res, _) = b.submit(key, vec![qid as f32; 4]).await.unwrap();
                assert_eq!(res.len(), 4);
                assert_eq!(res[0].id, qid * 100);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        let stats = batcher.stats();
        assert_eq!(stats.total_requests, 8);
        assert!(
            stats.total_batches < 8,
            "expected coalesced batches < 8, got {}",
            stats.total_batches
        );
        assert_eq!(
            batch_calls.load(Ordering::Relaxed) as u64,
            stats.total_batches
        );
    }

    #[tokio::test]
    async fn test_micro_batcher_separates_distinct_keys() {
        let batcher = Arc::new(SearchBatcher::with_executor(
            Duration::from_millis(2),
            16,
            Arc::new(|key, _queries, num_queries| {
                Ok(vec![
                    GpuSearchResult {
                        id: key.top_k as u32,
                        distance: 0.5,
                    };
                    num_queries * key.top_k
                ])
            }),
        ));

        let k1 = BatchKey {
            algorithm: SearchAlgorithm::Hnsw,
            top_k: 2,
            ef_or_nprobe: 64,
            rerank_k: 0,
        };
        let k2 = BatchKey {
            algorithm: SearchAlgorithm::IvfRabitq,
            top_k: 5,
            ef_or_nprobe: 8,
            rerank_k: 32,
        };

        let b1 = Arc::clone(&batcher);
        let b2 = Arc::clone(&batcher);
        let (r1, r2) = tokio::join!(b1.submit(k1, vec![1.0; 4]), b2.submit(k2, vec![2.0; 4]));
        assert_eq!(r1.unwrap().0.len(), 2);
        assert_eq!(r2.unwrap().0.len(), 5);
    }
}
