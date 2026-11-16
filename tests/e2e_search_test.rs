// Requires a CUDA device; compiled out in CPU-only builds (FLASHVECTOR_CPU_ONLY=1).
#![cfg(not(flashvector_cpu_only))]
#![allow(clippy::needless_range_loop)]

use engine::{
    init_gpu, DeviceBuffer, GpuVectorIndex, IndexConfig, IvfQuantMode, MetadataFilter,
    MetadataIndex, MetricType, RecallEvaluator, ShardAlgorithm, ShardedGpuIndex, VectorMetadata,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use std::collections::HashSet;
use std::sync::Arc;

#[test]
fn test_e2e_hnsw_recall_target() {
    assert!(init_gpu(0).is_ok(), "GPU init failed");

    let dim = 128;
    let num_vectors = 5000;
    let num_clusters = 20;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.2).unwrap();
    let mut cluster_centers = Vec::with_capacity(num_clusters * dim);
    for _ in 0..(num_clusters * dim) {
        cluster_centers.push(rand::Rng::gen_range(&mut rng, -1.0f32..1.0f32));
    }
    let mut dataset = Vec::with_capacity(num_vectors * dim);
    for i in 0..num_vectors {
        let c_id = i % num_clusters;
        let center = &cluster_centers[c_id * dim..(c_id + 1) * dim];
        for d in 0..dim {
            dataset.push(center[d] + rand_distr::Distribution::sample(&normal, &mut rng));
        }
    }

    let config = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 32,
        ef_construction: 200,
        nlist: 32,
        m_pq: 16,
        metric: 0,
    };

    let mut index = GpuVectorIndex::new(config).expect("Index init failed");
    index.build(&dataset).expect("Build failed");

    let num_queries = 50;
    let top_k = 10;
    let mut total_hnsw_recall = 0.0f32;

    for q in 0..num_queries {
        let c_id = q % num_clusters;
        let center = &cluster_centers[c_id * dim..(c_id + 1) * dim];
        let mut query = Vec::with_capacity(dim);
        for d in 0..dim {
            query.push(center[d] + rand_distr::Distribution::sample(&normal, &mut rng));
        }

        let gt = RecallEvaluator::exact_knn(&dataset, &query, dim, top_k);
        let (hnsw_res, hops) = index
            .search_with_trajectory(&query, top_k, 128)
            .expect("HNSW search failed");
        assert!(!hops.is_empty(), "Traversal hops must be recorded");

        let r = RecallEvaluator::compute_recall(&hnsw_res, &gt);
        total_hnsw_recall += r;
    }

    let avg_recall = total_hnsw_recall / (num_queries as f32);
    println!("Average HNSW Recall@{}: {:.4}", top_k, avg_recall);
    assert!(
        avg_recall >= 0.90,
        "HNSW Recall@{} ({:.4}) below target 0.90",
        top_k,
        avg_recall
    );
}

#[test]
fn test_device_pointer_zero_copy_search() {
    assert!(init_gpu(0).is_ok(), "GPU init failed");

    let dim = 64;
    let num_vectors = 512;
    let num_queries = 4;
    let top_k = 10;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut dataset = Vec::with_capacity(num_vectors * dim);
    for _ in 0..(num_vectors * dim) {
        dataset.push(rand_distr::Distribution::sample(&normal, &mut rng));
    }
    let mut queries = Vec::with_capacity(num_queries * dim);
    for _ in 0..(num_queries * dim) {
        queries.push(rand_distr::Distribution::sample(&normal, &mut rng));
    }

    let config = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 16,
        ef_construction: 64,
        nlist: 16,
        m_pq: 8,
        metric: 0,
    };
    let mut index = GpuVectorIndex::new(config).expect("Index init failed");
    index.build(&dataset).expect("Build failed");

    let host_results = index
        .batch_search(&queries, num_queries, top_k, 64)
        .expect("Host batch search failed");

    let stream = engine::CudaStream::new(0).expect("Stream init failed");
    let mut d_queries = DeviceBuffer::new(queries.len()).expect("Device query alloc failed");
    d_queries
        .copy_from_host_async(&queries, stream.raw())
        .expect("Query upload failed");
    stream.sync().expect("Stream sync failed");

    let device_results = index
        .batch_search_device_ptr(d_queries.as_ptr(), num_queries, top_k, 64)
        .expect("Device pointer batch search failed");

    assert_eq!(host_results.len(), device_results.len());
    for (host, device) in host_results.iter().zip(device_results.iter()) {
        assert_eq!(host.id, device.id, "Zero-copy HNSW result mismatch");
        assert!(
            (host.distance - device.distance).abs() < 1e-4,
            "Zero-copy HNSW distance mismatch"
        );
    }

    let host_ivf = index
        .batch_search_ivf(&queries[..dim], 1, top_k, 8, 0)
        .expect("Host IVF search failed");
    let device_ivf = index
        .batch_search_ivf_device_ptr(d_queries.as_ptr(), 1, top_k, 8, 0)
        .expect("Device pointer IVF search failed");

    assert_eq!(host_ivf.len(), device_ivf.len());
    for (host, device) in host_ivf.iter().zip(device_ivf.iter()) {
        assert_eq!(host.id, device.id, "Zero-copy IVF result mismatch");
        assert!(
            (host.distance - device.distance).abs() < 1e-4,
            "Zero-copy IVF distance mismatch"
        );
    }
}

fn gen_clustered(
    num_vectors: usize,
    dim: usize,
    num_clusters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.18).unwrap();
    let mut centers = Vec::with_capacity(num_clusters * dim);
    for _ in 0..(num_clusters * dim) {
        centers.push(rng.gen_range(-1.0f32..1.0f32));
    }
    let mut dataset = Vec::with_capacity(num_vectors * dim);
    for i in 0..num_vectors {
        let c = i % num_clusters;
        let center = &centers[c * dim..(c + 1) * dim];
        for d in 0..dim {
            dataset.push(center[d] + rand_distr::Distribution::sample(&normal, &mut rng));
        }
    }
    (dataset, centers)
}

#[test]
fn test_cagra_recall_and_beam_steps() {
    init_gpu(0).unwrap();
    let dim = 64;
    let num_vectors = 2000;
    let (dataset, _) = gen_clustered(num_vectors, dim, 16, 101);

    let cfg = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 16,
        ef_construction: 96,
        nlist: 16,
        m_pq: 8,
        metric: MetricType::L2 as u32,
    };
    let mut idx = GpuVectorIndex::new(cfg).unwrap();
    idx.build(&dataset).unwrap();

    let num_queries = 16;
    let top_k = 10;
    let queries: Vec<f32> = dataset[..num_queries * dim]
        .iter()
        .map(|&v| v + 0.02)
        .collect();
    let batch = idx
        .batch_search_cagra(&queries, num_queries, top_k, 64, None)
        .unwrap();
    assert_eq!(batch.len(), num_queries * top_k);

    let mut recall_sum = 0.0f32;
    for qi in 0..num_queries {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let gt = RecallEvaluator::exact_knn(&dataset, q, dim, top_k);
        let row = &batch[qi * top_k..(qi + 1) * top_k];
        recall_sum += RecallEvaluator::compute_recall(row, &gt);
    }
    let avg = recall_sum / num_queries as f32;
    assert!(avg >= 0.85, "CAGRA recall@{top_k} too low: {avg:.3}");

    let trace = idx
        .search_cagra_with_debug(&queries[..dim], top_k, 48, None)
        .unwrap();
    assert!(!trace.results.is_empty());
    assert!(
        !trace.steps.is_empty(),
        "CAGRA debug steps must be recorded"
    );
    assert!(!trace.steps[0].candidates.is_empty());
}

#[test]
fn test_ivf_modes_pq_sq8_rabitq_large_rerank_and_mapped() {
    init_gpu(0).unwrap();
    // 96 is non-power-of-two, exercising RaBitQ FWHT padding (to 128) cleanly
    let dim = 96;
    let num_vectors = 1500;
    let (dataset, _) = gen_clustered(num_vectors, dim, 12, 202);

    let cfg = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 16,
        ef_construction: 64,
        nlist: 24,
        m_pq: 12,
        metric: MetricType::L2 as u32,
    };
    let mut idx = GpuVectorIndex::new(cfg).unwrap();
    idx.build(&dataset).unwrap();

    let top_k = 10;
    let queries = &dataset[0..8 * dim];
    for mode in [IvfQuantMode::Pq, IvfQuantMode::Sq8, IvfQuantMode::Rabitq] {
        let res = idx
            .batch_search_ivf_mode(queries, 8, top_k, 12, 256, mode, None)
            .unwrap();
        assert_eq!(res.len(), 8 * top_k);
        let mut recall_sum = 0.0f32;
        for qi in 0..8 {
            let gt = RecallEvaluator::exact_knn(
                &dataset,
                &queries[qi * dim..(qi + 1) * dim],
                dim,
                top_k,
            );
            recall_sum += RecallEvaluator::compute_recall(&res[qi * top_k..(qi + 1) * top_k], &gt);
        }
        let avg = recall_sum / 8.0;
        assert!(avg >= 0.80, "{mode:?} recall@{top_k} too low: {avg:.3}");
    }

    // Exercise mapped pinned Stage-2 reranking with large shortlist (rerank_k = 512)
    idx.set_mapped_rerank(true);
    let mapped_res = idx
        .batch_search_ivf_mode(&queries[..dim], 1, top_k, 16, 512, IvfQuantMode::Pq, None)
        .unwrap();
    let gt0 = RecallEvaluator::exact_knn(&dataset, &queries[..dim], dim, top_k);
    assert!(RecallEvaluator::compute_recall(&mapped_res, &gt0) >= 0.80);

    // Probe telemetry for FlashVector Studio IVF scrubbing
    let (probe_res, probes) = idx
        .search_ivf_with_debug(&queries[..dim], top_k, 8, 64, IvfQuantMode::Rabitq)
        .unwrap();
    assert_eq!(probe_res.len(), top_k);
    assert_eq!(probes.len(), 8);
    assert!(probes
        .windows(2)
        .all(|w| w[0].centroid_distance <= w[1].centroid_distance));
}

#[test]
fn test_fp16_tensor_core_search_l2_and_cosine() {
    init_gpu(0).unwrap();
    let dim = 64;
    let num_vectors = 512;
    let (dataset, _) = gen_clustered(num_vectors, dim, 8, 303);
    let queries = &dataset[..4 * dim];
    let top_k = 10;

    for metric in [MetricType::L2, MetricType::Cosine] {
        let cfg = IndexConfig {
            dim: dim as u32,
            max_elements: num_vectors as u32,
            m: 16,
            ef_construction: 64,
            nlist: 8,
            m_pq: 8,
            metric: metric as u32,
        };
        let mut idx = GpuVectorIndex::new(cfg).unwrap();
        idx.build(&dataset).unwrap();

        let res = idx.batch_search_fp16_tc(queries, 4, top_k, None).unwrap();
        assert_eq!(res.len(), 4 * top_k);
        assert_eq!(
            res[0].id, 0,
            "self-query nearest neighbor must be 0 ({metric:?})"
        );
    }
}

#[test]
fn test_on_device_bitmask_filtering_all_engines() {
    init_gpu(0).unwrap();
    let dim = 64;
    let num_vectors = 600;
    let (dataset, _) = gen_clustered(num_vectors, dim, 6, 404);

    let mut meta = MetadataIndex::new();
    for id in 0..num_vectors as u32 {
        meta.set(
            id,
            &VectorMetadata {
                tenant_id: Some(if id % 3 == 0 { "acme" } else { "other" }.to_string()),
                timestamp: Some(1_000 + id as i64),
                acl_tags: if id % 2 == 0 {
                    vec!["public".to_string()]
                } else {
                    vec!["restricted".to_string()]
                },
            },
        );
    }
    let allowed_bm = meta.evaluate(&MetadataFilter {
        tenant_id: Some("acme".to_string()),
        min_timestamp: Some(1_100),
        max_timestamp: None,
        any_acl_tags: vec!["public".to_string()],
    });
    let mask_words = allowed_bm.to_packed_u32_words(num_vectors);
    let allowed_ids: HashSet<u32> = (0..num_vectors as u32)
        .filter(|&id| allowed_bm.contains(id))
        .collect();
    assert!(!allowed_ids.is_empty());

    let cfg = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 16,
        ef_construction: 64,
        nlist: 12,
        m_pq: 8,
        metric: MetricType::L2 as u32,
    };
    let mut idx = GpuVectorIndex::new(cfg).unwrap();
    idx.build(&dataset).unwrap();

    let query = &dataset[..dim];
    let top_k = 10;

    let all_runs = vec![
        (
            "hnsw",
            idx.batch_search_filtered(query, 1, top_k, 96, Some(&mask_words))
                .unwrap(),
        ),
        (
            "cagra",
            idx.batch_search_cagra(query, 1, top_k, 96, Some(&mask_words))
                .unwrap(),
        ),
        (
            "ivf_pq",
            idx.batch_search_ivf_mode(query, 1, top_k, 12, 64, IvfQuantMode::Pq, Some(&mask_words))
                .unwrap(),
        ),
        (
            "ivf_sq8",
            idx.batch_search_ivf_mode(
                query,
                1,
                top_k,
                12,
                64,
                IvfQuantMode::Sq8,
                Some(&mask_words),
            )
            .unwrap(),
        ),
        (
            "ivf_rabitq",
            idx.batch_search_ivf_mode(
                query,
                1,
                top_k,
                12,
                64,
                IvfQuantMode::Rabitq,
                Some(&mask_words),
            )
            .unwrap(),
        ),
        (
            "fp16_tc",
            idx.batch_search_fp16_tc(query, 1, top_k, Some(&mask_words))
                .unwrap(),
        ),
    ];

    for (name, hits) in all_runs {
        let valid: Vec<u32> = hits
            .iter()
            .filter(|h| h.id != u32::MAX)
            .map(|h| h.id)
            .collect();
        assert!(!valid.is_empty(), "{name}: expected filtered results");
        for id in valid {
            assert!(
                allowed_ids.contains(&id),
                "{name} returned masked-out id {id}"
            );
        }
    }
}

#[test]
fn test_mmap_snapshot_tombstones_delta_and_compaction() {
    init_gpu(0).unwrap();
    let dim = 32;
    let initial = 400;
    let (dataset, _) = gen_clustered(initial, dim, 4, 505);

    let cfg = IndexConfig {
        dim: dim as u32,
        max_elements: initial as u32,
        m: 16,
        ef_construction: 64,
        nlist: 8,
        m_pq: 8,
        metric: MetricType::L2 as u32,
    };
    let mut idx = GpuVectorIndex::new(cfg).unwrap();
    idx.build(&dataset).unwrap();

    // Add distinct marker vector in delta buffer
    let marker_vec = vec![50.0f32; dim];
    let delta_ids = idx.add_vectors(&marker_vec).unwrap();
    assert_eq!(delta_ids, vec![initial as u32]);

    // Delta vector must appear at rank 0 on main search
    let delta_hit = idx.batch_search(&marker_vec, 1, 5, 32).unwrap();
    assert_eq!(delta_hit[0].id, delta_ids[0]);

    // Delete vector 0 and confirm no engine returns id 0
    assert_eq!(idx.delete_ids(&[0]).unwrap(), 1);
    let q0 = &dataset[..dim];
    let hnsw_after_del = idx.batch_search(q0, 1, 10, 64).unwrap();
    assert!(hnsw_after_del.iter().all(|r| r.id != 0));
    let ivf_after_del = idx
        .batch_search_ivf_mode(q0, 1, 10, 8, 32, IvfQuantMode::Rabitq, None)
        .unwrap();
    assert!(ivf_after_del.iter().all(|r| r.id != 0));

    // Roundtrip mmap snapshot BEFORE compaction (preserves delta & tombstone)
    let mut snap_path = std::env::temp_dir();
    snap_path.push(format!("flashvector_e2e_snap_{}.fv", std::process::id()));
    idx.save_snapshot(&snap_path).unwrap();
    let loaded = GpuVectorIndex::load_snapshot(&snap_path).unwrap();
    let _ = std::fs::remove_file(&snap_path);

    let loaded_delta = loaded.batch_search(&marker_vec, 1, 5, 32).unwrap();
    assert_eq!(loaded_delta[0].id, delta_ids[0]);
    let loaded_del = loaded.batch_search_cagra(q0, 1, 10, 64, None).unwrap();
    assert!(loaded_del.iter().all(|r| r.id != 0));

    // Background compaction preserving surviving logical IDs and clearing tombstones/delta
    let shared = Arc::new(parking_lot::RwLock::new(loaded));
    let handle = GpuVectorIndex::compact_in_background(Arc::clone(&shared));
    handle.join().unwrap().unwrap();
    let compacted = shared.read();
    assert_eq!(compacted.tombstone_count(), 0);
    assert_eq!(compacted.delta_len(), 0);
    assert_eq!(compacted.num_vectors(), initial); // initial + 1 added - 1 deleted
    let post_compact = compacted.batch_search(&marker_vec, 1, 5, 32).unwrap();
    assert_eq!(post_compact[0].id, delta_ids[0]);
}

#[test]
fn test_sparse_csr_hybrid_rrf_and_sharded_merge_topk() {
    init_gpu(0).unwrap();
    let dim = 32;
    let num_vectors = 450;
    let (dataset, _) = gen_clustered(num_vectors, dim, 5, 606);

    let cfg = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 16,
        ef_construction: 64,
        nlist: 9,
        m_pq: 8,
        metric: MetricType::L2 as u32,
    };
    let mut idx = GpuVectorIndex::new(cfg).unwrap();
    idx.build(&dataset).unwrap();

    // Sparse CSR doc 42 carries unique high-weight token 9999
    let mut rows: Vec<Vec<(u32, f32)>> = (0..num_vectors)
        .map(|i| vec![((i % 7) as u32, 0.25)])
        .collect();
    rows[42].push((9999, 10.0));
    idx.build_sparse(&rows).unwrap();

    let sparse_hits = idx.search_sparse(&[&[(9999, 2.0)]], 5, None).unwrap();
    assert_eq!(sparse_hits[0][0].id, 42);

    let fused = idx
        .hybrid_search(
            &dataset[42 * dim..43 * dim],
            &[&[(9999, 2.0)]],
            5,
            32,
            48,
            60.0,
            1.0,
            1.0,
            None,
        )
        .unwrap();
    assert_eq!(fused[0][0].id, 42);
    assert!(fused[0]
        .windows(2)
        .all(|w| w[0].rrf_score >= w[1].rrf_score));

    // Sharded multi-shard index + device-side cuda_merge_topk
    let sharded = ShardedGpuIndex::build(cfg, &dataset, Some(3)).unwrap();
    assert_eq!(sharded.num_shards(), 3);
    assert_eq!(sharded.total_vectors(), num_vectors);

    let q_idx = 317usize; // resides in shard 2
    let query = &dataset[q_idx * dim..(q_idx + 1) * dim];
    for alg in [
        ShardAlgorithm::Hnsw,
        ShardAlgorithm::Cagra,
        ShardAlgorithm::IvfPq,
    ] {
        let shard_res = sharded.batch_search(query, 1, 10, 64, 6, alg).unwrap();
        assert_eq!(shard_res.len(), 10);
        assert!(shard_res.iter().any(|r| r.id == q_idx as u32));
        assert!(shard_res.windows(2).all(|w| w[0].distance <= w[1].distance));
    }
}
