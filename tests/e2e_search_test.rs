use engine::{init_gpu, DeviceBuffer, GpuVectorIndex, IndexConfig, RecallEvaluator};
use rand_distr::Normal;

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
        nbits_pq: 8,
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
        nbits_pq: 8,
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
        .batch_search_ivf(&queries[..dim], 1, top_k, 8)
        .expect("Host IVF search failed");
    let device_ivf = index
        .batch_search_ivf_device_ptr(d_queries.as_ptr(), 1, top_k, 8)
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
