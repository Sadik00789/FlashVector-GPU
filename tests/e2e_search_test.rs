use engine::{
    init_gpu, GpuVectorIndex, IndexConfig, RecallEvaluator,
};
use rand_distr::{Distribution, Normal};

#[test]
fn test_e2e_hnsw_and_ivf_recall() {
    assert!(init_gpu(0).is_ok(), "GPU init failed");

    let dim = 128;
    let num_vectors = 2000;
    let num_clusters = 10;

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
            dataset.push(center[d] + normal.sample(&mut rng));
        }
    }

    let config = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 32,
        ef_construction: 128,
        nlist: 16,
        m_pq: 16,
        nbits_pq: 8,
        metric: 0,
    };

    let mut index = GpuVectorIndex::new(config).expect("Index init failed");
    index.build(&dataset).expect("Build failed");

    // Test 10 random query searches
    let mut total_hnsw_recall = 0.0f32;
    let mut total_ivf_recall = 0.0f32;
    let num_queries = 10;
    let top_k = 10;

    for q in 0..num_queries {
        let c_id = q % num_clusters;
        let center = &cluster_centers[c_id * dim..(c_id + 1) * dim];
        let mut query = Vec::with_capacity(dim);
        for d in 0..dim {
            query.push(center[d] + normal.sample(&mut rng));
        }

        // Ground Truth
        let gt = RecallEvaluator::exact_knn(&dataset, &query, dim, top_k);

        // HNSW Search with Trajectory
        let (hnsw_res, hops) = index
            .search_with_trajectory(&query, top_k, 64)
            .expect("HNSW search failed");

        assert!(!hnsw_res.is_empty());
        assert!(!hops.is_empty(), "Trajectory hops must be recorded");

        let r_hnsw = RecallEvaluator::compute_recall(&hnsw_res, &gt);
        total_hnsw_recall += r_hnsw;

        // IVF-PQ Search
        let ivf_res = index.search_ivf_pq(&query, top_k, 8).expect("IVF search failed");
        assert!(!ivf_res.is_empty());

        let r_ivf = RecallEvaluator::compute_recall(&ivf_res, &gt);
        total_ivf_recall += r_ivf;
    }

    let avg_hnsw_recall = total_hnsw_recall / (num_queries as f32);
    let avg_ivf_recall = total_ivf_recall / (num_queries as f32);

    println!("Average HNSW Recall@10: {:.3}", avg_hnsw_recall);
    println!("Average IVF-PQ Recall@10: {:.3}", avg_ivf_recall);

    assert!(
        avg_hnsw_recall >= 0.80,
        "HNSW Recall@10 ({:.3}) should meet target >= 0.80",
        avg_hnsw_recall
    );
}
