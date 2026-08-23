use engine::{
    ffi::{gpu_bitonic_sort_test, gpu_compute_distances_warp, MetricType},
    init_gpu, DeviceBuffer, PinnedBuffer,
};

#[test]
fn test_gpu_initialization_and_memory() {
    assert!(init_gpu(0).is_ok(), "CUDA initialization must succeed on device 0");
    let (free_b, total_b) = engine::gpu_get_memory().expect("Should read GPU memory");
    println!("GPU VRAM - Free: {} MB, Total: {} MB", free_b / (1024 * 1024), total_b / (1024 * 1024));
    assert!(total_b > 0);
}

#[test]
fn test_pinned_device_memory_transfers() {
    init_gpu(0).unwrap();
    let stream = engine::CudaStream::new(0).unwrap();

    let data = vec![1.0f32, 2.5, -3.14, 42.0, 100.5];
    let h_buf = PinnedBuffer::from_slice(&data).unwrap();
    let mut d_buf = DeviceBuffer::new(data.len()).unwrap();
    let mut h_out = PinnedBuffer::new(data.len()).unwrap();

    d_buf.copy_from_pinned_async(&h_buf, stream.raw()).unwrap();
    d_buf.copy_to_pinned_async(&mut h_out, stream.raw()).unwrap();
    stream.sync().unwrap();

    for i in 0..data.len() {
        assert_eq!(h_out[i], data[i]);
    }
}

#[test]
fn test_warp_distance_reduction_correctness() {
    init_gpu(0).unwrap();
    let stream = engine::CudaStream::new(0).unwrap();

    let dim = 128;
    let num_queries = 4;
    let num_vectors = 8;

    let mut queries = Vec::with_capacity(num_queries * dim);
    let mut dataset = Vec::with_capacity(num_vectors * dim);

    for i in 0..(num_queries * dim) {
        queries.push((i as f32) * 0.01);
    }
    for i in 0..(num_vectors * dim) {
        dataset.push((i as f32) * 0.02);
    }

    let mut d_q = DeviceBuffer::new(queries.len()).unwrap();
    let mut d_v = DeviceBuffer::new(dataset.len()).unwrap();
    let mut d_out = DeviceBuffer::new(num_queries * num_vectors).unwrap();

    d_q.copy_from_host_async(&queries, stream.raw()).unwrap();
    d_v.copy_from_host_async(&dataset, stream.raw()).unwrap();

    gpu_compute_distances_warp(
        d_q.as_ptr(),
        d_v.as_ptr(),
        num_queries as u32,
        num_vectors as u32,
        dim as u32,
        MetricType::L2,
        d_out.as_mut_ptr(),
        stream.raw(),
    ).unwrap();

    let mut gpu_dists = vec![0.0f32; num_queries * num_vectors];
    d_out.copy_to_host_async(&mut gpu_dists, stream.raw()).unwrap();
    stream.sync().unwrap();

    // Verify against CPU floating-point baseline with relative error tolerance
    for q in 0..num_queries {
        for v in 0..num_vectors {
            let q_vec = &queries[q * dim..(q + 1) * dim];
            let v_vec = &dataset[v * dim..(v + 1) * dim];

            let cpu_dist: f32 = q_vec.iter().zip(v_vec.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
            let gpu_dist = gpu_dists[q * num_vectors + v];

            let rel_diff = (cpu_dist - gpu_dist).abs() / cpu_dist.max(1.0);
            assert!(
                rel_diff < 1e-4,
                "Relative distance error too high at q={}, v={}: CPU={}, GPU={}, rel_diff={}",
                q, v, cpu_dist, gpu_dist, rel_diff
            );
        }
    }
}

#[test]
fn test_bitonic_sort_correctness() {
    init_gpu(0).unwrap();
    let stream = engine::CudaStream::new(0).unwrap();

    let keys_in = vec![45.2f32, 1.2, 99.0, 3.14, 0.5, 12.8, 88.3, 7.7];
    let vals_in = vec![0u32, 1, 2, 3, 4, 5, 6, 7];

    let mut d_k_in = DeviceBuffer::new(keys_in.len()).unwrap();
    let mut d_v_in = DeviceBuffer::new(vals_in.len()).unwrap();
    let mut d_k_out = DeviceBuffer::new(keys_in.len()).unwrap();
    let mut d_v_out = DeviceBuffer::new(vals_in.len()).unwrap();

    d_k_in.copy_from_host_async(&keys_in, stream.raw()).unwrap();
    d_v_in.copy_from_host_async(&vals_in, stream.raw()).unwrap();

    gpu_bitonic_sort_test(
        d_k_in.as_ptr(),
        d_v_in.as_ptr(),
        d_k_out.as_mut_ptr(),
        d_v_out.as_mut_ptr(),
        keys_in.len() as u32,
        stream.raw(),
    ).unwrap();

    let mut keys_out = vec![0.0f32; keys_in.len()];
    let mut vals_out = vec![0u32; vals_in.len()];

    d_k_out.copy_to_host_async(&mut keys_out, stream.raw()).unwrap();
    d_v_out.copy_to_host_async(&mut vals_out, stream.raw()).unwrap();
    stream.sync().unwrap();

    // Verify sorted ascending
    for i in 1..keys_in.len() {
        assert!(keys_out[i] >= keys_out[i - 1], "Bitonic sort failed to sort ascending");
    }
}
