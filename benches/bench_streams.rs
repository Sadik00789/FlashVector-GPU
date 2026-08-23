use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::{init_gpu, CudaStreamPool, DeviceBuffer, PinnedBuffer};

fn bench_stream_transfers(c: &mut Criterion) {
    let _ = init_gpu(0);
    let pool = CudaStreamPool::new(4).unwrap();
    let size = 1024 * 1024; // 1M floats = 4MB
    let data = vec![1.0f32; size];

    let h_buf = PinnedBuffer::from_slice(&data).unwrap();
    let mut d_buf = DeviceBuffer::new(size).unwrap();
    let mut h_out = PinnedBuffer::new(size).unwrap();

    let mut group = c.benchmark_group("cuda_streams");
    group.bench_function("pinned_h2d_d2h_async_4mb", |b| {
        b.iter(|| {
            let stream = pool.get_stream();
            d_buf.copy_from_pinned_async(black_box(&h_buf), stream.raw()).unwrap();
            d_buf.copy_to_pinned_async(black_box(&mut h_out), stream.raw()).unwrap();
            stream.sync().unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, bench_stream_transfers);
criterion_main!(benches);
