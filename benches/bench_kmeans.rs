use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::KMeans;

fn bench_parallel_kmeans(c: &mut Criterion) {
    let dim = 128;
    let num_vectors = 10_000;
    let k = 64;
    let iters = 5;

    let mut data = Vec::with_capacity(num_vectors * dim);
    for i in 0..(num_vectors * dim) {
        data.push((i as f32) * 0.001);
    }

    let mut group = c.benchmark_group("kmeans_clustering");
    group.sample_size(10);
    group.bench_function("rayon_kmeans_10k_128d_k64", |b| {
        b.iter(|| {
            let res = KMeans::fit(black_box(&data), dim, k, iters);
            black_box(res);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_parallel_kmeans);
criterion_main!(benches);
