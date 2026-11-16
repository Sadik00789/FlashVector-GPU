use engine::ffi::gpu_flat_scan_batch;
use engine::{DeviceBuffer, GpuSearchResult, MetricType, Vector3D};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectionMethod {
    #[default]
    Pca,
    Umap,
}

impl ProjectionMethod {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "umap" | "parametric_umap" => Self::Umap,
            _ => Self::Pca,
        }
    }
}

pub struct PcaProjector3D {
    components: Vec<Vec<f32>>, // 3 components, each of length dim
    mean: Vec<f32>,
    dim: usize,
    scale: f32,
}

impl PcaProjector3D {
    /// Fit top 3 principal components using Power Iteration with Gram-Schmidt orthogonalization.
    pub fn fit(vectors: &[f32], dim: usize, max_samples: usize) -> Self {
        let num_vectors = vectors.len() / dim.max(1);
        if num_vectors == 0 || dim == 0 {
            return Self {
                components: vec![vec![0.0; dim.max(1)]; 3],
                mean: vec![0.0; dim.max(1)],
                dim: dim.max(1),
                scale: 50.0,
            };
        }
        let samples_to_use = num_vectors.min(max_samples.max(100));

        // 1. Compute Mean
        let mut mean = vec![0.0f32; dim];
        for i in 0..samples_to_use {
            let vec = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                mean[d] += vec[d];
            }
        }
        let inv_s = 1.0 / (samples_to_use as f32);
        for d in 0..dim {
            mean[d] *= inv_s;
        }

        // 2. Power Iteration for 3 components
        let mut rng = StdRng::seed_from_u64(0x50CA_3D00);
        let mut components: Vec<Vec<f32>> = Vec::with_capacity(3);

        for _comp_idx in 0..3 {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Self::normalize(&mut v);

            for _iter in 0..20 {
                let mut v_new = vec![0.0f32; dim];

                for i in 0..samples_to_use {
                    let vec = &vectors[i * dim..(i + 1) * dim];
                    let dot: f32 = (0..dim).map(|d| (vec[d] - mean[d]) * v[d]).sum();
                    for d in 0..dim {
                        v_new[d] += (vec[d] - mean[d]) * dot;
                    }
                }

                for prev in &components {
                    let dot: f32 = (0..dim).map(|d| v_new[d] * prev[d]).sum();
                    for d in 0..dim {
                        v_new[d] -= dot * prev[d];
                    }
                }

                Self::normalize(&mut v_new);
                v = v_new;
            }

            components.push(v);
        }

        Self {
            components,
            mean,
            dim,
            scale: 50.0,
        }
    }

    #[inline]
    fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }

    pub fn project_vector(&self, vec: &[f32]) -> Vector3D {
        assert_eq!(vec.len(), self.dim);
        let x: f32 = (0..self.dim)
            .map(|d| (vec[d] - self.mean[d]) * self.components[0][d])
            .sum();
        let y: f32 = (0..self.dim)
            .map(|d| (vec[d] - self.mean[d]) * self.components[1][d])
            .sum();
        let z: f32 = (0..self.dim)
            .map(|d| (vec[d] - self.mean[d]) * self.components[2][d])
            .sum();

        Vector3D {
            x: x * self.scale,
            y: y * self.scale,
            z: z * self.scale,
        }
    }

    pub fn project_batch(&self, vectors: &[f32]) -> Vec<Vector3D> {
        let num_vectors = vectors.len() / self.dim;
        (0..num_vectors)
            .into_par_iter()
            .map(|i| self.project_vector(&vectors[i * self.dim..(i + 1) * self.dim]))
            .collect()
    }
}

/// Parametric UMAP 3D encoder backed by a 2-hidden-layer ReLU MLP trained on the
/// fuzzy simplicial set k-NN graph (GPU-accelerated k-NN when CUDA is active).
pub struct ParametricUmap3D {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    dim: usize,
    h1: usize,
    h2: usize,
    w1: Vec<f32>, // [h1 x dim]
    b1: Vec<f32>, // [h1]
    w2: Vec<f32>, // [h2 x h1]
    b2: Vec<f32>, // [h2]
    w3: Vec<f32>, // [3 x h2]
    b3: Vec<f32>, // [3]
    out_scale: f32,
}

#[derive(Clone, Copy)]
struct UmapEdge {
    u: usize,
    v: usize,
    weight: f32,
}

struct MlpForward {
    z1: Vec<f32>,
    a1: Vec<f32>,
    z2: Vec<f32>,
    a2: Vec<f32>,
    out: [f32; 3],
}

impl ParametricUmap3D {
    const DISPLAY_SCALE: f32 = 16.0;

    /// Fit a parametric UMAP 3D neural encoder over up to `max_samples` vectors.
    pub fn fit(vectors: &[f32], dim: usize, max_samples: usize) -> Self {
        let total = vectors.len() / dim.max(1);
        if total < 8 || dim == 0 {
            let pca = PcaProjector3D::fit(vectors, dim, max_samples);
            return Self::from_pca_fallback(&pca, dim.max(1));
        }

        let n = total.min(max_samples.clamp(32, 1_500));
        let stride = (total / n).max(1);
        let mut samples = Vec::with_capacity(n * dim);
        let mut actual_n = 0usize;
        for i in (0..total).step_by(stride) {
            if actual_n >= n {
                break;
            }
            samples.extend_from_slice(&vectors[i * dim..(i + 1) * dim]);
            actual_n += 1;
        }
        let n = actual_n;

        // Input feature normalization
        let mut mean = vec![0.0f32; dim];
        for i in 0..n {
            let row = &samples[i * dim..(i + 1) * dim];
            for d in 0..dim {
                mean[d] += row[d];
            }
        }
        let inv_n = 1.0 / (n as f32);
        for d in 0..dim {
            mean[d] *= inv_n;
        }

        let mut var = vec![0.0f32; dim];
        for i in 0..n {
            let row = &samples[i * dim..(i + 1) * dim];
            for d in 0..dim {
                let diff = row[d] - mean[d];
                var[d] += diff * diff;
            }
        }
        let inv_std: Vec<f32> = var
            .into_iter()
            .map(|v| 1.0 / (v * inv_n).sqrt().max(1e-3))
            .collect();

        let mut x_norm = vec![0.0f32; n * dim];
        for i in 0..n {
            let src = &samples[i * dim..(i + 1) * dim];
            let dst = &mut x_norm[i * dim..(i + 1) * dim];
            for d in 0..dim {
                dst[d] = (src[d] - mean[d]) * inv_std[d];
            }
        }

        // k-NN graph on samples (uses GPU flat_scan when CUDA runtime is live, Rayon otherwise)
        let k = 15usize.min(n.saturating_sub(1)).max(2);
        let knn = Self::compute_knn(&samples, n, dim, k);
        let edges = Self::build_fuzzy_edges(&knn, n, k);

        // PCA warm-start targets scaled down to UMAP unit manifold scale (~2.5 radius)
        let pca = PcaProjector3D::fit(&samples, dim, n);
        let pca_targets: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let p = pca.project_vector(&samples[i * dim..(i + 1) * dim]);
                [p.x / 20.0, p.y / 20.0, p.z / 20.0]
            })
            .collect();

        let h1 = 48usize;
        let h2 = 24usize;
        let mut model = Self::init_mlp(mean, inv_std, dim, h1, h2);
        model.warm_start_pca(&x_norm, &pca_targets, n);
        model.train_umap_edges(&x_norm, &edges, n);
        model
    }

    fn from_pca_fallback(pca: &PcaProjector3D, dim: usize) -> Self {
        let mut model = Self::init_mlp(pca.mean.clone(), vec![1.0; dim], dim, 48, 24);
        model.out_scale = 1.0;
        model
    }

    fn init_mlp(mean: Vec<f32>, inv_std: Vec<f32>, dim: usize, h1: usize, h2: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9_554D_4190);
        let s1 = (2.0 / dim as f32).sqrt();
        let s2 = (2.0 / h1 as f32).sqrt();
        let s3 = (1.0 / h2 as f32).sqrt();
        Self {
            mean,
            inv_std,
            dim,
            h1,
            h2,
            w1: (0..h1 * dim).map(|_| rng.gen_range(-s1..s1)).collect(),
            b1: vec![0.0; h1],
            w2: (0..h2 * h1).map(|_| rng.gen_range(-s2..s2)).collect(),
            b2: vec![0.0; h2],
            w3: (0..3 * h2).map(|_| rng.gen_range(-s3..s3)).collect(),
            b3: vec![0.0; 3],
            out_scale: Self::DISPLAY_SCALE,
        }
    }

    fn compute_knn(samples: &[f32], n: usize, dim: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
        if !engine::CPU_ONLY {
            if let Ok(gpu_knn) = Self::compute_knn_gpu(samples, n, dim, k) {
                return gpu_knn;
            }
        }
        Self::compute_knn_cpu(samples, n, dim, k)
    }

    fn compute_knn_gpu(
        samples: &[f32],
        n: usize,
        dim: usize,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>, engine::CudaError> {
        let fetch_k = (k + 1).min(n);
        let mut d_db = DeviceBuffer::new(n * dim)?;
        let mut d_ids = DeviceBuffer::new(n)?;
        let h_ids: Vec<u32> = (0..n as u32).collect();
        d_db.copy_from_host_async(samples, std::ptr::null_mut())?;
        d_ids.copy_from_host_async(&h_ids, std::ptr::null_mut())?;

        let chunk_max = 256usize.min(n);
        let mut d_q = DeviceBuffer::new(chunk_max * dim)?;
        let mut d_out = DeviceBuffer::new(chunk_max * fetch_k)?;
        let mut all_knn = Vec::with_capacity(n);

        for start in (0..n).step_by(chunk_max) {
            let count = (n - start).min(chunk_max);
            d_q.copy_from_host_async(
                &samples[start * dim..(start + count) * dim],
                std::ptr::null_mut(),
            )?;
            gpu_flat_scan_batch(
                d_db.as_ptr(),
                d_ids.as_ptr(),
                n as u32,
                dim as u32,
                d_q.as_ptr(),
                count as u32,
                fetch_k as u32,
                MetricType::L2,
                std::ptr::null(),
                d_out.as_mut_ptr(),
                std::ptr::null_mut(),
            )?;
            let mut h_out = vec![GpuSearchResult::default(); count * fetch_k];
            d_out.copy_to_host_async(&mut h_out, std::ptr::null_mut())?;
            engine::gpu_device_synchronize()?;

            for qi in 0..count {
                let row_idx = start + qi;
                let mut nbrs = Vec::with_capacity(k);
                for r in &h_out[qi * fetch_k..(qi + 1) * fetch_k] {
                    if r.id != u32::MAX && r.id as usize != row_idx && nbrs.len() < k {
                        nbrs.push((r.id as usize, r.distance.max(0.0).sqrt()));
                    }
                }
                all_knn.push(nbrs);
            }
        }
        Ok(all_knn)
    }

    fn compute_knn_cpu(samples: &[f32], n: usize, dim: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let vi = &samples[i * dim..(i + 1) * dim];
                let mut dists: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let vj = &samples[j * dim..(j + 1) * dim];
                        let d2: f32 = vi
                            .iter()
                            .zip(vj.iter())
                            .map(|(&a, &b)| {
                                let d = a - b;
                                d * d
                            })
                            .sum();
                        (j, d2.sqrt())
                    })
                    .collect();
                if dists.len() > k {
                    dists.select_nth_unstable_by(k - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    dists.truncate(k);
                }
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists
            })
            .collect()
    }

    fn build_fuzzy_edges(knn: &[Vec<(usize, f32)>], n: usize, k: usize) -> Vec<UmapEdge> {
        let target = (k.max(2) as f32).log2();
        let mut directed: Vec<std::collections::HashMap<usize, f32>> =
            vec![std::collections::HashMap::new(); n];

        for (i, nbrs) in knn.iter().enumerate() {
            if nbrs.is_empty() {
                continue;
            }
            let rho = nbrs[0].1;
            let mut lo = 1e-5f32;
            let mut hi = 1e4f32;
            let mut sigma = 1.0f32;
            for _ in 0..24 {
                sigma = 0.5 * (lo + hi);
                let sum: f32 = nbrs
                    .iter()
                    .map(|&(_, d)| (-(d - rho).max(0.0) / sigma.max(1e-6)).exp())
                    .sum();
                if (sum - target).abs() < 1e-3 {
                    break;
                }
                if sum > target {
                    hi = sigma;
                } else {
                    lo = sigma;
                }
            }
            for &(j, d) in nbrs {
                let w = (-(d - rho).max(0.0) / sigma.max(1e-6)).exp();
                directed[i].insert(j, w);
            }
        }

        let mut sym = std::collections::HashMap::<(usize, usize), f32>::new();
        for (i, map_i) in directed.iter().enumerate() {
            for (&j, &w_ij) in map_i {
                let key = if i < j { (i, j) } else { (j, i) };
                let w_ji = directed[j].get(&i).copied().unwrap_or(0.0);
                let p = w_ij + w_ji - w_ij * w_ji;
                if p > 1e-3 {
                    sym.entry(key).and_modify(|v| *v = v.max(p)).or_insert(p);
                }
            }
        }
        sym.into_iter()
            .map(|((u, v), weight)| UmapEdge { u, v, weight })
            .collect()
    }

    fn forward_normalized(&self, x_norm: &[f32]) -> MlpForward {
        let mut z1 = vec![0.0f32; self.h1];
        let mut a1 = vec![0.0f32; self.h1];
        for r in 0..self.h1 {
            let w_row = &self.w1[r * self.dim..(r + 1) * self.dim];
            let acc: f32 = w_row.iter().zip(x_norm.iter()).map(|(&w, &x)| w * x).sum();
            let v = acc + self.b1[r];
            z1[r] = v;
            a1[r] = v.max(0.0);
        }

        let mut z2 = vec![0.0f32; self.h2];
        let mut a2 = vec![0.0f32; self.h2];
        for r in 0..self.h2 {
            let w_row = &self.w2[r * self.h1..(r + 1) * self.h1];
            let acc: f32 = w_row.iter().zip(a1.iter()).map(|(&w, &x)| w * x).sum();
            let v = acc + self.b2[r];
            z2[r] = v;
            a2[r] = v.max(0.0);
        }

        let mut out = [0.0f32; 3];
        for r in 0..3 {
            let w_row = &self.w3[r * self.h2..(r + 1) * self.h2];
            let acc: f32 = w_row.iter().zip(a2.iter()).map(|(&w, &x)| w * x).sum();
            out[r] = acc + self.b3[r];
        }

        MlpForward {
            z1,
            a1,
            z2,
            a2,
            out,
        }
    }

    fn accumulate_backward(
        &self,
        x_norm: &[f32],
        fwd: &MlpForward,
        grad_out: [f32; 3],
        grads: &mut MlpGrads,
    ) {
        let mut d_a2 = vec![0.0f32; self.h2];
        for r in 0..3 {
            let g = grad_out[r];
            grads.b3[r] += g;
            let w_row = &self.w3[r * self.h2..(r + 1) * self.h2];
            let gw_row = &mut grads.w3[r * self.h2..(r + 1) * self.h2];
            for c in 0..self.h2 {
                gw_row[c] += g * fwd.a2[c];
                d_a2[c] += g * w_row[c];
            }
        }

        let mut d_a1 = vec![0.0f32; self.h1];
        for r in 0..self.h2 {
            if fwd.z2[r] <= 0.0 {
                continue;
            }
            let g = d_a2[r];
            grads.b2[r] += g;
            let w_row = &self.w2[r * self.h1..(r + 1) * self.h1];
            let gw_row = &mut grads.w2[r * self.h1..(r + 1) * self.h1];
            for c in 0..self.h1 {
                gw_row[c] += g * fwd.a1[c];
                d_a1[c] += g * w_row[c];
            }
        }

        for r in 0..self.h1 {
            if fwd.z1[r] <= 0.0 {
                continue;
            }
            let g = d_a1[r];
            grads.b1[r] += g;
            let gw_row = &mut grads.w1[r * self.dim..(r + 1) * self.dim];
            for c in 0..self.dim {
                gw_row[c] += g * x_norm[c];
            }
        }
    }

    fn warm_start_pca(&mut self, x_norm: &[f32], targets: &[[f32; 3]], n: usize) {
        let mut adam = MlpAdam::new(self.dim, self.h1, self.h2, 0.025);
        let mut grads = MlpGrads::new(self.dim, self.h1, self.h2);
        let batch_size = n.min(64);
        let mut rng = StdRng::seed_from_u64(0xB105_F00D);

        for _epoch in 0..16 {
            grads.zero();
            for _ in 0..batch_size {
                let i = rng.gen_range(0..n);
                let x_i = &x_norm[i * self.dim..(i + 1) * self.dim];
                let fwd = self.forward_normalized(x_i);
                let t = targets[i];
                let g = [
                    (fwd.out[0] - t[0]) / batch_size as f32,
                    (fwd.out[1] - t[1]) / batch_size as f32,
                    (fwd.out[2] - t[2]) / batch_size as f32,
                ];
                self.accumulate_backward(x_i, &fwd, g, &mut grads);
            }
            adam.step(self, &grads);
        }
    }

    fn train_umap_edges(&mut self, x_norm: &[f32], edges: &[UmapEdge], n: usize) {
        if edges.is_empty() || n < 4 {
            return;
        }
        // Standard UMAP min_dist ~ 0.25 kernel parameters
        const A: f32 = 1.577;
        const B: f32 = 0.895;

        let mut adam = MlpAdam::new(self.dim, self.h1, self.h2, 0.015);
        let mut grads = MlpGrads::new(self.dim, self.h1, self.h2);
        let mut rng = StdRng::seed_from_u64(0xCAFE_BEEF);
        let steps_per_epoch = edges.len().min(640);
        let batch = 64usize;
        let neg_samples = 3usize;

        for _epoch in 0..22 {
            let mut offset = 0usize;
            while offset < steps_per_epoch {
                let cur_batch = (steps_per_epoch - offset).min(batch);
                grads.zero();
                let inv_bs = 1.0 / cur_batch as f32;

                for _ in 0..cur_batch {
                    let e = edges[rng.gen_range(0..edges.len())];
                    let x_u = &x_norm[e.u * self.dim..(e.u + 1) * self.dim];
                    let x_v = &x_norm[e.v * self.dim..(e.v + 1) * self.dim];
                    let f_u = self.forward_normalized(x_u);
                    let f_v = self.forward_normalized(x_v);

                    let d = [
                        f_u.out[0] - f_v.out[0],
                        f_u.out[1] - f_v.out[1],
                        f_u.out[2] - f_v.out[2],
                    ];
                    let d2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).max(1e-4);
                    let grad_coeff = (2.0 * A * B * d2.powf(B - 1.0)) / (1.0 + A * d2.powf(B));
                    let w = e.weight * inv_bs;
                    let g_attr = [
                        (w * grad_coeff * d[0]).clamp(-2.0, 2.0),
                        (w * grad_coeff * d[1]).clamp(-2.0, 2.0),
                        (w * grad_coeff * d[2]).clamp(-2.0, 2.0),
                    ];
                    self.accumulate_backward(x_u, &f_u, g_attr, &mut grads);
                    self.accumulate_backward(
                        x_v,
                        &f_v,
                        [-g_attr[0], -g_attr[1], -g_attr[2]],
                        &mut grads,
                    );

                    // Repulsive force to random negatives
                    for _ in 0..neg_samples {
                        let neg = rng.gen_range(0..n);
                        if neg == e.u || neg == e.v {
                            continue;
                        }
                        let x_neg = &x_norm[neg * self.dim..(neg + 1) * self.dim];
                        let f_neg = self.forward_normalized(x_neg);
                        let dn = [
                            f_u.out[0] - f_neg.out[0],
                            f_u.out[1] - f_neg.out[1],
                            f_u.out[2] - f_neg.out[2],
                        ];
                        let dn2 = (dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2]).max(1e-4);
                        let rep_coeff = (-2.0 * B) / ((0.001 + dn2) * (1.0 + A * dn2.powf(B)));
                        let g_rep = [
                            (inv_bs * rep_coeff * dn[0]).clamp(-2.0, 2.0),
                            (inv_bs * rep_coeff * dn[1]).clamp(-2.0, 2.0),
                            (inv_bs * rep_coeff * dn[2]).clamp(-2.0, 2.0),
                        ];
                        self.accumulate_backward(x_u, &f_u, g_rep, &mut grads);
                        self.accumulate_backward(
                            x_neg,
                            &f_neg,
                            [-g_rep[0], -g_rep[1], -g_rep[2]],
                            &mut grads,
                        );
                    }
                }

                adam.step(self, &grads);
                offset += cur_batch;
            }
        }
    }

    pub fn project_vector(&self, vec: &[f32]) -> Vector3D {
        assert_eq!(vec.len(), self.dim);
        let x_norm: Vec<f32> = (0..self.dim)
            .map(|d| (vec[d] - self.mean[d]) * self.inv_std[d])
            .collect();
        let fwd = self.forward_normalized(&x_norm);
        Vector3D {
            x: fwd.out[0] * self.out_scale,
            y: fwd.out[1] * self.out_scale,
            z: fwd.out[2] * self.out_scale,
        }
    }

    pub fn project_batch(&self, vectors: &[f32]) -> Vec<Vector3D> {
        let num_vectors = vectors.len() / self.dim;
        (0..num_vectors)
            .into_par_iter()
            .map(|i| self.project_vector(&vectors[i * self.dim..(i + 1) * self.dim]))
            .collect()
    }
}

struct MlpGrads {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
}

impl MlpGrads {
    fn new(dim: usize, h1: usize, h2: usize) -> Self {
        Self {
            w1: vec![0.0; h1 * dim],
            b1: vec![0.0; h1],
            w2: vec![0.0; h2 * h1],
            b2: vec![0.0; h2],
            w3: vec![0.0; 3 * h2],
            b3: vec![0.0; 3],
        }
    }

    fn zero(&mut self) {
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2.fill(0.0);
        self.w3.fill(0.0);
        self.b3.fill(0.0);
    }
}

struct MlpAdam {
    lr: f32,
    t: u32,
    m: MlpGrads,
    v: MlpGrads,
}

impl MlpAdam {
    fn new(dim: usize, h1: usize, h2: usize, lr: f32) -> Self {
        Self {
            lr,
            t: 0,
            m: MlpGrads::new(dim, h1, h2),
            v: MlpGrads::new(dim, h1, h2),
        }
    }

    fn apply_vec(
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        bc1: f32,
        bc2: f32,
    ) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        for i in 0..param.len() {
            let g = grad[i];
            m[i] = B1 * m[i] + (1.0 - B1) * g;
            v[i] = B2 * v[i] + (1.0 - B2) * g * g;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            param[i] -= lr * m_hat / (v_hat.sqrt() + 1e-6);
        }
    }

    fn step(&mut self, model: &mut ParametricUmap3D, grads: &MlpGrads) {
        self.t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.t as i32);
        let bc2 = 1.0 - 0.999f32.powi(self.t as i32);
        Self::apply_vec(
            &mut model.w1,
            &grads.w1,
            &mut self.m.w1,
            &mut self.v.w1,
            self.lr,
            bc1,
            bc2,
        );
        Self::apply_vec(
            &mut model.b1,
            &grads.b1,
            &mut self.m.b1,
            &mut self.v.b1,
            self.lr,
            bc1,
            bc2,
        );
        Self::apply_vec(
            &mut model.w2,
            &grads.w2,
            &mut self.m.w2,
            &mut self.v.w2,
            self.lr,
            bc1,
            bc2,
        );
        Self::apply_vec(
            &mut model.b2,
            &grads.b2,
            &mut self.m.b2,
            &mut self.v.b2,
            self.lr,
            bc1,
            bc2,
        );
        Self::apply_vec(
            &mut model.w3,
            &grads.w3,
            &mut self.m.w3,
            &mut self.v.w3,
            self.lr,
            bc1,
            bc2,
        );
        Self::apply_vec(
            &mut model.b3,
            &grads.b3,
            &mut self.m.b3,
            &mut self.v.b3,
            self.lr,
            bc1,
            bc2,
        );
    }
}

/// Unified 3D projector supporting fast linear PCA and topology-preserving Parametric UMAP.
pub enum Projector3D {
    Pca(PcaProjector3D),
    Umap(ParametricUmap3D),
}

impl Projector3D {
    pub fn fit(vectors: &[f32], dim: usize, method: ProjectionMethod) -> Self {
        match method {
            ProjectionMethod::Pca => Self::Pca(PcaProjector3D::fit(vectors, dim, 2_000)),
            ProjectionMethod::Umap => Self::Umap(ParametricUmap3D::fit(vectors, dim, 1_500)),
        }
    }

    pub fn method(&self) -> ProjectionMethod {
        match self {
            Self::Pca(_) => ProjectionMethod::Pca,
            Self::Umap(_) => ProjectionMethod::Umap,
        }
    }

    pub fn project_vector(&self, vec: &[f32]) -> Vector3D {
        match self {
            Self::Pca(p) => p.project_vector(vec),
            Self::Umap(u) => u.project_vector(vec),
        }
    }

    pub fn project_batch(&self, vectors: &[f32]) -> Vec<Vector3D> {
        match self {
            Self::Pca(p) => p.project_batch(vectors),
            Self::Umap(u) => u.project_batch(vectors),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separated_clusters(n_per_cluster: usize, dim: usize) -> (Vec<f32>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::with_capacity(3 * n_per_cluster * dim);
        let mut labels = Vec::with_capacity(3 * n_per_cluster);
        let centers = [0usize, 10, 20];
        for (c_idx, &offset) in centers.iter().enumerate() {
            for _ in 0..n_per_cluster {
                labels.push(c_idx);
                for d in 0..dim {
                    let base = if d >= offset && d < offset + 8 {
                        4.0
                    } else {
                        0.0
                    };
                    data.push(base + rng.gen_range(-0.25f32..0.25f32));
                }
            }
        }
        (data, labels)
    }

    fn avg_dist(a: &Vector3D, b: &Vector3D) -> f32 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[test]
    fn test_pca_and_umap_project_separated_clusters() {
        let dim = 32;
        let n_per = 40;
        let (data, labels) = separated_clusters(n_per, dim);

        for method in [ProjectionMethod::Pca, ProjectionMethod::Umap] {
            let proj = Projector3D::fit(&data, dim, method);
            assert_eq!(proj.method(), method);
            let pts = proj.project_batch(&data);
            assert_eq!(pts.len(), 3 * n_per);
            assert!(pts
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()));

            // Unseen test query near cluster 0 must also project finitely
            let q = proj.project_vector(&data[0..dim]);
            assert!(q.x.is_finite() && q.y.is_finite() && q.z.is_finite());

            // Mean intra-cluster distance must be strictly smaller than mean inter-cluster distance
            let mut intra_sum = 0.0f32;
            let mut intra_cnt = 0usize;
            let mut inter_sum = 0.0f32;
            let mut inter_cnt = 0usize;
            for i in 0..pts.len() {
                for j in (i + 1)..pts.len() {
                    let d = avg_dist(&pts[i], &pts[j]);
                    if labels[i] == labels[j] {
                        intra_sum += d;
                        intra_cnt += 1;
                    } else {
                        inter_sum += d;
                        inter_cnt += 1;
                    }
                }
            }
            let intra = intra_sum / intra_cnt as f32;
            let inter = inter_sum / inter_cnt as f32;
            assert!(
                intra < inter,
                "{:?} expected intra ({intra}) < inter ({inter})",
                method
            );
        }
    }
}
