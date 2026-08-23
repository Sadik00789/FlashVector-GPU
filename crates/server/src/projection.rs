use engine::Vector3D;
use rand::Rng;

pub struct PcaProjector3D {
    components: Vec<Vec<f32>>, // 3 components, each of length dim
    mean: Vec<f32>,
    dim: usize,
    scale: f32,
}

impl PcaProjector3D {
    /// Fit top 3 principal components using Power Iteration with Gram-Schmidt orthogonalization
    pub fn fit(vectors: &[f32], dim: usize, max_samples: usize) -> Self {
        let num_vectors = vectors.len() / dim;
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
        let mut rng = rand::thread_rng();
        let mut components: Vec<Vec<f32>> = Vec::with_capacity(3);

        for _comp_idx in 0..3 {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Self::normalize(&mut v);

            for _iter in 0..20 {
                let mut v_new = vec![0.0f32; dim];

                // Matrix-vector multiplication: X^T * (X * v)
                for i in 0..samples_to_use {
                    let vec = &vectors[i * dim..(i + 1) * dim];
                    let dot: f32 = (0..dim).map(|d| (vec[d] - mean[d]) * v[d]).sum();
                    for d in 0..dim {
                        v_new[d] += (vec[d] - mean[d]) * dot;
                    }
                }

                // Gram-Schmidt orthogonalization against previously found components
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
        let centered: Vec<f32> = (0..self.dim).map(|d| vec[d] - self.mean[d]).collect();

        let x: f32 = (0..self.dim).map(|d| centered[d] * self.components[0][d]).sum();
        let y: f32 = (0..self.dim).map(|d| centered[d] * self.components[1][d]).sum();
        let z: f32 = (0..self.dim).map(|d| centered[d] * self.components[2][d]).sum();

        Vector3D {
            x: x * self.scale,
            y: y * self.scale,
            z: z * self.scale,
        }
    }

    pub fn project_batch(&self, vectors: &[f32]) -> Vec<Vector3D> {
        let num_vectors = vectors.len() / self.dim;
        let mut results = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let vec = &vectors[i * self.dim..(i + 1) * dim_offset(i, self.dim)];
            results.push(self.project_vector(vec));
        }
        results
    }
}

#[inline]
fn dim_offset(_i: usize, dim: usize) -> usize {
    dim
}
