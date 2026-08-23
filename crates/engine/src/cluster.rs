use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<f32>,      // [k * dim]
    pub assignments: Vec<u32>,    // [num_vectors]
    pub cluster_sizes: Vec<usize>,// [k]
    pub k: usize,
    pub dim: usize,
}

pub struct KMeans;

impl KMeans {
    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    pub fn fit(
        vectors: &[f32],
        dim: usize,
        k: usize,
        max_iters: usize,
    ) -> KMeansResult {
        let num_vectors = vectors.len() / dim;
        assert!(num_vectors >= k, "Number of vectors must be >= k");

        let mut rng = rand::thread_rng();

        // 1. Compute global dataset mean vector for empty cluster recovery
        let mut global_mean = vec![0.0f32; dim];
        for i in 0..num_vectors {
            let vec = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                global_mean[d] += vec[d];
            }
        }
        let inv_n = 1.0 / (num_vectors as f32);
        for d in 0..dim {
            global_mean[d] *= inv_n;
        }

        // 2. K-Means++ / Random Centroid Initialization
        let mut sample_indices: Vec<usize> = (0..num_vectors).collect();
        sample_indices.shuffle(&mut rng);

        let mut centroids = Vec::with_capacity(k * dim);
        for &idx in &sample_indices[..k] {
            centroids.extend_from_slice(&vectors[idx * dim..(idx + 1) * dim]);
        }

        let mut assignments = vec![0u32; num_vectors];
        let mut cluster_sizes = vec![0usize; k];

        for _iter in 0..max_iters {
            // Parallel Assign Step
            assignments
                .par_chunks_mut(512)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_idx = chunk_idx * 512;
                    for (i, assign) in chunk.iter_mut().enumerate() {
                        let vec_idx = base_idx + i;
                        let vec = &vectors[vec_idx * dim..(vec_idx + 1) * dim];

                        let mut min_dist = f32::INFINITY;
                        let mut best_c = 0;

                        for c in 0..k {
                            let centroid = &centroids[c * dim..(c + 1) * dim];
                            let dist = Self::l2_sq(vec, centroid);
                            if dist < min_dist {
                                min_dist = dist;
                                best_c = c;
                            }
                        }
                        *assign = best_c as u32;
                    }
                });

            // Update Step: Compute new centroids
            let mut new_centroids = vec![0.0f32; k * dim];
            let mut counts = vec![0usize; k];

            for (vec_idx, &c) in assignments.iter().enumerate() {
                let cluster = c as usize;
                counts[cluster] += 1;
                let vec = &vectors[vec_idx * dim..(vec_idx + 1) * dim];
                let cent = &mut new_centroids[cluster * dim..(cluster + 1) * dim];
                for d in 0..dim {
                    cent[d] += vec[d];
                }
            }

            let mut max_delta = 0.0f32;
            for c in 0..k {
                let count = counts[c];
                if count > 0 {
                    let inv_count = 1.0 / (count as f32);
                    let cent = &mut new_centroids[c * dim..(c + 1) * dim];
                    let old_cent = &centroids[c * dim..(c + 1) * dim];
                    let mut delta = 0.0f32;
                    for d in 0..dim {
                        cent[d] *= inv_count;
                        let diff = cent[d] - old_cent[d];
                        delta += diff * diff;
                    }
                    if delta > max_delta {
                        max_delta = delta;
                    }
                } else {
                    // Empty cluster recovery: Find vector with greatest distance from global mean
                    let mut furthest_vec_idx = rng.gen_range(0..num_vectors);
                    let mut max_d = 0.0f32;
                    for _i in 0..num_vectors.min(500) {
                        let sample_idx = rng.gen_range(0..num_vectors);
                        let v = &vectors[sample_idx * dim..(sample_idx + 1) * dim];
                        let d = Self::l2_sq(v, &global_mean);
                        if d > max_d {
                            max_d = d;
                            furthest_vec_idx = sample_idx;
                        }
                    }
                    new_centroids[c * dim..(c + 1) * dim]
                        .copy_from_slice(&vectors[furthest_vec_idx * dim..(furthest_vec_idx + 1) * dim]);
                }
            }

            centroids = new_centroids;
            cluster_sizes = counts;

            if max_delta < 1e-4 {
                break;
            }
        }

        KMeansResult {
            centroids,
            assignments,
            cluster_sizes,
            k,
            dim,
        }
    }
}

/// Product Quantizer dividing vectors into M subspaces and learning 256 centroids per subspace
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    pub m_pq: usize,
    pub sub_dim: usize,
    pub dim: usize,
    pub codebooks: Vec<f32>, // [m_pq * 256 * sub_dim]
}

impl ProductQuantizer {
    pub fn train(
        vectors: &[f32],
        dim: usize,
        m_pq: usize,
        iters: usize,
    ) -> Self {
        assert_eq!(dim % m_pq, 0, "Dimension must be divisible by m_pq");
        let sub_dim = dim / m_pq;
        let num_vectors = vectors.len() / dim;
        let num_centroids = 256;

        let mut codebooks = vec![0.0f32; m_pq * num_centroids * sub_dim];

        // Train codebook for each subspace in parallel
        codebooks
            .par_chunks_exact_mut(num_centroids * sub_dim)
            .enumerate()
            .for_each(|(m, sub_codebook)| {
                let mut sub_vectors = Vec::with_capacity(num_vectors * sub_dim);
                for i in 0..num_vectors {
                    let start = i * dim + m * sub_dim;
                    sub_vectors.extend_from_slice(&vectors[start..start + sub_dim]);
                }

                let k_actual = if num_vectors < num_centroids { num_vectors } else { num_centroids };
                let res = KMeans::fit(&sub_vectors, sub_dim, k_actual, iters);

                sub_codebook[..k_actual * sub_dim].copy_from_slice(&res.centroids);
            });

        Self {
            m_pq,
            sub_dim,
            dim,
            codebooks,
        }
    }

    pub fn encode(&self, vectors: &[f32]) -> Vec<u8> {
        let num_vectors = vectors.len() / self.dim;
        let mut codes = vec![0u8; num_vectors * self.m_pq];

        codes
            .par_chunks_exact_mut(self.m_pq)
            .enumerate()
            .for_each(|(i, vec_codes)| {
                let vec = &vectors[i * self.dim..(i + 1) * self.dim];

                for m in 0..self.m_pq {
                    let sub_vec = &vec[m * self.sub_dim..(m + 1) * self.sub_dim];
                    let cb_offset = m * 256 * self.sub_dim;

                    let mut min_dist = f32::INFINITY;
                    let mut best_code = 0u8;

                    for c in 0..256 {
                        let centroid = &self.codebooks[cb_offset + c * self.sub_dim..cb_offset + (c + 1) * self.sub_dim];
                        let dist: f32 = sub_vec.iter().zip(centroid.iter()).map(|(&x, &y)| {
                            let d = x - y;
                            d * d
                        }).sum();

                        if dist < min_dist {
                            min_dist = dist;
                            best_code = c as u8;
                        }
                    }

                    vec_codes[m] = best_code;
                }
            });

        codes
    }
}
