//! Quantization encoders for IVF-SQ8, IVF-RaBitQ, and FP16 Tensor Core modes.

use rayon::prelude::*;

/// 8-bit Scalar Quantizer (SQ8) trained on Voronoi residuals.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer8 {
    pub dim: usize,
    pub mins: Vec<f32>,   // [dim]
    pub scales: Vec<f32>, // [dim]
}

impl ScalarQuantizer8 {
    pub fn fit(residuals: &[f32], dim: usize) -> Self {
        let num_vectors = residuals.len() / dim;
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for i in 0..num_vectors {
            let v = &residuals[i * dim..(i + 1) * dim];
            for d in 0..dim {
                if v[d] < mins[d] {
                    mins[d] = v[d];
                }
                if v[d] > maxs[d] {
                    maxs[d] = v[d];
                }
            }
        }

        if num_vectors == 0 {
            mins.fill(0.0);
            maxs.fill(0.0);
        }

        let scales = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| {
                let range = hi - lo;
                if range > 1e-12 {
                    range / 255.0
                } else {
                    0.0
                }
            })
            .collect();

        Self { dim, mins, scales }
    }

    pub fn encode(&self, residuals: &[f32]) -> Vec<u8> {
        let num_vectors = residuals.len() / self.dim;
        let mut codes = vec![0u8; residuals.len()];

        codes
            .par_chunks_exact_mut(self.dim)
            .enumerate()
            .for_each(|(i, dst)| {
                let src = &residuals[i * self.dim..(i + 1) * self.dim];
                for d in 0..self.dim {
                    let s = self.scales[d];
                    if s > 0.0 {
                        let q = ((src[d] - self.mins[d]) / s).round().clamp(0.0, 255.0);
                        dst[d] = q as u8;
                    } else {
                        dst[d] = 0;
                    }
                }
            });

        let _ = num_vectors;
        codes
    }
}

/// Encoded IVF-RaBitQ data ready for GPU upload.
#[derive(Debug, Clone)]
pub struct RabitqEncoded {
    pub dim: usize,
    pub padded_dim: usize,
    pub words_per_vec: usize,
    pub rot_signs: Vec<f32>,     // [3 * padded_dim]
    pub rot_centroids: Vec<f32>, // [nlist * padded_dim]
    pub bits: Vec<u32>,          // [num_vectors * words_per_vec]
    pub meta: Vec<f32>,          // [num_vectors * 4]
}

/// Randomized Binary Quantization (RaBitQ) with 3-stage Fast Walsh-Hadamard
/// orthogonal rotation (matches CUDA `fwht_rotate_query_shared` exactly).
pub struct RabitqEncoder;

impl RabitqEncoder {
    #[inline]
    pub fn padded_dim(dim: usize) -> usize {
        dim.max(32).next_power_of_two()
    }

    /// Generate deterministic Rademacher sign flips (+1 / -1) for 3 FWHT stages.
    pub fn generate_rot_signs(padded_dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        (0..3 * padded_dim)
            .map(|_| {
                s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                if (z & 1) == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect()
    }

    /// Apply the 3-stage orthogonal FWHT rotation to a `dim`-length vector.
    pub fn fwht_rotate(vec: &[f32], rot_signs: &[f32], padded_dim: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; padded_dim];
        let copy_len = vec.len().min(padded_dim);
        buf[..copy_len].copy_from_slice(&vec[..copy_len]);

        let inv_sqrt_p = 1.0f32 / (padded_dim as f32).sqrt();
        for stage in 0..3 {
            let signs = &rot_signs[stage * padded_dim..(stage + 1) * padded_dim];
            for d in 0..padded_dim {
                buf[d] *= signs[d];
            }

            let mut len = 1usize;
            while len < padded_dim {
                let step = len << 1;
                let mut blk = 0usize;
                while blk < padded_dim {
                    for off in 0..len {
                        let a = blk + off;
                        let b = a + len;
                        let x = buf[a];
                        let y = buf[b];
                        buf[a] = x + y;
                        buf[b] = x - y;
                    }
                    blk += step;
                }
                len <<= 1;
            }

            for x in buf.iter_mut() {
                *x *= inv_sqrt_p;
            }
        }
        buf
    }

    pub fn encode(
        vectors: &[f32],
        centroids: &[f32],
        assignments: &[u32],
        dim: usize,
        seed: u64,
    ) -> RabitqEncoded {
        let num_vectors = vectors.len() / dim;
        let nlist = centroids.len() / dim;
        let padded_dim = Self::padded_dim(dim);
        let words_per_vec = padded_dim / 32;
        let rot_signs = Self::generate_rot_signs(padded_dim, seed);

        let mut rot_centroids = Vec::with_capacity(nlist * padded_dim);
        for c in 0..nlist {
            let cent = &centroids[c * dim..(c + 1) * dim];
            rot_centroids.extend_from_slice(&Self::fwht_rotate(cent, &rot_signs, padded_dim));
        }

        let mut bits = vec![0u32; num_vectors * words_per_vec];
        let mut meta = vec![0.0f32; num_vectors * 4];
        let inv_sqrt_p = 1.0f32 / (padded_dim as f32).sqrt();

        bits.par_chunks_exact_mut(words_per_vec)
            .zip(meta.par_chunks_exact_mut(4))
            .enumerate()
            .for_each(|(i, (vec_bits, vec_meta))| {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let cluster = assignments[i] as usize;
                let centroid = &centroids[cluster * dim..(cluster + 1) * dim];

                let mut residual = vec![0.0f32; dim];
                let mut res_norm_sq = 0.0f32;
                let mut vec_norm_sq = 0.0f32;
                for d in 0..dim {
                    let r = vec[d] - centroid[d];
                    residual[d] = r;
                    res_norm_sq += r * r;
                    vec_norm_sq += vec[d] * vec[d];
                }

                let rot_r = Self::fwht_rotate(&residual, &rot_signs, padded_dim);
                let res_norm = res_norm_sq.sqrt();

                let mut abs_sum = 0.0f32;
                let mut popcnt_b = 0u32;
                for (d, &val) in rot_r.iter().enumerate() {
                    if val >= 0.0 {
                        vec_bits[d >> 5] |= 1u32 << (d & 31);
                        popcnt_b += 1;
                    }
                    abs_sum += val.abs();
                }

                let rescale = if res_norm > 1e-12 && abs_sum > 1e-12 {
                    // <x_bar, r_bar> = abs_sum / (sqrt(P) * res_norm)
                    // rescale = res_norm / (sqrt(P) * <x_bar, r_bar>) = res_norm^2 / abs_sum
                    (res_norm * res_norm) / abs_sum
                } else {
                    0.0
                };
                let _ = inv_sqrt_p;

                vec_meta[0] = res_norm_sq;
                vec_meta[1] = rescale;
                vec_meta[2] = popcnt_b as f32;
                vec_meta[3] = vec_norm_sq;
            });

        RabitqEncoded {
            dim,
            padded_dim,
            words_per_vec,
            rot_signs,
            rot_centroids,
            bits,
            meta,
        }
    }
}

/// Pure-Rust IEEE-754 binary16 (FP16) conversion helpers for Tensor Core search.
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let unbiased = exp - 127 + 15;
    if unbiased >= 31 {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (14 - unbiased);
        return sign | (m as u16);
    }

    let rounded = mant + 0x1000;
    if (rounded & 0x0080_0000) != 0 {
        let next_exp = unbiased + 1;
        if next_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((next_exp as u16) << 10);
    }

    sign | ((unbiased as u16) << 10) | ((rounded >> 13) as u16)
}

#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = -14i32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        let f_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f_exp | (m << 13));
    }
    if exp == 31 {
        let f_exp = 0xffu32 << 23;
        return f32::from_bits(sign | f_exp | (mant << 13));
    }
    let f_exp = (exp + 112) << 23;
    f32::from_bits(sign | f_exp | (mant << 13))
}

/// Encode FP32 vectors into 16-aligned FP16 matrix `[num_vectors, padded_dim]`
/// and precompute `||x||^2` per vector for `wmma` Tensor Core search.
pub fn encode_fp16_padded(vectors: &[f32], dim: usize) -> (Vec<u16>, Vec<f32>, usize) {
    let num_vectors = vectors.len() / dim;
    let padded_dim = dim.div_ceil(16) * 16;
    let mut fp16 = vec![0u16; num_vectors * padded_dim];
    let mut norms_sq = vec![0.0f32; num_vectors];

    fp16.par_chunks_exact_mut(padded_dim)
        .zip(norms_sq.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dst, norm_out))| {
            let src = &vectors[i * dim..(i + 1) * dim];
            let mut nsq = 0.0f32;
            for d in 0..dim {
                let v = src[d];
                nsq += v * v;
                dst[d] = f32_to_f16_bits(v);
            }
            *norm_out = nsq;
        });

    (fp16, norms_sq, padded_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_is_orthogonal() {
        let dim = 96;
        let p = RabitqEncoder::padded_dim(dim);
        assert_eq!(p, 128);
        let signs = RabitqEncoder::generate_rot_signs(p, 42);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.29).cos()).collect();
        let ra = RabitqEncoder::fwht_rotate(&a, &signs, p);
        let rb = RabitqEncoder::fwht_rotate(&b, &signs, p);

        let orig_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let rot_dot: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum();
        assert!((orig_dot - rot_dot).abs() < 1e-4);
    }

    #[test]
    fn test_sq8_roundtrip_accuracy() {
        let residuals = vec![-2.0f32, 0.0, 2.0, 4.0, -1.0, 0.5, 1.0, 2.0];
        let sq = ScalarQuantizer8::fit(&residuals, 4);
        let codes = sq.encode(&residuals);
        assert_eq!(codes.len(), residuals.len());
        for d in 0..4 {
            let rec = sq.mins[d] + sq.scales[d] * (codes[d] as f32);
            assert!((rec - residuals[d]).abs() < 0.03);
        }
    }

    #[test]
    fn test_fp16_conversion_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.5, 0.125, 42.25, -100.5] {
            let h = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(h);
            assert!((v - back).abs() < 1e-2, "v={v} back={back}");
        }
    }
}
