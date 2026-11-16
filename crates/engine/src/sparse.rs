//! CSR sparse lexical vector index (SPLADE / BM25) and hybrid RRF result types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ffi::GpuSearchResult;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparseHit {
    pub id: u32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HybridHit {
    pub id: u32,
    pub rrf_score: f32,
}

/// Host-side CSR sparse document matrix ready for GPU upload or persistence.
#[derive(Debug, Clone, Default)]
pub struct SparseCsrData {
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub num_docs: usize,
}

impl SparseCsrData {
    /// Build CSR arrays from per-document `(term_id, weight)` pairs, merging
    /// duplicate term IDs and sorting each row by `term_id`.
    pub fn from_rows(rows: &[Vec<(u32, f32)>]) -> Self {
        let num_docs = rows.len();
        let mut indptr = Vec::with_capacity(num_docs + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0u32);

        for row in rows {
            let mut merged: Vec<(u32, f32)> = row.clone();
            merged.sort_unstable_by_key(|&(term, _)| term);
            let mut out: Vec<(u32, f32)> = Vec::with_capacity(merged.len());
            for (term, weight) in merged {
                if weight == 0.0 {
                    continue;
                }
                if let Some(last) = out.last_mut() {
                    if last.0 == term {
                        last.1 += weight;
                        continue;
                    }
                }
                out.push((term, weight));
            }
            for (term, weight) in out {
                indices.push(term);
                values.push(weight);
            }
            indptr.push(indices.len() as u32);
        }

        Self {
            indptr,
            indices,
            values,
            num_docs,
        }
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Pack a batch of sparse queries into CSR arrays for `cuda_sparse_search_batch`.
pub fn pack_sparse_queries(queries: &[&[(u32, f32)]]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut indptr = Vec::with_capacity(queries.len() + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0u32);

    for q in queries {
        let mut sorted: Vec<(u32, f32)> = q.to_vec();
        sorted.sort_unstable_by_key(|&(term, _)| term);
        for (term, weight) in sorted {
            if weight != 0.0 {
                indices.push(term);
                values.push(weight);
            }
        }
        indptr.push(indices.len() as u32);
    }

    (indptr, indices, values)
}

/// CPU reference Reciprocal Rank Fusion (used for CPU unit tests and validation).
pub fn cpu_rrf_fuse(
    dense: &[GpuSearchResult],
    sparse: &[GpuSearchResult],
    top_k: usize,
    rrf_k: f32,
    dense_weight: f32,
    sparse_weight: f32,
) -> Vec<HybridHit> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for (rank, r) in dense.iter().enumerate() {
        if r.id != u32::MAX {
            *scores.entry(r.id).or_insert(0.0) += dense_weight / (rrf_k + (rank + 1) as f32);
        }
    }
    for (rank, r) in sparse.iter().enumerate() {
        if r.id != u32::MAX {
            *scores.entry(r.id).or_insert(0.0) += sparse_weight / (rrf_k + (rank + 1) as f32);
        }
    }
    let mut out: Vec<HybridHit> = scores
        .into_iter()
        .map(|(id, rrf_score)| HybridHit { id, rrf_score })
        .collect();
    out.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    out.truncate(top_k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_dedup_and_rrf() {
        let csr = SparseCsrData::from_rows(&[
            vec![(10, 0.5), (2, 1.0), (10, 0.25)],
            vec![(3, 0.0), (7, 2.0)],
        ]);
        assert_eq!(csr.indptr, vec![0, 2, 3]);
        assert_eq!(csr.indices, vec![2, 10, 7]);
        assert_eq!(csr.values, vec![1.0, 0.75, 2.0]);

        let dense = vec![
            GpuSearchResult {
                id: 5,
                distance: 0.1,
            },
            GpuSearchResult {
                id: 2,
                distance: 0.2,
            },
        ];
        let sparse = vec![
            GpuSearchResult {
                id: 2,
                distance: -3.0,
            },
            GpuSearchResult {
                id: 9,
                distance: -1.0,
            },
        ];
        let fused = cpu_rrf_fuse(&dense, &sparse, 3, 60.0, 1.0, 1.0);
        assert_eq!(fused[0].id, 2); // appears in both lists
        assert!(fused[0].rrf_score > fused[1].rrf_score);
    }
}
