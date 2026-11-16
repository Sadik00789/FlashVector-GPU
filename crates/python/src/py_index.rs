use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::dlpack::TensorView;
use engine::ffi::{gpu_device_synchronize, gpu_memcpy_d2h_async};
use engine::{GpuSearchResult, GpuVectorIndex, IndexConfig, IvfQuantMode, MetricType};

fn copy_view_to_host(view: &TensorView) -> PyResult<Vec<f32>> {
    let total_elements = view.num_vectors * view.dim;
    if total_elements == 0 {
        return Ok(Vec::new());
    }
    if view.is_cuda {
        let mut staging = vec![0.0f32; total_elements];
        gpu_device_synchronize().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA sync failed: {:?}", e))
        })?;
        gpu_memcpy_d2h_async(
            staging.as_mut_ptr(),
            view.ptr,
            total_elements,
            std::ptr::null_mut(),
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H copy failed: {:?}", e))
        })?;
        gpu_device_synchronize().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA sync failed: {:?}", e))
        })?;
        Ok(staging)
    } else {
        // SAFETY: TensorView::from_pyany guarantees contiguous float32 memory kept alive by caller.
        let slice = unsafe { std::slice::from_raw_parts(view.ptr, total_elements) };
        Ok(slice.to_vec())
    }
}

fn validate_query_view(expected_dim: usize, view: &TensorView) -> PyResult<()> {
    if view.dim != expected_dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: expected {}, got {}",
            expected_dim, view.dim
        )));
    }
    Ok(())
}

fn empty_pair(py: Python<'_>, top_k: usize) -> PyResult<(PyObject, PyObject)> {
    let empty_labels = PyArray1::from_vec_bound(py, Vec::<u32>::new()).reshape([0, top_k])?;
    let empty_dists = PyArray1::from_vec_bound(py, Vec::<f32>::new()).reshape([0, top_k])?;
    Ok((empty_labels.into(), empty_dists.into()))
}

fn results_to_pair(
    py: Python<'_>,
    results: Vec<GpuSearchResult>,
    num_queries: usize,
    top_k: usize,
) -> PyResult<(PyObject, PyObject)> {
    let mut all_labels = Vec::with_capacity(num_queries * top_k);
    let mut all_dists = Vec::with_capacity(num_queries * top_k);
    for r in results {
        all_labels.push(r.id);
        all_dists.push(r.distance);
    }
    let labels_2d = PyArray1::from_vec_bound(py, all_labels).reshape([num_queries, top_k])?;
    let dists_2d = PyArray1::from_vec_bound(py, all_dists).reshape([num_queries, top_k])?;
    Ok((labels_2d.into(), dists_2d.into()))
}

fn pack_bools_to_words(bools: &[bool]) -> Vec<u32> {
    let mut words = vec![0u32; bools.len().div_ceil(32)];
    for (i, &allowed) in bools.iter().enumerate() {
        if allowed {
            words[i >> 5] |= 1u32 << (i & 31);
        }
    }
    words
}

fn mask_from_pyany(mask: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u32>>> {
    let Some(obj) = mask else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(bool_arr) = obj.extract::<PyReadonlyArray1<'_, bool>>() {
        let slice = bool_arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("filter_mask bool array must be contiguous")
        })?;
        return Ok(Some(pack_bools_to_words(slice)));
    }
    if let Ok(u32_arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
        let slice = u32_arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("filter_mask uint32 array must be contiguous")
        })?;
        return Ok(Some(slice.to_vec()));
    }
    if let Ok(bool_vec) = obj.extract::<Vec<bool>>() {
        return Ok(Some(pack_bools_to_words(&bool_vec)));
    }
    if let Ok(u32_vec) = obj.extract::<Vec<u32>>() {
        return Ok(Some(u32_vec));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "filter_mask must be a 1D bool mask or packed uint32 bitset array",
    ))
}

fn ids_from_pyany(ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = ids.extract::<PyReadonlyArray1<'_, u32>>() {
        let slice = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("ids uint32 array must be contiguous")
        })?;
        return Ok(slice.to_vec());
    }
    if let Ok(arr) = ids.extract::<PyReadonlyArray1<'_, i64>>() {
        let slice = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("ids int64 array must be contiguous")
        })?;
        return slice
            .iter()
            .map(|&v| {
                u32::try_from(v).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!("id out of u32 range: {v}"))
                })
            })
            .collect();
    }
    ids.extract::<Vec<u32>>()
}

fn parse_ivf_mode(mode: &str) -> PyResult<IvfQuantMode> {
    match mode.to_lowercase().as_str() {
        "pq" | "ivf_pq" | "ivf-pq" => Ok(IvfQuantMode::Pq),
        "sq8" | "sq" | "ivf_sq8" | "ivf-sq8" => Ok(IvfQuantMode::Sq8),
        "rabitq" | "ivf_rabitq" | "ivf-rabitq" => Ok(IvfQuantMode::Rabitq),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported IVF mode {mode:?}; expected 'pq', 'sq8', or 'rabitq'"
        ))),
    }
}

fn engine_err(prefix: &str, err: engine::CudaError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{prefix}: {err:?}"))
}

#[pyclass]
pub struct FlashVectorGPU {
    inner: RwLock<GpuVectorIndex>,
    dim: usize,
}

#[pymethods]
impl FlashVectorGPU {
    #[new]
    #[pyo3(signature = (dim, max_elements=100_000, m=32, ef_construction=128, nlist=256, m_pq=16, metric="l2"))]
    pub fn new(
        dim: u32,
        max_elements: u32,
        m: u32,
        ef_construction: u32,
        nlist: u32,
        m_pq: u32,
        metric: &str,
    ) -> PyResult<Self> {
        let metric_enum = match metric.to_lowercase().as_str() {
            "l2" | "euclidean" => MetricType::L2,
            "cosine" => MetricType::Cosine,
            "ip" | "inner_product" | "dot" => MetricType::InnerProduct,
            _ => MetricType::L2,
        };

        let config = IndexConfig {
            dim,
            max_elements,
            m,
            ef_construction,
            nlist,
            m_pq,
            metric: metric_enum as u32,
        };

        let index = GpuVectorIndex::new(config).map_err(|e| engine_err("CUDA init error", e))?;

        Ok(Self {
            inner: RwLock::new(index),
            dim: dim as usize,
        })
    }

    /// Ingest and build index from PyTorch Tensor or NumPy array.
    pub fn build(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = TensorView::from_pyany(data)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dataset cannot be empty",
            ));
        }

        let host_data: Vec<f32>;
        let slice: &[f32] = if view.is_cuda {
            host_data = copy_view_to_host(&view)?;
            &host_data
        } else {
            // SAFETY: TensorView::from_pyany only returns host pointers for contiguous float32 tensors.
            unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
        };

        let mut idx = self.inner.write();
        idx.build(slice).map_err(|e| engine_err("Build error", e))?;
        Ok(())
    }

    /// Append vectors dynamically into the GPU delta buffer and return assigned uint32 IDs.
    pub fn add(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let view = TensorView::from_pyany(data)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return Ok(PyArray1::from_vec_bound(py, Vec::<u32>::new()).into());
        }
        let host_data: Vec<f32>;
        let slice: &[f32] = if view.is_cuda {
            host_data = copy_view_to_host(&view)?;
            &host_data
        } else {
            unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
        };

        let mut idx = self.inner.write();
        let assigned = idx
            .add_vectors(slice)
            .map_err(|e| engine_err("Add vectors error", e))?;
        Ok(PyArray1::from_vec_bound(py, assigned).into())
    }

    /// Tombstone-delete logical vector IDs without rebuilding the GPU index.
    pub fn delete(&self, ids: &Bound<'_, PyAny>) -> PyResult<usize> {
        let parsed = ids_from_pyany(ids)?;
        let mut idx = self.inner.write();
        idx.delete_ids(&parsed)
            .map_err(|e| engine_err("Delete error", e))
    }

    /// Synchronously compact tombstoned vectors and merge the live delta buffer into main VRAM.
    pub fn compact(&self) -> PyResult<()> {
        let mut idx = self.inner.write();
        idx.compact().map_err(|e| engine_err("Compact error", e))
    }

    /// Toggle Stage-2 exact reranking against mapped pinned host vectors instead of VRAM vectors.
    pub fn set_mapped_rerank(&self, enabled: bool) {
        self.inner.write().set_mapped_rerank(enabled);
    }

    /// Save the built GPU index (graphs, centroids, quantizers, tombstones, delta) to a 64B-aligned snapshot.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .read()
            .save_snapshot(path)
            .map_err(|e| engine_err("Snapshot save error", e))
    }

    /// Memory-map and restore an index snapshot directly into GPU memory without rebuilding.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let idx = GpuVectorIndex::load_snapshot(path)
            .map_err(|e| engine_err("Snapshot load error", e))?;
        let dim = idx.dim();
        Ok(Self {
            inner: RwLock::new(idx),
            dim,
        })
    }

    /// Batched GPU HNSW search with optional on-device boolean/bitmask filter.
    #[pyo3(signature = (query, top_k=10, ef_search=64, filter_mask=None))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;

        let idx = self.inner.read();
        let results = if view.is_cuda {
            idx.batch_search_device_ptr_filtered(
                view.ptr,
                view.num_vectors,
                top_k,
                ef_search,
                mask_words.as_deref(),
            )
            .map_err(|e| engine_err("Batch search error", e))?
        } else {
            let slice =
                unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) };
            idx.batch_search_filtered(
                slice,
                view.num_vectors,
                top_k,
                ef_search,
                mask_words.as_deref(),
            )
            .map_err(|e| engine_err("Batch search error", e))?
        };
        results_to_pair(py, results, view.num_vectors, top_k)
    }

    /// Batched GPU CAGRA flat fixed-degree graph search with optional on-device filter.
    #[pyo3(signature = (query, top_k=10, ef_search=64, filter_mask=None))]
    pub fn search_cagra(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        ef_search: usize,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;
        let host_data: Vec<f32>;
        let slice: &[f32] = if view.is_cuda {
            host_data = copy_view_to_host(&view)?;
            &host_data
        } else {
            unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
        };

        let results = self
            .inner
            .read()
            .batch_search_cagra(
                slice,
                view.num_vectors,
                top_k,
                ef_search,
                mask_words.as_deref(),
            )
            .map_err(|e| engine_err("CAGRA search error", e))?;
        results_to_pair(py, results, view.num_vectors, top_k)
    }

    /// Batched GPU IVF search supporting mode="pq" | "sq8" | "rabitq" and optional bitmask filtering.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, top_k=10, nprobe=8, rerank_k=32, mode="pq", filter_mask=None))]
    pub fn search_ivf(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
        mode: &str,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let quant_mode = parse_ivf_mode(mode)?;
        let view = TensorView::from_pyany(query)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;

        let idx = self.inner.read();
        let results = if view.is_cuda && quant_mode == IvfQuantMode::Pq && mask_words.is_none() {
            idx.batch_search_ivf_device_ptr(view.ptr, view.num_vectors, top_k, nprobe, rerank_k)
                .map_err(|e| engine_err("Batch IVF search error", e))?
        } else {
            let host_data: Vec<f32>;
            let slice: &[f32] = if view.is_cuda {
                host_data = copy_view_to_host(&view)?;
                &host_data
            } else {
                unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
            };
            idx.batch_search_ivf_mode(
                slice,
                view.num_vectors,
                top_k,
                nprobe,
                rerank_k,
                quant_mode,
                mask_words.as_deref(),
            )
            .map_err(|e| engine_err("Batch IVF search error", e))?
        };
        results_to_pair(py, results, view.num_vectors, top_k)
    }

    /// Exact batched Tensor Core FP16 WMMA search with optional on-device filter.
    #[pyo3(signature = (query, top_k=10, filter_mask=None))]
    pub fn search_fp16(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors == 0 {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;
        let host_data: Vec<f32>;
        let slice: &[f32] = if view.is_cuda {
            host_data = copy_view_to_host(&view)?;
            &host_data
        } else {
            unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
        };

        let results = self
            .inner
            .read()
            .batch_search_fp16_tc(slice, view.num_vectors, top_k, mask_words.as_deref())
            .map_err(|e| engine_err("FP16 TC search error", e))?;
        results_to_pair(py, results, view.num_vectors, top_k)
    }

    /// Build the GPU CSR sparse lexical index from per-document `(term_id, weight)` lists.
    pub fn build_sparse(&self, rows: Vec<Vec<(u32, f32)>>) -> PyResult<()> {
        self.inner
            .write()
            .build_sparse(&rows)
            .map_err(|e| engine_err("Sparse build error", e))
    }

    /// Batched GPU CSR sparse dot-product search (returns `(labels_2d, scores_2d)`).
    #[pyo3(signature = (queries, top_k=10, filter_mask=None))]
    pub fn search_sparse(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<(u32, f32)>>,
        top_k: usize,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        if queries.is_empty() {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;
        let q_refs: Vec<&[(u32, f32)]> = queries.iter().map(|q| q.as_slice()).collect();
        let hits = self
            .inner
            .read()
            .search_sparse(&q_refs, top_k, mask_words.as_deref())
            .map_err(|e| engine_err("Sparse search error", e))?;

        let num_queries = queries.len();
        let mut labels = Vec::with_capacity(num_queries * top_k);
        let mut scores = Vec::with_capacity(num_queries * top_k);
        for row in hits {
            for i in 0..top_k {
                if let Some(h) = row.get(i) {
                    labels.push(h.id);
                    scores.push(h.score);
                } else {
                    labels.push(u32::MAX);
                    scores.push(0.0);
                }
            }
        }
        let labels_2d = PyArray1::from_vec_bound(py, labels).reshape([num_queries, top_k])?;
        let scores_2d = PyArray1::from_vec_bound(py, scores).reshape([num_queries, top_k])?;
        Ok((labels_2d.into(), scores_2d.into()))
    }

    /// Hybrid dense + CSR sparse search fused directly on GPU with Reciprocal Rank Fusion (RRF).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dense_query,
        sparse_queries,
        top_k=10,
        candidate_k=64,
        ef_search=64,
        rrf_k=60.0,
        dense_weight=1.0,
        sparse_weight=1.0,
        filter_mask=None
    ))]
    pub fn hybrid_search(
        &self,
        py: Python<'_>,
        dense_query: &Bound<'_, PyAny>,
        sparse_queries: Vec<Vec<(u32, f32)>>,
        top_k: usize,
        candidate_k: usize,
        ef_search: usize,
        rrf_k: f32,
        dense_weight: f32,
        sparse_weight: f32,
        filter_mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(dense_query)?;
        validate_query_view(self.dim, &view)?;
        if view.num_vectors != sparse_queries.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Batch size mismatch: {} dense queries vs {} sparse queries",
                view.num_vectors,
                sparse_queries.len()
            )));
        }
        if view.num_vectors == 0 {
            return empty_pair(py, top_k);
        }
        let mask_words = mask_from_pyany(filter_mask)?;
        let host_data: Vec<f32>;
        let slice: &[f32] = if view.is_cuda {
            host_data = copy_view_to_host(&view)?;
            &host_data
        } else {
            unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) }
        };
        let q_refs: Vec<&[(u32, f32)]> = sparse_queries.iter().map(|q| q.as_slice()).collect();

        let fused = self
            .inner
            .read()
            .hybrid_search(
                slice,
                &q_refs,
                top_k,
                candidate_k,
                ef_search,
                rrf_k,
                dense_weight,
                sparse_weight,
                mask_words.as_deref(),
            )
            .map_err(|e| engine_err("Hybrid search error", e))?;

        let num_queries = view.num_vectors;
        let mut labels = Vec::with_capacity(num_queries * top_k);
        let mut scores = Vec::with_capacity(num_queries * top_k);
        for row in fused {
            for i in 0..top_k {
                if let Some(h) = row.get(i) {
                    labels.push(h.id);
                    scores.push(h.rrf_score);
                } else {
                    labels.push(u32::MAX);
                    scores.push(0.0);
                }
            }
        }
        let labels_2d = PyArray1::from_vec_bound(py, labels).reshape([num_queries, top_k])?;
        let scores_2d = PyArray1::from_vec_bound(py, scores).reshape([num_queries, top_k])?;
        Ok((labels_2d.into(), scores_2d.into()))
    }

    pub fn num_vectors(&self) -> usize {
        self.inner.read().num_vectors()
    }

    pub fn delta_len(&self) -> usize {
        self.inner.read().delta_len()
    }

    pub fn tombstone_count(&self) -> usize {
        self.inner.read().tombstone_count()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_stats(&self) -> PyResult<String> {
        let stats = self.inner.read().metrics().get_stats();
        serde_json::to_string(&stats)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
