use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use parking_lot::RwLock;

use engine::{GpuVectorIndex, IndexConfig, MetricType};
use crate::dlpack::TensorView;

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
            nbits_pq: 8,
            metric: metric_enum as u32,
        };

        let index = GpuVectorIndex::new(config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init error: {:?}", e)))?;

        Ok(Self {
            inner: RwLock::new(index),
            dim: dim as usize,
        })
    }

    /// Ingest and build index from PyTorch Tensor or NumPy array
    pub fn build(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = TensorView::from_pyany(data)?;
        if view.dim != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, view.dim
            )));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim)
        };

        let mut idx = self.inner.write();
        idx.build(slice)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Build error: {:?}", e)))?;

        Ok(())
    }

    /// Batched GPU HNSW search across all queries in a single CUDA grid launch
    #[pyo3(signature = (query, top_k=10, ef_search=64))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        ef_search: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        if view.dim != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, view.dim
            )));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim)
        };

        let idx = self.inner.read();
        let results = idx.batch_search(slice, view.num_vectors, top_k, ef_search)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Batch search error: {:?}", e)))?;

        let mut all_labels = Vec::with_capacity(view.num_vectors * top_k);
        let mut all_dists = Vec::with_capacity(view.num_vectors * top_k);

        for r in results {
            all_labels.push(r.id);
            all_dists.push(r.distance);
        }

        let labels_arr = numpy::PyArray1::from_vec_bound(py, all_labels);
        let dists_arr = numpy::PyArray1::from_vec_bound(py, all_dists);

        let labels_2d = labels_arr.reshape([view.num_vectors, top_k])?;
        let dists_2d = dists_arr.reshape([view.num_vectors, top_k])?;

        Ok((labels_2d.into(), dists_2d.into()))
    }

    /// Batched GPU IVF-PQ ADC search across all queries in a single CUDA grid launch
    #[pyo3(signature = (query, top_k=10, nprobe=8))]
    pub fn search_ivf(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        nprobe: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        if view.dim != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, view.dim
            )));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim)
        };

        let idx = self.inner.read();
        let results = idx.batch_search_ivf(slice, view.num_vectors, top_k, nprobe)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Batch IVF search error: {:?}", e)))?;

        let mut all_labels = Vec::with_capacity(view.num_vectors * top_k);
        let mut all_dists = Vec::with_capacity(view.num_vectors * top_k);

        for r in results {
            all_labels.push(r.id);
            all_dists.push(r.distance);
        }

        let labels_arr = numpy::PyArray1::from_vec_bound(py, all_labels);
        let dists_arr = numpy::PyArray1::from_vec_bound(py, all_dists);

        let labels_2d = labels_arr.reshape([view.num_vectors, top_k])?;
        let dists_2d = dists_arr.reshape([view.num_vectors, top_k])?;

        Ok((labels_2d.into(), dists_2d.into()))
    }

    pub fn num_vectors(&self) -> usize {
        self.inner.read().num_vectors()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_stats(&self) -> PyResult<String> {
        let stats = self.inner.read().metrics().get_stats();
        serde_json::to_string(&stats).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
