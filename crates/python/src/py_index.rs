use numpy::PyArrayMethods;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::dlpack::TensorView;
use engine::ffi::{gpu_device_synchronize, gpu_memcpy_d2h_async};
use engine::{GpuVectorIndex, IndexConfig, MetricType};

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

        let index = GpuVectorIndex::new(config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init error: {:?}", e))
        })?;

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
        if view.num_vectors == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dataset cannot be empty",
            ));
        }

        let mut host_data: Vec<f32>;
        let slice: &[f32];

        if view.is_cuda {
            let total_elements = view.num_vectors * view.dim;
            host_data = vec![0.0f32; total_elements];
            // The tensor may have been produced on another CUDA stream; drain
            // outstanding work before reading it from the host.
            gpu_device_synchronize().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA sync failed: {:?}", e))
            })?;
            // The device pointer belongs to the caller's live CUDA tensor and
            // host_data is a CPU destination of the same element count.
            gpu_memcpy_d2h_async(
                host_data.as_mut_ptr(),
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
            slice = &host_data;
        } else {
            // SAFETY: TensorView::from_pyany only returns host pointers for
            // contiguous float32 tensors, and the caller keeps the tensor alive
            // for the duration of this call.
            slice = unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) };
        }

        let mut idx = self.inner.write();
        idx.build(slice).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Build error: {:?}", e))
        })?;

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
        if view.num_vectors == 0 {
            let empty_labels =
                numpy::PyArray1::from_vec_bound(py, Vec::<u32>::new()).reshape([0, top_k])?;
            let empty_dists =
                numpy::PyArray1::from_vec_bound(py, Vec::<f32>::new()).reshape([0, top_k])?;
            return Ok((empty_labels.into(), empty_dists.into()));
        }

        let idx = self.inner.read();
        let results = if view.is_cuda {
            // Zero-copy path: the CUDA tensor's device pointer is consumed directly
            // by the HNSW kernel, avoiding a D2H copy followed by an H2D re-upload.
            idx.batch_search_device_ptr(view.ptr, view.num_vectors, top_k, ef_search)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batch search error: {:?}",
                        e
                    ))
                })?
        } else {
            // SAFETY: TensorView::from_pyany only returns host pointers for
            // contiguous float32 tensors, and the caller keeps the tensor alive
            // for the duration of this call.
            let slice =
                unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) };
            idx.batch_search(slice, view.num_vectors, top_k, ef_search)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batch search error: {:?}",
                        e
                    ))
                })?
        };

        let mut all_labels = Vec::with_capacity(view.num_vectors * top_k);
        let mut all_dists = Vec::with_capacity(view.num_vectors * top_k);
        for r in results {
            all_labels.push(r.id);
            all_dists.push(r.distance);
        }

        let labels_2d =
            numpy::PyArray1::from_vec_bound(py, all_labels).reshape([view.num_vectors, top_k])?;
        let dists_2d =
            numpy::PyArray1::from_vec_bound(py, all_dists).reshape([view.num_vectors, top_k])?;
        Ok((labels_2d.into(), dists_2d.into()))
    }

    /// Batched GPU IVF-PQ ADC search with exact re-ranking across all queries
    #[pyo3(signature = (query, top_k=10, nprobe=8, rerank_k=32))]
    pub fn search_ivf(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        nprobe: usize,
        rerank_k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = TensorView::from_pyany(query)?;
        if view.dim != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, view.dim
            )));
        }
        if view.num_vectors == 0 {
            let empty_labels =
                numpy::PyArray1::from_vec_bound(py, Vec::<u32>::new()).reshape([0, top_k])?;
            let empty_dists =
                numpy::PyArray1::from_vec_bound(py, Vec::<f32>::new()).reshape([0, top_k])?;
            return Ok((empty_labels.into(), empty_dists.into()));
        }

        let idx = self.inner.read();
        let results = if view.is_cuda {
            idx.batch_search_ivf_device_ptr(view.ptr, view.num_vectors, top_k, nprobe, rerank_k)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batch IVF search error: {:?}",
                        e
                    ))
                })?
        } else {
            // SAFETY: TensorView::from_pyany only returns host pointers for
            // contiguous float32 tensors, and the caller keeps the tensor alive
            // for the duration of this call.
            let slice =
                unsafe { std::slice::from_raw_parts(view.ptr, view.num_vectors * view.dim) };
            idx.batch_search_ivf(slice, view.num_vectors, top_k, nprobe, rerank_k)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batch IVF search error: {:?}",
                        e
                    ))
                })?
        };

        let mut all_labels = Vec::with_capacity(view.num_vectors * top_k);
        let mut all_dists = Vec::with_capacity(view.num_vectors * top_k);
        for r in results {
            all_labels.push(r.id);
            all_dists.push(r.distance);
        }

        let labels_2d =
            numpy::PyArray1::from_vec_bound(py, all_labels).reshape([view.num_vectors, top_k])?;
        let dists_2d =
            numpy::PyArray1::from_vec_bound(py, all_dists).reshape([view.num_vectors, top_k])?;
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
        serde_json::to_string(&stats)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
