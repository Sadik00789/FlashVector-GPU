use pyo3::prelude::*;
use pyo3::types::PyAny;

pub struct TensorView {
    pub ptr: *const f32,
    pub num_vectors: usize,
    pub dim: usize,
    #[allow(dead_code)]
    pub is_cuda: bool,
}

impl TensorView {
    /// Extract contiguous float32 data pointer and shape from a PyTorch Tensor (CPU/CUDA) or NumPy array
    pub fn from_pyany(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 1. Check for PyTorch Tensor (has .data_ptr(), .shape, .is_cuda)
        if obj.hasattr("data_ptr")? && obj.hasattr("shape")? {
            let dtype = obj.getattr("dtype")?.to_string();
            if !dtype.contains("float32") {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected float32 tensor (torch.float32)",
                ));
            }

            let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
            if shape.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Expected 2D tensor of shape [num_vectors, dim]",
                ));
            }

            let is_cuda: bool = if obj.hasattr("is_cuda")? {
                obj.getattr("is_cuda")?.extract().unwrap_or(false)
            } else {
                false
            };

            // Ensure contiguous memory layout
            let is_contiguous: bool = obj.call_method0("is_contiguous")?.extract()?;
            let tensor_obj = if !is_contiguous {
                obj.call_method0("contiguous")?
            } else {
                obj.clone()
            };

            let data_ptr: usize = tensor_obj.call_method0("data_ptr")?.extract()?;

            return Ok(Self {
                ptr: data_ptr as *const f32,
                num_vectors: shape[0],
                dim: shape[1],
                is_cuda,
            });
        }

        // 2. Check for NumPy ndarray (using __array_interface__)
        if obj.hasattr("__array_interface__")? {
            let dict = obj.getattr("__array_interface__")?;
            let typestr: String = dict.get_item("typestr")?.extract()?;
            if !typestr.contains("f4") {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected float32 numpy array (np.float32)",
                ));
            }

            let shape: Vec<usize> = dict.get_item("shape")?.extract()?;
            if shape.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Expected 2D array of shape [num_vectors, dim]",
                ));
            }

            let data_tuple: (usize, bool) = dict.get_item("data")?.extract()?;
            let ptr = data_tuple.0 as *const f32;

            return Ok(Self {
                ptr,
                num_vectors: shape[0],
                dim: shape[1],
                is_cuda: false,
            });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported tensor type. Expected PyTorch Tensor or NumPy array.",
        ))
    }
}
