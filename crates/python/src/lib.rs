use pyo3::prelude::*;

mod dlpack;
mod py_index;

use py_index::FlashVectorGPU;

#[pymodule]
fn gpu_vector_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlashVectorGPU>()?;
    Ok(())
}
