// pyo3 0.22's `#[pymethods]` expansion trips clippy::useless_conversion on
// `PyResult` return types; the conversion lives in generated code.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod dlpack;
mod py_index;

use py_index::FlashVectorGPU;

#[pymodule]
fn gpu_vector_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlashVectorGPU>()?;
    Ok(())
}
