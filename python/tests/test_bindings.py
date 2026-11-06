"""
FlashVector-GPU PyO3 Python & PyTorch Tensor Interop Verification Suite
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gpu_vector_index
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index binary module not built")
def test_index_numpy_build_and_search():
    dim = 128
    num_vectors = 1000
    top_k = 10

    data = np.random.randn(num_vectors, dim).astype(np.float32)
    query = np.random.randn(5, dim).astype(np.float32)

    index = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=num_vectors, m=16, ef_construction=64)
    index.build(data)

    labels, dists = index.search(query, top_k=top_k, ef_search=32)

    assert labels.shape == (5, top_k)
    assert dists.shape == (5, top_k)
    assert labels.dtype == np.uint32
    assert dists.dtype == np.float32


@pytest.mark.skipif(not HAS_BINDINGS or not HAS_TORCH, reason="PyTorch or bindings unavailable")
def test_index_pytorch_tensor_interop():
    dim = 128
    num_vectors = 500
    top_k = 5

    # CPU Tensor
    tensor_data = torch.randn(num_vectors, dim, dtype=torch.float32)
    tensor_query = torch.randn(2, dim, dtype=torch.float32)

    index = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=num_vectors, m=16)
    index.build(tensor_data)

    labels, dists = index.search(tensor_query, top_k=top_k, ef_search=32)
    assert labels.shape == (2, top_k)


@pytest.mark.skipif(not HAS_BINDINGS or not HAS_TORCH, reason="PyTorch or bindings unavailable")
def test_cuda_tensor_build_and_search():
    """Regression: CUDA tensors must not segfault (Bug #1)."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")
    dim = 128
    num_vectors = 500
    top_k = 5
    data = torch.randn(num_vectors, dim, dtype=torch.float32, device="cuda")
    query = torch.randn(2, dim, dtype=torch.float32, device="cuda")
    index = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=num_vectors, m=16)
    index.build(data)  # Must not segfault
    labels, dists = index.search(query, top_k=top_k, ef_search=32)
    assert labels.shape == (2, top_k)
    assert dists.shape == (2, top_k)


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index not built")
def test_empty_tensor_build():
    """Edge case: building with 0 vectors should raise, not crash."""
    dim = 128
    data = np.zeros((0, dim), dtype=np.float32)
    index = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=100, m=16)
    with pytest.raises(Exception):
        index.build(data)


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index not built")
def test_cosine_metric_search():
    """Regression: cosine metric must return valid cosine distances."""
    dim = 64
    num_vectors = 200
    data = np.random.randn(num_vectors, dim).astype(np.float32)
    query = np.random.randn(1, dim).astype(np.float32)
    index = gpu_vector_index.FlashVectorGPU(
        dim=dim, max_elements=num_vectors, m=16, metric="cosine"
    )
    index.build(data)
    labels, dists = index.search(query, top_k=5, ef_search=32)
    assert labels.shape == (1, 5)
    assert np.all(dists >= -0.01), f"Cosine distance out of range: {dists}"
    assert np.all(dists <= 2.01), f"Cosine distance out of range: {dists}"


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index not built")
def test_large_top_k():
    """Regression: top_k > 64 must return full requested result count."""
    dim = 64
    num_vectors = 500
    top_k = 100
    data = np.random.randn(num_vectors, dim).astype(np.float32)
    query = np.random.randn(1, dim).astype(np.float32)
    index = gpu_vector_index.FlashVectorGPU(dim=dim, max_elements=num_vectors, m=16)
    index.build(data)
    labels, dists = index.search(query, top_k=top_k, ef_search=256)
    assert labels.shape == (1, top_k)
    valid = np.sum(labels[0] != 0xFFFFFFFF)
    assert valid >= top_k * 0.9, f"Only {valid}/{top_k} valid results"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
