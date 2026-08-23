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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
