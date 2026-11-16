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

    index = gpu_vector_index.FlashVectorGPU(
        dim=dim, max_elements=num_vectors, m=16, ef_construction=64
    )
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
    with pytest.raises(ValueError):
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


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index not built")
def test_large_top_k_ivf():
    """Regression: block merge must not drop IVF-PQ candidates for top_k > 32."""
    dim = 64
    num_vectors = 1000
    top_k = 100
    data = np.random.randn(num_vectors, dim).astype(np.float32)
    query = np.random.randn(1, dim).astype(np.float32)
    index = gpu_vector_index.FlashVectorGPU(
        dim=dim, max_elements=num_vectors, m=16, nlist=16, m_pq=8
    )
    index.build(data)
    labels, dists = index.search_ivf(query, top_k=top_k, nprobe=16)
    assert labels.shape == (1, top_k)
    valid = np.sum(labels[0] != 0xFFFFFFFF)
    assert valid >= top_k * 0.9, f"Only {valid}/{top_k} valid IVF results"
    assert np.all(dists[0][:-1] <= dists[0][1:]), "IVF distances must be sorted"


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index not built")
def test_new_algorithms_filtering_delta_persistence_and_hybrid(tmp_path):
    """Verify CAGRA, IVF modes, FP16-TC, bitmasks, delta/compact, mmap & sparse RRF."""
    rng = np.random.default_rng(7)
    dim = 64
    num_vectors = 400
    data = rng.standard_normal((num_vectors, dim)).astype(np.float32)

    index = gpu_vector_index.FlashVectorGPU(
        dim=dim, max_elements=num_vectors, m=16, nlist=8, m_pq=8
    )
    index.build(data)

    q = data[:2].copy()
    mask = np.zeros(num_vectors, dtype=bool)
    mask[10:200] = True

    # CAGRA, IVF modes, FP16 Tensor Core with on-device bool filter_mask
    for labels, _dists in [
        index.search(q, top_k=5, ef_search=48, filter_mask=mask),
        index.search_cagra(q, top_k=5, ef_search=48, filter_mask=mask),
        index.search_ivf(q, top_k=5, nprobe=8, rerank_k=64, mode="sq8", filter_mask=mask),
        index.search_ivf(q, top_k=5, nprobe=8, rerank_k=64, mode="rabitq", filter_mask=mask),
        index.search_fp16(q, top_k=5, filter_mask=mask),
    ]:
        assert labels.shape == (2, 5)
        valid = labels[labels != 0xFFFFFFFF]
        assert len(valid) > 0
        assert np.all((valid >= 10) & (valid < 200))

    # Dynamic delta insert + tombstone delete + snapshot roundtrip + compact
    extra = np.full((1, dim), 25.0, dtype=np.float32)
    assigned = index.add(extra)
    assert assigned.tolist() == [num_vectors]
    labels, _ = index.search(extra, top_k=3, ef_search=32)
    assert labels[0, 0] == num_vectors

    assert index.delete(np.array([0], dtype=np.uint32)) == 1
    labels_del, _ = index.search(data[:1], top_k=5, ef_search=32)
    assert 0 not in labels_del[0]

    snap = tmp_path / "bindings_snap.fv"
    index.save(str(snap))
    reloaded = gpu_vector_index.FlashVectorGPU.load(str(snap))
    reloaded.compact()
    assert reloaded.tombstone_count() == 0
    assert reloaded.delta_len() == 0

    # Sparse CSR + GPU RRF hybrid search
    sparse_rows = [[(int(i % 5), 0.5)] for i in range(reloaded.num_vectors())]
    sparse_rows[42].append((777, 8.0))
    reloaded.build_sparse(sparse_rows)
    s_labels, _s_scores = reloaded.search_sparse([[(777, 1.5)]], top_k=5)
    assert s_labels[0, 0] == 42
    h_labels, h_scores = reloaded.hybrid_search(data[42:43], [[(777, 1.5)]], top_k=5)
    assert h_labels[0, 0] == 42
    assert np.all(h_scores[0, :-1] >= h_scores[0, 1:])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
