"""NumPy brute-force fake implementing the FlashVectorGPU backend protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class FakeBackend:
    """In-memory NumPy reference backend supporting add, delete, filter_mask, and 3 metrics."""

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        capacity: int = 100_000,
        *,
        supports_filter_mask: bool = True,
        supports_add: bool = True,
        supports_delete: bool = True,
    ) -> None:
        self._dim = int(dim)
        self._metric = str(metric).lower()
        self._capacity = int(capacity)
        self._supports_filter_mask = supports_filter_mask
        self._supports_add = supports_add
        self._supports_delete = supports_delete
        self._vectors: NDArray[np.float32] = np.empty((0, self._dim), dtype=np.float32)
        self._deleted: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self.filter_mask_calls = 0
        self.build_calls = 0

        if not supports_add:
            self.add = None  # type: ignore[assignment]
        if not supports_delete:
            self.delete = None  # type: ignore[assignment]

    def dim(self) -> int:
        return self._dim

    def num_vectors(self) -> int:
        return int(len(self._vectors))

    def build(self, data: Any) -> None:
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(f"expected shape (N, {self._dim}), got {arr.shape}")
        self._vectors = arr.copy()
        self._deleted = np.zeros(len(arr), dtype=bool)
        self.build_calls += 1

    def add(self, data: Any) -> NDArray[np.uint32]:
        if not self._supports_add:
            raise NotImplementedError("add disabled on this fake")
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        start = len(self._vectors)
        self._vectors = (
            arr.copy() if start == 0 else np.vstack([self._vectors, arr]).astype(np.float32)
        )
        self._deleted = np.concatenate([self._deleted, np.zeros(len(arr), dtype=bool)])
        return np.arange(start, start + len(arr), dtype=np.uint32)

    def delete(self, ids: Any) -> None:
        if not self._supports_delete:
            raise NotImplementedError("delete disabled on this fake")
        for idx in np.asarray(ids, dtype=np.int64):
            if 0 <= idx < len(self._deleted):
                self._deleted[idx] = True

    def _compute_distances(self, q: NDArray[np.float32]) -> NDArray[np.float32]:
        # q: (Q, D), vectors: (N, D) -> (Q, N)
        x = self._vectors.astype(np.float64)
        qf = q.astype(np.float64)
        if self._metric == "cosine":
            xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            qn = qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-12)
            return (1.0 - (qn @ xn.T)).astype(np.float32)
        if self._metric in ("ip", "inner_product", "dot"):
            return (-(qf @ x.T)).astype(np.float32)
        # Squared L2
        q2 = np.sum(qf * qf, axis=1, keepdims=True)
        x2 = np.sum(x * x, axis=1, keepdims=True).T
        dists = np.maximum(0.0, q2 + x2 - 2.0 * (qf @ x.T))
        return dists.astype(np.float32)

    def search(
        self,
        query: Any,
        top_k: int = 10,
        ef_search: int = 64,
        filter_mask: Any | None = None,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
        del ef_search
        if filter_mask is not None and not self._supports_filter_mask:
            raise TypeError("unexpected keyword argument 'filter_mask'")

        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        nq = q.shape[0]
        n = len(self._vectors)

        labels = np.full((nq, top_k), 0xFFFFFFFF, dtype=np.uint32)
        dists = np.full((nq, top_k), 1e30, dtype=np.float32)
        if n == 0 or top_k <= 0:
            return labels, dists

        all_dists = self._compute_distances(q)
        allowed = ~self._deleted
        if filter_mask is not None:
            self.filter_mask_calls += 1
            fm = np.asarray(filter_mask, dtype=bool)
            allowed = allowed & fm[:n]

        valid_indices = np.flatnonzero(allowed)
        if len(valid_indices) == 0:
            return labels, dists

        take = min(top_k, len(valid_indices))
        for qi in range(nq):
            sub_dists = all_dists[qi, valid_indices]
            order = np.argsort(sub_dists, kind="stable")[:take]
            labels[qi, :take] = valid_indices[order].astype(np.uint32)
            dists[qi, :take] = sub_dists[order]
        return labels, dists
