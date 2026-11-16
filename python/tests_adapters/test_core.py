"""Tests for flashvector_adapters.core (CRUD, filters, scroll, score helpers)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from flashvector_adapters import FlashVectorStore
from flashvector_adapters.core import (
    compile_filter,
    distance_to_qdrant_score,
    distance_to_relevance,
    distance_to_similarity,
    maximal_marginal_relevance,
    normalize_metric,
    prefix_filter,
)
from tests_adapters.fake_backend import FakeBackend


def make_store(
    dim: int | None = 8,
    metric: str = "cosine",
    *,
    supports_filter_mask: bool = True,
    supports_add: bool = True,
    supports_delete: bool = True,
) -> tuple[FlashVectorStore, list[FakeBackend]]:
    created: list[FakeBackend] = []

    def factory(d: int, m: str, cap: int) -> FakeBackend:
        b = FakeBackend(
            d,
            m,
            cap,
            supports_filter_mask=supports_filter_mask,
            supports_add=supports_add,
            supports_delete=supports_delete,
        )
        created.append(b)
        return b

    return FlashVectorStore(dim=dim, metric=metric, backend_factory=factory), created


def test_lazy_import_does_not_pull_frameworks():
    # Run in an isolated interpreter so pytest plugins (e.g. langsmith) cannot pre-import deps.
    code = (
        "import sys, flashvector_adapters; "
        "assert hasattr(flashvector_adapters, 'FlashVectorStore'); "
        "assert 'langchain_core' not in sys.modules; "
        "assert 'llama_index' not in sys.modules; "
        "assert 'flashvector_adapters.qdrant' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd="python",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_unknown_lazy_attribute_raises():
    import flashvector_adapters

    with pytest.raises(AttributeError):
        _ = flashvector_adapters.nonexistent_symbol


def test_add_search_get_count_and_dimension_inference():
    store, created = make_store(dim=None)
    assert store.dim is None
    vecs = np.eye(4, dtype=np.float32)
    ids = store.add(vecs, ids=["a", "b", "c", "d"], payloads=[{"i": i} for i in range(4)])
    assert ids == ["a", "b", "c", "d"]
    assert store.dim == 4
    assert len(store) == 4
    assert store.count() == 4
    assert "b" in store
    assert store.ids() == ["a", "b", "c", "d"]
    assert created == []  # backend created lazily on first search

    hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)[0]
    assert len(created) == 1
    assert hits[0].id == "a"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)

    pts = store.get(["b", "missing", "d"], with_vectors=True)
    assert pts[0] is not None and pts[0].payload == {"i": 1}
    assert pts[0].vector is not None and pts[0].vector[1] == 1.0
    assert pts[1] is None
    assert pts[2] is not None and pts[2].id == "d"


def test_add_auto_ids_and_validation_errors():
    store, _ = make_store(dim=3, metric="l2")
    ids = store.add([[1, 0, 0], [0, 1, 0]])
    assert len(ids) == 2 and ids[0] != ids[1]
    with pytest.raises(ValueError):
        store.add([[1, 0, 0, 0]])  # wrong dim
    with pytest.raises(ValueError):
        store.add([[1, 0, 0]], ids=["a", "b"])
    with pytest.raises(ValueError):
        store.add([[1, 0, 0]], payloads=[{}, {}])


def test_upsert_replaces_existing_id_and_batch_dedup():
    store, _ = make_store(dim=3, metric="l2")
    store.add([[1, 0, 0], [0, 1, 0]], ids=["x", "y"], payloads=[{"v": 1}, {"v": 2}])
    store.search([1, 0, 0], k=1)

    # Upsert "x" with new vector and payload; duplicate in single batch keeps last
    store.add(
        [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
        ids=["x", "z", "z"],
        payloads=[{"v": 10}, {"v": 20}, {"v": 30}],
    )
    assert len(store) == 3
    pts = store.get(["x", "z"], with_vectors=True)
    assert pts[0] is not None and pts[0].payload == {"v": 10}
    assert pts[1] is not None and pts[1].payload == {"v": 30}
    hits = store.search([0, 0, 1], k=2)[0]
    assert hits[0].id == "x"


def test_incremental_add_uses_backend_add_without_rebuild():
    store, created = make_store(dim=3, metric="l2", supports_add=True)
    store.add([[1, 0, 0], [0, 1, 0]], ids=["a", "b"])
    store.flush()
    assert created[0].build_calls == 1
    store.add([[0, 0, 1]], ids=["c"])
    hits = store.search([0, 0, 1], k=1)[0]
    assert hits[0].id == "c"
    assert len(created) == 1 and created[0].build_calls == 1


def test_delete_by_ids_and_by_filter_and_clear():
    store, _ = make_store(dim=3, metric="l2")
    store.add(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
        ids=[1, 2, 3, 4],
        payloads=[
            {"tenant": "a", "age": 10},
            {"tenant": "a", "age": 20},
            {"tenant": "b", "age": 30},
            {"tenant": "b", "age": 40},
        ],
    )
    assert store.delete([2]) == 1
    assert 2 not in store
    assert store.delete(filter={"tenant": "b", "age": {"$gte": 35}}) == 1
    assert store.count() == 2
    hits = store.search([0, 1, 0], k=5)[0]
    assert {h.id for h in hits} == {1, 3}

    store.clear()
    assert len(store) == 0
    assert store.search([1, 0, 0], k=3)[0] == []


def test_fallback_when_backend_lacks_add_delete_and_filter_mask():
    store, created = make_store(
        dim=4,
        metric="l2",
        supports_filter_mask=False,
        supports_add=False,
        supports_delete=False,
    )
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(24, 4)).astype(np.float32)
    ids = [f"id-{i}" for i in range(24)]
    payloads = [{"tenant": "t1" if i % 6 == 0 else "t2", "idx": i} for i in range(24)]
    store.add(vecs, ids=ids, payloads=payloads)

    hits = store.search(vecs[6], k=3, filter={"tenant": "t1"})[0]
    assert len(hits) == 3
    assert all(h.payload["tenant"] == "t1" for h in hits)
    assert hits[0].id == "id-6"

    # Incremental add without backend.add triggers rebuild on next search
    store.add([[9, 9, 9, 9]], ids=["extra"], payloads=[{"tenant": "t1", "idx": 99}])
    assert created[0].build_calls == 1
    hits2 = store.search([9, 9, 9, 9], k=1, filter={"tenant": "t1"})[0]
    assert hits2[0].id == "extra"
    assert len(created) == 2

    # Delete without backend.delete masks out tombstoned slot
    store.delete(["extra"])
    hits3 = store.search([9, 9, 9, 9], k=1, filter={"tenant": "t1"})[0]
    assert hits3[0].id != "extra"


def test_filter_mask_is_forwarded_to_backend_and_batched():
    store, created = make_store(dim=4, metric="cosine", supports_filter_mask=True)
    vecs = np.random.default_rng(1).normal(size=(16, 4)).astype(np.float32)
    store.add(vecs, ids=list(range(16)), payloads=[{"grp": i % 4} for i in range(16)])
    batch = store.search(vecs[:2], k=3, filter={"grp": 2})
    assert len(batch) == 2
    assert all(len(row) == 3 and all(h.payload["grp"] == 2 for h in row) for row in batch)
    assert created[0].filter_mask_calls == 1


@pytest.mark.parametrize(
    ("flt", "pid", "payload", "expected"),
    [
        (
            {"tenant": "acme", "year": {"$gte": 2022, "$lt": 2025}},
            "1",
            {"tenant": "acme", "year": 2023},
            True,
        ),
        (
            {"tenant": "acme", "year": {"$gte": 2022, "$lt": 2025}},
            "1",
            {"tenant": "acme", "year": 2025},
            False,
        ),
        (
            {"$or": [{"tag": {"$in": ["gpu", "cuda"]}}, {"score": {"$gt": 0.9}}]},
            "1",
            {"tag": ["rust", "cuda"]},
            True,
        ),
        (
            {"$or": [{"tag": {"$in": ["gpu", "cuda"]}}, {"score": {"$gt": 0.9}}]},
            "3",
            {"tag": "cpu", "score": 0.2},
            False,
        ),
        (
            {"$not": {"archived": True}, "user.role": "admin", "$id": {"$in": ["a", "b"]}},
            "a",
            {"archived": False, "user": {"role": "admin"}},
            True,
        ),
        ({"$not": {"archived": True}}, "a", {"archived": True}, False),
        ({"acl": {"$all": ["read", "write"]}}, "1", {"acl": ["read", "write", "admin"]}, True),
        ({"acl": {"$all": ["read", "write"]}}, "1", {"acl": ["read"]}, False),
        ({"title": {"$contains": "vector"}}, "1", {"title": "gpu vector db"}, True),
        ({"opt": {"$exists": True}}, "1", {"opt": None}, True),
        ({"opt": {"$null": True}}, "1", {"opt": None}, True),
        ({"opt": {"$empty": True}}, "1", {}, True),
        ({"tag": {"$nin": ["ban", "spam"]}}, "1", {"tag": "ok"}, True),
        ({"tag": {"$ne": "ban"}}, "1", {"tag": "ban"}, False),
    ],
)
def test_filter_compiler_operators(flt, pid, payload, expected):
    pred = compile_filter(flt)
    assert pred(pid, payload) is expected


def test_prefix_filter_nested():
    prefixed = prefix_filter({"tenant": "x", "$and": [{"year": 2024}], "$id": "1"}, "metadata.")
    assert prefixed == {"metadata.tenant": "x", "$and": [{"metadata.year": 2024}], "$id": "1"}
    assert prefix_filter(None, "metadata.") is None


def test_scroll_pagination_and_set_payload():
    store, _ = make_store(dim=2, metric="l2")
    store.add(
        [[i, 0] for i in range(7)],
        ids=[f"p{i}" for i in range(7)],
        payloads=[{"keep": i % 2 == 0} for i in range(7)],
    )
    store.set_payload(["p2"], {"extra": "ok"})
    store.set_payload(["p5"], {"replaced": True}, overwrite=True)
    page1, nxt = store.scroll(filter={"keep": True}, limit=2)
    assert [pt.id for pt in page1] == ["p0", "p2"]
    assert page1[1].payload == {"keep": True, "extra": "ok"}
    page2, nxt2 = store.scroll(filter={"keep": True}, limit=2, offset=nxt)
    assert [pt.id for pt in page2] == ["p4", "p6"]
    p5 = store.get(["p5"])[0]
    assert p5 is not None and p5.payload == {"replaced": True}
    assert nxt2 is None


def test_score_helpers_and_mmr():
    assert normalize_metric("Euclid") == "l2"
    assert normalize_metric("Dot") == "ip"
    assert distance_to_similarity(0.25, "cosine") == pytest.approx(0.75)
    assert distance_to_similarity(-4.0, "ip") == pytest.approx(4.0)
    assert distance_to_similarity(3.0, "l2") == pytest.approx(0.25)

    assert distance_to_qdrant_score(9.0, "l2") == pytest.approx(3.0)
    assert distance_to_qdrant_score(0.2, "cosine") == pytest.approx(0.8)
    assert distance_to_qdrant_score(-2.5, "ip") == pytest.approx(2.5)
    assert 0.0 <= distance_to_relevance(0.0, "l2") <= 1.0
    assert distance_to_relevance(0.2, "cosine") == pytest.approx(0.8)
    assert distance_to_relevance(-3.0, "ip") == pytest.approx(3.0)

    # MMR picks the exact match first and with low lambda_mult prefers the orthogonal vector
    chosen = maximal_marginal_relevance(
        [1, 0], [[1, 0], [0.99, 0.01], [0, 1]], lambda_mult=0.2, k=2
    )
    assert chosen == [0, 2]


def test_filter_validation_and_empty_store():
    store, _ = make_store(dim=3, metric="ip")
    assert store.backend_size() is None
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0], k=3)[0] == []
    with pytest.raises(ValueError):
        compile_filter({"$badop": 1})
    with pytest.raises(ValueError):
        compile_filter({"field": {"$badop": 1}})
    with pytest.raises(ValueError):
        compile_filter({"field": {"$eq": 1, "plain": 2}})
    with pytest.raises(ValueError):
        normalize_metric("hamming")

    store.add([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], ids=["a", "b"], payloads=[{"x": 1}, {"x": 2}])
    hits = store.search([1.0, 0.0, 0.0], k=2)[0]
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].distance == pytest.approx(-2.0, abs=1e-5)
    assert store.backend_size() == 2
