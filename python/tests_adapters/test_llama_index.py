"""Tests for flashvector_adapters.llama_index (runs when llama-index-core is installed)."""

from __future__ import annotations

import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from flashvector_adapters.llama_index import FlashVectorGPUVectorStore  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from tests_adapters.fake_backend import FakeBackend  # noqa: E402


def test_llama_index_add_query_filter_and_delete():
    store = FlashVectorGPUVectorStore(
        dim=4,
        metric="cosine",
        backend_factory=lambda d, m, cap: FakeBackend(d, m, cap),
    )
    nodes = [
        TextNode(
            id_="n1",
            text="CUDA HNSW traversal",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"tenant": "acme", "year": 2024},
            relationships={},
        ),
        TextNode(
            id_="n2",
            text="IVF-RaBitQ quantization",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={"tenant": "beta", "year": 2025},
        ),
    ]
    nodes[0].metadata["ref_doc_id"] = "doc-a"
    assigned = store.add(nodes)
    assert assigned == ["n1", "n2"]

    res = store.query(
        VectorStoreQuery(
            query_embedding=[0.9, 0.1, 0.0, 0.0],
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)]
            ),
        )
    )
    assert res.ids == ["n1", "n2"]
    assert res.nodes[0].get_content() == "CUDA HNSW traversal"

    store.delete_nodes(node_ids=["n1"])
    res2 = store.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2))
    assert res2.ids == ["n2"]
