"""End-to-end tests exercising adapters on the real CUDA `gpu_vector_index` binary."""

from __future__ import annotations

import numpy as np
import pytest
from flashvector_adapters import (
    FlashVectorGPU as LangChainVectorStore,
)
from flashvector_adapters import (
    FlashVectorGPUVectorStore,
    FlashVectorQdrantClient,
    FlashVectorStore,
)
from flashvector_adapters.qdrant import Distance, PointStruct, VectorParams

try:
    import gpu_vector_index  # noqa: F401

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False


@pytest.mark.skipif(not HAS_BINDINGS, reason="gpu_vector_index native module not built")
def test_qdrant_langchain_llamaindex_on_native_gpu_backend():
    dim = 32
    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((40, dim)).astype(np.float32)

    qdrant = FlashVectorQdrantClient()
    qdrant.create_collection("gpu_smoke", VectorParams(size=dim, distance=Distance.COSINE))
    qdrant.upsert(
        "gpu_smoke",
        [
            PointStruct(
                id=i,
                vector=vecs[i].tolist(),
                payload={"tenant": "alpha" if i % 2 == 0 else "beta", "rank": i},
            )
            for i in range(40)
        ],
    )
    hits = qdrant.search(
        "gpu_smoke",
        query_vector=vecs[0].tolist(),
        query_filter={"tenant": "beta"},
        limit=5,
    )
    assert len(hits) == 5
    assert all(h.payload is not None and h.payload["tenant"] == "beta" for h in hits)

    # Incremental tombstone delete on live native GPU store
    store = FlashVectorStore(dim=dim, metric="l2")
    store.add(vecs, ids=list(range(40)), payloads=[{"tenant": "t"} for _ in range(40)])
    store.flush()
    store.delete([0])
    after_del = store.search(vecs[0], k=5)[0]
    assert all(h.id != 0 for h in after_del)

    # LangChain & LlamaIndex smoke against native store with deterministic dummy embedder
    class _FakeEmb:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [vecs[i % len(vecs)].tolist() for i in range(len(texts))]

        def embed_query(self, text: str) -> list[float]:
            return vecs[1].tolist()

    lc = LangChainVectorStore(embedding=_FakeEmb(), metric="cosine")
    lc.add_texts(["hello", "world"], metadatas=[{"tag": "a"}, {"tag": "b"}])
    docs = lc.similarity_search("q", k=1, filter={"tag": "b"})
    assert len(docs) == 1
    assert docs[0].metadata["tag"] == "b"

    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import VectorStoreQuery

    li = FlashVectorGPUVectorStore(dim=dim, metric="cosine")
    li.add(
        [
            TextNode(id_="n0", text="alpha", embedding=vecs[0].tolist()),
            TextNode(id_="n1", text="beta", embedding=vecs[1].tolist()),
        ]
    )
    res = li.query(VectorStoreQuery(query_embedding=vecs[1].tolist(), similarity_top_k=1))
    assert res.ids == ["n1"]
