"""Tests for flashvector_adapters.langchain (runs when langchain-core is installed)."""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core")

from flashvector_adapters.langchain import FlashVectorGPU  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from tests_adapters.fake_backend import FakeBackend  # noqa: E402


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            low = t.lower()
            out.append(
                [
                    1.0 if "gpu" in low or "cuda" in low else 0.1,
                    1.0 if "rag" in low or "vector" in low else 0.1,
                    1.0 if "rust" in low else 0.1,
                    0.5,
                ]
            )
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _factory(dim: int, metric: str, cap: int) -> FakeBackend:
    return FakeBackend(dim, metric, cap)


def test_langchain_from_texts_search_filter_mmr_delete():
    emb = DeterministicEmbeddings()
    vs = FlashVectorGPU.from_texts(
        ["CUDA GPU kernels", "RAG vector store", "Rust host engine"],
        embedding=emb,
        metadatas=[{"lang": "cu"}, {"lang": "py"}, {"lang": "rs"}],
        ids=["d1", "d2", "d3"],
        backend_factory=_factory,
    )

    docs = vs.similarity_search("CUDA kernel", k=2)
    assert docs[0].id == "d1"

    filtered = vs.similarity_search_with_score("CUDA kernel", k=2, filter={"lang": "rs"})
    assert len(filtered) == 1
    assert filtered[0][0].page_content == "Rust host engine"

    mmr_docs = vs.max_marginal_relevance_search("CUDA GPU", k=2, fetch_k=3, lambda_mult=0.5)
    assert len(mmr_docs) == 2

    fetched = vs.get_by_ids(["d2", "missing"])
    assert len(fetched) == 1
    assert fetched[0].page_content == "RAG vector store"

    assert vs.delete(["d1"]) is True
    assert vs.get_by_ids(["d1"]) == []


def test_langchain_from_documents_and_retriever():
    emb = DeterministicEmbeddings()
    docs = [
        Document(page_content="GPU vector index", metadata={"tenant": "t1"}, id="x1"),
        Document(page_content="Rust server", metadata={"tenant": "t2"}, id="x2"),
    ]
    vs = FlashVectorGPU.from_documents(docs, emb, backend_factory=_factory)
    retriever = vs.as_retriever(search_kwargs={"k": 1})
    found = retriever.invoke("GPU")
    assert len(found) == 1
    assert found[0].id == "x1"
