"""LangChain ``VectorStore`` adapter backed by ``FlashVectorStore``."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .core import (
    FilterSpec,
    FlashVectorStore,
    Hit,
    maximal_marginal_relevance,
    normalize_metric,
)

if TYPE_CHECKING:  # pragma: no cover - static type checking only
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore as _BaseVectorStore
else:
    try:
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import VectorStore as _BaseVectorStore
    except ImportError:
        Document = Any  # type: ignore[misc, assignment]
        Embeddings = Any  # type: ignore[misc, assignment]

        class _BaseVectorStore:
            """Placeholder base when ``langchain-core`` is not installed."""


def _require_langchain() -> tuple[Any, Any]:
    try:
        from langchain_core.documents import Document as lc_doc
        from langchain_core.vectorstores import VectorStore as lc_vs
    except ImportError as exc:  # pragma: no cover - exercised when optional dep missing
        raise ImportError(
            "FlashVectorGPU LangChain adapter requires `langchain-core` "
            "(install via `pip install langchain-core`)."
        ) from exc
    return lc_doc, lc_vs


def _mmr_select(
    query_embedding: list[float],
    embedding_list: list[list[float]],
    lambda_mult: float,
    k: int,
) -> list[int]:
    try:
        from langchain_core.vectorstores.utils import (
            maximal_marginal_relevance as lc_mmr,
        )

        return list(
            lc_mmr(
                np.asarray(query_embedding, dtype=np.float32),
                embedding_list,
                lambda_mult=lambda_mult,
                k=k,
            )
        )
    except ImportError:
        return maximal_marginal_relevance(
            query_embedding,
            embedding_list,
            lambda_mult=lambda_mult,
            k=k,
        )


def _distance_to_lc_score(distance: float, metric: str) -> float:
    m = normalize_metric(metric)
    if m == "cosine":
        return float(max(0.0, min(1.0, 1.0 - 0.5 * distance)))
    if m == "ip":
        # Native IP distance is -dot; sigmoid maps (-inf, +inf) -> (0, 1]
        return float(1.0 / (1.0 + math.exp(distance)))
    # Squared L2 -> Euclidean-normalized relevance in [0, 1]
    euclid = math.sqrt(max(0.0, distance))
    return float(max(0.0, 1.0 - euclid / math.sqrt(2.0)))


class FlashVectorGPU(_BaseVectorStore):
    """LangChain ``VectorStore`` implementation powered by FlashVector-GPU."""

    _TEXT_KEY = "_text"

    def __init__(
        self,
        embedding: Embeddings,
        *,
        dim: int | None = None,
        metric: str = "cosine",
        backend_factory: Callable[..., Any] | None = None,
        ef_search: int = 64,
        store: FlashVectorStore | None = None,
    ) -> None:
        _require_langchain()
        self._embedding = embedding
        self._store = store or FlashVectorStore(
            dim=dim,
            metric=metric,
            backend_factory=backend_factory,
            ef_search=ef_search,
        )

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embedding

    @property
    def store(self) -> FlashVectorStore:
        return self._store

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        metric = self._store.metric
        return lambda d: _distance_to_lc_score(d, metric)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError(f"got {len(metadatas)} metadatas for {len(text_list)} texts")
        vectors = self._embedding.embed_documents(text_list)
        payloads: list[dict[str, Any]] = []
        for idx, text in enumerate(text_list):
            meta = dict(metadatas[idx]) if metadatas and metadatas[idx] else {}
            meta[self._TEXT_KEY] = text
            payloads.append(meta)
        assigned = self._store.add(vectors, ids=ids, payloads=payloads)
        return [str(x) for x in assigned]

    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        doc_ids = ids
        if doc_ids is None and all(getattr(d, "id", None) is not None for d in documents):
            doc_ids = [str(d.id) for d in documents]
        texts = [d.page_content for d in documents]
        metadatas = [dict(d.metadata) for d in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=doc_ids, **kwargs)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if ids is None and "filter" not in kwargs:
            self._store.clear()
            return True
        removed = self._store.delete(ids=ids, filter=kwargs.get("filter"))
        return removed > 0 or (ids is not None and len(ids) == 0)

    def get_by_ids(self, ids: Iterable[str], /) -> list[Document]:  # type: ignore[override]
        lc_doc, _ = _require_langchain()
        docs: list[Document] = []
        for point in self._store.get(ids):
            if point is None:
                continue
            meta = dict(point.payload)
            text = str(meta.pop(self._TEXT_KEY, ""))
            docs.append(lc_doc(page_content=text, metadata=meta, id=str(point.id)))
        return docs

    def _hits_to_documents(self, hits: list[Hit]) -> list[tuple[Document, float]]:
        lc_doc, _ = _require_langchain()
        out: list[tuple[Document, float]] = []
        for hit in hits:
            meta = dict(hit.payload)
            text = str(meta.pop(self._TEXT_KEY, ""))
            doc = lc_doc(page_content=text, metadata=meta, id=str(hit.id))
            out.append((doc, float(hit.distance)))
        return out

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        ef_search = kwargs.get("ef_search")
        hits = self._store.search(embedding, k=k, filter=filter, ef_search=ef_search)[0]
        return self._hits_to_documents(hits)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        q_vec = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(q_vec, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[Document]:
        pairs = self.similarity_search_with_score_by_vector(embedding, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in pairs]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[Document]:
        q_vec = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(q_vec, k=k, filter=filter, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[Document]:
        lc_doc, _ = _require_langchain()
        ef_search = kwargs.get("ef_search")
        hits = self._store.search(
            embedding,
            k=max(fetch_k, k),
            filter=filter,
            ef_search=ef_search,
            with_vectors=True,
        )[0]
        if not hits:
            return []
        cand_vecs = [hit.vector.tolist() for hit in hits if hit.vector is not None]
        if not cand_vecs:
            return [doc for doc, _ in self._hits_to_documents(hits[:k])]
        indices = _mmr_select(embedding, cand_vecs, lambda_mult=lambda_mult, k=k)
        docs: list[Document] = []
        for idx in indices:
            hit = hits[idx]
            meta = dict(hit.payload)
            text = str(meta.pop(self._TEXT_KEY, ""))
            docs.append(lc_doc(page_content=text, metadata=meta, id=str(hit.id)))
        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterSpec = None,
        **kwargs: Any,
    ) -> list[Document]:
        q_vec = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            q_vec,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FlashVectorGPU:
        instance = cls(embedding, **kwargs)
        instance.add_texts(texts, metadatas=metadatas, ids=ids)
        return instance

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FlashVectorGPU:
        instance = cls(embedding, **kwargs)
        instance.add_documents(documents, ids=ids)
        return instance
