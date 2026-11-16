"""LlamaIndex ``BasePydanticVectorStore`` adapter backed by :class:`FlashVectorStore`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, cast

from .core import (
    FilterSpec,
    FlashVectorStore,
    distance_to_similarity,
)

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore as _BaseLlamaStore,
    )
    from llama_index.core.vector_stores.types import (
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
else:
    try:
        from llama_index.core.vector_stores.types import (
            BasePydanticVectorStore as _BaseLlamaStore,
        )
    except ImportError:

        class _BaseLlamaStore:
            """Placeholder base when ``llama-index-core`` is not installed."""


def _require_llama() -> tuple[Any, Any, Any]:
    try:
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import (
            BasePydanticVectorStore,
            VectorStoreQueryResult,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FlashVectorGPUVectorStore requires `llama-index-core` "
            "(install via `pip install llama-index-core`)."
        ) from exc
    return BasePydanticVectorStore, TextNode, VectorStoreQueryResult


def _compile_llama_filters(filters: Any) -> FilterSpec:
    if filters is None or not getattr(filters, "filters", None):
        return None
    parts: list[dict[str, Any]] = []
    for item in filters.filters:
        if hasattr(item, "filters"):
            nested = _compile_llama_filters(item)
            if nested:
                parts.append(dict(nested))
            continue
        key = str(item.key)
        val = item.value
        op_raw = getattr(item.operator, "value", str(item.operator)).lower()
        if op_raw in ("==", "eq"):
            parts.append({key: val})
        elif op_raw in ("!=", "ne"):
            parts.append({key: {"$ne": val}})
        elif op_raw in (">", "gt"):
            parts.append({key: {"$gt": val}})
        elif op_raw in (">=", "gte"):
            parts.append({key: {"$gte": val}})
        elif op_raw in ("<", "lt"):
            parts.append({key: {"$lt": val}})
        elif op_raw in ("<=", "lte"):
            parts.append({key: {"$lte": val}})
        elif op_raw == "in":
            parts.append({key: {"$in": val}})
        elif op_raw == "nin":
            parts.append({key: {"$nin": val}})
        elif op_raw in ("any", "contains"):
            parts.append({key: {"$contains": val}})
        elif op_raw == "is_empty":
            parts.append({key: {"$empty": True}})
        else:
            raise ValueError(f"unsupported LlamaIndex filter operator {op_raw!r}")
    cond = getattr(getattr(filters, "condition", None), "value", "and")
    if str(cond).lower() == "or":
        return {"$or": parts}
    if str(cond).lower() == "not":
        return {"$not": {"$and": parts}}
    return {"$and": parts} if len(parts) > 1 else (parts[0] if parts else None)


class FlashVectorGPUVectorStore(_BaseLlamaStore):
    """LlamaIndex vector store powered by FlashVector-GPU."""

    stores_text: bool = True
    flat_metadata: bool = False
    _store: FlashVectorStore

    _TEXT_KEY = "_text"
    _REF_DOC_KEY = "ref_doc_id"

    def __init__(
        self,
        *,
        dim: int | None = None,
        metric: str = "cosine",
        backend_factory: Callable[..., Any] | None = None,
        ef_search: int = 64,
        store: FlashVectorStore | None = None,
        **kwargs: Any,
    ) -> None:
        _require_llama()
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "_store",
            store
            or FlashVectorStore(
                dim=dim,
                metric=metric,
                backend_factory=backend_factory,
                ef_search=ef_search,
            ),
        )

    @property
    def client(self) -> FlashVectorStore:
        return self._store

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        if not nodes:
            return []
        vectors: list[list[float]] = []
        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        for node in nodes:
            emb = node.get_embedding()
            vectors.append(list(emb))
            ids.append(str(node.node_id))
            meta = dict(getattr(node, "metadata", {}) or {})
            meta[self._TEXT_KEY] = node.get_content()
            ref_id = getattr(node, "ref_doc_id", None)
            if ref_id is not None:
                meta[self._REF_DOC_KEY] = str(ref_id)
            payloads.append(meta)
        assigned = self._store.add(vectors, ids=ids, payloads=payloads)
        return [str(x) for x in assigned]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._store.delete(filter={self._REF_DOC_KEY: ref_doc_id})

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        flt = _compile_llama_filters(filters)
        self._store.delete(ids=node_ids, filter=flt)

    def clear(self) -> None:
        self._store.clear()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        _, text_node_cls, result_cls = _require_llama()
        if query.query_embedding is None:
            raise ValueError("FlashVectorGPUVectorStore.query requires query_embedding")
        flt = _compile_llama_filters(getattr(query, "filters", None))
        k = int(getattr(query, "similarity_top_k", 10) or 10)
        ef_search = kwargs.get("ef_search")
        hits = self._store.search(query.query_embedding, k=k, filter=flt, ef_search=ef_search)[0]
        nodes: list[Any] = []
        similarities: list[float] = []
        ids: list[str] = []
        for hit in hits:
            meta = dict(hit.payload)
            text = str(meta.pop(self._TEXT_KEY, ""))
            meta.pop(self._REF_DOC_KEY, None)
            nodes.append(text_node_cls(id_=str(hit.id), text=text, metadata=meta))
            similarities.append(float(distance_to_similarity(hit.distance, self._store.metric)))
            ids.append(str(hit.id))
        return cast(
            "VectorStoreQueryResult",
            result_cls(nodes=nodes, similarities=similarities, ids=ids),
        )
