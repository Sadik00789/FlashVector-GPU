"""Qdrant-compatible client facade and lightweight Pydantic-free models.

Allows swapping ``QdrantClient`` for :class:`FlashVectorQdrantClient` in
existing retrieval code with zero external Qdrant server dependency.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .core import (
    FilterSpec,
    FlashVectorStore,
    Point,
    distance_to_qdrant_score,
    normalize_metric,
)


class Distance(str, Enum):
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


@dataclass
class VectorParams:
    size: int
    distance: Distance | str = Distance.COSINE


@dataclass
class PointStruct:
    id: int | str
    vector: Sequence[float]
    payload: dict[str, Any] | None = None


@dataclass
class MatchValue:
    value: Any


@dataclass
class MatchAny:
    any: list[Any]


@dataclass
class Range:
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None


@dataclass
class FieldCondition:
    key: str
    match: MatchValue | MatchAny | None = None
    range: Range | None = None


@dataclass
class Filter:
    must: list[FieldCondition | Filter] = field(default_factory=list)
    should: list[FieldCondition | Filter] = field(default_factory=list)
    must_not: list[FieldCondition | Filter] = field(default_factory=list)


@dataclass
class ScoredPoint:
    id: int | str
    version: int
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None


@dataclass
class Record:
    id: int | str
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None


@dataclass
class QueryResponse:
    points: list[ScoredPoint]


@dataclass
class UpdateResult:
    operation_id: int
    status: str = "completed"


@dataclass
class CountResult:
    count: int


@dataclass
class CollectionInfo:
    name: str
    points_count: int
    vectors_count: int
    config: VectorParams
    status: str = "green"


class _ModelsNamespace:
    Distance = Distance
    VectorParams = VectorParams
    PointStruct = PointStruct
    MatchValue = MatchValue
    MatchAny = MatchAny
    Range = Range
    FieldCondition = FieldCondition
    Filter = Filter
    ScoredPoint = ScoredPoint
    Record = Record
    QueryResponse = QueryResponse
    UpdateResult = UpdateResult
    CountResult = CountResult
    CollectionInfo = CollectionInfo


models = _ModelsNamespace()


def _qdrant_distance_to_metric(distance: Distance | str) -> str:
    raw = distance.value if isinstance(distance, Distance) else str(distance)
    return normalize_metric(raw)


def _condition_to_dict(cond: Any) -> dict[str, Any]:
    if isinstance(cond, Filter):
        compiled = _filter_to_spec(cond)
        return dict(compiled) if compiled is not None else {}
    if isinstance(cond, FieldCondition):
        key = cond.key
        ops: dict[str, Any] = {}
        if cond.match is not None:
            if isinstance(cond.match, MatchValue):
                return {key: cond.match.value}
            if isinstance(cond.match, MatchAny):
                return {key: {"$in": list(cond.match.any)}}
            if hasattr(cond.match, "value"):
                return {key: cond.match.value}
            if hasattr(cond.match, "any"):
                return {key: {"$in": list(cond.match.any)}}
        if cond.range is not None:
            if cond.range.gt is not None:
                ops["$gt"] = cond.range.gt
            if cond.range.gte is not None:
                ops["$gte"] = cond.range.gte
            if cond.range.lt is not None:
                ops["$lt"] = cond.range.lt
            if cond.range.lte is not None:
                ops["$lte"] = cond.range.lte
            return {key: ops}
        return {}
    if isinstance(cond, dict):
        return cond
    raise TypeError(f"unsupported Qdrant filter condition {type(cond).__name__}")


def _filter_to_spec(flt: Filter | dict[str, Any] | None) -> FilterSpec:
    if flt is None:
        return None
    if isinstance(flt, dict):
        return flt
    clauses: list[dict[str, Any]] = []
    if flt.must:
        must_parts = [_condition_to_dict(c) for c in flt.must]
        clauses.extend(p for p in must_parts if p)
    if flt.should:
        should_parts = [_condition_to_dict(c) for c in flt.should]
        should_parts = [p for p in should_parts if p]
        if should_parts:
            clauses.append({"$or": should_parts})
    if flt.must_not:
        not_parts = [_condition_to_dict(c) for c in flt.must_not]
        not_parts = [p for p in not_parts if p]
        if not_parts:
            clauses.append({"$nor": not_parts})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_point_id(point: Any) -> int | str:
    pid = point["id"] if isinstance(point, dict) else point.id
    if isinstance(pid, (int, str)):
        return pid
    return str(pid)


class FlashVectorQdrantClient:
    """In-process Qdrant-compatible client backed by :class:`FlashVectorStore`."""

    def __init__(
        self,
        *,
        backend_factory: Callable[..., Any] | None = None,
        ef_search: int = 64,
    ) -> None:
        self._backend_factory = backend_factory
        self._ef_search = ef_search
        self._collections: dict[str, tuple[FlashVectorStore, VectorParams]] = {}
        self._op_counter = 0
        self._lock = threading.RLock()

    def _next_op(self) -> int:
        self._op_counter += 1
        return self._op_counter

    def _get(self, collection_name: str) -> tuple[FlashVectorStore, VectorParams]:
        entry = self._collections.get(collection_name)
        if entry is None:
            raise KeyError(f"collection {collection_name!r} does not exist")
        return entry

    def collection_exists(self, collection_name: str) -> bool:
        with self._lock:
            return collection_name in self._collections

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | dict[str, Any],
    ) -> bool:
        with self._lock:
            if collection_name in self._collections:
                raise ValueError(f"collection {collection_name!r} already exists")
            cfg = (
                vectors_config
                if isinstance(vectors_config, VectorParams)
                else VectorParams(**vectors_config)
            )
            metric = _qdrant_distance_to_metric(cfg.distance)
            store = FlashVectorStore(
                dim=int(cfg.size),
                metric=metric,
                backend_factory=self._backend_factory,
                ef_search=self._ef_search,
            )
            self._collections[collection_name] = (store, cfg)
            return True

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | dict[str, Any],
    ) -> bool:
        with self._lock:
            self._collections.pop(collection_name, None)
            return self.create_collection(collection_name, vectors_config)

    def delete_collection(self, collection_name: str) -> bool:
        with self._lock:
            return self._collections.pop(collection_name, None) is not None

    def get_collection(self, collection_name: str) -> CollectionInfo:
        with self._lock:
            store, cfg = self._get(collection_name)
            n = len(store)
            return CollectionInfo(
                name=collection_name,
                points_count=n,
                vectors_count=n,
                config=cfg,
            )

    def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct | dict[str, Any]],
    ) -> UpdateResult:
        with self._lock:
            store, _ = self._get(collection_name)
            if not points:
                return UpdateResult(operation_id=self._next_op())
            ids: list[int | str] = []
            vectors: list[Sequence[float]] = []
            payloads: list[dict[str, Any] | None] = []
            for p in points:
                ids.append(_extract_point_id(p))
                if isinstance(p, dict):
                    vectors.append(p["vector"])
                    payloads.append(p.get("payload"))
                else:
                    vectors.append(p.vector)
                    payloads.append(p.payload)
            store.add(vectors, ids=ids, payloads=payloads)
            return UpdateResult(operation_id=self._next_op())

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        query_filter: Filter | dict[str, Any] | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[ScoredPoint]:
        with self._lock:
            store, _ = self._get(collection_name)
            flt = _filter_to_spec(query_filter)
            hits = store.search(
                query_vector,
                k=limit,
                filter=flt,
                ef_search=kwargs.get("ef_search"),
                with_vectors=with_vectors,
            )[0]
            results: list[ScoredPoint] = []
            for hit in hits:
                score = float(distance_to_qdrant_score(hit.distance, store.metric))
                if score_threshold is not None:
                    if store.metric == "l2" and score > score_threshold:
                        continue
                    if store.metric != "l2" and score < score_threshold:
                        continue
                results.append(
                    ScoredPoint(
                        id=hit.id,
                        version=0,
                        score=score,
                        payload=dict(hit.payload) if with_payload else None,
                        vector=hit.vector.tolist()
                        if with_vectors and hit.vector is not None
                        else None,
                    )
                )
            return results

    def query_points(
        self,
        collection_name: str,
        query: Sequence[float],
        query_filter: Filter | dict[str, Any] | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> QueryResponse:
        points = self.search(
            collection_name=collection_name,
            query_vector=query,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            **kwargs,
        )
        return QueryResponse(points=points)

    def delete(
        self,
        collection_name: str,
        points_selector: Iterable[int | str] | Filter | dict[str, Any],
    ) -> UpdateResult:
        with self._lock:
            store, _ = self._get(collection_name)
            if isinstance(points_selector, (Filter, dict)):
                store.delete(filter=_filter_to_spec(points_selector))
            else:
                store.delete(ids=points_selector)
            return UpdateResult(operation_id=self._next_op())

    def retrieve(
        self,
        collection_name: str,
        ids: Iterable[int | str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Record]:
        with self._lock:
            store, _ = self._get(collection_name)
            records: list[Record] = []
            for pt in store.get(ids, with_vectors=with_vectors):
                if pt is None:
                    continue
                records.append(
                    Record(
                        id=pt.id,
                        payload=dict(pt.payload) if with_payload else None,
                        vector=pt.vector.tolist()
                        if with_vectors and pt.vector is not None
                        else None,
                    )
                )
            return records

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Filter | dict[str, Any] | None = None,
        limit: int = 10,
        offset: int | str | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Record], int | str | None]:
        with self._lock:
            store, _ = self._get(collection_name)
            pts: list[Point]
            pts, nxt = store.scroll(
                filter=_filter_to_spec(scroll_filter),
                limit=limit,
                offset=offset,
                with_vectors=with_vectors,
            )
            records = [
                Record(
                    id=pt.id,
                    payload=dict(pt.payload) if with_payload else None,
                    vector=pt.vector.tolist() if with_vectors and pt.vector is not None else None,
                )
                for pt in pts
            ]
            return records, nxt

    def count(
        self,
        collection_name: str,
        count_filter: Filter | dict[str, Any] | None = None,
    ) -> CountResult:
        with self._lock:
            store, _ = self._get(collection_name)
            return CountResult(count=store.count(filter=_filter_to_spec(count_filter)))
