"""Framework-neutral store on top of the FlashVector-GPU native index.

:class:`FlashVectorStore` owns

* the native (or fake) backend, created lazily on first search,
* external ids (``str`` or ``int``), payload dictionaries and a retained host copy of the vectors,
* a small metadata filter language compiled to a boolean mask (see :func:`compile_filter`).

Backend protocol (see :class:`VectorBackend`): the native index is immutable after ``build``;
``add`` / ``delete`` / ``filter_mask`` are optional capabilities and the store transparently falls
back to (lazy) rebuilds from the host copy, tombstones and over-fetch + post-filtering.

Distances follow the native conventions: squared L2, cosine ``1 - cos`` and *negative* inner
product (smaller is better for all of them).
"""

from __future__ import annotations

import importlib
import inspect
import json
import operator
import threading
import uuid
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MISSING_DIST",
    "MISSING_LABEL",
    "FlashVectorStore",
    "Hit",
    "Point",
    "VectorBackend",
    "compile_filter",
    "distance_to_qdrant_score",
    "distance_to_relevance",
    "distance_to_similarity",
    "maximal_marginal_relevance",
    "normalize_metric",
    "prefix_filter",
]

MISSING_LABEL = 0xFFFFFFFF
MISSING_DIST = 1e30

PointId = Hashable
Payload = dict[str, Any]
FilterSpec = Optional[Mapping[str, Any]]
FloatArray = NDArray[np.float32]

_METRIC_ALIASES = {
    "l2": "l2",
    "euclid": "l2",
    "euclidean": "l2",
    "sqeuclidean": "l2",
    "cosine": "cosine",
    "cos": "cosine",
    "ip": "ip",
    "dot": "ip",
    "inner_product": "ip",
    "innerproduct": "ip",
    "max_inner_product": "ip",
}


def normalize_metric(metric: str) -> str:
    """Return the canonical metric name (``"l2"``, ``"cosine"`` or ``"ip"``)."""
    try:
        return _METRIC_ALIASES[str(metric).lower()]
    except KeyError:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of l2/euclid, cosine, ip/dot"
        ) from None


# ---------------------------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------------------------


def _as_f64(dist: Any) -> NDArray[np.float64]:
    return np.asarray(dist, dtype=np.float64)


def distance_to_similarity(dist: Any, metric: str) -> Any:
    """Distance -> "bigger is better" similarity.

    cosine: ``1 - d`` (the cosine similarity); ip: ``-d`` (the dot product);
    l2 (squared): ``1 / (1 + d)``.
    """
    d = _as_f64(dist)
    m = normalize_metric(metric)
    out = 1.0 - d if m == "cosine" else (-d if m == "ip" else 1.0 / (1.0 + d))
    return out if isinstance(dist, np.ndarray) or np.ndim(dist) else float(out)


def distance_to_qdrant_score(dist: Any, metric: str) -> Any:
    """Distance -> Qdrant style score (cosine similarity, dot product, euclidean distance)."""
    d = _as_f64(dist)
    m = normalize_metric(metric)
    out = np.sqrt(np.maximum(d, 0.0)) if m == "l2" else (1.0 - d if m == "cosine" else -d)
    return out if isinstance(dist, np.ndarray) or np.ndim(dist) else float(out)


def distance_to_relevance(dist: Any, metric: str) -> Any:
    """Distance -> LangChain-style relevance score (roughly in ``[0, 1]``, bigger is better)."""
    d = _as_f64(dist)
    m = normalize_metric(metric)
    if m == "cosine":
        out = 1.0 - d
    elif m == "ip":
        out = -d
    else:
        out = 1.0 - np.sqrt(np.maximum(d, 0.0)) / np.sqrt(2.0)
    return out if isinstance(dist, np.ndarray) or np.ndim(dist) else float(out)


def maximal_marginal_relevance(
    query_embedding: Any,
    embedding_list: Any,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Maximal marginal relevance (cosine based), returns indices into ``embedding_list``."""
    emb = np.asarray(embedding_list, dtype=np.float64)
    if emb.size == 0 or k <= 0:
        return []
    if emb.ndim == 1:
        emb = emb[None, :]
    q = np.asarray(query_embedding, dtype=np.float64).reshape(-1)

    def _unit(x: NDArray[np.float64]) -> NDArray[np.float64]:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(norms == 0.0, 1.0, norms)

    unit = _unit(emb)
    sim_q = unit @ _unit(q)
    selected = [int(np.argmax(sim_q))]
    while len(selected) < min(k, len(emb)):
        redundancy = (unit @ unit[selected].T).max(axis=1)
        score = lambda_mult * sim_q - (1.0 - lambda_mult) * redundancy
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


# ---------------------------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------------------------

Predicate = Callable[[Any, Mapping[str, Any]], bool]
_NOT_FOUND = object()
_SEQ_TYPES = (list, tuple, set, frozenset)


def _lookup(point_id: Any, payload: Mapping[str, Any], key: str) -> Any:
    if key == "$id":
        return point_id
    if key in payload:
        return payload[key]
    cur: Any = payload
    for part in key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return _NOT_FOUND
    return cur


def _eq(value: Any, expected: Any) -> bool:
    if value is _NOT_FOUND:
        return False
    try:
        if value == expected:
            return True
        return isinstance(value, _SEQ_TYPES) and expected in value
    except Exception:  # noqa: BLE001 - exotic payload values must never crash a search
        return False


def _compare(op: Callable[[Any, Any], bool], bound: Any) -> Callable[[Any], bool]:
    def single(v: Any) -> bool:
        if isinstance(v, bool) or v is None:
            return False
        try:
            return bool(op(v, bound))
        except TypeError:
            return False

    def check(value: Any) -> bool:
        if value is _NOT_FOUND:
            return False
        if isinstance(value, _SEQ_TYPES):
            return any(single(v) for v in value)
        return single(value)

    return check


def _op_predicate(op: str, arg: Any) -> Callable[[Any], bool]:
    if op == "$eq":
        return lambda v: _eq(v, arg)
    if op == "$ne":
        return lambda v: not _eq(v, arg)
    if op in ("$in", "$nin"):
        options = list(arg) if isinstance(arg, Iterable) and not isinstance(arg, str) else [arg]

        def is_in(v: Any) -> bool:
            return any(_eq(v, o) for o in options)

        return is_in if op == "$in" else (lambda v: not is_in(v))
    if op == "$gt":
        return _compare(operator.gt, arg)
    if op == "$gte":
        return _compare(operator.ge, arg)
    if op == "$lt":
        return _compare(operator.lt, arg)
    if op == "$lte":
        return _compare(operator.le, arg)
    if op == "$exists":
        return lambda v: (v is not _NOT_FOUND) == bool(arg)
    if op == "$null":
        return lambda v: (v is None) == bool(arg)
    if op == "$empty":
        return lambda v: (v is _NOT_FOUND or v is None or v in ([], {}, "", ())) == bool(arg)
    if op == "$all":
        needed = list(arg)
        return lambda v: isinstance(v, _SEQ_TYPES) and all(n in v for n in needed)
    if op == "$contains":

        def contains(v: Any) -> bool:
            if isinstance(v, str):
                return isinstance(arg, str) and arg in v
            return isinstance(v, _SEQ_TYPES) and arg in v

        return contains
    raise ValueError(f"unsupported filter operator {op!r}")


def _all_of(subs: list[Predicate]) -> Predicate:
    return lambda i, p: all(s(i, p) for s in subs)


def _any_of(subs: list[Predicate]) -> Predicate:
    return lambda i, p: any(s(i, p) for s in subs)


def _negate(sub: Predicate) -> Predicate:
    return lambda i, p: not sub(i, p)


def _field_predicate(key: str, cond: Any) -> Predicate:
    if isinstance(cond, Mapping) and cond and all(str(k).startswith("$") for k in cond):
        ops = [_op_predicate(str(op), arg) for op, arg in cond.items()]
    elif isinstance(cond, Mapping) and any(str(k).startswith("$") for k in cond):
        raise ValueError(f"cannot mix operators and plain keys in filter for field {key!r}")
    else:
        ops = [_op_predicate("$eq", cond)]
    return lambda i, p: all(op(_lookup(i, p, key)) for op in ops)


def _as_filter_list(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value]
    return list(value)


def compile_filter(flt: FilterSpec) -> Predicate:
    """Compile a metadata filter into a predicate ``(point_id, payload) -> bool``.

    Supported syntax (MongoDB / Qdrant flavoured)::

        {"lang": "en"}                          # equality (list payload values: membership)
        {"year": {"$gte": 2020, "$lt": 2024}}   # ranges: $gt $gte $lt $lte
        {"tag": {"$in": ["a", "b"]}}            # also $nin $ne $eq $exists $all $contains ...
        {"$and": [...]}, {"$or": [...]}, {"$not": {...}}
        {"user.name": "bob"}                    # dotted paths into nested dicts
        {"$id": {"$in": [1, 2]}}                # pseudo field: external point id

    Several keys in one dict are AND-ed.
    """
    if flt is None or (isinstance(flt, Mapping) and not flt):
        return lambda i, p: True
    if not isinstance(flt, Mapping):
        raise TypeError(f"filter must be a mapping, got {type(flt).__name__}")
    preds: list[Predicate] = []
    for key, value in flt.items():
        if key == "$and":
            preds.append(_all_of([compile_filter(f) for f in _as_filter_list(value)]))
        elif key == "$or":
            preds.append(_any_of([compile_filter(f) for f in _as_filter_list(value)]))
        elif key in ("$not", "$nor"):
            inner = [compile_filter(f) for f in _as_filter_list(value)]
            preds.append(_negate(_any_of(inner) if key == "$nor" else _all_of(inner)))
        elif isinstance(key, str) and key.startswith("$") and key != "$id":
            raise ValueError(f"unsupported top-level filter operator {key!r}")
        else:
            preds.append(_field_predicate(str(key), value))
    return _all_of(preds)


def prefix_filter(flt: FilterSpec, prefix: str) -> dict[str, Any] | None:
    """Prefix every field name of ``flt`` with ``prefix`` (e.g. ``"metadata."``)."""
    if flt is None:
        return None
    out: dict[str, Any] = {}
    for key, value in flt.items():
        if key in ("$and", "$or", "$nor") or key == "$not":
            out[key] = [prefix_filter(f, prefix) for f in _as_filter_list(value)]
        elif key == "$id":
            out[key] = value
        else:
            out[prefix + key] = value
    return out


# ---------------------------------------------------------------------------------------------
# Backend plumbing
# ---------------------------------------------------------------------------------------------


class VectorBackend(Protocol):
    """What :class:`FlashVectorStore` needs from the native index (``add``/``delete`` optional)."""

    def build(self, data: Any) -> None: ...

    def search(self, query: Any, top_k: int = ..., ef_search: int = ..., **kwargs: Any) -> Any: ...


def _as_int(value: Any) -> int:
    """``dim`` / ``num_vectors`` may be an int attribute or a zero-arg method."""
    return int(value() if callable(value) else value)


def _default_backend_factory(dim: int, metric: str, capacity: int) -> Any:
    try:
        mod = importlib.import_module("gpu_vector_index")
    except ImportError as exc:
        raise ImportError(
            "flashvector_adapters needs the native `gpu_vector_index` extension "
            "(build it with `maturin develop --release`, requires CUDA) or an explicit "
            "`backend_factory=` callable returning an object with build()/search()."
        ) from exc
    return mod.FlashVectorGPU(dim=dim, max_elements=capacity, metric=metric)


def _call_factory(factory: Callable[..., Any], dim: int, metric: str, capacity: int) -> Any:
    """Call ``factory(dim, metric, capacity)``, tolerating factories taking fewer arguments."""
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return factory(dim, metric, capacity)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return factory(dim, metric, capacity)
    n_pos = sum(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    return factory(*(dim, metric, capacity)[: max(n_pos, 0)])


def _as_matrix(vectors: Any, dim: int | None) -> FloatArray:
    arr = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {arr.shape}")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"vector dimension {arr.shape[1]} does not match store dimension {dim}")
    return arr


# ---------------------------------------------------------------------------------------------
# Public records
# ---------------------------------------------------------------------------------------------


@dataclass
class Hit:
    """One search result. ``distance`` follows the native metric convention."""

    id: Any
    distance: float
    payload: Payload
    vector: FloatArray | None = None


@dataclass
class Point:
    """A stored point as returned by ``get`` / ``scroll``."""

    id: Any
    payload: Payload
    vector: FloatArray | None = None


# ---------------------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------------------


class FlashVectorStore:
    """Vectors + ids + payloads + filters on top of a FlashVector-GPU backend.

    Args:
        dim: vector dimension; inferred from the first ``add`` when omitted.
        metric: ``"cosine"`` (default), ``"l2"`` (squared) or ``"ip"``.
        backend_factory: ``factory(dim, metric, capacity) -> backend``; defaults to
            ``gpu_vector_index.FlashVectorGPU``. Factories taking fewer arguments are accepted.
        ef_search: default ``ef_search`` for queries.
        overfetch: over-fetch multiplier when the backend lacks ``filter_mask`` support.
    """

    def __init__(
        self,
        dim: int | None = None,
        metric: str = "cosine",
        *,
        backend_factory: Callable[..., Any] | None = None,
        ef_search: int = 64,
        overfetch: int = 4,
    ) -> None:
        self.metric = normalize_metric(metric)
        self.ef_search = ef_search
        self.overfetch = max(2, overfetch)
        self._dim = dim
        self._factory = backend_factory or _default_backend_factory
        self._backend: Any = None
        self._dirty = True
        self._unsynced_deletes = False
        self._mask_ok: bool | None = None
        self._n = 0  # slots used (alive or tombstoned)
        self._vecs: FloatArray = np.empty((0, dim or 0), dtype=np.float32)
        self._alive = np.zeros(0, dtype=bool)
        self._ids: list[Any] = []
        self._payloads: list[Payload | None] = []
        self._id2slot: dict[Any, int] = {}
        self._version = 0
        self._mask_cache: dict[str, NDArray[np.bool_]] = {}
        self._lock = threading.RLock()

    # -- introspection ------------------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def backend(self) -> Any:
        """The backend instance (``None`` until the first search builds it)."""
        return self._backend

    def __len__(self) -> int:
        return len(self._id2slot)

    def __contains__(self, point_id: object) -> bool:
        return point_id in self._id2slot

    def ids(self) -> list[Any]:
        """All live ids in insertion order."""
        with self._lock:
            return [self._ids[s] for s in np.flatnonzero(self._alive[: self._n])]

    # -- mutation -----------------------------------------------------------------------------

    def _reserve(self, extra: int, dim: int) -> None:
        need = self._n + extra
        if self._vecs.shape[1] != dim:
            self._vecs = np.empty((0, dim), dtype=np.float32)
        if need > self._vecs.shape[0]:
            cap = max(need, 2 * self._vecs.shape[0], 64)
            grown = np.empty((cap, dim), dtype=np.float32)
            grown[: self._n] = self._vecs[: self._n]
            self._vecs = grown
            alive = np.zeros(cap, dtype=bool)
            alive[: self._n] = self._alive[: self._n]
            self._alive = alive

    def _touch(self) -> None:
        self._version += 1
        self._mask_cache.clear()

    def _tombstone(self, slots: Sequence[int]) -> None:
        if not slots:
            return
        for s in slots:
            self._id2slot.pop(self._ids[s], None)
            self._ids[s] = None
            self._payloads[s] = None
            self._alive[s] = False
        self._touch()
        if self._backend is None or self._dirty:
            return
        deleter = getattr(self._backend, "delete", None)
        if deleter is None:
            self._unsynced_deletes = True
            return
        try:
            deleter(np.asarray(slots, dtype=np.uint32))
        except (AttributeError, NotImplementedError):
            self._unsynced_deletes = True

    def add(
        self,
        vectors: Any,
        ids: Sequence[Any] | None = None,
        payloads: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[Any]:
        """Insert or replace (upsert) points; returns the ids (generated when ``ids`` is None)."""
        mat = _as_matrix(vectors, self._dim)
        m = mat.shape[0]
        if ids is None:
            id_list: list[Any] = [uuid.uuid4().hex for _ in range(m)]
        else:
            id_list = list(ids)
            if len(id_list) != m:
                raise ValueError(f"got {len(id_list)} ids for {m} vectors")
        if payloads is not None and len(payloads) != m:
            raise ValueError(f"got {len(payloads)} payloads for {m} vectors")
        pl: list[Payload] = [dict(p) if p else {} for p in (payloads or [None] * m)]
        for pid in id_list:
            if not isinstance(pid, Hashable):
                raise TypeError(f"point ids must be hashable, got {type(pid).__name__}")

        with self._lock:
            if m == 0:
                return []
            if self._dim is None:
                self._dim = int(mat.shape[1])
            # last occurrence wins for duplicates inside one batch
            last = {pid: i for i, pid in enumerate(id_list)}
            keep = sorted(last.values())
            if len(keep) != m:
                mat = mat[keep]
                kept_ids = [id_list[i] for i in keep]
                pl = [pl[i] for i in keep]
            else:
                kept_ids = id_list
            self._tombstone([self._id2slot[p] for p in kept_ids if p in self._id2slot])
            count = len(kept_ids)
            self._reserve(count, int(self._dim))
            start = self._n
            self._vecs[start : start + count] = mat
            self._alive[start : start + count] = True
            self._ids.extend(kept_ids)
            self._payloads.extend(pl)
            for off, pid in enumerate(kept_ids):
                self._id2slot[pid] = start + off
            self._n += count
            self._touch()
            self._push_to_backend(start, count)
            return id_list

    def _push_to_backend(self, start: int, count: int) -> None:
        if self._backend is None or self._dirty:
            self._dirty = True
            return
        adder = getattr(self._backend, "add", None)
        if adder is None:
            self._dirty = True
            return
        try:
            labels = adder(self._vecs[start : start + count])
        except (AttributeError, NotImplementedError):
            self._dirty = True
            return
        expected = np.arange(start, start + count)
        if labels is not None and not np.array_equal(np.asarray(labels).astype(np.int64), expected):
            raise RuntimeError(
                "backend.add() returned ids that differ from the store's slot numbers; "
                "the backend must assign consecutive ids continuing after the last label"
            )

    def delete(self, ids: Iterable[Any] | None = None, *, filter: FilterSpec = None) -> int:
        """Delete points by id and/or by metadata filter. Returns the number removed."""
        with self._lock:
            slots: set[int] = set()
            if ids is not None:
                slots.update(self._id2slot[i] for i in ids if i in self._id2slot)
            if filter is not None:
                slots.update(int(s) for s in np.flatnonzero(self._filter_mask(filter)))
            self._tombstone(sorted(slots))
            return len(slots)

    def clear(self) -> None:
        """Remove everything (the dimension is kept)."""
        with self._lock:
            dim = self._dim or 0
            self._backend = None
            self._dirty = True
            self._unsynced_deletes = False
            self._n = 0
            self._vecs = np.empty((0, dim), dtype=np.float32)
            self._alive = np.zeros(0, dtype=bool)
            self._ids, self._payloads, self._id2slot = [], [], {}
            self._touch()

    def set_payload(
        self, ids: Iterable[Any], payload: Mapping[str, Any], *, overwrite: bool = False
    ) -> None:
        """Merge (or with ``overwrite=True`` replace) the payload of the given points."""
        with self._lock:
            for pid in ids:
                slot = self._id2slot.get(pid)
                if slot is None:
                    continue
                current = self._payloads[slot]
                if overwrite or current is None:
                    self._payloads[slot] = dict(payload)
                else:
                    current.update(payload)
            self._touch()

    # -- backend synchronisation --------------------------------------------------------------

    def _compact(self) -> None:
        keep = np.flatnonzero(self._alive[: self._n])
        self._vecs = np.ascontiguousarray(self._vecs[keep])
        self._ids = [self._ids[s] for s in keep]
        self._payloads = [self._payloads[s] for s in keep]
        self._alive = np.ones(len(keep), dtype=bool)
        self._n = len(keep)
        self._id2slot = {pid: i for i, pid in enumerate(self._ids)}
        self._touch()

    def _sync(self) -> None:
        if self._backend is not None and not self._dirty:
            return
        if len(self._id2slot) == 0:
            self._backend = None
            return
        if len(self._id2slot) < self._n:
            self._compact()
        assert self._dim is not None
        capacity = max(2 * self._n, 100_000)
        backend = _call_factory(self._factory, int(self._dim), self.metric, capacity)
        backend.build(self._vecs[: self._n])
        self._backend = backend
        self._dirty = False
        self._unsynced_deletes = False

    def flush(self) -> None:
        """Eagerly (re)build the backend if there are pending changes."""
        with self._lock:
            self._sync()

    # -- queries ------------------------------------------------------------------------------

    def _filter_mask(self, flt: FilterSpec) -> NDArray[np.bool_]:
        """Boolean mask over all slots (length ``_n``); dead slots are always False."""
        key: str | None
        try:
            key = json.dumps(flt, sort_keys=True, default=repr)
        except (TypeError, ValueError):
            key = None
        if key is not None and key in self._mask_cache:
            return self._mask_cache[key]
        pred = compile_filter(flt)
        mask = np.zeros(self._n, dtype=bool)
        for slot in np.flatnonzero(self._alive[: self._n]):
            payload = self._payloads[slot]
            if pred(self._ids[slot], payload if payload is not None else {}):
                mask[slot] = True
        if key is not None:
            if len(self._mask_cache) >= 16:
                self._mask_cache.clear()
            self._mask_cache[key] = mask
        return mask

    def count(self, filter: FilterSpec = None) -> int:
        """Number of live points (matching ``filter`` when given)."""
        with self._lock:
            if filter is None:
                return len(self._id2slot)
            return int(self._filter_mask(filter).sum())

    def _point(self, slot: int, with_vector: bool) -> Point:
        payload = self._payloads[slot]
        vec = self._vecs[slot].copy() if with_vector else None
        return Point(self._ids[slot], dict(payload or {}), vec)

    def get(self, ids: Iterable[Any], *, with_vectors: bool = False) -> list[Point | None]:
        """Fetch points by id; the result is aligned with ``ids`` (``None`` for unknown ids)."""
        with self._lock:
            out: list[Point | None] = []
            for pid in ids:
                slot = self._id2slot.get(pid)
                out.append(None if slot is None else self._point(slot, with_vectors))
            return out

    def scroll(
        self,
        filter: FilterSpec = None,
        limit: int = 10,
        offset: Any | None = None,
        *,
        with_vectors: bool = False,
    ) -> tuple[list[Point], Any | None]:
        """Page through points in insertion order.

        ``offset`` is the id of the first point to return (``None`` = from the start). The second
        return value is the id to pass as the next ``offset`` (``None`` when exhausted).
        """
        with self._lock:
            if offset is None:
                first = 0
            elif offset in self._id2slot:
                first = self._id2slot[offset]
            else:
                raise KeyError(f"scroll offset {offset!r} is not a known point id")
            mask = self._alive[: self._n] if filter is None else self._filter_mask(filter)
            slots = np.flatnonzero(mask[first:]) + first
            page = slots[:limit]
            nxt = self._ids[slots[limit]] if len(slots) > limit else None
            return [self._point(int(s), with_vectors) for s in page], nxt

    def _search_backend(
        self,
        q: FloatArray,
        k: int,
        ef: int,
        allowed: NDArray[np.bool_] | None,
    ) -> list[tuple[NDArray[Any], NDArray[Any]]]:
        """Return per-query (slots, distances) already restricted to ``allowed`` slots."""
        backend = self._backend
        n = self._n
        if allowed is not None and self._mask_ok is not False:
            try:
                labels, dists = backend.search(q, top_k=k, ef_search=ef, filter_mask=allowed)
                self._mask_ok = True
            except TypeError:
                if self._mask_ok:
                    raise
                self._mask_ok = False
            else:
                return self._clean_rows(labels, dists, allowed)
        if allowed is None:
            labels, dists = backend.search(q, top_k=k, ef_search=ef)
            return self._clean_rows(labels, dists, None)
        assert allowed is not None
        n_allowed = int(allowed.sum())
        want = min(k, n_allowed)
        fetch = min(max(k * self.overfetch, k), n)
        while True:
            labels, dists = backend.search(q, top_k=fetch, ef_search=max(ef, fetch))
            rows = self._clean_rows(labels, dists, allowed)
            if fetch >= n or all(len(r[0]) >= want for r in rows):
                return [(r[0][:k], r[1][:k]) for r in rows]
            fetch = min(fetch * self.overfetch, n)

    def _clean_rows(
        self,
        labels: Any,
        dists: Any,
        allowed: NDArray[np.bool_] | None,
    ) -> list[tuple[NDArray[Any], NDArray[Any]]]:
        labels = np.asarray(labels).astype(np.int64)
        dists = np.asarray(dists, dtype=np.float32)
        if labels.ndim == 1:
            labels, dists = labels[None, :], dists[None, :]
        alive = self._alive[: self._n]
        rows = []
        for lab, dist in zip(labels, dists):
            ok = (lab >= 0) & (lab < self._n) & (dist < MISSING_DIST)
            ok &= lab != MISSING_LABEL
            safe = np.where(ok, lab, 0)
            ok &= alive[safe]
            if allowed is not None:
                ok &= allowed[safe]
            rows.append((lab[ok], dist[ok]))
        return rows

    def search(
        self,
        query_vectors: Any,
        k: int = 10,
        filter: FilterSpec = None,
        *,
        ef_search: int | None = None,
        with_vectors: bool = False,
    ) -> list[list[Hit]]:
        """k-NN search; returns one list of :class:`Hit` per query (best first)."""
        q = _as_matrix(query_vectors, self._dim)
        with self._lock:
            empty: list[list[Hit]] = [[] for _ in range(q.shape[0])]
            if k <= 0:
                return empty
            self._sync()
            if self._backend is None:
                return empty
            allowed: NDArray[np.bool_] | None = None
            if filter is not None:
                allowed = self._filter_mask(filter)
            elif self._unsynced_deletes:
                allowed = self._alive[: self._n].copy()
            n_allowed = len(self._id2slot) if allowed is None else int(allowed.sum())
            if n_allowed == 0:
                return empty
            ef = ef_search or self.ef_search
            rows = self._search_backend(q, min(k, n_allowed), max(ef, k), allowed)
            results: list[list[Hit]] = []
            for slots, dists in rows:
                hits = []
                for slot, dist in zip(slots[:k], dists[:k]):
                    s = int(slot)
                    vec = self._vecs[s].copy() if with_vectors else None
                    hits.append(Hit(self._ids[s], float(dist), dict(self._payloads[s] or {}), vec))
                results.append(hits)
            return results

    def backend_size(self) -> int | None:
        """``num_vectors()`` as reported by the backend (``None`` before it is built)."""
        with self._lock:
            if self._backend is None:
                return None
            fn = getattr(self._backend, "num_vectors", None)
            return None if fn is None else _as_int(fn)
