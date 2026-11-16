"""Tests for flashvector_adapters.qdrant."""

from __future__ import annotations

import numpy as np
import pytest
from flashvector_adapters.qdrant import (
    Distance,
    FieldCondition,
    Filter,
    FlashVectorQdrantClient,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from tests_adapters.fake_backend import FakeBackend


def make_client() -> FlashVectorQdrantClient:
    return FlashVectorQdrantClient(
        backend_factory=lambda d, m, cap: FakeBackend(d, m, cap),
    )


def test_collection_lifecycle():
    client = make_client()
    assert not client.collection_exists("docs")
    assert client.create_collection("docs", VectorParams(size=4, distance=Distance.COSINE))
    assert client.collection_exists("docs")
    with pytest.raises(ValueError):
        client.create_collection("docs", VectorParams(size=4, distance=Distance.COSINE))

    info = client.get_collection("docs")
    assert info.name == "docs"
    assert info.points_count == 0
    assert info.config.size == 4

    assert client.recreate_collection("docs", VectorParams(size=8, distance=Distance.DOT))
    assert client.get_collection("docs").config.size == 8
    assert client.delete_collection("docs")
    assert not client.collection_exists("docs")


def test_upsert_query_points_and_filters_int_and_uuid_ids():
    client = make_client()
    client.create_collection("rag", VectorParams(size=4, distance=Distance.COSINE))
    client.upsert(
        "rag",
        points=[
            PointStruct(id=1, vector=[1, 0, 0, 0], payload={"tenant": "a", "ts": 100}),
            PointStruct(
                id="550e8400-e29b-41d4-a716-446655440000",
                vector=[0.9, 0.1, 0, 0],
                payload={"tenant": "b", "ts": 200},
            ),
            PointStruct(id=3, vector=[0, 1, 0, 0], payload={"tenant": "a", "ts": 300}),
            PointStruct(id=4, vector=[0, 0, 1, 0], payload={"tenant": "a", "ts": 400}),
        ],
    )
    assert client.count("rag").count == 4
    assert (
        client.count(
            "rag",
            count_filter=Filter(must=[FieldCondition(key="tenant", match=MatchValue("a"))]),
        ).count
        == 3
    )

    resp = client.query_points(
        "rag",
        query=[1, 0, 0, 0],
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant", match=MatchAny(["a", "c"])),
                FieldCondition(key="ts", range=Range(gte=100, lt=350)),
            ]
        ),
        limit=5,
        with_vectors=True,
    )
    assert [pt.id for pt in resp.points] == [1, 3]
    assert resp.points[0].score == pytest.approx(1.0, abs=1e-5)
    assert resp.points[0].vector == [1.0, 0.0, 0.0, 0.0]


def test_qdrant_must_not_should_retrieve_scroll_and_delete():
    client = make_client()
    client.create_collection("items", VectorParams(size=3, distance=Distance.DOT))
    client.upsert(
        "items",
        points=[
            PointStruct(id="a", vector=[2, 0, 0], payload={"acl": "pub", "ban": False}),
            PointStruct(id="b", vector=[5, 0, 0], payload={"acl": "priv", "ban": True}),
            PointStruct(id="c", vector=[3, 0, 0], payload={"acl": "team", "ban": False}),
        ],
    )
    hits = client.search(
        "items",
        query_vector=[1, 0, 0],
        query_filter=Filter(
            should=[
                FieldCondition(key="acl", match=MatchValue("pub")),
                FieldCondition(key="acl", match=MatchValue("team")),
            ],
            must_not=[FieldCondition(key="ban", match=MatchValue(True))],
        ),
        limit=5,
    )
    assert [h.id for h in hits] == ["c", "a"]
    assert hits[0].score == pytest.approx(3.0, abs=1e-5)

    recs = client.retrieve("items", ["a", "missing", "c"])
    assert [r.id for r in recs] == ["a", "c"]

    page, nxt = client.scroll("items", limit=2)
    assert [r.id for r in page] == ["a", "b"]
    assert nxt == "c"

    client.delete(
        "items", points_selector=Filter(must=[FieldCondition(key="ban", match=MatchValue(True))])
    )
    assert client.count("items").count == 2


def test_qdrant_euclid_distance_scoring():
    client = make_client()
    client.create_collection("euclid", VectorParams(size=2, distance=Distance.EUCLID))
    client.upsert(
        "euclid",
        points=[
            PointStruct(id=10, vector=[3, 4]),
            PointStruct(id=20, vector=[0, 1]),
        ],
    )
    hits = client.search("euclid", query_vector=[0, 0], limit=2, score_threshold=3.0)
    assert len(hits) == 1
    assert hits[0].id == 20
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    # Without threshold, [3,4] has Euclidean distance 5.0
    all_hits = client.search("euclid", query_vector=[0, 0], limit=2)
    assert np.isclose(all_hits[1].score, 5.0, atol=1e-5)


def test_qdrant_error_cases_and_options():
    client = make_client()
    with pytest.raises(KeyError):
        client.get_collection("missing")

    client.create_collection("c", VectorParams(size=3, distance=Distance.COSINE))
    assert client.upsert("c", points=[]).status == "completed"
    client.upsert(
        "c",
        points=[
            {"id": 1, "vector": [1.0, 0.0, 0.0], "payload": {"k": "a"}},
            {"id": 2, "vector": [0.0, 1.0, 0.0], "payload": {"k": "b"}},
        ],
    )
    hits = client.search(
        "c",
        query_vector=[1.0, 0.0, 0.0],
        with_payload=False,
        score_threshold=0.5,
    )
    assert [h.id for h in hits] == [1]
    assert hits[0].payload is None

    client.delete("c", points_selector=[1])
    assert client.count("c").count == 1
    assert not client.delete_collection("missing")
