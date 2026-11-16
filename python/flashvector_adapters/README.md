# FlashVector-GPU Python Adapters

Zero-boilerplate adapters over `gpu_vector_index.FlashVectorGPU` for **LangChain**, **LlamaIndex**, and **Qdrant-style** workflows.

## Core Store (Framework-Neutral)

```python
from flashvector_adapters import FlashVectorStore
import numpy as np

store = FlashVectorStore(dim=128, metric="cosine")
store.add(
    np.random.randn(100, 128).astype(np.float32),
    ids=[f"doc-{i}" for i in range(100)],
    payloads=[{"tenant_id": "acme" if i % 2 == 0 else "beta", "year": 2024} for i in range(100)],
)

hits = store.search(
    np.random.randn(128).astype(np.float32),
    k=5,
    filter={"tenant_id": "acme", "year": {"$gte": 2023}},
)
```

## Qdrant-Compatible Client

```python
from flashvector_adapters.qdrant import (
    Distance,
    FieldCondition,
    Filter,
    FlashVectorQdrantClient,
    MatchValue,
    PointStruct,
    VectorParams,
)

client = FlashVectorQdrantClient()
client.create_collection("chunks", VectorParams(size=128, distance=Distance.COSINE))
client.upsert(
    "chunks",
    points=[PointStruct(id=1, vector=[0.1] * 128, payload={"tenant_id": "acme"})],
)
resp = client.query_points(
    "chunks",
    query=[0.1] * 128,
    query_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue("acme"))]),
    limit=5,
)
```

## LangChain VectorStore

```python
from flashvector_adapters.langchain import FlashVectorGPU

vs = FlashVectorGPU.from_texts(
    ["GPU vector search", "Reciprocal rank fusion"],
    embedding=my_embeddings,
    metadatas=[{"topic": "cuda"}, {"topic": "rag"}],
)
docs = vs.similarity_search("fast retrieval", k=2, filter={"topic": "cuda"})
```

## LlamaIndex VectorStore

```python
from flashvector_adapters.llama_index import FlashVectorGPUVectorStore
from llama_index.core.vector_stores import VectorStoreQuery

store = FlashVectorGPUVectorStore(dim=128, metric="cosine")
store.add(nodes)
result = store.query(VectorStoreQuery(query_embedding=q, similarity_top_k=5))
```
