"""Zero-boilerplate adapters for LangChain, LlamaIndex, and Qdrant-compatible workflows."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "FlashVectorGPU",
    "FlashVectorGPUVectorStore",
    "FlashVectorQdrantClient",
    "FlashVectorStore",
    "Hit",
    "Point",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "FlashVectorStore": (".core", "FlashVectorStore"),
    "Hit": (".core", "Hit"),
    "Point": (".core", "Point"),
    "FlashVectorGPU": (".langchain", "FlashVectorGPU"),
    "FlashVectorGPUVectorStore": (".llama_index", "FlashVectorGPUVectorStore"),
    "FlashVectorQdrantClient": (".qdrant", "FlashVectorQdrantClient"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
