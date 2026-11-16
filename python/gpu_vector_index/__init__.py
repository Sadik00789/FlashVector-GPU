"""Native CUDA extension wrapper for FlashVector-GPU."""

from __future__ import annotations

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from .gpu_vector_index import FlashVectorGPU  # type: ignore[import-not-found]

__all__ = ["FlashVectorGPU"]
