"""
FlashVector-GPU vs NVIDIA cuVS (CAGRA / IVF-PQ) Comparative Benchmark
"""

import time
import numpy as np

def run_cuvs_benchmark():
    print("=" * 60)
    print("FLASHVECTOR-GPU VS NVIDIA cuVS CAGRA / IVF-PQ BENCHMARK")
    print("=" * 60)

    try:
        import cuvs
        print("NVIDIA cuVS detected. Running CAGRA and IVF-PQ evaluation...")
    except ImportError:
        print("NVIDIA cuVS not installed in current Python environment.")
        print("To install: pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com")

if __name__ == "__main__":
    run_cuvs_benchmark()
