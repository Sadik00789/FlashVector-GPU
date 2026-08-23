#!/usr/bin/env bash
set -euo pipefail

# NVIDIA Nsight Systems timeline profiler
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
NSYS="${CUDA_ROOT}/bin/nsys"

if [ ! -f "${NSYS}" ]; then
    NSYS="nsys"
fi

echo "==> Profiling CUDA stream timeline with NVIDIA Nsight Systems..."
export PATH="$HOME/.cargo/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

cargo test --test e2e_search_test --no-run

TEST_BIN=$(find target/debug/deps -name "e2e_search_test-*" -type f -executable | head -n 1)

if [ -n "${TEST_BIN}" ]; then
    ${NSYS} profile \
          --trace=cuda,nvtx,osrt \
          --output=flashvector_timeline \
          --force-overwrite=true \
          "${TEST_BIN}" --nocapture || true
    echo "==> Nsight Systems report saved to flashvector_timeline.nsys-rep"
else
    echo "Error: Test binary not found. Run cargo test --no-run first."
fi
