#!/usr/bin/env bash
set -euo pipefail

# NVIDIA compute-sanitizer for memory leaks, out-of-bounds, and race conditions
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
SANITIZER="${CUDA_ROOT}/bin/compute-sanitizer"

if [ ! -f "${SANITIZER}" ]; then
    SANITIZER="compute-sanitizer"
fi

echo "==> Running NVIDIA compute-sanitizer memcheck..."
export PATH="$HOME/.cargo/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

cargo test --test cuda_sanity_test --no-run

TEST_BIN=$(find target/debug/deps -name "cuda_sanity_test-*" -type f -executable | head -n 1)

if [ -n "${TEST_BIN}" ]; then
    ${SANITIZER} --tool memcheck "${TEST_BIN}" --nocapture
    echo "==> Memcheck passed with 0 errors!"

    echo "==> Running NVIDIA compute-sanitizer racecheck..."
    ${SANITIZER} --tool racecheck "${TEST_BIN}" --nocapture
    echo "==> Racecheck passed with 0 errors!"
else
    echo "Error: Test binary not found. Run cargo test --no-run first."
fi
