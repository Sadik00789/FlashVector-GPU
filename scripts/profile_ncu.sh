#!/usr/bin/env bash
set -euo pipefail

# NVIDIA Nsight Compute Profiler for CUDA Search Kernels
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
NCU="${CUDA_ROOT}/bin/ncu"

if [ ! -f "${NCU}" ]; then
    NCU="ncu"
fi

echo "==> Profiling FlashVector-GPU Kernels with NVIDIA Nsight Compute..."
export PATH="$HOME/.cargo/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

cargo test --test e2e_search_test --no-run

TEST_BIN=$(find target/debug/deps -name "e2e_search_test-*" -type f -executable | head -n 1)

if [ -n "${TEST_BIN}" ]; then
    echo "==> Running ncu --set full on ${TEST_BIN}..."
    ${NCU} --set full \
          --target-processes all \
          --kernel-name-base function \
          --kernel-regex ".*(hnsw_warp_traverse|ivf_pq_adc).*" \
          "${TEST_BIN}" --nocapture || true
else
    echo "Error: Test binary not found. Run cargo test --no-run first."
fi
