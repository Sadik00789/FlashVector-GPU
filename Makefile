# FlashVector-GPU Unified Developer CLI
SHELL := /bin/bash

# Export environment paths for Fedora / local toolchain
export PATH := $(HOME)/.local/share/fnm/node-versions/v24.19.0/installation/bin:$(HOME)/.cargo/bin:/usr/local/cuda/bin:/usr/local/cuda-12.6/bin:$(PATH)
export CUDA_HOME ?= $(shell if [ -d "/usr/local/cuda-12.6" ]; then echo "/usr/local/cuda-12.6"; else echo "/usr/local/cuda"; fi)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)

.PHONY: all build build-kernels build-rust build-web test bench profile-ncu profile-nsys dev clean help

all: build

help:
	@echo "FlashVector-GPU Developer CLI"
	@echo "----------------------------------------------------"
	@echo "make build         - Compile CUDA kernels & release Rust workspace"
	@echo "make build-kernels - Compile CUDA static library (libgpukernels.a)"
	@echo "make build-rust    - Compile Rust crates (engine, server, python)"
	@echo "make build-web     - Build Next.js 3D visualizer frontend"
	@echo "make test          - Run cargo unit tests & CUDA sanity tests"
	@echo "make bench         - Run Criterion Rust microbenchmarks"
	@echo "make dev           - Start Axum backend (8080) & Next.js UI (3000)"
	@echo "make profile-ncu   - Run NVIDIA Nsight Compute kernel profiler"
	@echo "make profile-nsys  - Run NVIDIA Nsight Systems timeline profiler"
	@echo "make check-san     - Run NVIDIA compute-sanitizer for race detection"
	@echo "make clean         - Clean build artifacts"

build-kernels:
	@echo "==> Building CUDA Kernels (sm_86)..."
	mkdir -p kernels/build
	cd kernels/build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j$$(nproc)

build-rust: build-kernels
	@echo "==> Building Rust Workspace (Release)..."
	cargo build --workspace --release

build-web:
	@echo "==> Building Next.js Visualizer..."
	cd web && pnpm install --frozen-lockfile=false && pnpm build

build: build-rust build-web

test: build-kernels
	@echo "==> Running Test Suite..."
	cargo test --workspace -- --nocapture
	cargo test --test cuda_sanity_test -- --nocapture
	cargo test --test e2e_search_test -- --nocapture

bench: build-kernels
	@echo "==> Running Criterion Micro-benchmarks..."
	cargo bench --bench bench_streams
	cargo bench --bench bench_kmeans

profile-ncu:
	@echo "==> Launching NVIDIA Nsight Compute Profiler..."
	bash scripts/profile_ncu.sh

profile-nsys:
	@echo "==> Launching NVIDIA Nsight Systems Profiler..."
	bash scripts/profile_nsys.sh

check-san:
	@echo "==> Running compute-sanitizer memory & race check..."
	bash scripts/check_sanitizer.sh

dev: build-kernels
	@echo "==> Starting FlashVector-GPU Dev Services..."
	@trap 'kill 0' EXIT; \
	(cargo run --release -p server) & \
	(cd web && pnpm dev) & \
	wait

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf target/
	rm -rf kernels/build/
	rm -rf web/.next/
	rm -rf web/node_modules/
	rm -rf python/build/
