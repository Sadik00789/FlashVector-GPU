cmake_minimum_required(VERSION 3.24)
project(gpukernels LANGUAGES CXX)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6" CACHE PATH "CUDA Toolkit root directory")
if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()

find_program(CLANG_CXX clang++)
find_program(CUDA_NVCC nvcc PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin" NO_DEFAULT_PATH)

set(CUDA_ARCH "sm_86" CACHE STRING "CUDA GPU Architecture")
set(KERNEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KERNEL_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(SRC_FILES
    "${KERNEL_SRC_DIR}/hnsw_traverse.cu"
    "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu"
)

set(OBJ_FILES
    "${BUILD_DIR}/hnsw_traverse.o"
    "${BUILD_DIR}/ivf_pq_lookup.o"
)

# Custom commands using clang++ (native LLVM CUDA backend) for maximum compiler compatibility
add_custom_command(
    OUTPUT "${BUILD_DIR}/hnsw_traverse.o"
    COMMAND clang++ -x cuda --cuda-gpu-arch=${CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -O3 -ffast-math -I${KERNEL_INC_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include -fPIC -c "${KERNEL_SRC_DIR}/hnsw_traverse.cu" -o "${BUILD_DIR}/hnsw_traverse.o"
    DEPENDS "${KERNEL_SRC_DIR}/hnsw_traverse.cu" "${KERNEL_SRC_DIR}/hnsw_traverse.cuh" "${KERNEL_SRC_DIR}/distance_metrics.cuh" "${KERNEL_SRC_DIR}/bitonic_topk.cuh" "${KERNEL_INC_DIR}/types.h" "${KERNEL_INC_DIR}/cuda_bridge.h"
    COMMENT "Compiling hnsw_traverse.cu (sm_86)"
)

add_custom_command(
    OUTPUT "${BUILD_DIR}/ivf_pq_lookup.o"
    COMMAND clang++ -x cuda --cuda-gpu-arch=${CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -O3 -ffast-math -I${KERNEL_INC_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include -fPIC -c "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu" -o "${BUILD_DIR}/ivf_pq_lookup.o"
    DEPENDS "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu" "${KERNEL_SRC_DIR}/ivf_pq_lookup.cuh" "${KERNEL_SRC_DIR}/distance_metrics.cuh" "${KERNEL_SRC_DIR}/bitonic_topk.cuh" "${KERNEL_INC_DIR}/types.h" "${KERNEL_INC_DIR}/cuda_bridge.h"
    COMMENT "Compiling ivf_pq_lookup.cu (sm_86)"
)

add_custom_command(
    OUTPUT "${BUILD_DIR}/libgpukernels.a"
    COMMAND ${CMAKE_AR} rcs "${BUILD_DIR}/libgpukernels.a" ${OBJ_FILES}
    DEPENDS ${OBJ_FILES}
    COMMENT "Archiving libgpukernels.a"
)

add_custom_target(gpukernels ALL DEPENDS "${BUILD_DIR}/libgpukernels.a")
