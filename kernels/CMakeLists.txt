cmake_minimum_required(VERSION 3.24)
project(gpukernels LANGUAGES CXX)

set(DEFAULT_CUDA_ROOT "/usr/local/cuda-12.6")
if(NOT EXISTS "${DEFAULT_CUDA_ROOT}")
    set(DEFAULT_CUDA_ROOT "/usr/local/cuda")
endif()
set(CUDA_TOOLKIT_ROOT_DIR "${DEFAULT_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root directory")

find_program(CUDA_NVCC nvcc PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin" NO_DEFAULT_PATH)
if(NOT CUDA_NVCC)
    find_program(CUDA_NVCC nvcc)
endif()
if(NOT CUDA_NVCC)
    message(FATAL_ERROR "nvcc not found. Set CUDA_TOOLKIT_ROOT_DIR or pass -DCUDA_NVCC=...")
endif()

option(FLASHVECTOR_TRACE "Enable kernel tracing" OFF)
set(TRACE_DEFINE "")
if(FLASHVECTOR_TRACE)
    set(TRACE_DEFINE "-DFLASHVECTOR_TRACE_MODE=1")
    add_compile_definitions(FLASHVECTOR_TRACE_MODE=1)
endif()

set(CUDA_ARCH "86" CACHE STRING "CUDA GPU Architecture (e.g. 86)")
set(NVCC_GENCODE "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

set(KERNEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KERNEL_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(WIN32)
    set(NVCC_HOST_FLAGS "-Xcompiler=/MD")
    set(OBJ_EXT "obj")
else()
    set(NVCC_HOST_FLAGS "-Xcompiler=-fPIC")
    set(OBJ_EXT "o")
endif()

set(SRC_FILES
    "${KERNEL_SRC_DIR}/hnsw_traverse.cu"
    "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu"
)

set(OBJ_FILES
    "${BUILD_DIR}/hnsw_traverse.${OBJ_EXT}"
    "${BUILD_DIR}/ivf_pq_lookup.${OBJ_EXT}"
)

# Compile the CUDA translation units with nvcc (native CUDA compiler driver).
add_custom_command(
    OUTPUT "${BUILD_DIR}/hnsw_traverse.${OBJ_EXT}"
    COMMAND ${CUDA_NVCC} -x cu ${NVCC_GENCODE} -O3 --use_fast_math -std=c++17
        -I${KERNEL_INC_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include ${NVCC_HOST_FLAGS} ${TRACE_DEFINE}
        -c "${KERNEL_SRC_DIR}/hnsw_traverse.cu" -o "${BUILD_DIR}/hnsw_traverse.${OBJ_EXT}"
    DEPENDS "${KERNEL_SRC_DIR}/hnsw_traverse.cu" "${KERNEL_SRC_DIR}/hnsw_traverse.cuh"
            "${KERNEL_SRC_DIR}/distance_metrics.cuh" "${KERNEL_SRC_DIR}/bitonic_topk.cuh"
            "${KERNEL_INC_DIR}/types.h" "${KERNEL_INC_DIR}/cuda_bridge.h"
    COMMENT "Compiling hnsw_traverse.cu (sm_${CUDA_ARCH})"
)

add_custom_command(
    OUTPUT "${BUILD_DIR}/ivf_pq_lookup.${OBJ_EXT}"
    COMMAND ${CUDA_NVCC} -x cu ${NVCC_GENCODE} -O3 --use_fast_math -std=c++17
        -I${KERNEL_INC_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include ${NVCC_HOST_FLAGS} ${TRACE_DEFINE}
        -c "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu" -o "${BUILD_DIR}/ivf_pq_lookup.${OBJ_EXT}"
    DEPENDS "${KERNEL_SRC_DIR}/ivf_pq_lookup.cu" "${KERNEL_SRC_DIR}/ivf_pq_lookup.cuh"
            "${KERNEL_SRC_DIR}/distance_metrics.cuh" "${KERNEL_SRC_DIR}/bitonic_topk.cuh"
            "${KERNEL_INC_DIR}/types.h" "${KERNEL_INC_DIR}/cuda_bridge.h"
    COMMENT "Compiling ivf_pq_lookup.cu (sm_${CUDA_ARCH})"
)

if(WIN32)
    set(KERNEL_LIB "${BUILD_DIR}/gpukernels.lib")
    add_custom_command(
        OUTPUT "${KERNEL_LIB}"
        COMMAND ${CMAKE_AR} /NOLOGO /OUT:"${KERNEL_LIB}" ${OBJ_FILES}
        DEPENDS ${OBJ_FILES}
        COMMENT "Archiving gpukernels.lib"
    )
else()
    set(KERNEL_LIB "${BUILD_DIR}/libgpukernels.a")
    add_custom_command(
        OUTPUT "${KERNEL_LIB}"
        COMMAND ${CMAKE_AR} rcs "${KERNEL_LIB}" ${OBJ_FILES}
        DEPENDS ${OBJ_FILES}
        COMMENT "Archiving libgpukernels.a"
    )
endif()

add_custom_target(gpukernels ALL DEPENDS "${KERNEL_LIB}")
