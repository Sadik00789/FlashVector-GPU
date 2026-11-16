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

# Build a fat binary spanning Turing through Hopper, plus a compute_90 PTX
# fallback so future architectures (sm_100+) can JIT from the same archive.
set(CUDA_ARCH_LIST "75;80;86;89;90" CACHE STRING "CUDA architectures to build (semicolon-separated)")

set(NVCC_GENCODES "")
foreach(ARCH ${CUDA_ARCH_LIST})
    list(APPEND NVCC_GENCODES "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
endforeach()
list(APPEND NVCC_GENCODES "-gencode=arch=compute_90,code=compute_90")

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

set(COMMON_HDRS
    "${KERNEL_SRC_DIR}/distance_metrics.cuh"
    "${KERNEL_SRC_DIR}/bitonic_topk.cuh"
    "${KERNEL_SRC_DIR}/ivf_rerank.cuh"
    "${KERNEL_INC_DIR}/types.h"
    "${KERNEL_INC_DIR}/cuda_bridge.h"
)

set(KERNEL_UNITS
    hnsw_traverse
    cagra_search
    ivf_pq_lookup
    ivf_quant_lookup
    sparse_hybrid
)

set(OBJ_FILES "")
foreach(UNIT ${KERNEL_UNITS})
    set(SRC_CU "${KERNEL_SRC_DIR}/${UNIT}.cu")
    set(HDR_CUH "${KERNEL_SRC_DIR}/${UNIT}.cuh")
    set(OUT_OBJ "${BUILD_DIR}/${UNIT}.${OBJ_EXT}")
    list(APPEND OBJ_FILES "${OUT_OBJ}")

    add_custom_command(
        OUTPUT "${OUT_OBJ}"
        COMMAND ${CUDA_NVCC} -x cu ${NVCC_GENCODES} -O3 --use_fast_math -std=c++17
            -I${KERNEL_INC_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include ${NVCC_HOST_FLAGS} ${TRACE_DEFINE}
            -c "${SRC_CU}" -o "${OUT_OBJ}"
        DEPENDS "${SRC_CU}" "${HDR_CUH}" ${COMMON_HDRS}
        COMMENT "Compiling ${UNIT}.cu (arch ${CUDA_ARCH_LIST}, multi-arch fatbin)"
    )
endforeach()

if(WIN32)
    set(KERNEL_LIB "${BUILD_DIR}/gpukernels.lib")
    add_custom_command(
        OUTPUT "${KERNEL_LIB}"
        COMMAND ${CMAKE_AR} /NOLOGO /OUT:"${KERNEL_LIB}" ${OBJ_FILES}
        DEPENDS ${OBJ_FILES}
        COMMENT "Archiving multi-arch gpukernels.lib"
    )
else()
    set(KERNEL_LIB "${BUILD_DIR}/libgpukernels.a")
    add_custom_command(
        OUTPUT "${KERNEL_LIB}"
        COMMAND ${CMAKE_AR} rcs "${KERNEL_LIB}" ${OBJ_FILES}
        DEPENDS ${OBJ_FILES}
        COMMENT "Archiving multi-arch libgpukernels.a"
    )
endif()

add_custom_target(gpukernels ALL DEPENDS "${KERNEL_LIB}")
