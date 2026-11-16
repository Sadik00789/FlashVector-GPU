/*
 * CPU-only stand-in for the CUDA bridge (kernels/include/cuda_bridge.h).
 *
 * Compiled instead of the real kernels when FLASHVECTOR_CPU_ONLY=1, so the
 * Rust workspace can be type-checked, linted and unit-tested on machines
 * without a CUDA toolkit (e.g. standard GitHub-hosted CI runners).
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FV_NO_DEVICE 100

static int fv_stream_token;

int cuda_init_device(int device_id) {
    (void)device_id;
    return FV_NO_DEVICE;
}

int cuda_set_device(int device_id) {
    (void)device_id;
    return FV_NO_DEVICE;
}

int cuda_get_device_count(int* count) {
    if (!count) return -1;
    *count = 0;
    return 0;
}

int cuda_get_device_memory(size_t* free_bytes, size_t* total_bytes) {
    (void)free_bytes;
    (void)total_bytes;
    return FV_NO_DEVICE;
}

int cuda_device_synchronize(void) { return 0; }

int cuda_enable_peer_access(int device, int peer) {
    (void)device;
    (void)peer;
    return FV_NO_DEVICE;
}

int cuda_memcpy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                           size_t bytes, void* stream) {
    (void)dst_device;
    (void)src_device;
    (void)stream;
    if (!dst || !src) return -1;
    memcpy(dst, src, bytes);
    return 0;
}

int cuda_malloc_device(void** ptr, size_t bytes) {
    if (!ptr || bytes == 0) return -1;
    *ptr = malloc(bytes);
    return *ptr ? 0 : 2;
}

int cuda_free_device(void* ptr) {
    free(ptr);
    return 0;
}

int cuda_malloc_host(void** ptr, size_t bytes) { return cuda_malloc_device(ptr, bytes); }

int cuda_free_host(void* ptr) { return cuda_free_device(ptr); }

int cuda_host_get_device_pointer(void** dev_ptr, void* host_ptr) {
    if (!dev_ptr || !host_ptr) return -1;
    *dev_ptr = host_ptr;
    return 0;
}

int cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    (void)stream;
    if (!dst || !src) return -1;
    memcpy(dst, src, bytes);
    return 0;
}

int cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    return cuda_memcpy_h2d_async(dst, src, bytes, stream);
}

int cuda_memset_device_async(void* dst, int value, size_t bytes, void* stream) {
    (void)stream;
    if (!dst) return -1;
    memset(dst, value, bytes);
    return 0;
}

int cuda_create_stream(void** stream) {
    if (!stream) return -1;
    *stream = &fv_stream_token;
    return 0;
}

int cuda_destroy_stream(void* stream) {
    (void)stream;
    return 0;
}

int cuda_sync_stream(void* stream) {
    (void)stream;
    return 0;
}

/* Kernel entry points: unavailable without a device. */
#define FV_KERNEL_STUB(name) \
    int name() { return FV_NO_DEVICE; }

FV_KERNEL_STUB(cuda_hnsw_search_batch)
FV_KERNEL_STUB(cuda_hnsw_search_batch_debug)
FV_KERNEL_STUB(cuda_cagra_search_batch)
FV_KERNEL_STUB(cuda_ivf_pq_search_batch)
FV_KERNEL_STUB(cuda_ivf_sq_search_batch)
FV_KERNEL_STUB(cuda_ivf_rabitq_search_batch)
FV_KERNEL_STUB(cuda_fp16_tc_search_batch)
FV_KERNEL_STUB(cuda_flat_scan_batch)
FV_KERNEL_STUB(cuda_merge_topk)
FV_KERNEL_STUB(cuda_sparse_search_batch)
FV_KERNEL_STUB(cuda_rrf_fuse)
FV_KERNEL_STUB(cuda_compute_distances_warp)
FV_KERNEL_STUB(cuda_bitonic_sort_test)
