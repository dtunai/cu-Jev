#include "util.h"
#include <cuda_runtime.h>

static __thread char g_last_error[1024];

cujev_status cujev_fail(cujev_status s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof g_last_error, fmt, ap);
    va_end(ap);
    return s;
}

const char *cujev_last_error(void) { return g_last_error; }

void cujev_log(int verbose, const char *fmt, ...) {
    if (!verbose)
        return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[cujev] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

const char *cujev_status_string(cujev_status s) {
    switch (s) {
    case CUJEV_OK: return "ok";
    case CUJEV_ERR_INVALID_ARGUMENT: return "invalid argument";
    case CUJEV_ERR_IO: return "i/o error";
    case CUJEV_ERR_FORMAT: return "format error";
    case CUJEV_ERR_CUDA: return "cuda error";
    case CUJEV_ERR_OOM: return "out of memory";
    case CUJEV_ERR_CAPACITY: return "capacity exceeded";
    case CUJEV_ERR_UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

const char *cujev_version(void) { return "0.1.0"; }

cujev_status cujev_arena_init(cujev_arena *a, size_t bytes) {
    a->base = NULL;
    a->size = 0;
    a->used = 0;
    cudaError_t e = cudaMalloc(&a->base, bytes);
    if (e != cudaSuccess)
        return cujev_fail(CUJEV_ERR_OOM, "cudaMalloc(%zu bytes): %s", bytes, cudaGetErrorString(e));
    a->size = bytes;
    return CUJEV_OK;
}

void *cujev_arena_alloc(cujev_arena *a, size_t bytes) {
    size_t off = cujev_align_up(a->used, 256);
    if (off + bytes > a->size)
        return NULL;
    a->used = off + bytes;
    return (char *)a->base + off;
}

void cujev_arena_free(cujev_arena *a) {
    if (a->base)
        cudaFree(a->base);
    a->base = NULL;
    a->size = a->used = 0;
}
