#ifndef CUJEV_UTIL_H
#define CUJEV_UTIL_H

#include "cujev/cujev.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Record an error message for cujev_last_error() and return the status. */
cujev_status cujev_fail(cujev_status s, const char *fmt, ...);
void cujev_log(int verbose, const char *fmt, ...);

/* Bump allocator for device memory: one cudaMalloc, many carve-outs. */
typedef struct cujev_arena {
    void *base;
    size_t size;
    size_t used;
} cujev_arena;

cujev_status cujev_arena_init(cujev_arena *a, size_t bytes);
void *cujev_arena_alloc(cujev_arena *a, size_t bytes); /* 256-byte aligned; NULL if full */
void cujev_arena_free(cujev_arena *a);

static inline size_t cujev_align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

#define CUJEV_TRY(expr)                                                                          \
    do {                                                                                         \
        cujev_status _s = (expr);                                                                \
        if (_s != CUJEV_OK)                                                                      \
            return _s;                                                                           \
    } while (0)

#ifdef __cplusplus
}
#endif
#endif
