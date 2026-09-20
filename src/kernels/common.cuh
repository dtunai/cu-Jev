#pragma once
#include "kernels.h"
#include <math.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Block-wide sum for blockDim.x <= 1024, result broadcast to all threads. */
__device__ __forceinline__ float block_sum(float v, float *scratch /* [32] */) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0)
        scratch[wid] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nw; i++)
        r += scratch[i];
    return r;
}
