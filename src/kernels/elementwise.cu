#include "common.cuh"

__global__ void embed_kernel(const int32_t *tokens, const bf16 *E, float *x, int N, int H) {
    int n = blockIdx.x;
    const bf16 *row = E + (size_t)tokens[n] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        x[(size_t)n * H + i] = bf2f(row[i]);
}
void k_embed(const int32_t *tokens, const bf16 *E, float *x, int N, int H, cudaStream_t s) {
    embed_kernel<<<N, 256, 0, s>>>(tokens, E, x, N, H);
}

__global__ void rmsnorm_kernel(const float *x, const float *w, bf16 *out, int H, float eps,
                               const int32_t *rows) {
    __shared__ float scratch[32];
    int r = blockIdx.x;
    int src = rows ? rows[r] : r;
    const float *xr = x + (size_t)src * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += xr[i] * xr[i];
    ss = block_sum(ss, scratch);
    float inv = rsqrtf(ss / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[(size_t)r * H + i] = f2bf(xr[i] * inv * w[i]);
}
void k_rmsnorm(const float *x, const float *w, bf16 *out, int N, int H, float eps,
               const int32_t *rows, int R, cudaStream_t s) {
    rmsnorm_kernel<<<rows ? R : N, 256, 0, s>>>(x, w, out, H, eps, rows);
}

__global__ void add_bf16_kernel(float *x, const bf16 *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += bf2f(y[i]);
}
void k_add_bf16(float *x, const bf16 *y, int N, int H, cudaStream_t s) {
    size_t n = (size_t)N * H;
    add_bf16_kernel<<<(unsigned)CEIL_DIV(n, 256), 256, 0, s>>>(x, y, n);
}

__global__ void silu_mul_kernel(const bf16 *gu, bf16 *out, int N, int I) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * I)
        return;
    int n = (int)(idx / I), i = (int)(idx % I);
    float g = bf2f(gu[(size_t)n * 2 * I + i]);
    float u = bf2f(gu[(size_t)n * 2 * I + I + i]);
    out[idx] = f2bf(silu_f(g) * u);
}
void k_silu_mul(const bf16 *gu, bf16 *out, int N, int I, cudaStream_t s) {
    size_t n = (size_t)N * I;
    silu_mul_kernel<<<(unsigned)CEIL_DIV(n, 256), 256, 0, s>>>(gu, out, N, I);
}

/* Y[N,M] = X[N,K] W[M,K]^T. In cuBLAS column-major terms: C(MxN) = op(A) op(B)
 * with A = W seen as (K x M) col-major -> op T, B = X seen as (K x N) -> op N. */
cudaError_t k_gemm_bf16(cublasHandle_t h, const bf16 *X, const bf16 *W, bf16 *Y, int N, int M,
                        int K) {
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st =
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16BF, K, X,
                     CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}
cudaError_t k_gemm_bf16_f32out(cublasHandle_t h, const bf16 *X, const bf16 *W, float *Y, int N,
                               int M, int K) {
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st =
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16BF, K, X,
                     CUDA_R_16BF, K, &beta, Y, CUDA_R_32F, M, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

/* readout: one block per (branch, candidate) */
__global__ void readout_kernel(const bf16 *h, const bf16 *E, const int32_t *cand,
                               const int32_t *ncand, int cand_ld, float *logits, int ld, int H) {
    __shared__ float scratch[32];
    int b = blockIdx.y, i = blockIdx.x;
    if (i >= ncand[b])
        return;
    const bf16 *hr = h + (size_t)b * H;
    const bf16 *er = E + (size_t)cand[(size_t)b * cand_ld + i] * H;
    float acc = 0.f;
    for (int j = threadIdx.x; j < H; j += blockDim.x)
        acc += bf2f(hr[j]) * bf2f(er[j]);
    acc = block_sum(acc, scratch);
    if (threadIdx.x == 0)
        logits[(size_t)b * ld + i] = acc;
}
void k_readout(const bf16 *h, const bf16 *E, const int32_t *cand, const int32_t *ncand, int cand_ld,
               float *logits, int ld, int B, int H, cudaStream_t s) {
    dim3 grid(cand_ld, B);
    readout_kernel<<<grid, 256, 0, s>>>(h, E, cand, ncand, cand_ld, logits, ld, H);
}
