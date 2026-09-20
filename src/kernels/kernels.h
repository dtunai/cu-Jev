/* C-callable launchers for every CUDA kernel. All pointers are device
 * pointers unless stated. All launches go to `stream`. Row-major throughout. */
#ifndef CUJEV_KERNELS_H
#define CUJEV_KERNELS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __CUDACC__
#include <cuda_bf16.h>
typedef __nv_bfloat16 bf16;
#else
typedef uint16_t bf16; /* same size and layout; C code never dereferences it */
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Per-token metadata of one forward call (see engine/forward.cu). */
typedef struct cujev_tokmeta {
    const int32_t *pos;       /* [N] global position (RoPE)                     */
    const int32_t *own_base;  /* [N] index of the token's segment start in own KV */
    const int32_t *own_local; /* [N] index of the token inside its segment       */
    const int32_t *seg_start; /* [B] first token index of every segment          */
    const int32_t *seg_len;   /* [B] length of every segment                     */
    const int32_t *tile_n0;   /* [T] first token of every 16-token query tile     */
    const int32_t *tile_len;  /* [T] tokens in the tile (1..16)                    */
    int N;                    /* total tokens */
    int B;                    /* segments (branches) */
    int num_tiles;            /* T */
} cujev_tokmeta;

/* x[N,H] = E[tokens]  (bf16 table -> fp32 residual) */
void k_embed(const int32_t *tokens, const bf16 *E, float *x, int N, int H, cudaStream_t s);

/* out[N,H] = bf16( x * rsqrt(mean(x^2)+eps) * w )   (w already includes the +1 of
 * zero-centred norms). rows: optional [R] row indices to normalise instead of all N. */
void k_rmsnorm(const float *x, const float *w, bf16 *out, int N, int H, float eps,
               const int32_t *rows, int R, cudaStream_t s);

/* x[N,H] += y[N,H] */
void k_add_bf16(float *x, const bf16 *y, int N, int H, cudaStream_t s);

/* out[N,I] = silu(gu[N, :I]) * gu[N, I:2I] */
void k_silu_mul(const bf16 *gu, bf16 *out, int N, int I, cudaStream_t s);

/* Y[N,M] = X[N,K] . W[M,K]^T, bf16 in, fp32 accumulate. */
cudaError_t k_gemm_bf16(cublasHandle_t h, const bf16 *X, const bf16 *W, bf16 *Y, int N, int M,
                        int K);
cudaError_t k_gemm_bf16_f32out(cublasHandle_t h, const bf16 *X, const bf16 *W, float *Y, int N,
                               int M, int K);

/* ---- full attention ---------------------------------------------------- */
/* qkv[N, Hq*2D + Hkv*D + Hkv*D] is the fused projection output. Per head the q
 * block is [q(D) | gate(D)]. Writes q (normed+roped) as fp32 [N,Hq,D], the gate
 * stays where it is, and K/V (normed+roped) go to kv_k/kv_v[Hkv][slot][D] with
 * slot = own_base+own_local. */
void k_attn_prep(const bf16 *qkv, const float *q_norm_w, const float *k_norm_w, float *q,
                 bf16 *kv_k, bf16 *kv_v, int kv_stride_tokens, const cujev_tokmeta *m, int Hq,
                 int Hkv, int D, int rot, float eps, double theta, cudaStream_t s);

/* out[N, Hq*D] = sigmoid(gate) * softmax(q.K^T*scale).V over keys
 *   [prefix 0..P) from pk/pv  and  [own_base .. own_base+own_local] from ok/ov. */
void k_attention(const float *q, const bf16 *qkv_for_gate, int qkv_stride, const bf16 *pk,
                 const bf16 *pv, int P, int prefix_stride_tokens, const bf16 *ok, const bf16 *ov,
                 int own_stride_tokens, const cujev_tokmeta *m, bf16 *out, int Hq, int Hkv, int D,
                 float scale, cudaStream_t s);

/* ---- gated delta net --------------------------------------------------- */
/* in[N, C] (row stride `in_stride`, bf16) is the qkv part of the fused
 * projection. Depthwise causal conv, kernel 4, then SiLU -> out fp32 [N, C].
 * conv_state_in: [C][3] last three pre-conv inputs of the prefix (or NULL =
 * zeros) shared by every segment. If conv_state_out != NULL the last three
 * inputs of segment 0 are written there. */
void k_gdn_conv(const bf16 *in, int in_stride, const float *conv_w, const float *conv_state_in,
                float *conv_state_out, float *out, const cujev_tokmeta *m, int C, cudaStream_t s);

/* qkv fp32 [N, 2*Hk*128 + Hv*128] -> q,k l2-normalised (q scaled by 1/sqrt(128))
 * and expanded to Hv heads: q,k,v fp32 [N][Hv][128]. Also g[N][Hv] =
 * -exp(A_log) * softplus(a + dt_bias) and beta[N][Hv] = sigmoid(b) from the
 * fused projection tail ab[N, 2*Hv] (bf16, row stride ab_stride, b first). */
void k_gdn_prep(const float *qkv, const bf16 *ab, int ab_stride, const float *A_neg_exp,
                const float *dt_bias, float *q, float *k, float *v, float *g, float *beta, int N,
                int Hk, int Hv, cudaStream_t s);

/* Recurrent scan, one block per (segment, head). S_in: [Hv][128][128] shared
 * initial state (NULL = zeros). S_out (may be NULL): final state of segment 0.
 * out fp32 [N][Hv][128]. */
void k_gdn_scan(const float *q, const float *k, const float *v, const float *g,
                const float *beta, const float *S_in, float *S_out, float *out,
                const cujev_tokmeta *m, int Hv, cudaStream_t s);

/* y[N, Hv*128] = bf16( rmsnorm_head(out) * w * silu(z) ), z bf16 [N, Hv*128] with row stride. */
void k_gdn_norm_gate(const float *out, const bf16 *z, int z_stride, const float *w, bf16 *y,
                     int N, int Hv, float eps, cudaStream_t s);

/* ---- read-out ----------------------------------------------------------- */
/* logits[b*ld + i] = h[b] . E[cand[b*cand_ld + i]]  for i < ncand[b]. h bf16 [B,H]. */
void k_readout(const bf16 *h, const bf16 *E, const int32_t *cand, const int32_t *ncand, int cand_ld,
               float *logits, int ld, int B, int H, cudaStream_t s);

#ifdef __cplusplus
}
#endif
#endif
