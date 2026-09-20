#include "common.cuh"

/* Depthwise causal conv (kernel 4) + SiLU. Threads over (token, channel). */
__global__ void gdn_conv_kernel(const bf16 *in, int in_stride, const float *w,
                                const float *state_in, float *out, const int32_t *own_local, int N,
                                int C) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * C)
        return;
    int n = (int)(idx / C), c = (int)(idx % C);
    int local = own_local[n];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int t = local - 3 + j; /* local index of the tap */
        float xv;
        if (t >= 0)
            xv = bf2f(in[(size_t)(n - (local - t)) * in_stride + c]);
        else
            xv = state_in ? state_in[(size_t)c * 3 + (t + 3)] : 0.f;
        acc += w[c * 4 + j] * xv;
    }
    out[idx] = silu_f(acc);
}

__global__ void gdn_conv_tail_kernel(const bf16 *in, int in_stride, const float *state_in,
                                     float *state_out, const int32_t *seg_start_p,
                                     const int32_t *seg_len_p, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    const int seg_start = seg_start_p[0], seg_len = seg_len_p[0];
#pragma unroll
    for (int j = 0; j < 3; j++) {
        int t = seg_len - 3 + j; /* local index */
        float xv;
        if (t >= 0)
            xv = bf2f(in[(size_t)(seg_start + t) * in_stride + c]);
        else
            xv = state_in ? state_in[(size_t)c * 3 + (t + 3)] : 0.f;
        state_out[(size_t)c * 3 + j] = xv;
    }
}

void k_gdn_conv(const bf16 *in, int in_stride, const float *conv_w, const float *conv_state_in,
                float *conv_state_out, float *out, const cujev_tokmeta *m, int C, cudaStream_t s) {
    size_t n = (size_t)m->N * C;
    gdn_conv_kernel<<<(unsigned)CEIL_DIV(n, 256), 256, 0, s>>>(in, in_stride, conv_w, conv_state_in,
                                                              out, m->own_local, m->N, C);
    if (conv_state_out)
        gdn_conv_tail_kernel<<<CEIL_DIV(C, 256), 256, 0, s>>>(in, in_stride, conv_state_in,
                                                              conv_state_out, m->seg_start,
                                                              m->seg_len, C);
}

/* One warp per (token, v-head): l2-normalise q and k, expand k-heads, g, beta. */
__global__ void gdn_prep_kernel(const float *qkv, const bf16 *ab, int ab_stride,
                                const float *A_neg_exp, const float *dt_bias, float *q, float *k,
                                float *v, float *g, float *beta, int N, int Hk, int Hv) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= N * Hv)
        return;
    int n = gw / Hv, hv = gw % Hv;
    int hk = hv / (Hv / Hk);
    int row = (2 * Hk + Hv) * 128;
    const float *qs = qkv + (size_t)n * row + hk * 128 + lane * 4;
    const float *ks = qkv + (size_t)n * row + Hk * 128 + hk * 128 + lane * 4;
    const float *vs = qkv + (size_t)n * row + 2 * Hk * 128 + hv * 128 + lane * 4;
    float qv[4], kv[4];
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        qv[i] = qs[i];
        kv[i] = ks[i];
        sq += qv[i] * qv[i];
        sk += kv[i] * kv[i];
    }
    sq = warp_sum(sq);
    sk = warp_sum(sk);
    float iq = rsqrtf(sq + 1e-6f) * rsqrtf(128.f);
    float ik = rsqrtf(sk + 1e-6f);
    size_t o = ((size_t)n * Hv + hv) * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        q[o + i] = qv[i] * iq;
        k[o + i] = kv[i] * ik;
        v[o + i] = vs[i];
    }
    if (lane == 0) {
        float b = bf2f(ab[(size_t)n * ab_stride + hv]);
        float a = bf2f(ab[(size_t)n * ab_stride + Hv + hv]);
        beta[(size_t)n * Hv + hv] = sigmoid_f(b);
        g[(size_t)n * Hv + hv] = A_neg_exp[hv] * softplus_f(a + dt_bias[hv]);
    }
}

void k_gdn_prep(const float *qkv, const bf16 *ab, int ab_stride, const float *A_neg_exp,
                const float *dt_bias, float *q, float *k, float *v, float *g, float *beta, int N,
                int Hk, int Hv, cudaStream_t s) {
    int warps = N * Hv;
    gdn_prep_kernel<<<CEIL_DIV(warps * 32, 256), 256, 0, s>>>(qkv, ab, ab_stride, A_neg_exp,
                                                             dt_bias, q, k, v, g, beta, N, Hk, Hv);
}

/* Gated delta rule, recurrent form. Block = (segment, head). Each of the 128
 * state columns S[:, v] is split over SPLIT consecutive lanes, each holding
 * 128/SPLIT rows in registers; the two dot products per token (k.S[:,v] and
 * q.S[:,v]) are finished with xor-shuffles between the partners. q_t and k_t
 * are staged in ping-pong shared buffers so one barrier per token suffices. */
template <int SPLIT>
__global__ void __launch_bounds__(128 * SPLIT) gdn_scan_kernel(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ g, const float *__restrict__ beta, const float *__restrict__ S_in,
    float *__restrict__ S_out, float *__restrict__ out, const int32_t *__restrict__ seg_start,
    const int32_t *__restrict__ seg_len, int Hv) {
    constexpr int ROWS = 128 / SPLIT;
    const int seg = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    const int vi = tid / SPLIT, part = tid % SPLIT;
    const int r0 = part * ROWS; /* first state row owned by this thread */
    __shared__ float qs[2][128], ks[2][128];
    float s[ROWS];
    if (S_in) {
#pragma unroll
        for (int i = 0; i < ROWS; i++)
            s[i] = S_in[((size_t)h * 128 + r0 + i) * 128 + vi];
    } else {
#pragma unroll
        for (int i = 0; i < ROWS; i++)
            s[i] = 0.f;
    }
    const int start = seg_start[seg], len = seg_len[seg];
    /* stage token 0 */
    if (tid < 128)
        qs[0][tid] = q[((size_t)start * Hv + h) * 128 + tid];
    else if (tid < 256)
        ks[0][tid - 128] = k[((size_t)start * Hv + h) * 128 + tid - 128];
    __syncthreads();
    for (int t = 0; t < len; t++) {
        const int n = start + t, buf = t & 1;
        const size_t o = ((size_t)n * Hv + h) * 128;
        if (t + 1 < len) { /* prefetch token t+1 into the other buffer */
            const size_t o1 = o + (size_t)Hv * 128;
            if (tid < 128)
                qs[buf ^ 1][tid] = q[o1 + tid];
            else if (tid < 256)
                ks[buf ^ 1][tid - 128] = k[o1 + tid - 128];
        }
        const float vt = v[o + vi];
        const float dec = __expf(g[(size_t)n * Hv + h]);
        const float bt = beta[(size_t)n * Hv + h];
        const float *kk = ks[buf] + r0, *qq = qs[buf] + r0;
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < ROWS; i++) {
            s[i] *= dec;
            kv += s[i] * kk[i];
        }
#pragma unroll
        for (int m = 1; m < SPLIT; m <<= 1)
            kv += __shfl_xor_sync(0xffffffffu, kv, m);
        const float delta = (vt - kv) * bt;
        float ov = 0.f;
#pragma unroll
        for (int i = 0; i < ROWS; i++) {
            s[i] += kk[i] * delta;
            ov += s[i] * qq[i];
        }
#pragma unroll
        for (int m = 1; m < SPLIT; m <<= 1)
            ov += __shfl_xor_sync(0xffffffffu, ov, m);
        if (part == 0)
            out[o + vi] = ov;
        __syncthreads(); /* buf^1 is now complete, buf is free for the next prefetch */
    }
    if (S_out && seg == 0) {
#pragma unroll
        for (int i = 0; i < ROWS; i++)
            S_out[((size_t)h * 128 + r0 + i) * 128 + vi] = s[i];
    }
}

void k_gdn_scan(const float *q, const float *k, const float *v, const float *g,
                const float *beta, const float *S_in, float *S_out, float *out,
                const cujev_tokmeta *m, int Hv, cudaStream_t s) {
    dim3 grid(m->B, Hv);
    gdn_scan_kernel<2><<<grid, 256, 0, s>>>(q, k, v, g, beta, S_in, S_out, out, m->seg_start,
                                            m->seg_len, Hv);
}

/* One warp per (token, head): y = bf16(w * rmsnorm(out)) * silu(z). */
__global__ void gdn_norm_gate_kernel(const float *out, const bf16 *z, int z_stride, const float *w,
                                     bf16 *y, int N, int Hv, float eps) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= N * Hv)
        return;
    int n = gw / Hv, h = gw % Hv;
    const float *src = out + ((size_t)n * Hv + h) * 128 + lane * 4;
    float x[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        x[i] = src[i];
        ss += x[i] * x[i];
    }
    ss = warp_sum(ss);
    float inv = rsqrtf(ss / 128.f + eps);
    const bf16 *zs = z + (size_t)n * z_stride + h * 128 + lane * 4;
    bf16 *dst = y + ((size_t)n * Hv + h) * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float hn = bf2f(f2bf(x[i] * inv)) * w[lane * 4 + i]; /* matches the bf16 round-trip */
        dst[i] = f2bf(hn * silu_f(bf2f(zs[i])));
    }
}

void k_gdn_norm_gate(const float *out, const bf16 *z, int z_stride, const float *w, bf16 *y,
                     int N, int Hv, float eps, cudaStream_t s) {
    int warps = N * Hv;
    gdn_norm_gate_kernel<<<CEIL_DIV(warps * 32, 256), 256, 0, s>>>(out, z, z_stride, w, y, N, Hv,
                                                                  eps);
}
