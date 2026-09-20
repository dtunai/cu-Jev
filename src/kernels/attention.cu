#include "common.cuh"
#include <stdint.h>

/* ======================================================================
 * Q/K/V preparation: per-head zero-centred RMSNorm on q and k, partial RoPE
 * (rotate_half over the first `rot` dims), K/V written into their cache.
 * One warp per head; lane l owns dims [8l, 8l+8). D == 256.
 * ==================================================================== */
__constant__ float c_inv_freq[64];

__device__ __forceinline__ void head_norm_rope(float *x, const float *w, float eps, int pos,
                                               int rot) {
    int lane = threadIdx.x & 31;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        ss += x[i] * x[i];
    ss = warp_sum(ss);
    float inv = rsqrtf(ss / 256.f + eps);
#pragma unroll
    for (int i = 0; i < 8; i++)
        x[i] = x[i] * inv * w[lane * 8 + i];
    int half_lanes = rot / 16; /* lanes holding the first half of the rotary dims */
    float partner[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
        partner[i] = __shfl_xor_sync(0xffffffffu, x[i], half_lanes);
    if (lane < 2 * half_lanes) {
        int first = lane < half_lanes;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane * 8 + i;
            int fi = first ? d : d - rot / 2;
            float c, s;
            sincosf((float)pos * c_inv_freq[fi], &s, &c);
            x[i] = first ? x[i] * c - partner[i] * s : x[i] * c + partner[i] * s;
        }
    }
}

__global__ void attn_prep_kernel(const bf16 *qkv, const float *q_norm_w, const float *k_norm_w,
                                 float *q, bf16 *kv_k, bf16 *kv_v, int kv_stride, const int32_t *pos,
                                 const int32_t *own_base, const int32_t *own_local, int Hq, int Hkv,
                                 float eps, int rot) {
    const int D = 256;
    int n = blockIdx.x;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row_stride = Hq * 2 * D + 2 * Hkv * D;
    const bf16 *row = qkv + (size_t)n * row_stride;
    int p = pos[n];
    float x[8];
    if (warp < Hq) {
        const bf16 *src = row + warp * 2 * D + lane * 8;
        uint4 raw = *(const uint4 *)src;
        const bf16 *b = (const bf16 *)&raw;
#pragma unroll
        for (int i = 0; i < 8; i++)
            x[i] = bf2f(b[i]);
        head_norm_rope(x, q_norm_w, eps, p, rot);
        float4 *dst = (float4 *)(q + ((size_t)n * Hq + warp) * D + lane * 8);
        dst[0] = make_float4(x[0], x[1], x[2], x[3]);
        dst[1] = make_float4(x[4], x[5], x[6], x[7]);
    } else if (warp < Hq + Hkv) {
        int kh = warp - Hq;
        int slot = own_base[n] + own_local[n];
        uint4 raw = *(const uint4 *)(row + Hq * 2 * D + kh * D + lane * 8);
        const bf16 *b = (const bf16 *)&raw;
#pragma unroll
        for (int i = 0; i < 8; i++)
            x[i] = bf2f(b[i]);
        head_norm_rope(x, k_norm_w, eps, p, rot);
        uint4 packed;
        bf16 *pb = (bf16 *)&packed;
#pragma unroll
        for (int i = 0; i < 8; i++)
            pb[i] = f2bf(x[i]);
        *(uint4 *)(kv_k + ((size_t)kh * kv_stride + slot) * D + lane * 8) = packed;
        *(uint4 *)(kv_v + ((size_t)kh * kv_stride + slot) * D + lane * 8) =
            *(const uint4 *)(row + Hq * 2 * D + Hkv * D + kh * D + lane * 8);
    }
}

void k_attn_prep(const bf16 *qkv, const float *q_norm_w, const float *k_norm_w, float *q,
                 bf16 *kv_k, bf16 *kv_v, int kv_stride_tokens, const cujev_tokmeta *m, int Hq,
                 int Hkv, int D, int rot, float eps, double theta, cudaStream_t s) {
    (void)D;
    static double cached_theta = 0;
    static int cached_rot = 0;
    if (cached_theta != theta || cached_rot != rot) {
        float inv[64];
        for (int i = 0; i < rot / 2; i++)
            inv[i] = (float)(1.0 / pow(theta, 2.0 * i / rot));
        cudaMemcpyToSymbol(c_inv_freq, inv, sizeof(float) * (rot / 2));
        cached_theta = theta;
        cached_rot = rot;
    }
    attn_prep_kernel<<<m->N, 32 * (Hq + Hkv), 0, s>>>(qkv, q_norm_w, k_norm_w, q, kv_k, kv_v,
                                                      kv_stride_tokens, m->pos, m->own_base,
                                                      m->own_local, Hq, Hkv, eps, rot);
}

/* ======================================================================
 * Attention, FlashAttention-2 style on tensor cores (mma.sync m16n8k16 bf16).
 *
 * One block = one kv-head x one tile of 16 consecutive query tokens of one
 * segment. The 4 q-heads that share the kv-head are the block's 4 warps, so
 * a K/V tile loaded into shared memory serves 64 query rows (16 tokens x 4
 * heads). Keys are [prefix 0..P) from the state cache followed by the
 * segment's own keys [0 .. own_local] from the branch cache; the own part is
 * causally masked per query token. Output = sigmoid(gate) * softmax(QK^T)V.
 * ==================================================================== */
#define ATT_BR 64      /* query rows per block: 16 tokens x 4 heads */
#define ATT_TQ 16      /* query tokens per block */
#define ATT_BC 64      /* keys per tile */
#define ATT_D 256
#define ATT_LD 264     /* padded smem row length (bf16) -> conflict-free fragment loads */
#define ATT_SMEM_BYTES ((ATT_BR + 2 * ATT_BC) * ATT_LD * 2)

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(uint32_t *)&v;
}

__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t *r, const void *smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__global__ void __launch_bounds__(128, 1) attention_tc_kernel(
    const float *__restrict__ q, const bf16 *__restrict__ qkv, int qkv_stride,
    const bf16 *__restrict__ pk, const bf16 *__restrict__ pv, int P, int p_stride,
    const bf16 *__restrict__ ok, const bf16 *__restrict__ ov, int o_stride,
    const int32_t *__restrict__ own_base, const int32_t *__restrict__ own_local,
    const int32_t *__restrict__ tile_n0, const int32_t *__restrict__ tile_len,
    bf16 *__restrict__ out, int Hq, int Hkv, float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16 *Qs = (bf16 *)smem_raw;                /* [64][264] */
    bf16 *Ks = Qs + ATT_BR * ATT_LD;            /* [64][264] */
    bf16 *Vs = Ks + ATT_BC * ATT_LD;            /* [64][264] */

    const int tile = blockIdx.x, kh = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tid = threadIdx.x;
    const int n0 = tile_n0[tile], tlen = tile_len[tile];
    const int base = own_base[n0], local0 = own_local[n0];
    const int head = kh * 4 + warp;
    const int nown = local0 + tlen; /* own keys visible to the last token of the tile */
    const int total_keys = P + nown;

    /* ---- Q tile -> bf16 smem (pre-scaled by scale*log2e) ---- */
    for (int i = tid; i < ATT_BR * (ATT_D / 8); i += 128) {
        int r = i / (ATT_D / 8), c8 = (i % (ATT_D / 8)) * 8;
        int w = r >> 4, t = r & 15;
        uint4 packed;
        if (t < tlen) {
            const float4 *src = (const float4 *)(q + ((size_t)(n0 + t) * Hq + kh * 4 + w) * ATT_D + c8);
            float4 a = src[0], b = src[1];
            uint32_t *pp = (uint32_t *)&packed;
            pp[0] = pack_bf16x2(a.x * scale_log2, a.y * scale_log2);
            pp[1] = pack_bf16x2(a.z * scale_log2, a.w * scale_log2);
            pp[2] = pack_bf16x2(b.x * scale_log2, b.y * scale_log2);
            pp[3] = pack_bf16x2(b.z * scale_log2, b.w * scale_log2);
        } else {
            packed = make_uint4(0, 0, 0, 0);
        }
        *(uint4 *)(Qs + r * ATT_LD + c8) = packed;
    }

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++)
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int row0 = lane >> 2;        /* token index (0..15) of rows c0/c1 */
    const int row1 = row0 + 8;         /* token index of rows c2/c3 */
    const int lim0 = local0 + row0;    /* last visible own key for row0 */
    const int lim1 = local0 + row1;

    for (int k0 = 0; k0 < total_keys; k0 += ATT_BC) {
        __syncthreads(); /* previous tile fully consumed */
        /* ---- load K and V tiles (zero-filled past the end) ---- */
        for (int i = tid; i < ATT_BC * (ATT_D / 8); i += 128) {
            int r = i / (ATT_D / 8), c8 = (i % (ATT_D / 8)) * 8;
            int j = k0 + r;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < total_keys) {
                const bf16 *krow, *vrow;
                if (j < P) {
                    krow = pk + ((size_t)kh * p_stride + j) * ATT_D;
                    vrow = pv + ((size_t)kh * p_stride + j) * ATT_D;
                } else {
                    krow = ok + ((size_t)kh * o_stride + base + (j - P)) * ATT_D;
                    vrow = ov + ((size_t)kh * o_stride + base + (j - P)) * ATT_D;
                }
                kk = *(const uint4 *)(krow + c8);
                vv = *(const uint4 *)(vrow + c8);
            }
            *(uint4 *)(Ks + r * ATT_LD + c8) = kk;
            *(uint4 *)(Vs + r * ATT_LD + c8) = vv;
        }
        __syncthreads();

        /* ---- S = Q K^T for this warp's 16 rows x 64 keys ---- */
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++)
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        const bf16 *qw = Qs + (warp * 16) * ATT_LD;
#pragma unroll
        for (int kk = 0; kk < ATT_D / 16; kk++) {
            uint32_t a[4];
            const bf16 *qa = qw + row0 * ATT_LD + kk * 16 + (lane & 3) * 2;
            a[0] = *(const uint32_t *)(qa);
            a[1] = *(const uint32_t *)(qa + 8 * ATT_LD);
            a[2] = *(const uint32_t *)(qa + 8);
            a[3] = *(const uint32_t *)(qa + 8 * ATT_LD + 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t b[2];
                const bf16 *kb = Ks + (j * 8 + row0) * ATT_LD + kk * 16 + (lane & 3) * 2;
                b[0] = *(const uint32_t *)(kb);
                b[1] = *(const uint32_t *)(kb + 8);
                mma16816(s[j], a, b);
            }
        }

        /* ---- mask + online softmax (base 2) ---- */
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; j++) {
#pragma unroll
            for (int c = 0; c < 2; c++) {
                int key = k0 + j * 8 + (lane & 3) * 2 + c;
                bool vis = key < total_keys;
                bool vis0 = vis && (key < P || key - P <= lim0);
                bool vis1 = vis && (key < P || key - P <= lim1);
                if (!vis0) s[j][c] = -INFINITY;
                if (!vis1) s[j][2 + c] = -INFINITY;
                mt0 = fmaxf(mt0, s[j][c]);
                mt1 = fmaxf(mt1, s[j][2 + c]);
            }
        }
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffffu, mt0, 1));
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffffu, mt0, 2));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffffu, mt1, 1));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffffu, mt1, 2));
        float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
        float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        float sc0 = exp2f(m0 - ms0), sc1 = exp2f(m1 - ms1);
        l0 *= sc0;
        l1 *= sc1;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            o[j][0] *= sc0;
            o[j][1] *= sc0;
            o[j][2] *= sc1;
            o[j][3] *= sc1;
        }
        uint32_t pa[4][4]; /* P as A fragments: 4 k-steps of 16 keys */
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float p0 = exp2f(s[j][0] - ms0), p1 = exp2f(s[j][1] - ms0);
            float p2 = exp2f(s[j][2] - ms1), p3 = exp2f(s[j][3] - ms1);
            l0 += p0 + p1;
            l1 += p2 + p3;
            int kk = j >> 1, hi = j & 1;
            pa[kk][hi ? 2 : 0] = pack_bf16x2(p0, p1);
            pa[kk][hi ? 3 : 1] = pack_bf16x2(p2, p3);
        }
        m0 = mn0;
        m1 = mn1;

        /* ---- O += P V  (V B-fragments via ldmatrix.trans) ---- */
        {
            int mat = lane >> 3, rr = lane & 7;
#pragma unroll
            for (int kk = 0; kk < ATT_BC / 16; kk++) {
                const bf16 *vbase = Vs + (kk * 16 + (mat & 1) * 8 + rr) * ATT_LD + (mat >> 1) * 8;
#pragma unroll
                for (int j = 0; j < 32; j += 2) {
                    uint32_t b[4];
                    ldmatrix_x4_trans(b, vbase + j * 8);
                    mma16816(o[j], pa[kk], b);
                    mma16816(o[j + 1], pa[kk], b + 2);
                }
            }
        }
    }

    /* ---- epilogue: normalise, gate, store ---- */
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int col = (lane & 3) * 2;
    if (row0 < tlen) {
        int n = n0 + row0;
        const bf16 *g = qkv + (size_t)n * qkv_stride + head * 2 * ATT_D + ATT_D;
        bf16 *dst = out + (size_t)n * Hq * ATT_D + head * ATT_D;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            int d = j * 8 + col;
            __nv_bfloat162 gg = *(const __nv_bfloat162 *)(g + d);
            float v0 = o[j][0] * inv0 * sigmoid_f(__bfloat162float(gg.x));
            float v1 = o[j][1] * inv0 * sigmoid_f(__bfloat162float(gg.y));
            *(__nv_bfloat162 *)(dst + d) = __floats2bfloat162_rn(v0, v1);
        }
    }
    if (row1 < tlen) {
        int n = n0 + row1;
        const bf16 *g = qkv + (size_t)n * qkv_stride + head * 2 * ATT_D + ATT_D;
        bf16 *dst = out + (size_t)n * Hq * ATT_D + head * ATT_D;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            int d = j * 8 + col;
            __nv_bfloat162 gg = *(const __nv_bfloat162 *)(g + d);
            float v0 = o[j][2] * inv1 * sigmoid_f(__bfloat162float(gg.x));
            float v1 = o[j][3] * inv1 * sigmoid_f(__bfloat162float(gg.y));
            *(__nv_bfloat162 *)(dst + d) = __floats2bfloat162_rn(v0, v1);
        }
    }
}

void k_attention(const float *q, const bf16 *qkv_for_gate, int qkv_stride, const bf16 *pk,
                 const bf16 *pv, int P, int prefix_stride_tokens, const bf16 *ok, const bf16 *ov,
                 int own_stride_tokens, const cujev_tokmeta *m, bf16 *out, int Hq, int Hkv, int D,
                 float scale, cudaStream_t s) {
    (void)D;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attention_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             ATT_SMEM_BYTES);
        configured = true;
    }
    dim3 grid(m->num_tiles, Hkv);
    attention_tc_kernel<<<grid, 128, ATT_SMEM_BYTES, s>>>(
        q, qkv_for_gate, qkv_stride, pk, pv, P, prefix_stride_tokens, ok, ov, own_stride_tokens,
        m->own_base, m->own_local, m->tile_n0, m->tile_len, out, Hq, Hkv,
        scale * 1.4426950408889634f);
}
