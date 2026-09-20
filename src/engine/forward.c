/*
 * The forward pass. One function handles both uses:
 *
 *   PREFILL  N tokens, 1 segment. Full-attention K/V go into the state's
 *            per-layer cache; every linear layer's final recurrent state and
 *            conv tail are stored in the state.
 *   EVAL     N tokens in B segments (branches). Every token attends to the
 *            state's prefix plus its own segment. Linear layers start every
 *            segment from the state's recurrent state (read-only, never
 *            copied) and the state's conv tail.
 */
#include "model.h"

#define CUDA_TRY(expr)                                                                           \
    do {                                                                                         \
        cudaError_t _e = (expr);                                                                 \
        if (_e != cudaSuccess)                                                                   \
            return cujev_fail(CUJEV_ERR_CUDA, "%s: %s", #expr, cudaGetErrorString(_e));          \
    } while (0)

/* CUJEV_DEBUG_SYNC=1 synchronises after every kernel and reports the first
 * failing one; the substitute for compute-sanitizer where it is unavailable. */
static int debug_sync_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CUJEV_DEBUG_SYNC");
        cached = e && *e && *e != '0';
    }
    return cached;
}

static cujev_status debug_check(cujev_model *m, const char *what, int layer) {
    if (!debug_sync_enabled())
        return CUJEV_OK;
    cudaError_t e = cudaStreamSynchronize(m->stream);
    if (e == cudaSuccess)
        e = cudaGetLastError();
    if (e != cudaSuccess)
        return cujev_fail(CUJEV_ERR_CUDA, "%s (layer %d): %s", what, layer, cudaGetErrorString(e));
    return CUJEV_OK;
}

static cujev_status gemm(cujev_model *m, const bf16 *X, const bf16 *W, bf16 *Y, int N, int M,
                         int K) {
    if (k_gemm_bf16(m->cublas, X, W, Y, N, M, K) != cudaSuccess)
        return cujev_fail(CUJEV_ERR_CUDA, "cublasGemmEx failed (N=%d M=%d K=%d)", N, M, K);
    return CUJEV_OK;
}

cujev_status cujev_forward(cujev_model *m, cujev_state *st, cujev_fwd_mode mode,
                           const int32_t *tokens, int N, const int32_t *seg_start,
                           const int32_t *seg_len, int B) {
    const cujev_config *c = &m->cfg;
    const int H = c->hidden_size, I = c->intermediate_size;
    const int P = mode == CUJEV_FWD_EVAL ? st->num_tokens : 0;
    cudaStream_t s = m->stream;

    /* ---- per-token metadata, built on the host, uploaded once ---- */
    int32_t *h = m->h_meta;
    int32_t *pos = h, *base = h + N, *local = h + 2 * N, *rows = h + 3 * N;
    int32_t *tile_n0 = h + (size_t)m->max_tokens * 4 + (size_t)m->opts.max_branches * 3 +
                       (size_t)m->opts.max_branches * m->opts.max_candidates + m->opts.max_branches;
    int32_t *tile_len = tile_n0 + (m->max_tokens / 16 + m->opts.max_branches + 1);
    int T = 0;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < seg_len[b]; t++) {
            int n = seg_start[b] + t;
            pos[n] = P + t;
            base[n] = seg_start[b];
            local[n] = t;
        }
        rows[b] = seg_start[b] + seg_len[b] - 1;
        for (int t = 0; t < seg_len[b]; t += 16) {
            tile_n0[T] = seg_start[b] + t;
            tile_len[T] = seg_len[b] - t < 16 ? seg_len[b] - t : 16;
            T++;
        }
    }
    CUDA_TRY(cudaMemcpyAsync(m->d_tokens, tokens, (size_t)N * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_pos, pos, (size_t)N * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_base, base, (size_t)N * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_local, local, (size_t)N * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_rows, rows, (size_t)B * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_seg_start, seg_start, (size_t)B * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_seg_len, seg_len, (size_t)B * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_tile_n0, tile_n0, (size_t)T * 4, cudaMemcpyHostToDevice, s));
    CUDA_TRY(cudaMemcpyAsync(m->d_tile_len, tile_len, (size_t)T * 4, cudaMemcpyHostToDevice, s));

    cujev_tokmeta meta = {m->d_pos,       m->d_base,   m->d_local,   m->d_seg_start, m->d_seg_len,
                          m->d_tile_n0,   m->d_tile_len, N, B, T};

    k_embed(m->d_tokens, m->embed, m->x, N, H, s);
    CUJEV_TRY(debug_check(m, "embed", -1));

    for (int l = 0; l < c->num_layers; l++) {
        cujev_layer *ly = &m->layers[l];
        k_rmsnorm(m->x, ly->in_norm, m->xb, N, H, c->rms_norm_eps, NULL, 0, s);

        if (ly->type == CUJEV_LAYER_FULL) {
            CUJEV_TRY(gemm(m, m->xb, ly->w_qkv, m->proj, N, m->w_attn, H));
            bf16 *kdst, *vdst;
            int kstride;
            if (mode == CUJEV_FWD_PREFILL) {
                kdst = st->kv_k[l];
                vdst = st->kv_v[l];
                kstride = m->opts.max_state_tokens;
            } else {
                kdst = m->br_k;
                vdst = m->br_v;
                kstride = m->opts.max_branch_tokens;
            }
            k_attn_prep(m->proj, ly->q_norm, ly->k_norm, m->q, kdst, vdst, kstride, &meta, m->Hq,
                        m->Hkv, m->D, c->rotary_dim, c->rms_norm_eps, c->rope_theta, s);
            k_attention(m->q, m->proj, m->w_attn, st->kv_k[l], st->kv_v[l], P,
                        m->opts.max_state_tokens, kdst, vdst, kstride, &meta, m->mix_out, m->Hq,
                        m->Hkv, m->D, 1.0f / sqrtf((float)m->D), s);
            CUJEV_TRY(debug_check(m, "attention", l));
            CUJEV_TRY(gemm(m, m->mix_out, ly->w_o, m->y, N, H, m->Hq * m->D));
        } else {
            CUJEV_TRY(gemm(m, m->xb, ly->w_in, m->proj, N, m->w_gdn, H));
            const float *conv_in = mode == CUJEV_FWD_EVAL ? st->conv[l] : NULL;
            float *conv_out = mode == CUJEV_FWD_PREFILL ? st->conv[l] : NULL;
            k_gdn_conv(m->proj, m->w_gdn, ly->conv_w, conv_in, conv_out, m->conv_out, &meta, m->C,
                       s);
            const bf16 *ab = m->proj + m->C + m->V; /* b then a, per row */
            k_gdn_prep(m->conv_out, ab, m->w_gdn, ly->A_neg_exp, ly->dt_bias, m->gq, m->gk, m->gv,
                       m->gg, m->gbeta, N, m->Hk, m->Hv, s);
            const float *S_in = mode == CUJEV_FWD_EVAL ? st->S[l] : NULL;
            float *S_out = mode == CUJEV_FWD_PREFILL ? st->S[l] : NULL;
            k_gdn_scan(m->gq, m->gk, m->gv, m->gg, m->gbeta, S_in, S_out, m->gout, &meta, m->Hv, s);
            CUJEV_TRY(debug_check(m, "gdn conv/prep/scan", l));
            k_gdn_norm_gate(m->gout, m->proj + m->C, m->w_gdn, ly->g_norm, m->mix_out, N, m->Hv,
                            c->rms_norm_eps, s);
            CUJEV_TRY(gemm(m, m->mix_out, ly->w_out, m->y, N, H, m->V));
        }
        k_add_bf16(m->x, m->y, N, H, s);

        k_rmsnorm(m->x, ly->post_norm, m->xb, N, H, c->rms_norm_eps, NULL, 0, s);
        CUJEV_TRY(gemm(m, m->xb, ly->w_gate_up, m->proj, N, 2 * I, H));
        k_silu_mul(m->proj, m->act, N, I, s);
        CUJEV_TRY(gemm(m, m->act, ly->w_down, m->y, N, H, I));
        k_add_bf16(m->x, m->y, N, H, s);
        CUJEV_TRY(debug_check(m, "mlp", l));
    }
    k_rmsnorm(m->x, m->final_norm, m->hsel, N, H, c->rms_norm_eps, m->d_rows, B, s);
    CUJEV_TRY(debug_check(m, "final norm", -1));
    CUDA_TRY(cudaGetLastError());
    return CUJEV_OK;
}
