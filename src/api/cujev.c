#include "engine/model.h"
#include <math.h>

#define CUDA_TRY(expr)                                                                           \
    do {                                                                                         \
        cudaError_t _e = (expr);                                                                 \
        if (_e != cudaSuccess)                                                                   \
            return cujev_fail(CUJEV_ERR_CUDA, "%s: %s", #expr, cudaGetErrorString(_e));          \
    } while (0)

void cujev_options_default(cujev_options *o) {
    memset(o, 0, sizeof *o);
    o->device = 0;
    o->max_state_tokens = 8192;
    o->max_branch_tokens = 4096;
    o->max_branches = 256;
    o->max_candidates = 256;
}

cujev_status cujev_model_open(const char *dir, const cujev_options *opts, cujev_model **out) {
    if (!dir || !out)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "null argument");
    cujev_model *m = (cujev_model *)calloc(1, sizeof *m);
    if (opts)
        m->opts = *opts;
    else
        cujev_options_default(&m->opts);
    cujev_status s = cujev_config_load(dir, &m->cfg);
    if (s != CUJEV_OK) {
        free(m);
        return s;
    }
    const cujev_config *c = &m->cfg;
    m->D = c->head_dim;
    m->Hq = c->num_heads;
    m->Hkv = c->num_kv_heads;
    m->Hk = c->lin_num_k_heads;
    m->Hv = c->lin_num_v_heads;
    m->K = m->Hk * c->lin_k_dim;
    m->V = m->Hv * c->lin_v_dim;
    m->C = 2 * m->K + m->V;
    m->w_attn = m->Hq * 2 * m->D + 2 * m->Hkv * m->D;
    m->w_gdn = m->C + m->V + 2 * m->Hv;
    m->w_mlp = 2 * c->intermediate_size;
    m->w_proj = m->w_attn;
    if (m->w_gdn > m->w_proj)
        m->w_proj = m->w_gdn;
    if (m->w_mlp > m->w_proj)
        m->w_proj = m->w_mlp;
    m->max_tokens = m->opts.max_state_tokens > m->opts.max_branch_tokens
                        ? m->opts.max_state_tokens
                        : m->opts.max_branch_tokens;

    if (cudaSetDevice(m->opts.device) != cudaSuccess) {
        int dev = m->opts.device;
        free(m);
        return cujev_fail(CUJEV_ERR_CUDA, "cudaSetDevice(%d)", dev);
    }
    cudaStreamCreate(&m->stream);
    cudaEventCreate(&m->ev0);
    cudaEventCreate(&m->ev1);
    if (cublasCreate(&m->cublas) != CUBLAS_STATUS_SUCCESS) {
        free(m);
        return cujev_fail(CUJEV_ERR_CUDA, "cublasCreate");
    }
    cublasSetStream(m->cublas, m->stream);

    s = cujev_model_load_weights(m, dir);
    if (s == CUJEV_OK)
        s = cujev_model_alloc_workspace(m);
    if (s != CUJEV_OK) {
        cujev_model_close(m);
        return s;
    }
    cudaStreamSynchronize(m->stream);
    *out = m;
    return CUJEV_OK;
}

void cujev_model_close(cujev_model *m) {
    if (!m)
        return;
    if (m->full_logits)
        cudaFree(m->full_logits);
    if (m->h_meta)
        cudaFreeHost(m->h_meta);
    cujev_arena_free(&m->work);
    cujev_arena_free(&m->weights);
    if (m->cublas)
        cublasDestroy(m->cublas);
    if (m->stream)
        cudaStreamDestroy(m->stream);
    if (m->ev0)
        cudaEventDestroy(m->ev0);
    if (m->ev1)
        cudaEventDestroy(m->ev1);
    free(m);
}

cujev_status cujev_model_info_get(const cujev_model *m, cujev_model_info *o) {
    if (!m || !o)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "null argument");
    memset(o, 0, sizeof *o);
    snprintf(o->name, sizeof o->name, "%s", m->cfg.name);
    o->hidden_size = m->cfg.hidden_size;
    o->num_layers = m->cfg.num_layers;
    o->num_full_attention_layers = m->cfg.num_full_layers;
    o->vocab_size = m->cfg.vocab_size;
    o->max_state_tokens = m->opts.max_state_tokens;
    o->max_branch_tokens = m->opts.max_branch_tokens;
    o->max_branches = m->opts.max_branches;
    o->device_bytes_weights = m->weights.used;
    o->device_bytes_workspace = m->work.used;
    return CUJEV_OK;
}

cujev_status cujev_state_create(cujev_model *m, cujev_state **out) {
    if (!m || !out)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "null argument");
    cujev_state *st = (cujev_state *)calloc(1, sizeof *st);
    st->model = m;
    int L = m->cfg.num_layers;
    st->kv_k = (bf16 **)calloc((size_t)L, sizeof(bf16 *));
    st->kv_v = (bf16 **)calloc((size_t)L, sizeof(bf16 *));
    st->S = (float **)calloc((size_t)L, sizeof(float *));
    st->conv = (float **)calloc((size_t)L, sizeof(float *));
    size_t kv_bytes = (size_t)m->Hkv * m->opts.max_state_tokens * m->D * 2;
    size_t S_bytes = (size_t)m->Hv * 128 * 128 * 4;
    size_t conv_bytes = (size_t)m->C * 3 * 4;
    size_t total = 0;
    for (int l = 0; l < L; l++)
        total += m->cfg.layer_types[l] == CUJEV_LAYER_FULL
                     ? 2 * cujev_align_up(kv_bytes, 256)
                     : cujev_align_up(S_bytes, 256) + cujev_align_up(conv_bytes, 256);
    cujev_status s = cujev_arena_init(&st->mem, total + 4096);
    if (s != CUJEV_OK) {
        cujev_state_destroy(st);
        return s;
    }
    for (int l = 0; l < L; l++) {
        if (m->cfg.layer_types[l] == CUJEV_LAYER_FULL) {
            st->kv_k[l] = (bf16 *)cujev_arena_alloc(&st->mem, kv_bytes);
            st->kv_v[l] = (bf16 *)cujev_arena_alloc(&st->mem, kv_bytes);
        } else {
            st->S[l] = (float *)cujev_arena_alloc(&st->mem, S_bytes);
            st->conv[l] = (float *)cujev_arena_alloc(&st->mem, conv_bytes);
        }
    }
    *out = st;
    return CUJEV_OK;
}

void cujev_state_destroy(cujev_state *st) {
    if (!st)
        return;
    cujev_arena_free(&st->mem);
    free(st->kv_k);
    free(st->kv_v);
    free(st->S);
    free(st->conv);
    free(st);
}

int cujev_state_num_tokens(const cujev_state *st) { return st ? st->num_tokens : 0; }

cujev_status cujev_state_prefill(cujev_model *m, cujev_state *st, const int32_t *tokens, int N) {
    if (!m || !st || !tokens || N <= 0)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "bad prefill arguments");
    if (st->model != m)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "state belongs to another model");
    if (N > m->opts.max_state_tokens)
        return cujev_fail(CUJEV_ERR_CAPACITY, "state of %d tokens exceeds max_state_tokens=%d", N,
                          m->opts.max_state_tokens);
    for (int i = 0; i < N; i++)
        if (tokens[i] < 0 || tokens[i] >= m->cfg.vocab_size)
            return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "token id %d out of range", tokens[i]);
    int32_t ss = 0, sl = N;
    st->num_tokens = 0;
    cudaEventRecord(m->ev0, m->stream);
    CUJEV_TRY(cujev_forward(m, st, CUJEV_FWD_PREFILL, tokens, N, &ss, &sl, 1));
    cudaEventRecord(m->ev1, m->stream);
    CUDA_TRY(cudaStreamSynchronize(m->stream));
    cudaEventElapsedTime(&m->timing.prefill_ms, m->ev0, m->ev1);
    m->timing.prefill_tokens = N;
    st->num_tokens = N;
    return CUJEV_OK;
}

static cujev_status pack_branches(cujev_model *m, const cujev_branch *br, int B, int32_t **tok,
                                  int32_t *seg_start, int32_t *seg_len, int *N_out) {
    int N = 0;
    for (int b = 0; b < B; b++) {
        if (br[b].num_tokens <= 0 || !br[b].tokens)
            return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "branch %d has no tokens", b);
        if (br[b].num_candidates > m->opts.max_candidates)
            return cujev_fail(CUJEV_ERR_CAPACITY, "branch %d: %d candidates > max_candidates=%d",
                              b, br[b].num_candidates, m->opts.max_candidates);
        seg_start[b] = N;
        seg_len[b] = br[b].num_tokens;
        N += br[b].num_tokens;
    }
    if (N > m->opts.max_branch_tokens)
        return cujev_fail(CUJEV_ERR_CAPACITY, "%d branch tokens exceed max_branch_tokens=%d", N,
                          m->opts.max_branch_tokens);
    int32_t *t = (int32_t *)malloc((size_t)N * 4);
    for (int b = 0; b < B; b++)
        memcpy(t + seg_start[b], br[b].tokens, (size_t)br[b].num_tokens * 4);
    for (int i = 0; i < N; i++)
        if (t[i] < 0 || t[i] >= m->cfg.vocab_size) {
            int32_t bad = t[i];
            free(t);
            return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "token id %d out of range", bad);
        }
    *tok = t;
    *N_out = N;
    return CUJEV_OK;
}

static cujev_status check_eval_args(cujev_model *m, cujev_state *st, const cujev_branch *br,
                                    int B) {
    if (!m || !st || !br || B <= 0)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "bad eval arguments");
    if (st->model != m)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "state belongs to another model");
    if (st->num_tokens == 0)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "state has not been prefilled");
    if (B > m->opts.max_branches)
        return cujev_fail(CUJEV_ERR_CAPACITY, "%d branches > max_branches=%d", B,
                          m->opts.max_branches);
    return CUJEV_OK;
}

cujev_status cujev_eval(cujev_model *m, cujev_state *st, const cujev_branch *br, int B,
                        float *logits_out, int ld) {
    CUJEV_TRY(check_eval_args(m, st, br, B));
    if (!logits_out)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "logits_out is NULL");
    int32_t seg_start[B], seg_len[B];
    int32_t *tok;
    int N;
    CUJEV_TRY(pack_branches(m, br, B, &tok, seg_start, seg_len, &N));
    int maxc = 0;
    for (int b = 0; b < B; b++)
        if (br[b].num_candidates > maxc)
            maxc = br[b].num_candidates;
    if (ld < maxc) {
        free(tok);
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "ld=%d < max candidates=%d", ld, maxc);
    }
    /* candidates -> pinned staging (after the meta region) -> device */
    int32_t *hc = m->h_meta + (size_t)m->max_tokens * 4 + (size_t)m->opts.max_branches * 3;
    int32_t *hn = hc + (size_t)B * maxc;
    for (int b = 0; b < B; b++) {
        int n = br[b].num_candidates;
        hn[b] = n;
        for (int i = 0; i < n; i++) {
            int32_t id = br[b].candidates[i];
            if (id < 0 || id >= m->cfg.vocab_size) {
                free(tok);
                return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "candidate id %d out of range", id);
            }
            hc[(size_t)b * maxc + i] = id;
        }
    }
    cudaEventRecord(m->ev0, m->stream);
    cujev_status s = cujev_forward(m, st, CUJEV_FWD_EVAL, tok, N, seg_start, seg_len, B);
    free(tok);
    CUJEV_TRY(s);
    if (maxc > 0) {
        CUDA_TRY(cudaMemcpyAsync(m->d_cand, hc, (size_t)B * maxc * 4, cudaMemcpyHostToDevice,
                                 m->stream));
        CUDA_TRY(cudaMemcpyAsync(m->d_ncand, hn, (size_t)B * 4, cudaMemcpyHostToDevice, m->stream));
        k_readout(m->hsel, m->lm_head, m->d_cand, m->d_ncand, maxc, m->d_logits, maxc, B,
                  m->cfg.hidden_size, m->stream);
        CUDA_TRY(cudaMemcpy2DAsync(logits_out, (size_t)ld * 4, m->d_logits, (size_t)maxc * 4,
                                   (size_t)maxc * 4, (size_t)B, cudaMemcpyDeviceToHost,
                                   m->stream));
    }
    cudaEventRecord(m->ev1, m->stream);
    CUDA_TRY(cudaStreamSynchronize(m->stream));
    cudaEventElapsedTime(&m->timing.eval_ms, m->ev0, m->ev1);
    m->timing.eval_tokens = N;
    m->timing.eval_branches = B;
    return CUJEV_OK;
}

cujev_status cujev_eval_full_logits(cujev_model *m, cujev_state *st, const cujev_branch *br,
                                    int B, float *logits_out) {
    CUJEV_TRY(check_eval_args(m, st, br, B));
    if (!logits_out)
        return cujev_fail(CUJEV_ERR_INVALID_ARGUMENT, "logits_out is NULL");
    int32_t seg_start[B], seg_len[B];
    int32_t *tok;
    int N;
    CUJEV_TRY(pack_branches(m, br, B, &tok, seg_start, seg_len, &N));
    int V = m->cfg.vocab_size, H = m->cfg.hidden_size;
    if (m->full_logits_rows < B) {
        if (m->full_logits)
            cudaFree(m->full_logits);
        if (cudaMalloc((void **)&m->full_logits, (size_t)B * V * 4) != cudaSuccess) {
            free(tok);
            m->full_logits = NULL;
            m->full_logits_rows = 0;
            return cujev_fail(CUJEV_ERR_OOM, "full logits buffer");
        }
        m->full_logits_rows = B;
    }
    cudaEventRecord(m->ev0, m->stream);
    cujev_status s = cujev_forward(m, st, CUJEV_FWD_EVAL, tok, N, seg_start, seg_len, B);
    free(tok);
    CUJEV_TRY(s);
    if (k_gemm_bf16_f32out(m->cublas, m->hsel, m->lm_head, m->full_logits, B, V, H) != cudaSuccess)
        return cujev_fail(CUJEV_ERR_CUDA, "lm head gemm");
    CUDA_TRY(cudaMemcpyAsync(logits_out, m->full_logits, (size_t)B * V * 4, cudaMemcpyDeviceToHost,
                             m->stream));
    cudaEventRecord(m->ev1, m->stream);
    CUDA_TRY(cudaStreamSynchronize(m->stream));
    cudaEventElapsedTime(&m->timing.eval_ms, m->ev0, m->ev1);
    m->timing.eval_tokens = N;
    m->timing.eval_branches = B;
    return CUJEV_OK;
}

void cujev_last_timing(const cujev_model *m, cujev_timing *out) {
    if (m && out)
        *out = m->timing;
}
