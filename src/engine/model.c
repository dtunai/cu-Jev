#include "model.h"
#include "core/safetensors.h"
#include <math.h>

/* ---- host-side helpers ---------------------------------------------------- */

static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0)
            u = sign << 31;
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            u = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31)
        u = (sign << 31) | 0x7f800000 | (mant << 13);
    else
        u = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1, bias = 0x7fff + lsb; /* round to nearest even */
    return (uint16_t)((u + bias) >> 16);
}

/* Read any float tensor as fp32 into a host buffer. */
static cujev_status tensor_to_f32(const st_tensor *t, float *dst) {
    size_t n = st_numel(t);
    if (t->dtype == ST_F32)
        memcpy(dst, t->data, n * 4);
    else if (t->dtype == ST_BF16)
        for (size_t i = 0; i < n; i++)
            dst[i] = bf16_to_f32(((const uint16_t *)t->data)[i]);
    else if (t->dtype == ST_F16)
        for (size_t i = 0; i < n; i++)
            dst[i] = f16_to_f32(((const uint16_t *)t->data)[i]);
    else
        return cujev_fail(CUJEV_ERR_UNSUPPORTED, "tensor %s: unsupported dtype", t->name);
    return CUJEV_OK;
}

static cujev_status tensor_to_bf16(const st_tensor *t, uint16_t *dst) {
    size_t n = st_numel(t);
    if (t->dtype == ST_BF16)
        memcpy(dst, t->data, n * 2);
    else {
        float *tmp = (float *)malloc(n * 4);
        cujev_status s = tensor_to_f32(t, tmp);
        if (s != CUJEV_OK) {
            free(tmp);
            return s;
        }
        for (size_t i = 0; i < n; i++)
            dst[i] = f32_to_bf16(tmp[i]);
        free(tmp);
    }
    return CUJEV_OK;
}

typedef struct loader {
    cujev_model *m;
    const st_bundle *b;
    const char *prefix; /* "model.language_model." or "model." */
    uint16_t *stage;    /* host staging buffer */
    size_t stage_cap;
} loader;

static const st_tensor *find(loader *L, const char *fmt, int layer) {
    char name[256], full[320];
    snprintf(name, sizeof name, fmt, layer);
    snprintf(full, sizeof full, "%s%s", L->prefix, name);
    return st_bundle_find(L->b, full);
}

static cujev_status want(loader *L, const char *fmt, int layer, const st_tensor **out) {
    *out = find(L, fmt, layer);
    if (!*out) {
        char name[256];
        snprintf(name, sizeof name, fmt, layer);
        return cujev_fail(CUJEV_ERR_FORMAT, "missing tensor %s%s", L->prefix, name);
    }
    return CUJEV_OK;
}

static cujev_status stage_reserve(loader *L, size_t elems) {
    if (elems > L->stage_cap) {
        free(L->stage);
        L->stage = (uint16_t *)malloc(elems * 2);
        if (!L->stage)
            return cujev_fail(CUJEV_ERR_OOM, "host staging %zu elems", elems);
        L->stage_cap = elems;
    }
    return CUJEV_OK;
}

/* Upload a row-concatenation of 2-D tensors (all [rows_i, K]) as one bf16 matrix. */
static cujev_status upload_fused(loader *L, const st_tensor **parts, int nparts, int K,
                                 bf16 **out) {
    size_t rows = 0;
    for (int i = 0; i < nparts; i++) {
        if (parts[i]->ndim != 2 || parts[i]->shape[1] != K)
            return cujev_fail(CUJEV_ERR_FORMAT, "tensor %s: expected [*, %d]", parts[i]->name, K);
        rows += (size_t)parts[i]->shape[0];
    }
    CUJEV_TRY(stage_reserve(L, rows * K));
    size_t off = 0;
    for (int i = 0; i < nparts; i++) {
        CUJEV_TRY(tensor_to_bf16(parts[i], L->stage + off));
        off += st_numel(parts[i]);
    }
    bf16 *d = (bf16 *)cujev_arena_alloc(&L->m->weights, rows * K * 2);
    if (!d)
        return cujev_fail(CUJEV_ERR_OOM, "weight arena exhausted");
    if (cudaMemcpy(d, L->stage, rows * K * 2, cudaMemcpyHostToDevice) != cudaSuccess)
        return cujev_fail(CUJEV_ERR_CUDA, "cudaMemcpy weights");
    *out = d;
    return CUJEV_OK;
}

/* Upload a small fp32 vector, optionally transformed. */
static cujev_status upload_f32(loader *L, const st_tensor *t, float add, int negexp, float **out) {
    size_t n = st_numel(t);
    float *h = (float *)malloc(n * 4);
    cujev_status s = tensor_to_f32(t, h);
    if (s != CUJEV_OK) {
        free(h);
        return s;
    }
    for (size_t i = 0; i < n; i++)
        h[i] = negexp ? -expf(h[i]) : h[i] + add;
    float *d = (float *)cujev_arena_alloc(&L->m->weights, n * 4);
    if (!d) {
        free(h);
        return cujev_fail(CUJEV_ERR_OOM, "weight arena exhausted");
    }
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    free(h);
    *out = d;
    return CUJEV_OK;
}

static size_t tensor_bytes_bf16(const st_tensor *t) { return t ? st_numel(t) * 2 : 0; }

cujev_status cujev_model_load_weights(cujev_model *m, const char *dir) {
    st_bundle b;
    CUJEV_TRY(st_bundle_open(dir, &b));
    loader L = {m, &b, "model.language_model.", NULL, 0};
    if (!st_bundle_find(&b, "model.language_model.embed_tokens.weight"))
        L.prefix = "model.";
    const cujev_config *c = &m->cfg;

    /* Size the arena: sum of everything we will upload (+ small slack). */
    size_t total = 0;
    for (int i = 0; i < b.num_tensors; i++) {
        const st_tensor *t = &b.tensors[i];
        if (strncmp(t->name, L.prefix, strlen(L.prefix)) == 0 || strcmp(t->name, "lm_head.weight") == 0)
            total += cujev_align_up(st_numel(t) * (t->ndim == 1 ? 4 : 2), 256); /* vectors go up as fp32 */
    }
    CUJEV_TRY(cujev_arena_init(&m->weights, total + (16u << 20)));
    cujev_log(m->opts.verbose, "weight arena %.2f GiB (%d tensors in %d shards)",
              total / 1073741824.0, b.num_tensors, b.num_files);

    const st_tensor *t, *parts[4];
    CUJEV_TRY(want(&L, "embed_tokens.weight", 0, &t));
    if (t->shape[0] != c->vocab_size || t->shape[1] != c->hidden_size) {
        st_bundle_close(&b);
        return cujev_fail(CUJEV_ERR_FORMAT, "embed_tokens shape mismatch");
    }
    CUJEV_TRY(upload_fused(&L, &t, 1, c->hidden_size, &m->embed));
    m->lm_head = m->embed;
    if (!c->tie_word_embeddings) {
        const st_tensor *lm = st_bundle_find(&b, "lm_head.weight");
        if (!lm) {
            st_bundle_close(&b);
            return cujev_fail(CUJEV_ERR_FORMAT, "tie_word_embeddings is false but lm_head.weight is missing");
        }
        CUJEV_TRY(upload_fused(&L, &lm, 1, c->hidden_size, &m->lm_head));
    }
    CUJEV_TRY(want(&L, "norm.weight", 0, &t));
    CUJEV_TRY(upload_f32(&L, t, 1.f, 0, &m->final_norm));

    for (int l = 0; l < c->num_layers; l++) {
        cujev_layer *ly = &m->layers[l];
        ly->type = c->layer_types[l];
        CUJEV_TRY(want(&L, "layers.%d.input_layernorm.weight", l, &t));
        CUJEV_TRY(upload_f32(&L, t, 1.f, 0, &ly->in_norm));
        CUJEV_TRY(want(&L, "layers.%d.post_attention_layernorm.weight", l, &t));
        CUJEV_TRY(upload_f32(&L, t, 1.f, 0, &ly->post_norm));
        if (ly->type == CUJEV_LAYER_FULL) {
            CUJEV_TRY(want(&L, "layers.%d.self_attn.q_proj.weight", l, &parts[0]));
            CUJEV_TRY(want(&L, "layers.%d.self_attn.k_proj.weight", l, &parts[1]));
            CUJEV_TRY(want(&L, "layers.%d.self_attn.v_proj.weight", l, &parts[2]));
            CUJEV_TRY(upload_fused(&L, parts, 3, c->hidden_size, &ly->w_qkv));
            CUJEV_TRY(want(&L, "layers.%d.self_attn.o_proj.weight", l, &t));
            CUJEV_TRY(upload_fused(&L, &t, 1, c->num_heads * c->head_dim, &ly->w_o));
            CUJEV_TRY(want(&L, "layers.%d.self_attn.q_norm.weight", l, &t));
            CUJEV_TRY(upload_f32(&L, t, 1.f, 0, &ly->q_norm));
            CUJEV_TRY(want(&L, "layers.%d.self_attn.k_norm.weight", l, &t));
            CUJEV_TRY(upload_f32(&L, t, 1.f, 0, &ly->k_norm));
        } else {
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.in_proj_qkv.weight", l, &parts[0]));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.in_proj_z.weight", l, &parts[1]));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.in_proj_b.weight", l, &parts[2]));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.in_proj_a.weight", l, &parts[3]));
            CUJEV_TRY(upload_fused(&L, parts, 4, c->hidden_size, &ly->w_in));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.out_proj.weight", l, &t));
            CUJEV_TRY(upload_fused(&L, &t, 1, m->V, &ly->w_out));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.conv1d.weight", l, &t));
            if (st_numel(t) != (size_t)m->C * 4)
                return cujev_fail(CUJEV_ERR_FORMAT, "conv1d weight shape");
            CUJEV_TRY(upload_f32(&L, t, 0.f, 0, &ly->conv_w));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.A_log", l, &t));
            CUJEV_TRY(upload_f32(&L, t, 0.f, 1, &ly->A_neg_exp));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.dt_bias", l, &t));
            CUJEV_TRY(upload_f32(&L, t, 0.f, 0, &ly->dt_bias));
            CUJEV_TRY(want(&L, "layers.%d.linear_attn.norm.weight", l, &t));
            CUJEV_TRY(upload_f32(&L, t, 0.f, 0, &ly->g_norm));
        }
        CUJEV_TRY(want(&L, "layers.%d.mlp.gate_proj.weight", l, &parts[0]));
        CUJEV_TRY(want(&L, "layers.%d.mlp.up_proj.weight", l, &parts[1]));
        CUJEV_TRY(upload_fused(&L, parts, 2, c->hidden_size, &ly->w_gate_up));
        CUJEV_TRY(want(&L, "layers.%d.mlp.down_proj.weight", l, &t));
        CUJEV_TRY(upload_fused(&L, &t, 1, c->intermediate_size, &ly->w_down));
        (void)tensor_bytes_bf16;
    }
    free(L.stage);
    st_bundle_close(&b);
    cujev_log(m->opts.verbose, "weights on device: %.2f GiB", m->weights.used / 1073741824.0);
    return CUJEV_OK;
}

cujev_status cujev_model_alloc_workspace(cujev_model *m) {
    const cujev_config *c = &m->cfg;
    int N = m->max_tokens, H = c->hidden_size, I = c->intermediate_size;
    int B = m->opts.max_branches, Nb = m->opts.max_branch_tokens;
    size_t mix = (size_t)N * ((m->Hq * m->D > m->V) ? m->Hq * m->D : m->V);
    size_t bytes = 0;
#define ADD(n) bytes += cujev_align_up((n), 256)
    ADD((size_t)N * H * 4);           /* x */
    ADD((size_t)N * H * 2);           /* xb */
    ADD((size_t)N * m->w_proj * 2);   /* proj */
    ADD((size_t)N * H * 2);           /* y */
    ADD((size_t)N * m->Hq * m->D * 4); /* q */
    ADD(mix * 2);                     /* mix_out */
    ADD((size_t)m->Hkv * Nb * m->D * 2 * 2); /* br_k, br_v */
    ADD((size_t)N * m->C * 4);        /* conv_out */
    ADD((size_t)N * m->Hv * 128 * 4 * 4); /* gq gk gv gout */
    ADD((size_t)N * m->Hv * 4 * 2);   /* gg gbeta */
    ADD((size_t)N * I * 2);           /* act */
    ADD((size_t)B * H * 2);           /* hsel */
    ADD((size_t)(N * 4 + B * 3) * 4); /* meta ints */
    ADD((size_t)(N / 16 + B + 1) * 4 * 2); /* tile table */
    ADD((size_t)B * m->opts.max_candidates * 4 * 2); /* cand + logits */
    ADD((size_t)B * 4);               /* ncand */
#undef ADD
    CUJEV_TRY(cujev_arena_init(&m->work, bytes + (1u << 20)));
    cujev_arena *a = &m->work;
    m->x = (float *)cujev_arena_alloc(a, (size_t)N * H * 4);
    m->xb = (bf16 *)cujev_arena_alloc(a, (size_t)N * H * 2);
    m->proj = (bf16 *)cujev_arena_alloc(a, (size_t)N * m->w_proj * 2);
    m->y = (bf16 *)cujev_arena_alloc(a, (size_t)N * H * 2);
    m->q = (float *)cujev_arena_alloc(a, (size_t)N * m->Hq * m->D * 4);
    m->mix_out = (bf16 *)cujev_arena_alloc(a, mix * 2);
    m->br_k = (bf16 *)cujev_arena_alloc(a, (size_t)m->Hkv * Nb * m->D * 2);
    m->br_v = (bf16 *)cujev_arena_alloc(a, (size_t)m->Hkv * Nb * m->D * 2);
    m->conv_out = (float *)cujev_arena_alloc(a, (size_t)N * m->C * 4);
    m->gq = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 128 * 4);
    m->gk = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 128 * 4);
    m->gv = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 128 * 4);
    m->gout = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 128 * 4);
    m->gg = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 4);
    m->gbeta = (float *)cujev_arena_alloc(a, (size_t)N * m->Hv * 4);
    m->act = (bf16 *)cujev_arena_alloc(a, (size_t)N * I * 2);
    m->hsel = (bf16 *)cujev_arena_alloc(a, (size_t)B * H * 2);
    m->d_tokens = (int32_t *)cujev_arena_alloc(a, (size_t)N * 4);
    m->d_pos = (int32_t *)cujev_arena_alloc(a, (size_t)N * 4);
    m->d_base = (int32_t *)cujev_arena_alloc(a, (size_t)N * 4);
    m->d_local = (int32_t *)cujev_arena_alloc(a, (size_t)N * 4);
    m->d_seg_start = (int32_t *)cujev_arena_alloc(a, (size_t)B * 4);
    m->d_seg_len = (int32_t *)cujev_arena_alloc(a, (size_t)B * 4);
    m->d_rows = (int32_t *)cujev_arena_alloc(a, (size_t)B * 4);
    m->d_tile_n0 = (int32_t *)cujev_arena_alloc(a, (size_t)(N / 16 + B + 1) * 4);
    m->d_tile_len = (int32_t *)cujev_arena_alloc(a, (size_t)(N / 16 + B + 1) * 4);
    m->d_cand = (int32_t *)cujev_arena_alloc(a, (size_t)B * m->opts.max_candidates * 4);
    m->d_logits = (float *)cujev_arena_alloc(a, (size_t)B * m->opts.max_candidates * 4);
    m->d_ncand = (int32_t *)cujev_arena_alloc(a, (size_t)B * 4);
    if (!m->d_ncand)
        return cujev_fail(CUJEV_ERR_OOM, "workspace arena exhausted (bug: undersized)");
    m->h_meta_ints = (size_t)N * 4 + (size_t)B * 3 + (size_t)B * m->opts.max_candidates + B +
                     (size_t)(N / 16 + B + 1) * 2;
    if (cudaMallocHost((void **)&m->h_meta, m->h_meta_ints * 4) != cudaSuccess)
        return cujev_fail(CUJEV_ERR_OOM, "pinned host buffer");
    cujev_log(m->opts.verbose, "workspace: %.2f GiB for %d tokens, %d branches",
              m->work.used / 1073741824.0, N, B);
    return CUJEV_OK;
}
