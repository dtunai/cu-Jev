#ifndef CUJEV_MODEL_H
#define CUJEV_MODEL_H

#include "core/config.h"
#include "core/util.h"
#include "cujev/cujev.h"
#include "kernels/kernels.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cujev_layer {
    cujev_layer_type type;
    float *in_norm;   /* [H] fp32, includes the +1 */
    float *post_norm; /* [H] */
    /* full attention */
    bf16 *w_qkv; /* [Hq*2D + 2*Hkv*D, H] */
    float *q_norm, *k_norm; /* [D] */
    bf16 *w_o;   /* [H, Hq*D] */
    /* linear attention */
    bf16 *w_in;   /* [2K + V + V + 2Hv, H] : qkv | z | b | a */
    float *conv_w;    /* [C][4] */
    float *A_neg_exp; /* [Hv] = -exp(A_log) */
    float *dt_bias;   /* [Hv] */
    float *g_norm;    /* [128] */
    bf16 *w_out;  /* [H, V] */
    /* mlp */
    bf16 *w_gate_up; /* [2I, H] */
    bf16 *w_down;    /* [H, I] */
} cujev_layer;

struct cujev_model {
    cujev_config cfg;
    cujev_options opts;
    cujev_arena weights;
    cujev_arena work;
    cublasHandle_t cublas;
    cudaStream_t stream;
    cudaEvent_t ev0, ev1;
    cujev_timing timing;

    bf16 *embed;      /* [V, H] */
    bf16 *lm_head;    /* [V, H]; == embed when tie_word_embeddings */
    float *final_norm; /* [H] */
    cujev_layer layers[CUJEV_MAX_LAYERS];

    /* derived widths */
    int D, Hq, Hkv, Hk, Hv, K, V, C; /* C = 2K+V (conv width) */
    int w_attn, w_gdn, w_mlp, w_proj; /* fused projection widths */
    int max_tokens;                    /* workspace rows */

    /* workspace */
    float *x;       /* [N,H] residual */
    bf16 *xb;       /* [N,H] normed input */
    bf16 *proj;     /* [N, w_proj] */
    bf16 *y;        /* [N,H] projection output */
    float *q;       /* [N,Hq,D] */
    bf16 *mix_out;  /* [N, max(Hq*D, V)] */
    bf16 *br_k, *br_v; /* [Hkv][max_branch_tokens][D] */
    float *conv_out;   /* [N, C] */
    float *gq, *gk, *gv, *gout; /* [N,Hv,128] */
    float *gg, *gbeta;          /* [N,Hv] */
    bf16 *act;      /* [N, I] */
    bf16 *hsel;     /* [B, H] */
    float *full_logits; /* [B, V] lazily allocated */
    int full_logits_rows;

    int32_t *d_tokens, *d_pos, *d_base, *d_local, *d_seg_start, *d_seg_len, *d_rows;
    int32_t *d_tile_n0, *d_tile_len;
    int32_t *d_cand, *d_ncand;
    float *d_logits; /* [B, max_candidates] */
    int32_t *h_meta; /* pinned host staging for the meta arrays */
    size_t h_meta_ints;
};

struct cujev_state {
    cujev_model *model;
    cujev_arena mem;
    int num_tokens;
    bf16 **kv_k, **kv_v; /* per layer (NULL for linear layers) [Hkv][max_state][D] */
    float **S;           /* per layer (NULL for full layers)  [Hv][128][128] */
    float **conv;        /* per layer [C][3] */
};

cujev_status cujev_model_load_weights(cujev_model *m, const char *dir);
cujev_status cujev_model_alloc_workspace(cujev_model *m);

typedef enum { CUJEV_FWD_PREFILL = 0, CUJEV_FWD_EVAL = 1 } cujev_fwd_mode;

/* Run the network over `N` tokens organised in `B` segments (see forward.cu).
 * meta arrays are host pointers; they are uploaded inside. On return the
 * final-norm output of every segment's last token is in m->hsel[B,H]. */
cujev_status cujev_forward(cujev_model *m, cujev_state *st, cujev_fwd_mode mode,
                           const int32_t *tokens, int N, const int32_t *seg_start,
                           const int32_t *seg_len, int B);

#ifdef __cplusplus
}
#endif
#endif
