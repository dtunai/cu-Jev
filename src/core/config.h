#ifndef CUJEV_CONFIG_H
#define CUJEV_CONFIG_H

#include "cujev/cujev.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUJEV_MAX_LAYERS 128

typedef enum { CUJEV_LAYER_LINEAR = 0, CUJEV_LAYER_FULL = 1 } cujev_layer_type;

/* The text_config of a Qwen3.5 checkpoint, the parts the engine needs. */
typedef struct cujev_config {
    char name[64];
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int vocab_size;
    float rms_norm_eps;
    int tie_word_embeddings;
    cujev_layer_type layer_types[CUJEV_MAX_LAYERS];
    int num_full_layers;
    int num_linear_layers;

    /* full attention */
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int rotary_dim; /* head_dim * partial_rotary_factor */
    double rope_theta;
    int attn_output_gate;

    /* linear attention (gated delta net) */
    int lin_num_k_heads;
    int lin_num_v_heads;
    int lin_k_dim;
    int lin_v_dim;
    int lin_conv_kernel;
} cujev_config;

cujev_status cujev_config_load(const char *checkpoint_dir, cujev_config *out);

#ifdef __cplusplus
}
#endif
#endif
