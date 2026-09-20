/*
 * cu-Jev — a CUDA-native System One decision engine.
 *
 * Public C API. Everything the Python layer needs is here; the Python layer
 * owns tokenisation, prompt layout and HTTP, this library owns the model.
 *
 * Model of execution
 * ------------------
 *   1. cujev_state_prefill()  runs the shared prefix ("state" in Jev terms)
 *      once and keeps what later tokens need: the KV cache of every
 *      full-attention layer and the recurrent + conv state of every
 *      linear-attention (Gated DeltaNet) layer.
 *   2. cujev_eval()           runs M independent branches ("questions") in one
 *      batched forward. Every branch token sees the shared prefix and its own
 *      branch, never another branch. At the last token of every branch it
 *      returns the logits of the candidate token ids you asked for.
 *
 * Nothing is ever decoded. Output tokens are free.
 */
#ifndef CUJEV_H
#define CUJEV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUJEV_VERSION_MAJOR 0
#define CUJEV_VERSION_MINOR 1
#define CUJEV_VERSION_PATCH 0

typedef struct cujev_model cujev_model;
typedef struct cujev_state cujev_state;

typedef enum cujev_status {
    CUJEV_OK = 0,
    CUJEV_ERR_INVALID_ARGUMENT = 1,
    CUJEV_ERR_IO = 2,
    CUJEV_ERR_FORMAT = 3,
    CUJEV_ERR_CUDA = 4,
    CUJEV_ERR_OOM = 5,
    CUJEV_ERR_CAPACITY = 6, /* more tokens/branches than the model was opened for */
    CUJEV_ERR_UNSUPPORTED = 7,
} cujev_status;

typedef struct cujev_options {
    int device;               /* CUDA device ordinal, default 0 */
    int max_state_tokens;     /* capacity of the shared prefix, default 8192 */
    int max_branch_tokens;    /* capacity of all branch tokens of one eval, default 4096 */
    int max_branches;         /* default 256 */
    int max_candidates;       /* per branch, default 256 */
    int verbose;              /* log to stderr */
} cujev_options;

/* Description of the loaded network. Filled by cujev_model_info(). */
typedef struct cujev_model_info {
    char name[64];            /* e.g. "Qwen3.5-4B" */
    int hidden_size;
    int num_layers;
    int num_full_attention_layers;
    int vocab_size;
    int max_state_tokens;
    int max_branch_tokens;
    int max_branches;
    size_t device_bytes_weights;
    size_t device_bytes_workspace;
} cujev_model_info;

/* One branch of an eval call. Tokens are the branch's own tokens, appended
 * after the shared prefix. Candidates are the vocabulary ids whose logits
 * you want at the branch's last token. */
typedef struct cujev_branch {
    const int32_t *tokens;
    int num_tokens;
    const int32_t *candidates;
    int num_candidates;
} cujev_branch;

void cujev_options_default(cujev_options *opts);

/* Load a Hugging Face checkpoint directory (config.json + *.safetensors).
 * Only the text model is loaded; vision tower and MTP head are ignored. */
cujev_status cujev_model_open(const char *checkpoint_dir, const cujev_options *opts,
                              cujev_model **out);
void cujev_model_close(cujev_model *model);
cujev_status cujev_model_info_get(const cujev_model *model, cujev_model_info *out);

/* Allocate a state buffer on the device. A state is reusable: prefill it
 * once, call cujev_eval() as many times as you like. */
cujev_status cujev_state_create(cujev_model *model, cujev_state **out);
void cujev_state_destroy(cujev_state *state);

/* Run the shared prefix. Overwrites whatever the state held before. */
cujev_status cujev_state_prefill(cujev_model *model, cujev_state *state, const int32_t *tokens,
                                 int num_tokens);
int cujev_state_num_tokens(const cujev_state *state);

/* Evaluate branches against a prefilled state.
 *   logits_out : row-major [num_branches][max over branches of num_candidates];
 *                entry (b, i) is the raw logit of branches[b].candidates[i] at
 *                the last token of branch b. Unused entries are left untouched.
 *   ld         : row stride of logits_out in floats (>= max num_candidates).
 * Logits are the model's own next-token logits (pre-softmax, full-vocabulary
 * scale), so softmax over any subset is exactly the reference model's
 * renormalised distribution over that subset. */
cujev_status cujev_eval(cujev_model *model, cujev_state *state, const cujev_branch *branches,
                        int num_branches, float *logits_out, int ld);

/* Full-vocabulary logits at the last token of every branch, row-major
 * [num_branches][vocab_size]. Slow path (one GEMM against the whole embedding
 * matrix per call); meant for tests and for computing normalisers such as the
 * probability mass on the candidates. */
cujev_status cujev_eval_full_logits(cujev_model *model, cujev_state *state,
                                    const cujev_branch *branches, int num_branches,
                                    float *logits_out);

/* Timing of the last prefill / eval on this model, in milliseconds (GPU time). */
typedef struct cujev_timing {
    float prefill_ms;
    float eval_ms;
    int prefill_tokens;
    int eval_tokens;
    int eval_branches;
} cujev_timing;
void cujev_last_timing(const cujev_model *model, cujev_timing *out);

const char *cujev_status_string(cujev_status s);
/* Last error message for this thread (empty string if none). */
const char *cujev_last_error(void);
const char *cujev_version(void);

#ifdef __cplusplus
}
#endif
#endif /* CUJEV_H */
