/* Raw engine benchmark: synthetic tokens, no tokenizer.
 *   cujev-bench <checkpoint_dir> [state_tokens] [branches] [tokens_per_branch] [candidates] */
#include "cujev/cujev.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <checkpoint_dir> [P=1024] [B=16] [L=24] [C=8] [iters=5]\n",
                argv[0]);
        return 2;
    }
    int P = argc > 2 ? atoi(argv[2]) : 1024;
    int B = argc > 3 ? atoi(argv[3]) : 16;
    int L = argc > 4 ? atoi(argv[4]) : 24;
    int C = argc > 5 ? atoi(argv[5]) : 8;
    int iters = argc > 6 ? atoi(argv[6]) : 5;
    cujev_options o;
    cujev_options_default(&o);
    o.verbose = 1;
    if (P > o.max_state_tokens)
        o.max_state_tokens = P;
    if (B * L > o.max_branch_tokens)
        o.max_branch_tokens = B * L;
    cujev_model *m;
    cujev_status s = cujev_model_open(argv[1], &o, &m);
    if (s != CUJEV_OK) {
        fprintf(stderr, "open: %s: %s\n", cujev_status_string(s), cujev_last_error());
        return 1;
    }
    cujev_model_info info;
    cujev_model_info_get(m, &info);
    printf("model %s: %d layers (%d full attention), H=%d, V=%d, weights %.2f GiB, workspace %.2f GiB\n",
           info.name, info.num_layers, info.num_full_attention_layers, info.hidden_size,
           info.vocab_size, info.device_bytes_weights / 1073741824.0,
           info.device_bytes_workspace / 1073741824.0);
    cujev_state *st;
    cujev_state_create(m, &st);
    int32_t *tok = malloc(sizeof(int32_t) * (P + B * L));
    for (int i = 0; i < P + B * L; i++)
        tok[i] = 1000 + (i * 7919) % 50000;
    cujev_branch *br = calloc(B, sizeof *br);
    int32_t cand[256];
    for (int i = 0; i < C; i++)
        cand[i] = 32 + i;
    for (int b = 0; b < B; b++) {
        br[b].tokens = tok + P + b * L;
        br[b].num_tokens = L;
        br[b].candidates = cand;
        br[b].num_candidates = C;
    }
    float *logits = malloc(sizeof(float) * B * C);
    cujev_timing t;
    /* warm-up */
    if ((s = cujev_state_prefill(m, st, tok, P)) != CUJEV_OK ||
        (s = cujev_eval(m, st, br, B, logits, C)) != CUJEV_OK) {
        fprintf(stderr, "run: %s: %s\n", cujev_status_string(s), cujev_last_error());
        return 1;
    }
    float pre = 0, ev = 0;
    for (int i = 0; i < iters; i++) {
        cujev_state_prefill(m, st, tok, P);
        cujev_last_timing(m, &t);
        pre += t.prefill_ms;
        cujev_eval(m, st, br, B, logits, C);
        cujev_last_timing(m, &t);
        ev += t.eval_ms;
    }
    pre /= iters;
    ev /= iters;
    printf("prefill: %d tokens  %.2f ms  (%.0f tok/s)\n", P, pre, P / pre * 1000);
    printf("eval:    %d branches x %d tokens, %d candidates  %.2f ms  (%.0f decisions/s, %.0f tok/s)\n",
           B, L, C, ev, B / ev * 1000, B * L / ev * 1000);
    printf("logits[0]:");
    for (int i = 0; i < C && i < 8; i++)
        printf(" %.3f", logits[i]);
    printf("\n");
    free(logits);
    free(br);
    free(tok);
    cujev_state_destroy(st);
    cujev_model_close(m);
    return 0;
}
