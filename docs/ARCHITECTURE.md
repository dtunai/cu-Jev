# cu-Jev architecture

## The workload

A System One request is *one* long shared context (the state) and *many* short,
independent questions about it. Per request the model has to:

1. ingest the state once,
2. run every question against that state without letting questions see each
   other,
3. read, at the last token of each question, the logits of a handful of
   vocabulary ids (the option labels).

There is no decoding, no sampling and no KV growth. This is a batched-prefill
problem with a shared prefix and a tiny gather at the end — the opposite of what
autoregressive serving engines optimise for.

## Qwen3.5 in one paragraph

Qwen3.5 is a hybrid: every 4th layer is full (GQA) attention, the other three are
Gated DeltaNet linear attention. Full attention has an output gate
(`q_proj` emits `[q | gate]` per head), zero-centred RMSNorm on q and k, and
partial RoPE (first 64 of 256 dims, θ = 1e7). A linear layer projects to
`qkv | z | b | a`, runs a depth-wise causal conv (kernel 4) + SiLU over `qkv`,
l2-normalises q and k, forms `β = σ(b)` and `g = −exp(A_log)·softplus(a + dt_bias)`,
runs the gated delta rule on a per-head 128×128 fp32 state, applies a gated
RMSNorm with `silu(z)`, and projects out. All norms are zero-centred
(`(1 + w)`), embeddings are tied to the LM head, vocab is 248 320.

## Execution model

```
cujev_state_prefill(tokens)        one segment, N = |state|
   embed -> 32 x [norm, mixer, +res, norm, MLP, +res] -> per-layer side effects:
      full layer   : K,V (post-norm, post-RoPE) -> state.kv[l][Hkv][P][256]   (bf16)
      linear layer : final recurrent S -> state.S[l][Hv][128][128]           (fp32)
                     last 3 pre-conv rows  -> state.conv[l][C][3]

cujev_eval(branches)               B segments packed into one token stream
   same layer loop over all branch tokens at once; per layer:
      full layer   : K,V of branch tokens -> scratch [Hkv][Nb][256];
                     attention over (state.kv[l] ++ own segment, causal)
      linear layer : conv starts from state.conv[l]; the scan of every
                     segment starts from state.S[l] (read-only, never copied)
   final norm at the last token of every segment -> hsel[B][H]
   readout: logits[b][i] = hsel[b] . E[cand[b][i]]
```

Metadata per token (`pos`, `own_base`, `own_local`) and per segment
(`seg_start`, `seg_len`) plus a table of 16-token query tiles is built on the
host per call and uploaded once. No kernel ever branches on "prefill vs eval":
prefill is simply an eval with `P = 0` whose side outputs go to the state.

## Kernels

| kernel | grid | notes |
|---|---|---|
| `attn_prep` | token × (Hq+Hkv) warps | per-head RMSNorm, RoPE from a constant inv-freq table, K/V to cache |
| `attention_tc` | 16-token tile × kv-head, 128 threads | FA2 on `mma.sync.m16n8k16` bf16→fp32. Q tile (64 rows = 16 tokens × 4 heads) and 64-key K/V tiles in padded smem (row 264 bf16 → conflict-free fragment loads). V B-fragments via `ldmatrix.x4.trans`. Online softmax in base 2, P kept in registers as A-fragments. Keys iterate over `[0,P) ++ own[0..local]`, own part causally masked per token. Epilogue multiplies by `sigmoid(gate)`. |
| `gdn_conv` | token × channel | depth-wise causal conv, taps before the segment start come from the state's conv tail |
| `gdn_prep` | warp per (token, v-head) | l2-norm q,k (fp32, eps 1e-6), q·128^-½, k-head expansion, g, β |
| `gdn_scan` | (segment, v-head), 256 threads | column `S[:, v]` split over 2 lanes (64 fp32 registers each, no spills); per token: decay, `kv = Sᵀk` (partner xor-shuffle), `Δ = β(v − kv)`, `S += kΔᵀ`, `o = Sᵀq`; q/k staged in ping-pong smem, one barrier per token. Exactly the reference recurrent rule. |
| `gdn_norm_gate` | warp per (token, head) | RMSNorm(128)·w·silu(z) |
| `rmsnorm`, `add_bf16`, `silu_mul`, `embed`, `readout` | — | elementwise; `rmsnorm` takes an optional row list for the final norm |
| cuBLAS `GemmEx` | — | bf16 in, fp32 accumulate, bf16 out; weights row-fused at load (`q|k|v`, `qkv|z|b|a`, `gate|up`) so each layer is 4 GEMMs |

## Numerics

* Residual stream fp32; GEMM inputs/outputs bf16 (as in the bf16 reference);
  norms, softmax, RoPE, GDN state in fp32. The LM head is the embedding matrix
  when `tie_word_embeddings` is set (0.8B/2B/4B) and `lm_head.weight` otherwise (9B).
* Against `transformers` at the read-out position: identical argmax and top-5
  on every probe. Against an fp32 run of the reference, cu-Jev is closer than
  the reference's own bf16 run (4B: L1 ≤ 0.009 vs ≤ 0.060; 0.8B: ≤ 0.026 vs
  ≤ 0.025) — the fp32 residual stream buys accuracy at no cost.
* cuBLAS chooses different reduction orders for different batch shapes, so the
  same branch evaluated alone vs. with siblings differs by ≤ 0.03 logits; the
  isolation test bounds this and checks that an adversarial sibling does not
  move a branch beyond it.

## Verification

* `tests/python/test_oracle.py` — three real prompts, full-vocabulary logits at
  the read-out position vs. `transformers`; plus branch isolation with an
  adversarial sibling.
* `tests/python/test_engine.py` — 55 (prefix, suffix) length pairs straddling
  every tile boundary (1…200 × 1…33), 40 packed branches, 256 candidates vs.
  full logits, state reuse determinism, capacity errors. Run with
  `CUJEV_DEBUG_SYNC=1` to synchronise and error-check after every kernel
  (compute-sanitizer is not available under WSL2, so this is the substitute).
* `tests/python/oracle_two_phase.py` — the same oracle for checkpoints that do
  not fit on the GPU twice (9B).
* `benchmarks/benefit.py` — shared-prefix eval vs. naive re-prefill vs. HF
  transformers on the same tokens, with argmax agreement.
* `benchmarks/typed_decisions.py` — accuracy / Brier / ECE on the independent
  typed-decisions benchmark.
* Every kernel compiles with 0 bytes of spills (`nvcc -Xptxas -v`).

## Why not just vLLM/SGLang with `logprob_token_ids`?

That is what jevfire, openjev-sglang and simple-jev do, and it works. The
engine-level differences are: (1) the prefix is one KV copy plus one recurrent
state that every branch reads, instead of prefix-cache blocks that must be
aligned to hit; (2) the read-out is a gather-dot over ≤255 rows of the
embedding, not a 248k-wide LM head per position; (3) no scheduler, no
tokenizer, no Python between the request and the GEMMs — the whole eval is one
synchronous C call on one stream. The price is generality: it runs Qwen3.5 and
nothing else.

## Roadmap

1. Chunked Gated-DeltaNet (WY form, chunk 64) on tensor cores for prefill — the
   recurrent scan is ~18 % of prefill time at 600–8 000 tokens.
2. Fuse `add + rmsnorm` and `silu_mul` into GEMM epilogues (cublasLt).
3. CUDA graph capture of the eval path for fixed (B, L) buckets.
4. Teacher-forced multi-token option scoring for callers that want option
   *names* read verbatim instead of labels.
5. Calibration (temperature / isotonic on typed-decisions) reported per
   question type; `confidence` currently is `1 − H(p)/ln K` of the raw
   distribution.
6. INT8 weight path (Ampere has 2× int8 tensor throughput).
