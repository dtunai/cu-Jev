"""Does the shared prefix actually buy anything? Three ways to answer the same
questions, same model, same tokens:

  cujev-shared   prefill the state once, eval all questions as branches   (cu-Jev)
  cujev-naive    prefill state+question as one fresh sequence per question (cu-Jev, no sharing)
  hf-batched     Hugging Face transformers, all full prompts padded into one batch,
                 read the same candidate logits at the last position

    .venv/bin/python benchmarks/benefit.py --model /path/to/Qwen3.5-4B --state-tokens 1024 --questions 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cujev import Branch, Engine  # noqa: E402


def timeit(fn, iters):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--state-tokens", type=int, default=1024)
    ap.add_argument("--questions", type=int, default=16)
    ap.add_argument("--question-tokens", type=int, default=40)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-hf", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    P, B, L = a.state_tokens, a.questions, a.question_tokens
    state = [int(x) for x in rng.integers(1000, 60000, size=P)]
    qs = [[int(x) for x in rng.integers(1000, 60000, size=L)] for _ in range(B)]
    cands = [32 + i for i in range(8)]

    eng = Engine(a.model, max_state_tokens=P + L + 16, max_branch_tokens=max(B * L, P + L + 16),
                 max_branches=max(B, 1))
    print(f"{eng.name}: state {P} tokens, {B} questions x {L} tokens, 8 candidates")

    def shared():
        eng.prefill(state)
        return eng.eval([Branch(q, cands) for q in qs])

    def naive():
        out = []
        for q in qs:
            eng.prefill(state + q[:-1])
            out.append(eng.eval([Branch(q[-1:], cands)])[0])
        return out

    ref = shared()
    t_shared = timeit(shared, a.iters)
    eng.prefill(state)
    t_eval_only = timeit(lambda: eng.eval([Branch(q, cands) for q in qs]), a.iters)
    t_naive = timeit(naive, max(1, a.iters // 2))
    nv = naive()
    agree = np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(ref, nv)])
    print(f"cujev-shared   {t_shared:8.1f} ms/request  ({B / t_shared * 1000:7.0f} decisions/s)   "
          f"eval-only after a cached state: {t_eval_only:.1f} ms ({B / t_eval_only * 1000:.0f}/s)")
    print(f"cujev-naive    {t_naive:8.1f} ms/request  ({B / t_naive * 1000:7.0f} decisions/s)   "
          f"argmax agreement with shared: {agree * 100:.0f}%")
    print(f"  shared prefix speed-up: {t_naive / t_shared:.1f}x per request, "
          f"{t_naive / t_eval_only:.1f}x when the state is already cached")
    if a.no_hf:
        return
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    ids = torch.tensor([state + q for q in qs], device="cuda")
    cand_t = torch.tensor(cands, device="cuda")

    def hf():
        with torch.no_grad():
            lg = model(ids, logits_to_keep=1).logits[:, -1, :]
            return lg[:, cand_t].float().cpu().numpy()
    t_hf = timeit(hf, max(1, a.iters // 2))
    hv = hf()
    agree_hf = np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(ref, hv)])
    print(f"hf-batched     {t_hf:8.1f} ms/request  ({B / t_hf * 1000:7.0f} decisions/s)   "
          f"argmax agreement with cujev: {agree_hf * 100:.0f}%   -> cu-Jev is {t_hf / t_shared:.1f}x faster "
          f"({t_hf / t_eval_only:.1f}x with cached state)")


if __name__ == "__main__":
    main()
