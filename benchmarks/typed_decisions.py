"""Replay LocalLLaMA/typed-decisions (test split) through cu-Jev and score it.

    .venv/bin/python benchmarks/typed_decisions.py --model /path/to/Qwen3.5-4B [--limit 50]

Every row is literally a POST /v1/systemone body (state + questions) with gold
answers. Reports, per workflow and overall:
  acc     top-1 accuracy of our argmax against the gold label
  brier   multi-class Brier score against the gold distribution, sum_k (p_k - g_k)^2
          averaged over questions (and the same divided by K, which some papers report)
  ece     expected calibration error of max-probability vs correctness, 10 bins
  latency wall-clock per case (5 questions) and GPU prefill/eval time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cujev import SystemOne  # noqa: E402


def dist(ans: dict, gold: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Our and gold distributions over the same option order, plus the option names."""
    t = ans["type"]
    if t == "noul":
        keys = ["false", "true"]
        ours = np.array([1 - ans["noul"], ans["noul"]])
    else:
        keys = list(gold["probabilities"].keys())
        ours = np.array([ans["probabilities"][k] for k in keys])
    g = np.array([gold["probabilities"][k] for k in keys])
    return ours, g, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "typed-decisions-test.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.groupby("workflow", group_keys=False).head(args.limit // 4 or 1)
    so = SystemOne(args.model)
    rows = []
    t_start = time.perf_counter()
    for i, r in enumerate(df.itertuples()):
        state = json.loads(r.state)
        questions = json.loads(r.questions)
        gold = json.loads(r.gold)
        t0 = time.perf_counter()
        res = so.evaluate(state, "jev-latest", questions)
        wall = (time.perf_counter() - t0) * 1000
        for qid, ans in res.answers.items():
            ours, g, keys = dist(ans, gold[qid])
            gold_label = str(gold[qid]["label"])
            pred = keys[int(ours.argmax())]
            rows.append(dict(
                workflow=r.workflow, case=r.id, q=qid, type=ans["type"], pred=pred, gold=gold_label,
                correct=int(pred == gold_label), conf=float(ours.max()),
                brier=float(((ours - g) ** 2).sum()), brier_k=float(((ours - g) ** 2).mean()),
                wall_ms=wall, prefill_ms=res.timing.prefill_ms, eval_ms=res.timing.eval_ms,
                in_tokens=res.usage["input_tokens"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} cases, {time.perf_counter() - t_start:.0f}s", file=sys.stderr)
    out = pd.DataFrame(rows)

    def ece(d: pd.DataFrame, bins: int = 10) -> float:
        e, n = 0.0, len(d)
        edges = np.linspace(0, 1, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d.conf > lo) & (d.conf <= hi)
            if m.any():
                e += m.sum() / n * abs(d.correct[m].mean() - d.conf[m].mean())
        return float(e)

    def report(name: str, d: pd.DataFrame):
        cases = d.drop_duplicates("case")
        print(f"{name:28s} n={len(d):4d}  acc={d.correct.mean() * 100:5.2f}%  brier={d.brier.mean():.4f} "
              f"(/K {d.brier_k.mean():.4f})  ece={ece(d):.4f}  "
              f"wall/case={cases.wall_ms.mean():6.1f} ms  gpu prefill={cases.prefill_ms.mean():5.1f} "
              f"eval={cases.eval_ms.mean():5.1f}  tok/case={cases.in_tokens.mean():.0f}")

    print(f"\ncu-Jev {so.served.name} on {Path(args.data).name}: {len(df)} cases, {len(out)} decisions")
    for wf, d in out.groupby("workflow"):
        report(wf, d)
    for t, d in out.groupby("type"):
        report(f"  type={t}", d)
    report("ALL", out)
    if args.out:
        out.to_json(args.out, orient="records", lines=True)


if __name__ == "__main__":
    main()
