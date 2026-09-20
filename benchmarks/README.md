# Benchmarks

* `typed_decisions.py` — replays the `LocalLLaMA/typed-decisions` test split
  (download `all/test-00000-of-00001.parquet` into `data/`) and reports accuracy,
  Brier, ECE and latency per workflow / question type.
* `../build/cujev-bench` — raw engine timing on synthetic tokens:
  `cujev-bench <checkpoint> <state_tokens> <branches> <tokens_per_branch> <candidates> [iters]`.

Results (RTX 5090 and RTX 3090, CUDA 13.3, 2026-09-20) are in the top-level
README. Raw per-decision rows: `data/results-<model>.jsonl` (gitignored).

## Other GPUs

Ada, Hopper and A100: `make bench ARCH=89` (or `80` / `90`) and
`benchmarks/benefit.py --no-hf`, with GPU name, driver and nvcc version.
The 5090 build is `ARCH=120`, the 3090 build is `ARCH=86`.
