#!/usr/bin/env python3
"""Download a supported Qwen3.5 checkpoint (text weights + tokenizer only).

    uv run scripts/download_model.py Qwen3.5-4B            # -> models/Qwen3.5-4B
    uv run scripts/download_model.py Qwen/Qwen3.5-0.8B --dest /data/models

Only config.json, tokenizer files and the safetensors shards are fetched. The
vision tower and MTP head live inside the same shards, so they come along; the
engine ignores them at load time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED = {
    "Qwen3.5-0.8B": "tested",
    "Qwen3.5-2B": "tested",
    "Qwen3.5-4B": "tested",
    "Qwen3.5-9B": "tested (needs ~20 GB VRAM)",
}
PATTERNS = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "model.safetensors.index.json", "*.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="e.g. Qwen3.5-4B or Qwen/Qwen3.5-4B")
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"),
                    help="directory that will contain <model name>/ (default: ./models)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--list", action="store_true", help="list supported models and exit")
    args = ap.parse_args()
    if args.list:
        for k, v in SUPPORTED.items():
            print(f"{k:16s} {v}")
        return 0
    repo = args.model if "/" in args.model else f"Qwen/{args.model}"
    name = repo.split("/")[-1]
    if name not in SUPPORTED:
        print(f"warning: {name} is not in the tested list ({', '.join(SUPPORTED)}); "
              "the engine checks the architecture at load time", file=sys.stderr)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing: run `uv sync` first", file=sys.stderr)
        return 2
    dest = Path(args.dest) / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo} -> {dest}")
    snapshot_download(repo_id=repo, revision=args.revision, local_dir=str(dest),
                      allow_patterns=PATTERNS)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"done: {size / 2**30:.2f} GiB in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
