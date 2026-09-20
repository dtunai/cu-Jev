"""Command line: serve, ask, bench."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="cujev", description="cu-Jev: CUDA-native System One decisions")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", required=True, help="Qwen3.5 checkpoint directory")
        sp.add_argument("--device", type=int, default=0)
        sp.add_argument("--max-state-tokens", type=int, default=8192)
        sp.add_argument("--max-branch-tokens", type=int, default=4096)
        sp.add_argument("--max-branches", type=int, default=256)
        sp.add_argument("--verbose", action="store_true")

    s = sub.add_parser("serve", help="run the /v1/systemone HTTP server")
    common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)

    a = sub.add_parser("ask", help="evaluate one request from a JSON file or stdin")
    common(a)
    a.add_argument("request", nargs="?", help="path to a request JSON (default: stdin)")

    args = p.parse_args(argv)
    kw = dict(device=args.device, max_state_tokens=args.max_state_tokens,
              max_branch_tokens=args.max_branch_tokens, max_branches=args.max_branches,
              verbose=args.verbose)
    if args.cmd == "serve":
        from .server import serve
        serve(args.model, host=args.host, port=args.port, **kw)
    elif args.cmd == "ask":
        from .systemone import SystemOne
        req = json.load(open(args.request)) if args.request else json.load(sys.stdin)
        so = SystemOne(args.model, **kw)
        res = so.evaluate(req.get("state"), req.get("model", "jev-latest"), req.get("questions"))
        json.dump({"model": res.model, "answers": res.answers, "usage": res.usage}, sys.stdout,
                  indent=2, ensure_ascii=False)
        print(f"\n# prefill {res.timing.prefill_ms:.1f} ms ({res.timing.prefill_tokens} tok), "
              f"eval {res.timing.eval_ms:.1f} ms ({res.timing.eval_branches} questions, "
              f"{res.timing.eval_tokens} tok)", file=sys.stderr)


if __name__ == "__main__":
    main()
