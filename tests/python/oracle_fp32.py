"""Three-way oracle: HF fp32 (ground truth) vs HF bf16 vs cu-Jev, sequentially so
each model has the GPU to itself. Prints L1 distances between next-token
distributions at the read-out position.

    uv run tests/python/oracle_fp32.py models/Qwen3.5-4B
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_oracle import PREFIX, SUFFIXES  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from cujev import Branch, Engine  # noqa: E402


def probs(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main(model_dir: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_dir)
    prefix = tok.encode(PREFIX, add_special_tokens=False)
    sufs = [tok.encode(s, add_special_tokens=False) for s in SUFFIXES]
    refs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt).to("cuda").eval()
        out = []
        for s in sufs:
            with torch.no_grad():
                out.append(m(torch.tensor([prefix + s], device="cuda"), logits_to_keep=1)
                           .logits[0, -1].float().cpu().numpy())
        refs[dt] = out
        del m
        torch.cuda.empty_cache()
    e = Engine(model_dir, max_state_tokens=2048, max_branch_tokens=1024)
    e.prefill(prefix)
    ours = e.eval_full_logits([Branch(s, []) for s in sufs])
    worst_ours, worst_bf16 = 0.0, 0.0
    for i in range(len(sufs)):
        p32, p16, po = probs(refs[torch.float32][i]), probs(refs[torch.bfloat16][i]), probs(ours[i])
        d16, do = np.abs(p16 - p32).sum(), np.abs(po - p32).sum()
        worst_ours, worst_bf16 = max(worst_ours, do), max(worst_bf16, d16)
        print(f"{e.name} branch {i}: L1(HF-bf16, fp32)={d16:.4f}  L1(cu-Jev, fp32)={do:.4f}  "
              f"argmax fp32/bf16/ours={p32.argmax()}/{p16.argmax()}/{po.argmax()}")
    print(f"worst L1 to fp32: cu-Jev {worst_ours:.4f}, HF-bf16 {worst_bf16:.4f}")
    return 0 if worst_ours < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
