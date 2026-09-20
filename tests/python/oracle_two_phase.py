"""HF-vs-cu-Jev oracle in two phases, for checkpoints too large to hold twice on one GPU.

    python tests/python/oracle_two_phase.py hf   /path/to/model  ref.npz   # phase 1 (HF only)
    python tests/python/oracle_two_phase.py cujev /path/to/model ref.npz   # phase 2 (engine only)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from test_oracle import PREFIX, SUFFIXES  # noqa: E402

phase, model_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
prefix = tok.encode(PREFIX, add_special_tokens=False).ids
suffixes = [tok.encode(s, add_special_tokens=False).ids for s in SUFFIXES]

if phase == "hf":
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    refs = []
    for s in suffixes:
        with torch.no_grad():
            refs.append(model(torch.tensor([prefix + s], device="cuda"), logits_to_keep=1)
                        .logits[0, -1].float().cpu().numpy())
    np.savez(out, *refs)
    print("saved", out)
else:
    from cujev import Branch, Engine
    e = Engine(model_dir, max_state_tokens=2048, max_branch_tokens=1024)
    e.prefill(prefix)
    ours = e.eval_full_logits([Branch(s, []) for s in suffixes])
    refs = np.load(out)
    ok = True
    for i, key in enumerate(refs.files):
        a, b = ours[i], refs[key]
        pa = np.exp(a - a.max()); pa /= pa.sum()
        pb = np.exp(b - b.max()); pb /= pb.sum()
        same = a.argmax() == b.argmax() and set(np.argsort(-a)[:5]) == set(np.argsort(-b)[:5])
        ok &= bool(same) and np.abs(pa - pb).sum() < 0.05
        print(f"{e.name} branch {i}: argmax {a.argmax()}/{b.argmax()} top5 {np.argsort(-a)[:5].tolist()} "
              f"vs {np.argsort(-b)[:5].tolist()} L1(prob)={np.abs(pa - pb).sum():.4f} {'OK' if same else 'MISMATCH'}")
    sys.exit(0 if ok else 1)
