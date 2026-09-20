"""Engine edge cases against Hugging Face on Qwen3.5-0.8B (or CUJEV_MODEL).

Covers the shapes the kernels special-case: query tiles of 16, key tiles of
64, segments shorter than the conv kernel, many branches packed into one
call, the candidate cap, capacity errors, and state reuse.
Run:  CUJEV_MODEL=/path/to/Qwen3.5-0.8B pytest -q tests/python/test_engine.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cujev import Branch, Engine, EngineError  # noqa: E402

MODEL = os.environ.get("CUJEV_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set CUJEV_MODEL")


@pytest.fixture(scope="module")
def hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


@pytest.fixture(scope="module")
def engine():
    e = Engine(MODEL, max_state_tokens=1024, max_branch_tokens=512, max_branches=64,
               max_candidates=256)
    yield e
    e.close()


def ref_probs(hf, ids, cands):
    import torch
    _, model = hf
    with torch.no_grad():
        lg = model(torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu().numpy()
    sub = lg[cands]
    p = np.exp(sub - sub.max())
    return p / p.sum()


def our_probs(engine, prefix, suffix, cands):
    engine.prefill(prefix)
    lg = engine.eval([Branch(suffix, cands)])[0]
    p = np.exp(lg - lg.max())
    return p / p.sum()


def rand_ids(rng, n):
    # printable-ish vocabulary range, avoids special tokens
    return [int(x) for x in rng.integers(1000, 60000, size=n)]


@pytest.mark.parametrize("plen", [1, 2, 3, 15, 16, 17, 63, 64, 65, 129, 200])
@pytest.mark.parametrize("slen", [1, 4, 16, 17, 33])
def test_lengths_match_hf(hf, engine, plen, slen):
    rng = np.random.default_rng(plen * 100 + slen)
    prefix, suffix = rand_ids(rng, plen), rand_ids(rng, slen)
    cands = [int(x) for x in rng.integers(1000, 60000, size=8)]
    a = our_probs(engine, prefix, suffix, cands)
    b = ref_probs(hf, prefix + suffix, cands)
    assert np.abs(a - b).sum() < 0.08, (a, b)


def test_many_branches_packed(hf, engine):
    rng = np.random.default_rng(7)
    prefix = rand_ids(rng, 50)
    branches = [Branch(rand_ids(rng, int(l)), [int(x) for x in rng.integers(1000, 60000, size=6)])
                for l in rng.integers(1, 12, size=40)]
    engine.prefill(prefix)
    outs = engine.eval(branches)
    for b, lg in zip(branches[::7], outs[::7]):
        p = np.exp(lg - lg.max()); p /= p.sum()
        assert np.abs(p - ref_probs(hf, prefix + list(b.tokens), list(b.candidates))).sum() < 0.08


def test_max_candidates_and_full_logits(hf, engine):
    rng = np.random.default_rng(3)
    prefix, suffix = rand_ids(rng, 20), rand_ids(rng, 5)
    cands = sorted(set(int(x) for x in rng.integers(0, engine.vocab_size, size=300)))[:256]
    engine.prefill(prefix)
    sub = engine.eval([Branch(suffix, cands)])[0]
    full = engine.eval_full_logits([Branch(suffix, [])])[0]
    np.testing.assert_allclose(sub, full[cands], rtol=1e-3, atol=2e-2)
    with pytest.raises(EngineError):
        engine.eval([Branch(suffix, list(range(257)))])


def test_state_is_reusable_and_deterministic(engine):
    rng = np.random.default_rng(11)
    prefix, suffix = rand_ids(rng, 40), rand_ids(rng, 9)
    cands = [5, 6, 7]
    engine.prefill(prefix)
    a = engine.eval([Branch(suffix, cands)])[0]
    b = engine.eval([Branch(suffix, cands)])[0]          # same state, second eval
    engine.prefill(rand_ids(rng, 30))                     # overwrite
    engine.prefill(prefix)                                # and restore
    c = engine.eval([Branch(suffix, cands)])[0]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, c, atol=1e-5)


def test_capacity_errors(engine):
    with pytest.raises(EngineError):
        engine.prefill([1] * 1025)
    engine.prefill([1, 2, 3])
    with pytest.raises(EngineError):
        engine.eval([Branch([4] * 513, [1])])
    with pytest.raises(EngineError):
        engine.eval([Branch([4], [1])] * 65)
    with pytest.raises(EngineError):
        engine.eval([Branch([engine.vocab_size], [1])])
    with pytest.raises(EngineError):
        engine.eval([Branch([4], [engine.vocab_size])])
