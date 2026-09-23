"""GPU-free tests of the prompt layer: labels, request validation, answer math."""
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cujev.prompt import PromptBuilder  # noqa: E402
from cujev.systemone import _confidence, _softmax  # noqa: E402

TOK = None
for p in (Path.home() / "OS/dthinky/models/Qwen3.5-0.8B/tokenizer.json",
          Path.home() / "OS/dthinky/models/Qwen3.5-4B/tokenizer.json"):
    if p.exists():
        TOK = p
        break
pytestmark = pytest.mark.skipif(TOK is None, reason="no Qwen3.5 tokenizer.json found")


@pytest.fixture(scope="module")
def pb():
    return PromptBuilder(Tokenizer.from_file(str(TOK)))


def test_labels_are_single_tokens(pb):
    assert len(pb.choice_labels.ids) == 255 and len(set(pb.choice_labels.ids)) == 255
    assert pb.noul_labels.texts == ("yes", "no")
    assert pb.score_labels.texts == tuple(str(i) for i in range(10))
    for ls in (pb.choice_labels, pb.noul_labels, pb.score_labels):
        for t, i in zip(ls.texts, ls.ids):
            assert pb.tok.encode(t, add_special_tokens=False).ids == [i]


def test_suffix_ends_at_answer_position(pb):
    text, cands = pb.choice_suffix("Which team?", {"billing": "money", "tech": None})
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert cands == list(pb.choice_labels.ids[:2])
    assert "A: billing — money" in text and "B: tech" in text
    text, cands = pb.score_suffix("How bad?", ["fine", "bad", "worse"])
    assert cands == list(pb.score_labels.ids[:3]) and "2: worse" in text
    text, cands = pb.noul_suffix("Is it urgent?", {"true": "yes means urgent", "false": "calm"})
    assert cands == list(pb.noul_labels.ids) and "yes means: yes means urgent" in text


def test_structured_state_and_instructions_render_as_json(pb):
    pre = pb.prefix_text({"ticket": "x", "tier": "enterprise"})
    assert '"ticket": "x"' in pre and pre.endswith("\n\n")
    text, _ = pb.choice_suffix({"question": "same person as `dup`?", "dup": {"name": "J"}}, {"a": None})
    assert '"dup": {"name": "J"}' in text


def test_prefix_and_suffix_tokenise_independently(pb):
    """Prefix tokens must not change when a suffix is appended (boundary at a newline)."""
    pre = pb.prefix_text("Hello world.")
    suf, _ = pb.noul_suffix("Is it a greeting?", None)
    a = pb.encode(pre)
    b = pb.encode(pre + suf)
    assert b[:len(a)] == a


def test_confidence_and_softmax():
    assert _confidence(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert _confidence(np.array([0.25] * 4)) == pytest.approx(0.0)
    assert _confidence(np.array([1.0])) == 1.0
    p = _softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 2
