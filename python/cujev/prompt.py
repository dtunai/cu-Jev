"""Prompt layout: how a Jev request becomes a shared prefix + one suffix per question.

The prefix carries the system instructions and the STATE; every question is a
short suffix that ends where the assistant's answer starts. The model reads
one token there. Option labels are verified single tokens so that reading one
position is exact regardless of how long the option names are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

SYSTEM = (
    "You are a System One decision model. You are given a STATE and one QUESTION about it. "
    "Read the state literally; do not assume facts that are not in it. "
    "Answer with exactly one option label and nothing else."
)

THINK_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _text(x: Any) -> str:
    """Render a string / object / array the way the docs describe: JSON for structure."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


def _inline(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


@dataclass(frozen=True)
class LabelSet:
    """Verified single-token labels for one question type."""
    texts: tuple[str, ...]
    ids: tuple[int, ...]


class PromptBuilder:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.choice_labels = self._verify(self._letter_labels(), 255, "choice")
        self.noul_labels = self._verify(["yes", "no"], 2, "noul")
        self.score_labels = self._verify([str(i) for i in range(10)], 10, "score")

    # -- labels ---------------------------------------------------------------
    @staticmethod
    def _letter_labels() -> list[str]:
        import itertools
        import string
        one = list(string.ascii_uppercase)
        two = ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
        return one + two + list(string.ascii_lowercase)

    def _verify(self, candidates: list[str], need: int, what: str) -> LabelSet:
        texts, ids, seen = [], [], set()
        for t in candidates:
            enc = self.tok.encode(t, add_special_tokens=False).ids
            if len(enc) == 1 and enc[0] not in seen and self.tok.decode(enc) == t:
                texts.append(t)
                ids.append(enc[0])
                seen.add(enc[0])
            if len(texts) == need:
                break
        if len(texts) < need:
            raise ValueError(f"tokenizer yields only {len(texts)} single-token {what} labels, need {need}")
        return LabelSet(tuple(texts), tuple(ids))

    # -- text -----------------------------------------------------------------
    def prefix_text(self, state: Any) -> str:
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nSTATE:\n{_text(state)}\n\n"

    def choice_suffix(self, instructions: Any, criteria: dict[str, Any]) -> tuple[str, list[int]]:
        n = len(criteria)
        labels = self.choice_labels.texts[:n]
        lines = [f"QUESTION: {_inline(instructions)}", "Options:"]
        for lab, (name, desc) in zip(labels, criteria.items()):
            d = _inline(desc)
            lines.append(f"{lab}: {name}" + (f" — {d}" if d else ""))
        lines.append(f"Answer with the option label ({labels[0]}–{labels[-1]}).<|im_end|>\n" + THINK_OFF)
        return "\n".join(lines), list(self.choice_labels.ids[:n])

    def noul_suffix(self, instructions: Any, criteria: dict[str, Any] | None) -> tuple[str, list[int]]:
        lines = [f"QUESTION: {_inline(instructions)}"]
        if criteria:
            t, f = criteria.get("true"), criteria.get("false")
            if t is not None:
                lines.append(f"yes means: {_inline(t)}")
            if f is not None:
                lines.append(f"no means: {_inline(f)}")
        lines.append("Answer yes or no.<|im_end|>\n" + THINK_OFF)
        return "\n".join(lines), list(self.noul_labels.ids)

    def score_suffix(self, instructions: Any, criteria: list[Any]) -> tuple[str, list[int]]:
        n = len(criteria)
        lines = [f"QUESTION: {_inline(instructions)}", "Levels (in increasing order):"]
        for i, desc in enumerate(criteria):
            lines.append(f"{i}: {_inline(desc)}")
        lines.append(f"Answer with the level number (0–{n - 1}).<|im_end|>\n" + THINK_OFF)
        return "\n".join(lines), list(self.score_labels.ids[:n])

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids
