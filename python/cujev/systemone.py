"""Jev / TypeSafe System One semantics on top of the engine.

Request:  {state, model, questions: {id: {type, instructions, criteria}}}
Response: {model, answers: {id: Answer}, usage: {input_tokens, output_tokens}}
Exactly the shapes documented at https://docs.typesafe.ai/api.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from ._lib import Branch, Engine, Timing
from .prompt import PromptBuilder

JEV_ALIASES = ("jev-latest", "jev-preview", "jev-1.13", "jev-1.13.0", "typesafe/jev-1.13",
               "~typesafe/jev-latest", "cujev-latest")


class RequestError(ValueError):
    """A request that fails validation (HTTP 422)."""

    def __init__(self, loc: list[Any], msg: str):
        super().__init__(msg)
        self.loc = loc
        self.msg = msg


@dataclass
class ServedModel:
    name: str          # e.g. "cujev/qwen3.5-0.8b"
    version: str       # engine version
    description: str
    release_date: str


@dataclass
class Result:
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    model: str
    timing: Timing
    prefix_cached: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _confidence(p: np.ndarray) -> float:
    """1 - H(p)/ln K: 1 when certain, 0 when uniform. K=1 -> 1."""
    k = len(p)
    if k <= 1:
        return 1.0
    h = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


class SystemOne:
    def __init__(self, checkpoint_dir: str, *, max_state_tokens: int = 8192,
                 max_branch_tokens: int = 4096, max_branches: int = 256, device: int = 0,
                 verbose: bool = False):
        d = Path(checkpoint_dir)
        self.engine = Engine(d, device=device, max_state_tokens=max_state_tokens,
                             max_branch_tokens=max_branch_tokens, max_branches=max_branches,
                             max_candidates=256, verbose=verbose)
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.prompt = PromptBuilder(self.tokenizer)
        self.lock = threading.Lock()
        base = self.engine.name.lower().replace("qwen3.5", "qwen3.5")
        self.served = ServedModel(
            name=f"cujev/{base}",
            version=self.engine.version,
            description=f"cu-Jev {self.engine.version}: open System One decision engine on {self.engine.name} (CUDA)",
            release_date="2026-09-20",
        )
        self._cached_prefix: list[int] | None = None

    # -- validation -----------------------------------------------------------
    def resolve_model(self, name: Any) -> str:
        if not isinstance(name, str) or not name:
            raise RequestError(["body", "model"], "model must be a non-empty string")
        n = name.strip()
        if n in JEV_ALIASES or n == self.served.name or n.startswith("cujev"):
            return self.served.name
        raise RequestError(["body", "model"], f"unknown model {name!r}; use {self.served.name} or jev-latest")

    def _build(self, qid: str, q: Any) -> tuple[str, list[int], dict[str, Any]]:
        if not isinstance(q, dict):
            raise RequestError(["body", "questions", qid], "question must be an object")
        t = q.get("type")
        instr = q.get("instructions")
        if instr is None or (isinstance(instr, str) and not instr.strip()):
            raise RequestError(["body", "questions", qid, "instructions"], "instructions are required")
        crit = q.get("criteria")
        if t == "choice":
            if not isinstance(crit, dict) or not crit:
                raise RequestError(["body", "questions", qid, "criteria"], "choice needs a non-empty criteria object")
            if len(crit) > 255:
                raise RequestError(["body", "questions", qid, "criteria"], "at most 255 options")
            text, cands = self.prompt.choice_suffix(instr, crit)
            return text, cands, {"type": "choice", "options": list(crit.keys())}
        if t == "score":
            if not isinstance(crit, list) or len(crit) < 2:
                raise RequestError(["body", "questions", qid, "criteria"], "score needs a criteria array of at least 2 levels")
            if len(crit) > 10:
                raise RequestError(["body", "questions", qid, "criteria"], "at most 10 levels")
            text, cands = self.prompt.score_suffix(instr, crit)
            return text, cands, {"type": "score", "levels": [c for c in crit]}
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise RequestError(["body", "questions", qid, "criteria"], "noul criteria must be an object with true/false")
            text, cands = self.prompt.noul_suffix(instr, crit)
            return text, cands, {"type": "noul"}
        raise RequestError(["body", "questions", qid, "type"], "type must be one of choice, score, noul")

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, state: Any, model: Any, questions: Any) -> Result:
        served = self.resolve_model(model)
        if state is None:
            raise RequestError(["body", "state"], "state is required")
        if not isinstance(questions, dict) or not questions:
            raise RequestError(["body", "questions"], "questions must be a non-empty object")
        prefix_ids = self.prompt.encode(self.prompt.prefix_text(state))
        if len(prefix_ids) > self.engine.max_state_tokens:
            raise RequestError(["body", "state"],
                               f"state is {len(prefix_ids)} tokens; this server accepts at most {self.engine.max_state_tokens}")
        branches, metas, ids = [], [], []
        for qid, q in questions.items():
            text, cands, meta = self._build(str(qid), q)
            toks = self.prompt.encode(text)
            branches.append(Branch(toks, cands))
            metas.append(meta)
            ids.append(str(qid))
        longest = max(len(b.tokens) for b in branches)
        if longest > self.engine.max_branch_tokens:
            raise RequestError(["body", "questions"], f"a question is {longest} tokens; at most {self.engine.max_branch_tokens}")

        with self.lock:
            cached = self._cached_prefix == prefix_ids
            if not cached:
                self.engine.prefill(prefix_ids)
                self._cached_prefix = prefix_ids
            t_prefill = self.engine.last_timing().prefill_ms if not cached else 0.0
            logits: list[np.ndarray] = []
            eval_ms = 0.0
            # pack branches into eval calls that fit the branch-token budget
            i = 0
            while i < len(branches):
                j, ntok = i, 0
                while j < len(branches) and j - i < self.engine.max_branches and \
                        ntok + len(branches[j].tokens) <= self.engine.max_branch_tokens:
                    ntok += len(branches[j].tokens)
                    j += 1
                logits.extend(self.engine.eval(branches[i:j]))
                eval_ms += self.engine.last_timing().eval_ms
                i = j

        answers: dict[str, dict[str, Any]] = {}
        out_tokens = 0
        for qid, meta, lg in zip(ids, metas, logits):
            p = _softmax(lg)
            out_tokens += len(lg)
            if meta["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": _r(p[0])}
            elif meta["type"] == "choice":
                opts = meta["options"]
                probs = {o: _r(p[i]) for i, o in enumerate(opts)}
                answers[qid] = {"type": "choice", "choice": opts[int(p.argmax())],
                                "probabilities": probs, "confidence": _r(_confidence(p))}
            else:
                levels = meta["levels"]
                probs = {str(i): _r(p[i]) for i in range(len(levels))}
                legend = {str(i): (lv if isinstance(lv, str) else lv) for i, lv in enumerate(levels)}
                score = float(np.sum(p * np.arange(len(levels))))
                answers[qid] = {"type": "score", "score": _r(score), "legend": legend,
                                "probabilities": probs, "confidence": _r(_confidence(p))}
        in_tokens = len(prefix_ids) + sum(len(b.tokens) for b in branches)
        timing = Timing(t_prefill, eval_ms, len(prefix_ids), sum(len(b.tokens) for b in branches), len(branches))
        return Result(answers=answers, usage={"input_tokens": in_tokens, "output_tokens": out_tokens},
                      model=served, timing=timing, prefix_cached=cached)

    def models(self) -> list[dict[str, str]]:
        s = self.served
        rows = [{"name": s.name, "description": s.description, "release_date": s.release_date}]
        for alias in ("jev-latest", "cujev-latest"):
            rows.append({"name": alias, "description": f"alias of {s.name}", "release_date": s.release_date})
        return rows
