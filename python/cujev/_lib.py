"""ctypes binding to libcujev. Thin: arrays in, logits out."""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class EngineError(RuntimeError):
    pass


class _Options(C.Structure):
    _fields_ = [
        ("device", C.c_int),
        ("max_state_tokens", C.c_int),
        ("max_branch_tokens", C.c_int),
        ("max_branches", C.c_int),
        ("max_candidates", C.c_int),
        ("verbose", C.c_int),
    ]


class _Info(C.Structure):
    _fields_ = [
        ("name", C.c_char * 64),
        ("hidden_size", C.c_int),
        ("num_layers", C.c_int),
        ("num_full_attention_layers", C.c_int),
        ("vocab_size", C.c_int),
        ("max_state_tokens", C.c_int),
        ("max_branch_tokens", C.c_int),
        ("max_branches", C.c_int),
        ("device_bytes_weights", C.c_size_t),
        ("device_bytes_workspace", C.c_size_t),
    ]


class _Branch(C.Structure):
    _fields_ = [
        ("tokens", C.POINTER(C.c_int32)),
        ("num_tokens", C.c_int),
        ("candidates", C.POINTER(C.c_int32)),
        ("num_candidates", C.c_int),
    ]


class _Timing(C.Structure):
    _fields_ = [
        ("prefill_ms", C.c_float),
        ("eval_ms", C.c_float),
        ("prefill_tokens", C.c_int),
        ("eval_tokens", C.c_int),
        ("eval_branches", C.c_int),
    ]


@dataclass
class Timing:
    prefill_ms: float
    eval_ms: float
    prefill_tokens: int
    eval_tokens: int
    eval_branches: int


@dataclass
class Branch:
    tokens: Sequence[int]
    candidates: Sequence[int]


def _find_library() -> str:
    env = os.environ.get("CUJEV_LIBRARY")
    if env:
        return env
    here = Path(__file__).resolve().parent
    candidates = [here / "libcujev.so", here.parent.parent / "build" / "libcujev.so"]
    # editable installs (`uv sync`): scikit-build-core puts the library in site-packages/cujev/
    try:
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            candidates.append(Path(sp) / "cujev" / "libcujev.so")
    except Exception:
        pass
    candidates.append(Path("/usr/local/lib/libcujev.so"))
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise EngineError(
        "libcujev.so not found; run `uv sync` (or cmake -B build && cmake --build build) "
        "or set CUJEV_LIBRARY=/path/to/libcujev.so"
    )


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = C.CDLL(_find_library())
    lib.cujev_options_default.argtypes = [C.POINTER(_Options)]
    lib.cujev_model_open.argtypes = [C.c_char_p, C.POINTER(_Options), C.POINTER(C.c_void_p)]
    lib.cujev_model_open.restype = C.c_int
    lib.cujev_model_close.argtypes = [C.c_void_p]
    lib.cujev_model_info_get.argtypes = [C.c_void_p, C.POINTER(_Info)]
    lib.cujev_state_create.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    lib.cujev_state_create.restype = C.c_int
    lib.cujev_state_destroy.argtypes = [C.c_void_p]
    lib.cujev_state_prefill.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_int32), C.c_int]
    lib.cujev_state_prefill.restype = C.c_int
    lib.cujev_state_num_tokens.argtypes = [C.c_void_p]
    lib.cujev_eval.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(_Branch), C.c_int,
                               C.POINTER(C.c_float), C.c_int]
    lib.cujev_eval.restype = C.c_int
    lib.cujev_eval_full_logits.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(_Branch), C.c_int,
                                           C.POINTER(C.c_float)]
    lib.cujev_eval_full_logits.restype = C.c_int
    lib.cujev_last_timing.argtypes = [C.c_void_p, C.POINTER(_Timing)]
    lib.cujev_status_string.restype = C.c_char_p
    lib.cujev_last_error.restype = C.c_char_p
    lib.cujev_version.restype = C.c_char_p
    _lib = lib
    return lib


def _check(lib, status: int, what: str):
    if status != 0:
        raise EngineError(
            f"{what}: {lib.cujev_status_string(status).decode()}: {lib.cujev_last_error().decode()}"
        )


class Engine:
    """A loaded model plus one reusable state buffer."""

    def __init__(self, checkpoint_dir: str | os.PathLike, *, device: int = 0,
                 max_state_tokens: int = 8192, max_branch_tokens: int = 4096,
                 max_branches: int = 256, max_candidates: int = 256, verbose: bool = False):
        self._lib = _load()
        opts = _Options(device, max_state_tokens, max_branch_tokens, max_branches,
                        max_candidates, int(verbose))
        self._model = C.c_void_p()
        _check(self._lib, self._lib.cujev_model_open(str(checkpoint_dir).encode(), C.byref(opts),
                                                     C.byref(self._model)), "model_open")
        self._state = C.c_void_p()
        _check(self._lib, self._lib.cujev_state_create(self._model, C.byref(self._state)),
               "state_create")
        info = _Info()
        self._lib.cujev_model_info_get(self._model, C.byref(info))
        self.name = info.name.decode()
        self.vocab_size = info.vocab_size
        self.hidden_size = info.hidden_size
        self.num_layers = info.num_layers
        self.max_state_tokens = info.max_state_tokens
        self.max_branch_tokens = info.max_branch_tokens
        self.max_branches = info.max_branches
        self.max_candidates = max_candidates
        self.weight_bytes = info.device_bytes_weights
        self.workspace_bytes = info.device_bytes_workspace
        self._state_tokens: list[int] | None = None

    def close(self):
        if getattr(self, "_state", None) and self._state.value:
            self._lib.cujev_state_destroy(self._state)
            self._state = C.c_void_p()
        if getattr(self, "_model", None) and self._model.value:
            self._lib.cujev_model_close(self._model)
            self._model = C.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def version(self) -> str:
        return self._lib.cujev_version().decode()

    def prefill(self, tokens: Sequence[int]) -> None:
        arr = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
        _check(self._lib, self._lib.cujev_state_prefill(
            self._model, self._state, arr.ctypes.data_as(C.POINTER(C.c_int32)), len(arr)),
            "prefill")
        self._state_tokens = list(int(t) for t in arr)

    @property
    def state_tokens(self) -> list[int] | None:
        return self._state_tokens

    def _branches(self, branches: Sequence[Branch]):
        keep = []
        arr = (_Branch * len(branches))()
        for i, b in enumerate(branches):
            t = np.ascontiguousarray(np.asarray(b.tokens, dtype=np.int32))
            c = np.ascontiguousarray(np.asarray(b.candidates, dtype=np.int32))
            keep.append((t, c))
            arr[i].tokens = t.ctypes.data_as(C.POINTER(C.c_int32))
            arr[i].num_tokens = len(t)
            arr[i].candidates = c.ctypes.data_as(C.POINTER(C.c_int32))
            arr[i].num_candidates = len(c)
        return arr, keep

    def eval(self, branches: Sequence[Branch]) -> list[np.ndarray]:
        """Candidate logits at the last token of every branch."""
        if self._state_tokens is None:
            raise EngineError("prefill() first")
        arr, keep = self._branches(branches)
        ld = max((len(b.candidates) for b in branches), default=1)
        ld = max(ld, 1)
        out = np.zeros((len(branches), ld), dtype=np.float32)
        _check(self._lib, self._lib.cujev_eval(
            self._model, self._state, arr, len(branches),
            out.ctypes.data_as(C.POINTER(C.c_float)), ld), "eval")
        return [out[i, :len(b.candidates)].copy() for i, b in enumerate(branches)]

    def eval_full_logits(self, branches: Sequence[Branch]) -> np.ndarray:
        """[B, vocab] logits at the last token of every branch (slow path)."""
        if self._state_tokens is None:
            raise EngineError("prefill() first")
        arr, keep = self._branches(branches)
        out = np.zeros((len(branches), self.vocab_size), dtype=np.float32)
        _check(self._lib, self._lib.cujev_eval_full_logits(
            self._model, self._state, arr, len(branches),
            out.ctypes.data_as(C.POINTER(C.c_float))), "eval_full_logits")
        return out

    def last_timing(self) -> Timing:
        t = _Timing()
        self._lib.cujev_last_timing(self._model, C.byref(t))
        return Timing(t.prefill_ms, t.eval_ms, t.prefill_tokens, t.eval_tokens, t.eval_branches)
