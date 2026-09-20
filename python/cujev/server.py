"""HTTP server with TypeSafe's wire format: POST /v1/systemone, GET /v1/models.

    python -m cujev serve --model /path/to/Qwen3.5-4B --port 8080
    TYPESAFE_BASE_URL=http://127.0.0.1:8080 TYPESAFE_API_KEY=any python your_app.py
"""
from __future__ import annotations

import os
import time
from typing import Any

from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .systemone import RequestError, SystemOne


def _validation_error(loc: list[Any], msg: str, typ: str = "value_error") -> JSONResponse:
    # Same shape as FastAPI/pydantic (and Jev): a list of {loc, msg, type}
    return JSONResponse(status_code=422, content={"detail": [{"loc": loc, "msg": msg, "type": typ}]})


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def create_app(engine: SystemOne, api_key: str | None = None, examples: bool = True) -> FastAPI:
    app = FastAPI(title="cu-Jev", version=engine.served.version,
                  description="CUDA-native System One decision engine, Jev-compatible API")
    # browser examples (and any page on another origin) may call the API directly
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    if examples and EXAMPLES_DIR.is_dir():
        app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")

        @app.get("/play", include_in_schema=False)
        def play(request: Request):
            q = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(f"/examples/starfighter/index.html{q}")

    def _auth(authorization: str | None):
        if api_key is None:
            return None
        if not authorization or not authorization.startswith("Bearer ") or \
                authorization[7:].strip() != api_key:
            return JSONResponse(status_code=401, content={"detail": {
                "error_type": "authentication_error",
                "message": "Missing or invalid API key. Send Authorization: Bearer <key>."}})
        return None

    @app.get("/health")
    def health():
        e = engine.engine
        return {"status": "ok", "model": engine.served.name, "engine": engine.engine.name,
                "version": engine.served.version, "max_state_tokens": e.max_state_tokens,
                "max_branch_tokens": e.max_branch_tokens, "max_branches": e.max_branches}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        if (r := _auth(authorization)) is not None:
            return r
        return {"models": engine.models()}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        if (r := _auth(authorization)) is not None:
            return r
        try:
            body = await request.json()
        except Exception:
            return _validation_error(["body"], "request body must be JSON", "json_invalid")
        if not isinstance(body, dict):
            return _validation_error(["body"], "request body must be an object")
        for k in ("state", "model", "questions"):
            if k not in body:
                return _validation_error(["body", k], "field required", "missing")
        t0 = time.perf_counter()
        try:
            res = engine.evaluate(body["state"], body["model"], body["questions"])
        except RequestError as e:
            return _validation_error(e.loc, e.msg)
        wall_ms = (time.perf_counter() - t0) * 1000
        out = {"model": res.model, "answers": res.answers, "usage": res.usage}
        headers = {
            "x-cujev-prefill-ms": f"{res.timing.prefill_ms:.2f}",
            "x-cujev-eval-ms": f"{res.timing.eval_ms:.2f}",
            "x-cujev-wall-ms": f"{wall_ms:.2f}",
            "x-cujev-prefix-cached": "1" if res.prefix_cached else "0",
        }
        return JSONResponse(content=out, headers=headers)

    return app


def serve(model_dir: str, host: str = "127.0.0.1", port: int = 8080, **engine_kwargs):
    import uvicorn
    api_key = os.environ.get("CUJEV_API_KEY") or None
    engine = SystemOne(model_dir, **engine_kwargs)
    app = create_app(engine, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")
