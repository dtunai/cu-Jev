"""End-to-end HTTP tests: FastAPI app in-process, official typesafe-sdk as the client.
Run:  CUJEV_MODEL=/path/to/Qwen3.5-0.8B pytest -q tests/python/test_server.py
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

MODEL = os.environ.get("CUJEV_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set CUJEV_MODEL")
PORT = 18081


@pytest.fixture(scope="module")
def server():
    import uvicorn
    from cujev.server import create_app
    from cujev.systemone import SystemOne
    so = SystemOne(MODEL, max_state_tokens=2048, max_branch_tokens=1024)
    app = create_app(so, api_key="secret")
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{PORT}"
    srv.should_exit = True
    th.join(timeout=5)


def test_official_sdk_end_to_end(server):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    client = TypeSafeClient(api_key="secret", base_url=server)
    r = client.system_one(
        state="Everything is down and we have a demo with our biggest client at noon.",
        questions={
            "urgent": Noul(instructions="Does the customer need a reply within the hour?"),
            "team": Choice(instructions="Which team should handle it?",
                           criteria={"outage": "service down", "billing": "charges, refunds",
                                     "feature": "requests, how-to"}),
            "tone": Score(instructions="How upset is the customer?",
                          criteria=["calm", "annoyed", "furious"]),
        })
    # shapes and invariants only: the 0.8B's opinions are not what is under test here
    assert r.model.startswith("cujev/")
    assert 0.0 <= r.answers["urgent"].noul <= 1.0
    probs = r.answers["team"].probabilities
    assert set(probs) == {"outage", "billing", "feature"}
    assert r.answers["team"].choice == max(probs, key=probs.get)
    assert 0.0 <= r.answers["team"].confidence <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert 0.0 <= r.answers["tone"].score <= 2.0
    assert r.answers["tone"].legend == {0: "calm", 1: "annoyed", 2: "furious"} or \
        r.answers["tone"].legend == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert r.usage.input_tokens > 0
    names = [m.name for m in client.models.list().models]
    assert "jev-latest" in names


def test_http_shapes_and_errors(server):
    import httpx
    h = {"Authorization": "Bearer secret"}
    body = {"model": "jev-latest", "state": {"ticket": "Charged twice this month."},
            "questions": {"refund": {"type": "noul", "instructions": "Asking for money back?"}}}
    r = httpx.post(f"{server}/v1/systemone", json=body, headers=h)
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"model", "answers", "usage"}
    assert j["answers"]["refund"]["type"] == "noul" and 0 <= j["answers"]["refund"]["noul"] <= 1
    assert set(j["usage"]) == {"input_tokens", "output_tokens"}
    assert r.headers["x-cujev-prefix-cached"] in ("0", "1")
    # second identical state hits the prefix cache
    r2 = httpx.post(f"{server}/v1/systemone", json=body, headers=h)
    assert r2.headers["x-cujev-prefix-cached"] == "1"
    # auth
    assert httpx.post(f"{server}/v1/systemone", json=body).status_code == 401
    # validation: Jev's 422 list shape
    bad = dict(body, questions={"q": {"type": "score", "instructions": "?", "criteria": ["one"]}})
    r3 = httpx.post(f"{server}/v1/systemone", json=bad, headers=h)
    assert r3.status_code == 422
    d = r3.json()["detail"]
    assert isinstance(d, list) and d[0]["loc"][:2] == ["body", "questions"] and "msg" in d[0]
    assert httpx.post(f"{server}/v1/systemone", json={"state": "x"}, headers=h).status_code == 422
    assert httpx.post(f"{server}/v1/systemone", json=dict(body, model="gpt-9"), headers=h).status_code == 422
    assert httpx.get(f"{server}/health").json()["status"] == "ok"
    # the browser example is served by the same process; /play keeps the query string
    r4 = httpx.get(f"{server}/play?autostart=1")
    assert r4.status_code in (302, 307) and r4.headers["location"].endswith("index.html?autostart=1")
    assert httpx.get(f"{server}/examples/starfighter/game.js").status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in httpx.options(
        f"{server}/v1/systemone", headers={"Origin": "http://x", "Access-Control-Request-Method": "POST"}).headers}
