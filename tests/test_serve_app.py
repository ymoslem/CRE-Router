"""HTTP layer smoke test: drives the serving app end-to-end with a router built
from fake backends (no GPU, no model servers). Needs the `serve` extra."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from cre_router.server.app import build_app  # noqa: E402
from cre_router.server.cascade_router import CascadeRouter  # noqa: E402

CENTROIDS = np.array([[1.0, 0.0], [0.0, 1.0]])
EMBEDDINGS = {"easy": [0.9, 0.1], "hard": [0.1, 0.9]}


def embed_fn(texts):
    return np.array([EMBEDDINGS[t] for t in texts])


class Backend:
    async def __call__(self, model, request):
        return {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": f"answer from {model}"}}],
            "usage": {"completion_tokens": 5},
        }


class Escalate:
    accept = False
    p_accept = 0.2


def make_client(qe_predict_fns=None):
    router = CascadeRouter(
        centroids=CENTROIDS,
        routing_table={"0": "weak", "1": "strong"},
        embed_fn=embed_fn,
        completion_fn=Backend(),
        escalation_order=["weak", "strong"],
        qe_predict_fns=qe_predict_fns,
    )
    return TestClient(build_app(router))


def chat(text):
    return {"messages": [{"role": "user", "content": text}]}


def test_health_and_models():
    client = make_client()
    assert client.get("/health").json()["status"] == "ok"
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert ids == {"weak", "strong"}


def test_chat_completion_returns_content_and_headers():
    client = make_client()
    resp = client.post("/v1/chat/completions", json=chat("easy"))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "answer from weak"
    assert resp.headers["x-cre-cluster"] == "0"
    assert resp.headers["x-cre-stage1-model"] == "weak"
    assert resp.headers["x-cre-final-model"] == "weak"
    assert resp.headers["x-cre-escalated"] == "false"
    assert resp.headers["x-cre-path"] == "weak"


def test_escalation_reported_in_headers():
    client = make_client(qe_predict_fns={"weak": lambda q, o, n: Escalate()})
    resp = client.post("/v1/chat/completions", json=chat("easy"))
    assert resp.json()["model"] == "strong"
    assert resp.headers["x-cre-escalated"] == "true"
    assert resp.headers["x-cre-path"] == "weak -> strong"
    assert resp.headers["x-cre-final-model"] == "strong"


def test_streaming_rejected():
    client = make_client()
    resp = client.post("/v1/chat/completions", json={**chat("easy"), "stream": True})
    assert resp.status_code == 400
    assert "streaming" in resp.json()["detail"].lower()


def test_stats_tally():
    client = make_client(qe_predict_fns={"weak": lambda q, o, n: Escalate()})
    client.post("/v1/chat/completions", json=chat("easy"))  # weak -> strong (escalated)
    client.post("/v1/chat/completions", json=chat("hard"))  # strong directly
    stats = client.get("/stats").json()
    assert stats["total_requests"] == 2
    assert stats["escalations"] == 1
    assert stats["escalation_rate"] == 0.5
    assert stats["by_cluster"] == {"0": 1, "1": 1}
    assert stats["by_final_model"]["strong"] == 2
