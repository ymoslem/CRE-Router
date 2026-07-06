"""HTTP front-end for the cascade router.

Exposes POST /v1/chat/completions; a standard chat-completions client works
by pointing its base URL at this server. Routing decisions are reported in
x-cre-* response headers. Streaming is not supported because Stage 2 QE must
inspect the complete response before it can be accepted (set
``qe.enabled: false`` and use Stage 1 alone if streaming matters more than
escalation).

Observability adds no latency to the request path: GET /stats returns
in-memory tallies (O(1) counter updates per request), and an optional
per-request decision log is written by a background task fed through a
non-blocking queue, so no disk I/O happens while a request is in flight.
Enable the log with ``decision_log: <path>`` in the serving config.
"""

# NB: no ``from __future__ import annotations`` here. FastAPI resolves route
# handler annotations at request time against module globals, but fastapi is
# imported lazily inside the functions below; stringized annotations would fail
# to resolve (the Request param would be mistaken for a query param).

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


DECISION_LOG_MAXSIZE = 10_000


@dataclass
class RouterMetrics:
    """In-memory request tallies. Every update is an O(1) counter increment."""

    total: int = 0
    escalations: int = 0
    log_dropped: int = 0
    by_cluster: Counter = field(default_factory=Counter)
    by_final_model: Counter = field(default_factory=Counter)
    by_path: Counter = field(default_factory=Counter)

    def record(self, meta) -> None:
        self.total += 1
        if meta.escalated:
            self.escalations += 1
        self.by_cluster[str(meta.cluster)] += 1
        self.by_final_model[meta.final_model] += 1
        self.by_path[" -> ".join(meta.path)] += 1

    def snapshot(self) -> dict:
        return {
            "total_requests": self.total,
            "escalations": self.escalations,
            "escalation_rate": self.escalations / self.total if self.total else 0.0,
            "decision_log_dropped": self.log_dropped,
            "by_cluster": dict(self.by_cluster),
            "by_final_model": dict(self.by_final_model),
            "by_path": dict(self.by_path),
        }


def decision_record(meta) -> dict:
    return {
        "ts": time.time(),
        "cluster": meta.cluster,
        "path": meta.path,
        "final_model": meta.final_model,
        "escalated": meta.escalated,
        "p_accept": meta.p_accept,
    }


async def _drain_decision_log(queue, path: str | Path) -> None:
    """Background writer: append queued decision records to a JSONL file,
    flushing when the queue empties. A None sentinel stops it. Disk writes
    happen here, never on the request path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        while True:
            item = await queue.get()
            if item is None:
                f.flush()
                return
            f.write(json.dumps(item) + "\n")
            if queue.empty():
                f.flush()


def create_app(config: str | Path | dict):
    """Build the serving app from a config: load the CascadeRouter, then wire
    the HTTP layer around it."""
    import yaml

    from cre_router.server.cascade_router import CascadeRouter

    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    router = CascadeRouter.from_config(config)
    return build_app(router, decision_log=config.get("decision_log"))


def build_app(router, decision_log: str | Path | None = None):
    """Wire the HTTP endpoints around an already-built router. Kept separate
    from ``create_app`` so tests can inject a router with fake backends."""
    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    metrics = RouterMetrics()
    log_queue: asyncio.Queue | None = (
        asyncio.Queue(maxsize=DECISION_LOG_MAXSIZE) if decision_log else None
    )

    @asynccontextmanager
    async def lifespan(app):
        writer = (
            asyncio.create_task(_drain_decision_log(log_queue, decision_log))
            if log_queue is not None
            else None
        )
        try:
            yield
        finally:
            if writer is not None:
                await log_queue.put(None)  # stop the writer and flush
                await writer

    app = FastAPI(title="cre-router", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "clusters": len(router.centroids)}

    @app.get("/stats")
    async def stats():
        return metrics.snapshot()

    @app.get("/v1/models")
    async def models():
        pool = sorted(set(router.routing_table.values()) | set(router.escalation_order))
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in pool]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported: the QE cascade inspects complete responses.",
            )
        try:
            response, meta = await router.acompletion(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # O(1) in-memory tally; non-blocking hand-off to the background writer.
        # If the writer has fallen behind and the queue is full, drop the record
        # (and count it) rather than block the request or grow memory unbounded.
        metrics.record(meta)
        if log_queue is not None:
            try:
                log_queue.put_nowait(decision_record(meta))
            except asyncio.QueueFull:
                metrics.log_dropped += 1

        content = response.model_dump() if hasattr(response, "model_dump") else response
        headers = {
            "x-cre-cluster": str(meta.cluster),
            "x-cre-stage1-model": meta.stage1_model,
            "x-cre-final-model": meta.final_model,
            "x-cre-escalated": str(meta.escalated).lower(),
            "x-cre-path": " -> ".join(meta.path),
        }
        if meta.p_accept is not None:
            headers["x-cre-p-accept"] = f"{meta.p_accept:.4f}"
        return JSONResponse(content=content, headers=headers)

    return app


def serve(config_path: str | Path, host: str = "0.0.0.0", port: int = 4000) -> None:
    import uvicorn

    uvicorn.run(create_app(config_path), host=host, port=port)
