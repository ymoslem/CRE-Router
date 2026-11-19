"""Hold a vLLM server at a fixed load while a benchmark measures a small batch.

A benchmark batch is at full load only while ``target`` requests are in flight.
A batch with fewer requests than that never reaches it, and every batch drains
below it at the end, so the requests measured there share the GPU with fewer
others and look cheaper than they would in a server that stays busy. An
escalated subset of a few questions is the extreme case.

:class:`LoadFiller` runs beside the benchmark and tops the server up with filler
requests whenever fewer than ``target`` are running or waiting, from the moment
the benchmark has dispatched all its own requests until it returns. Fillers are
replays of the batch's own prompts, so the load has the same length mix. They are
never measured: the benchmark records only its own requests, and the fillers
still in flight when it returns are cancelled, which makes vLLM abort them.
Serve with prefix caching off: a filler replays a measured prompt, so with the
cache on it would hand the measured request a cached prefix.

The server's in-flight count is read from its Prometheus ``/metrics`` endpoint
(``vllm:num_requests_running`` and ``vllm:num_requests_waiting``); how many of the
benchmark's requests have finished is ``vllm:request_success_total`` since the
start, less the fillers this client saw finish. Every poll is kept in a trace,
written next to the benchmark's detail file, so the load each measured request
saw can be checked afterwards rather than assumed.

Usage::

    with LoadFiller(base_url, model, prompts, n_measured=len(items), target=32,
                    sampling=..., trace_path=...):
        result = run_vllm_benchmark(...)
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each Prometheus sample over its labels: ``name -> value``.

    vLLM labels its gauges and counters by model and engine (and the success
    counter by finish reason); the filler needs only their totals.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        name = name_part.split("{", 1)[0].strip()
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
SUCCESS = "vllm:request_success_total"


def fillers_to_launch(running: float, waiting: float, measured_done: float,
                      n_measured: int, target: int) -> int:
    """How many fillers to send now.

    None while the benchmark may still have requests queued on its side: it
    dispatches whenever fewer than ``target`` of its own are in flight, so a
    filler sent then could push the server above ``target``. Its queue is empty
    once all but ``target`` of its requests have finished. After that, enough to
    bring running plus waiting back to ``target``.
    """
    if measured_done < n_measured - target:
        return 0
    return max(0, int(target - running - waiting))


@dataclass
class FillerStats:
    launched: int = 0
    finished: int = 0
    cancelled: int = 0
    failed: int = 0
    trace: list[dict] = field(default_factory=list)


class LoadFiller:
    """Context manager: fill the server to ``target`` in flight while the body runs.

    ``prompts`` are the batch's own prompts, replayed round-robin. With ``chat``
    they go to ``/v1/chat/completions`` as one user message, so the server applies
    the model's chat template as the benchmark's loader does for a task that is
    not pre-rendered; without it they go to ``/v1/completions`` verbatim, as a
    pre-rendered prompt is served. ``sampling``
    holds the request fields the benchmark sends (``max_tokens``, ``temperature``,
    ``top_p``, ``top_k``, ``min_p`` and any extra body), so a filler generates
    like a measured request. ``fetch_metrics`` and ``send`` are injectable for
    tests; by default they talk to the server over HTTP.
    """

    def __init__(self, base_url: str, model: str, prompts: Sequence[str], *,
                 n_measured: int, target: int = 32, sampling: Mapping | None = None,
                 chat: bool = False,
                 trace_path: str | Path | None = None, poll_s: float = 0.1,
                 fetch_metrics: Callable | None = None, send: Callable | None = None):
        if not prompts:
            raise ValueError("LoadFiller needs at least one prompt to replay")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompts = list(prompts)
        self.n_measured = n_measured
        self.target = target
        self.sampling = {k: v for k, v in dict(sampling or {}).items() if v is not None}
        self.chat = chat
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.poll_s = poll_s
        self._fetch = fetch_metrics
        self._send = send
        self.stats = FillerStats()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    # -- HTTP defaults (vLLM env only) ------------------------------------
    async def _http_metrics(self, session) -> dict[str, float]:
        async with session.get(f"{self.base_url}/metrics") as r:
            r.raise_for_status()
            return parse_metrics(await r.text())

    def request(self, prompt: str) -> tuple[str, dict]:
        """(URL, JSON body) of one filler request."""
        if self.chat:
            url = f"{self.base_url}/v1/chat/completions"
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        else:
            url = f"{self.base_url}/v1/completions"
            body = {"model": self.model, "prompt": prompt}
        body.update(stream=False, **self.sampling)
        return url, body

    async def _http_send(self, session, prompt: str) -> None:
        url, body = self.request(prompt)
        async with session.post(url, json=body) as r:
            r.raise_for_status()
            await r.read()

    # -- the loop -----------------------------------------------------------
    async def _main(self) -> None:
        session = None
        if self._fetch is None or self._send is None:
            import aiohttp
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        fetch = self._fetch or (lambda: self._http_metrics(session))
        send = self._send or (lambda p: self._http_send(session, p))
        tasks: set[asyncio.Task] = set()
        nxt = 0

        def done(t: asyncio.Task) -> None:
            tasks.discard(t)
            if t.cancelled():
                self.stats.cancelled += 1
            elif t.exception() is not None:
                self.stats.failed += 1
            else:
                self.stats.finished += 1

        try:
            base = (await fetch()).get(SUCCESS, 0.0)
            self._ready.set()
            t0 = time.monotonic()
            while not self._stop.is_set():
                m = await fetch()
                running, waiting = m.get(RUNNING, 0.0), m.get(WAITING, 0.0)
                measured_done = m.get(SUCCESS, 0.0) - base - self.stats.finished
                k = fillers_to_launch(running, waiting, measured_done,
                                      self.n_measured, self.target)
                for _ in range(k):
                    t = asyncio.ensure_future(send(self.prompts[nxt % len(self.prompts)]))
                    nxt += 1
                    self.stats.launched += 1
                    tasks.add(t)
                    t.add_done_callback(done)
                self.stats.trace.append({
                    "t": round(time.monotonic() - t0, 3), "running": running,
                    "waiting": waiting, "measured_done": measured_done,
                    "fillers_in_flight": len(tasks), "launched_now": k})
                await asyncio.sleep(self.poll_s)
        finally:
            for t in list(tasks):
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if session is not None:
                await session.close()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as e:  # surfaced in __enter__ or __exit__, never swallowed
            self._error = e
        finally:
            # Set only after any error is recorded, so __enter__ cannot see the
            # thread as started when its first metrics read has failed.
            self._ready.set()

    def __enter__(self) -> "LoadFiller":
        self._thread = threading.Thread(target=self._run, name="load-filler", daemon=True)
        self._thread.start()
        # The success counter's baseline must be read before the benchmark sends
        # anything, or its first completions would be miscounted as earlier ones.
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError("load filler failed to start") from self._error
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text(json.dumps({
                "target": self.target, "n_measured": self.n_measured,
                "launched": self.stats.launched, "finished": self.stats.finished,
                "cancelled": self.stats.cancelled, "failed": self.stats.failed,
                "trace": self.stats.trace}))
        if self._error is not None and exc[0] is None:
            raise RuntimeError("load filler failed while the benchmark ran") from self._error
