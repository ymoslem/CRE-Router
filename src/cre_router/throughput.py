"""Output throughput of a served batch, whole-run and at full load (steady-state), always together.

A benchmark that sends a fixed batch of requests ends with a drain: once every
request has been handed out, the server works on fewer and fewer, and the last
few finish alone while the other slots idle. vLLM's ``output_throughput`` is
total output tokens over the run's wall-clock duration, so it includes that
drain. Where one request runs far longer than the rest, for instance an answer
that never stops until the output cap, the drain can dominate the run and the
whole-run figure mostly measures the wait for it.

**Full-load throughput**, called steady-state output throughput on the slides
and in the paper, is our own construction for that case, not a vLLM field: output tokens emitted while every slot is busy, over the time every slot
is busy. It measures what the server sustains under load. No request is removed.
A long-running request keeps its slot and its tokens inside the window; only the
stretch of time when too few requests remain to fill the slots is left out.

The two are returned together, never one alone, because the gap between them is
itself the finding: it says how much of the run was drain. Under a benchmark
with continuous arrivals there is no drain, the two agree, and this measure is
unnecessary.

Usage::

    from cre_router.throughput import measure
    t = measure(detail, cap=32)       # detail: one vLLM per-request dump
    t.whole_run, t.full_load, t.full_load_share
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

__all__ = ["Throughput", "measure"]


@dataclass(frozen=True)
class Throughput:
    """Output tokens per second for one served batch, two ways."""

    #: vLLM's ``output_throughput``: all output tokens over the whole run.
    whole_run: float
    #: Output tokens emitted while every slot was busy, over that time: the
    #: steady-state output throughput of the slides and the paper.
    full_load: float
    #: Share of the run's duration during which every slot was busy.
    full_load_share: float


def _finish_times(detail: dict) -> list[float]:
    return [start + ttft + sum(itl) for start, ttft, itl in
            zip(detail["start_times"], detail["ttfts"], detail["itls"])]


def _full_windows(detail: dict, cap: int) -> list[tuple[float, float]]:
    """The intervals during which at least ``cap`` requests were in progress."""
    events = sorted([(t, 1) for t in detail["start_times"]]
                    + [(t, -1) for t in _finish_times(detail)])
    busy, previous, windows = 0, events[0][0], []
    for time, step in events:
        if busy >= cap and time > previous:
            windows.append((previous, time))
        busy += step
        previous = time
    return windows


def _token_times(detail: dict) -> list[float]:
    """When every output token was emitted: the first at TTFT, the rest by ITL."""
    stamps = []
    for start, ttft, itl in zip(detail["start_times"], detail["ttfts"], detail["itls"]):
        time = start + ttft
        stamps.append(time)
        for gap in itl:
            time += gap
            stamps.append(time)
    stamps.sort()
    return stamps


def measure(detail: dict, cap: int) -> Throughput:
    """Whole-run and full-load output throughput of one served batch.

    ``detail`` is one per-request dump as ``cre evaluate`` saves it, carrying
    ``start_times``, ``ttfts`` and ``itls``; ``cap`` is the concurrency limit the
    batch was served at. Raises ``ValueError`` if the batch never had every slot
    busy, since then there is no full-load window to measure.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    if detail.get("output_throughput") is not None:
        whole = float(detail["output_throughput"])
    else:
        whole = detail["total_output_tokens"] / detail["duration"]

    windows = _full_windows(detail, cap)
    seconds = sum(b - a for a, b in windows)
    if seconds <= 0:
        raise ValueError(f"the batch never had all {cap} slots busy")
    stamps = _token_times(detail)
    tokens = sum(bisect.bisect_right(stamps, b) - bisect.bisect_left(stamps, a)
                 for a, b in windows)
    span = max(_finish_times(detail)) - min(detail["start_times"])
    return Throughput(whole_run=whole, full_load=tokens / seconds,
                      full_load_share=seconds / span if span > 0 else 1.0)
