"""Whole-run and full-load throughput: they agree without a drain, and diverge by
exactly the drain when one request outlasts the rest."""
from __future__ import annotations

import pytest

from cre_router.throughput import measure

# Binary fractions throughout, so token times land exactly on window edges.
STEP = 0.125


def request(tokens: int) -> tuple[float, list[float]]:
    """TTFT and inter-token gaps for a request emitting one token every STEP s."""
    return STEP, [STEP] * (tokens - 1)


def batch(*lengths: int, starts=None) -> dict:
    starts = starts or [0.0] * len(lengths)
    ttfts, itls = zip(*(request(n) for n in lengths))
    duration = max(s + t + sum(i) for s, t, i in zip(starts, ttfts, itls)) - min(starts)
    return {"start_times": list(starts), "ttfts": list(ttfts), "itls": list(itls),
            "total_output_tokens": sum(lengths), "duration": duration}


def test_without_a_drain_the_two_measures_agree():
    t = measure(batch(8, 8), cap=2)
    assert t.whole_run == pytest.approx(16.0)
    assert t.full_load == pytest.approx(16.0)
    assert t.full_load_share == pytest.approx(1.0)


def test_a_straggler_lowers_whole_run_but_not_full_load():
    """One request runs ten times longer: the server is full for the first second only."""
    t = measure(batch(8, 80), cap=2)
    # While both run, 16 tokens in 1 s. Over the whole run, 88 tokens in 10 s.
    assert t.full_load == pytest.approx(16.0)
    assert t.whole_run == pytest.approx(8.8)
    assert t.full_load_share == pytest.approx(0.1)


def test_the_straggler_keeps_its_tokens_inside_the_window():
    """Nothing is removed: the long request's tokens during full load count."""
    without = measure(batch(8, 8), cap=2).full_load
    with_straggler = measure(batch(8, 80), cap=2).full_load
    assert with_straggler == pytest.approx(without)


def test_vllm_figure_is_used_when_present():
    d = batch(8, 8)
    d["output_throughput"] = 15.5
    assert measure(d, cap=2).whole_run == 15.5


def test_a_batch_that_never_fills_is_refused():
    with pytest.raises(ValueError, match="never had all 3 slots busy"):
        measure(batch(8, 8), cap=3)


def test_cap_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        measure(batch(8), cap=0)
