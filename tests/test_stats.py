"""Stats files rebuilt from captures: one measurement per batch, the same path as cre evaluate."""
from __future__ import annotations

import pytest

from cre_router.stats import build_stats, measurements


def capture(errors_by_cluster, runs=2, per_cluster=4, tpot=10.0):
    """Questions q<c><i>; in each cluster the first `wrong` answers are wrong."""
    cap = {}
    for c, wrong in errors_by_cluster.items():
        for r in range(runs):
            for i in range(per_cluster):
                cap[(f"q{c}{i}", r)] = {
                    "cluster": c, "correct": i >= wrong,
                    "batch_tpot_ms": tpot + r, "batch_ttft_ms": 100.0, "batch_e2el_ms": 900.0 + r}
    return cap


def test_one_measurement_per_cluster_and_run_with_error_regraded():
    ms = measurements(capture({"0": 1, "1": 2}))
    assert [(m.cluster, m.run) for m in ms] == [("0", 0), ("0", 1), ("1", 0), ("1", 1)]
    assert [m.error for m in ms] == [0.25, 0.25, 0.5, 0.5]
    assert ms[1].tpot_ms == 11.0 and ms[1].e2el_ms == 901.0


def test_rows_of_one_batch_must_share_its_means():
    cap = capture({"0": 0})
    cap[("q00", 0)] = {**cap[("q00", 0)], "batch_tpot_ms": 99.0}
    with pytest.raises(ValueError, match="not served as one batch"):
        measurements(cap)


def test_build_stats_averages_runs_like_cre_evaluate():
    pool = {"name": "p", "_note": "n", "models": {"A": "a", "B": "b"}}
    caps = {"a": capture({"0": 1, "1": 2}), "b": capture({"0": 0, "1": 4}, tpot=20.0)}
    stats = build_stats(pool, caps.__getitem__)
    assert stats["cluster_sizes"] == {"0": 4, "1": 4}
    assert stats["models"]["A"]["errors"] == {"0": 0.25, "1": 0.5}
    assert stats["models"]["A"]["cluster_tpot_ms"] == {"0": 10.5, "1": 10.5}
    assert stats["models"]["B"]["cluster_e2el_ms"] == {"0": 900.5, "1": 900.5}
    assert stats["_note"] == "n" and stats["built_from"] == {"A": "a", "B": "b"}


def test_models_over_different_questions_are_refused():
    pool = {"models": {"A": "a", "B": "b"}}
    caps = {"a": capture({"0": 0}), "b": capture({"0": 0}, per_cluster=5)}
    with pytest.raises(ValueError, match="cluster sizes"):
        build_stats(pool, caps.__getitem__)


def test_a_run_missing_questions_is_refused():
    cap = capture({"0": 0})
    del cap[("q01", 1)]
    with pytest.raises(ValueError, match="not every run answered every question"):
        build_stats({"models": {"A": "a"}}, {"a": cap}.__getitem__)
