"""Exhaustive selection: the most accurate routing under the budget, by search."""
from __future__ import annotations

import math

import pytest

from cre_router.routing import ModelStats, select_exhaustive, select_lambda, system_metrics

SIZES = {"0": 50, "1": 50}


def pool():
    # S is accurate and expensive; E is cheap; M sits between them in C1 only.
    return [ModelStats("S", 30.0, {"0": 0.1, "1": 0.1}),
            ModelStats("E", 10.0, {"0": 0.5, "1": 0.4}),
            ModelStats("M", 20.0, {"0": 0.6, "1": 0.2})]


def test_it_returns_the_most_accurate_routing_within_the_budget():
    sel = select_exhaustive(pool(), SIZES, 20.0)
    acc, cost = system_metrics(pool(), sel.region.assignment, SIZES)
    assert sel.region.assignment == {"0": "S", "1": "E"}
    assert (sel.accuracy, sel.tpot_ms) == (acc, cost) and cost <= 20.0
    assert sel.lambda_star is None and math.isnan(sel.region.lam_min)


def test_it_is_never_less_accurate_than_the_sweep():
    for budget in (10.0, 15.0, 20.0, 25.0, 30.0):
        assert (select_exhaustive(pool(), SIZES, budget).accuracy
                >= select_lambda(pool(), SIZES, budget).accuracy - 1e-12)


def test_routings_within_the_error_tolerance_go_to_the_cheaper():
    models = [ModelStats("A", 10.0, {"0": 0.2000}), ModelStats("B", 12.0, {"0": 0.1995})]
    assert select_exhaustive(models, {"0": 1}, 20.0, error_tol=0.001).region.assignment == {"0": "A"}
    assert select_exhaustive(models, {"0": 1}, 20.0, error_tol=0.0).region.assignment == {"0": "B"}


def test_an_impossible_budget_is_refused():
    with pytest.raises(ValueError, match="No routing satisfies"):
        select_exhaustive(pool(), SIZES, 5.0)


def test_a_search_too_large_is_refused(monkeypatch):
    import cre_router.routing as r
    monkeypatch.setattr(r, "EXHAUSTIVE_LIMIT", 8)
    with pytest.raises(ValueError, match="routings, more than"):
        select_exhaustive(pool(), SIZES, 20.0)
