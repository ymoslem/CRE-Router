"""`estimate_stage2_cost` reproduces the Stage 2 cost estimates the paper reports.

For each reported routing, the budget selects the routing from the shipped stats,
`stage2_plan` gates its clusters, and the estimate of what Stage 2 adds must
match the reported values at the precision they are printed. A small hand-built
pool checks the arithmetic itself, including an escalated query's TPOT.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cre_router.routing import (estimate_stage2_cost, error_tol_from_stats, models_from_stats,
                                select_lambda, stage2_plan)

CONFIGS = Path(__file__).parents[1] / "configs"

#: (stats file, cost term, budget in ms, added TPOT ms, added E2EL s, E2EL digits)
REPORTED = [
    ("telemath_stats_1xA100_Sep2026", "tpot", 20, 3.2, 23.3, 1),
    ("telemath_stats_1xA100_Sep2026", "tpot", 25, 2.1, 13.6, 1),
    ("telemath_stats_1xA100_Sep2026", "e2el", 25_000, 3.1, 3.3, 1),
    ("telemath_stats_1xA100_Sep2026", "e2el", 120_000, 0.3, 22.5, 1),
    ("telemath_stats_2xA100_Sep2026", "tpot", 20, 1.9, 45.2, 1),
    ("teleqna_stats_1xA100_Oct2026", "tpot", 20, 5.0, 0.25, 2),
    ("teleqna_stats_2xA100_Oct2026", "tpot", 15, 3.2, 0.17, 2),
    ("aime_stats_1xA100_Sep2026", "tpot", 30, 1.5, 18.5, 1),
    ("aime_stats_2xA100_Sep2026", "tpot", 20, 1.0, 11.8, 1),
]


def _estimate(stats: dict, term: str, budget_ms: float):
    models, sizes = models_from_stats(stats, term)
    selection = select_lambda(models, sizes, budget_ms, error_tol_from_stats(stats))
    assignment = selection.region.assignment
    return estimate_stage2_cost(stats, assignment, stage2_plan(models, assignment, sizes))


@pytest.mark.parametrize("name,term,budget,tpot,e2el,digits", REPORTED,
                         ids=[f"{r[0]}-{r[1]}{r[2]}" for r in REPORTED])
def test_reproduces_the_reported_estimate(name, term, budget, tpot, e2el, digits):
    stats = json.loads((CONFIGS / f"{name}.json").read_text())
    est = _estimate(stats, term, budget)
    assert round(est.added_tpot_ms, 1) == tpot
    assert round(est.added_e2el_s, digits) == e2el


def test_arithmetic_on_a_hand_built_pool():
    # Two clusters of 30 and 10 queries. The efficient model serves cluster 0 and is
    # wrong on a quarter of it; the strong model serves cluster 1.
    stats = {"cluster_sizes": {"0": 30, "1": 10}, "models": {
        "eff": {"errors": {"0": 0.25, "1": 0.6},
                "cluster_tpot_ms": {"0": 10.0, "1": 10.0},
                "cluster_e2el_ms": {"0": 1100.0, "1": 1100.0},
                "cluster_ttft_ms": {"0": 100.0, "1": 100.0}},
        "strong": {"errors": {"0": 0.05, "1": 0.1},
                   "cluster_tpot_ms": {"0": 20.0, "1": 20.0},
                   "cluster_e2el_ms": {"0": 4100.0, "1": 4100.0},
                   "cluster_ttft_ms": {"0": 100.0, "1": 100.0}}}}
    models, sizes = models_from_stats(stats, "tpot")
    assignment = {"0": "eff", "1": "strong"}
    plan = stage2_plan(models, assignment, sizes)
    assert plan.target == "strong" and plan.gated == {"0": "eff"}
    est = estimate_stage2_cost(stats, assignment, plan)
    # 100 decode steps for the efficient answer, 200 for the strong one, so an
    # escalated query's TPOT is (10 * 100 + 20 * 200) / 200 = 25 ms.
    assert est.escalated == pytest.approx(0.25 * 30 / 40)
    assert est.stage1_tpot_ms == pytest.approx((30 * 10 + 10 * 20) / 40)
    assert est.added_tpot_ms == pytest.approx(30 * 0.25 * (25 - 10) / 40)
    assert est.stage1_e2el_s == pytest.approx((30 * 1.1 + 10 * 4.1) / 40)
    assert est.added_e2el_s == pytest.approx(30 * 0.25 * 4.1 / 40)
    assert est.tpot_ms == pytest.approx(est.stage1_tpot_ms + est.added_tpot_ms)


def test_nothing_gated_adds_nothing():
    stats = {"cluster_sizes": {"0": 5}, "models": {
        "only": {"errors": {"0": 0.3}, "cluster_tpot_ms": {"0": 12.0},
                 "cluster_e2el_ms": {"0": 600.0}, "cluster_ttft_ms": {"0": 50.0}}}}
    models, sizes = models_from_stats(stats, "tpot")
    plan = stage2_plan(models, {"0": "only"}, sizes)
    est = estimate_stage2_cost(stats, {"0": "only"}, plan)
    assert plan.target is None
    assert est.escalated == est.added_tpot_ms == est.added_e2el_s == 0.0
