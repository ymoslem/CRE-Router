"""Cross-cluster near-ties: the crossover-gap diagnostic and --crossover-tol.

The two-model pools are a fine-tuned and an off-the-shelf Qwen3-4B each paired
with Gemma4-26B on TeleQnA, whose cluster order flips between the two pools on
a 0.2 pp difference in Gemma's advantage.
"""
import json
from pathlib import Path

import pytest

from cre_router.routing import (
    ModelStats,
    crossover_gaps,
    error_tol_from_stats,
    is_tied,
    models_from_stats,
    pareto_prune,
    routing_regions,
    select_lambda,
    swapped_candidates,
    system_metrics,
)

SIZES = {"0": 5211.0, "1": 3789.0}
GEMMA = ({"0": 0.269445, "1": 0.298392}, {"0": 19.73, "1": 19.63})
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def pool(errors, tpot):
    def model(name, e, t):
        return ModelStats(name=name, errors=e, cluster_tpot_ms=t,
                          tpot_ms=sum(SIZES[c] * t[c] for c in t) / sum(SIZES.values()))
    return [model("Qwen", errors, tpot), model("Gemma", *GEMMA)]


OTS = pool({"0": 0.336395, "1": 0.363244}, {"0": 8.61, "1": 8.47})


def test_the_ots_pool_has_one_unresolved_crossover_order():
    (g,) = crossover_gaps(OTS, SIZES)
    assert (g.lower.cluster, g.upper.cluster) == ("1", "0")
    assert g.gap == pytest.approx(0.066950 - 0.064852, abs=1e-6)
    assert not g.resolvable and g.ratio < 1


def test_clearly_separated_crossovers_are_resolvable():
    far = pool({"0": 0.30, "1": 0.45}, {"0": 8.6, "1": 8.5})
    (g,) = crossover_gaps(far, SIZES)
    assert g.resolvable


def test_the_diagnostic_never_alters_the_regions():
    before = routing_regions(OTS)
    crossover_gaps(OTS, SIZES)
    assert routing_regions(OTS) == before


def test_zero_tolerance_adds_no_candidate():
    assert swapped_candidates(OTS, SIZES, 0.0) == []


def test_a_tie_adds_the_routing_no_lambda_reaches():
    (cand,) = swapped_candidates(OTS, SIZES, "auto")
    assert cand.assignment == {"0": "Qwen", "1": "Gemma"}
    assert cand.swapped == ("1", "0")
    assert cand.assignment not in [r.assignment for r in routing_regions(OTS)]
    assert swapped_candidates(OTS, SIZES, 0.003) == [cand]
    assert swapped_candidates(OTS, SIZES, 0.001) == []


def test_the_swapped_routing_fills_the_budget_gap():
    """Between the swapped routing's cost and the fitted one's, the published fit
    can only fall back to the cheap model everywhere; the tie makes the swapped
    routing available, and it is more accurate."""
    (cand,) = swapped_candidates(OTS, SIZES, "auto")
    _, cost = system_metrics(OTS, cand.assignment, SIZES)
    budget = cost + 0.1
    plain = select_lambda(OTS, SIZES, budget)
    tied = select_lambda(OTS, SIZES, budget, crossover_tol="auto")
    assert set(plain.region.assignment.values()) == {"Qwen"}
    assert tied.region.assignment == cand.assignment
    assert tied.lambda_star is None and tied.accuracy > plain.accuracy


def test_a_looser_budget_keeps_the_fitted_routing():
    fitted = select_lambda(OTS, SIZES, 15.1)
    assert select_lambda(OTS, SIZES, 15.1, crossover_tol="auto").region == fitted.region


@pytest.mark.parametrize("stats", sorted(CONFIGS.glob("*_stats_*.json")), ids=lambda p: p.stem)
def test_the_default_reproduces_every_shipped_fit(stats):
    st = json.loads(stats.read_text())
    models, sizes = models_from_stats(st)
    tol = error_tol_from_stats(st)
    efficient, _ = pareto_prune(models, tol)
    for region in routing_regions(efficient, tol):
        _, cost = system_metrics(efficient, region.assignment, sizes)
        default = select_lambda(efficient, sizes, cost, tol)
        explicit = select_lambda(efficient, sizes, cost, tol, crossover_tol=0.0)
        assert explicit.region == default.region and explicit.lambda_star == default.lambda_star


def test_a_numeric_tolerance_of_zero_ties_nothing():
    (g,) = crossover_gaps(OTS, SIZES)
    assert not is_tied(g, 0.0) and is_tied(g, "auto") and is_tied(g, 0.003)
