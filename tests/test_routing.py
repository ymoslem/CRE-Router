"""Routing math, checked against the shipped stats and routing configs.

Fixtures are the per-cluster training statistics in `configs/`. Where a test
pins an outcome, the expected value is recomputed from the raw JSON inside the
test, or read from the routing spec in `configs/routings/` that the outcome
must reproduce, so a changed config cannot leave a stale number passing.
"""

import json
import math
from pathlib import Path

import pytest

from cre_router.routing import (
    DEFAULT_ERROR_TOL,
    ModelStats,
    _nice_lambda,
    assign,
    cascade_system_accuracy,
    cascade_system_metrics,
    cascade_system_metrics_ntier,
    cluster_cascade_accuracy,
    crossover_candidates,
    dominates,
    error_tol_from_stats,
    eta,
    models_from_stats,
    normalized_costs,
    pareto_prune,
    routing_regions,
    select_lambda,
    system_metrics,
    selection_margins,
)

CONFIGS = Path(__file__).parent.parent / "configs"

V = "VibeThinker-1.5B"
Q = "Qwen3-30B-A3B"


# Every config names its basis, so a fixture has to as well. Most of this file
# runs on two cards; the one-card pool has different crossovers and different
# pruning, covered by TestOneCardBasis.
TWO_CARD = {"aime": "aime_stats_2xA100_Sep2026.json",
            "teleqna": "teleqna_stats_2xA100_Sep2026.json"}
ONE_CARD = {"aime": "aime_stats_1xA100_Sep2026.json",
            "teleqna": "teleqna_stats_1xA100_Sep2026.json"}
#: The two-card routing specs the fits below must reproduce, with their budgets.
AIME_ROUTING = ("aime_tpot_b20ms_2xA100_Sep2026.json", 20.0)
TELEQNA_ROUTING = ("teleqna_tpot_b15ms_2xA100_Sep2026.json", 15.0)
#: Routing specs name a model by its tier; the stats files use a shorter name.
STATS_NAME = {"Qwen3-4B-Instruct": "Qwen3-4B"}


def _raw(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text())


def _spec_assignment(name: str) -> dict[str, str]:
    spec = json.loads((CONFIGS / "routings" / name).read_text())
    return {c: STATS_NAME.get(m, m) for c, m in spec["assign"].items()}


def _weighted(raw: dict, assignment: dict[str, str]) -> tuple[float, float]:
    """Accuracy and TPOT of an assignment, straight from the stats JSON."""
    sizes, models = raw["cluster_sizes"], raw["models"]
    total = sum(sizes.values())
    acc = sum(sizes[c] * (1 - models[m]["errors"][c]) for c, m in assignment.items()) / total
    tpot = sum(sizes[c] * models[m]["cluster_tpot_ms"][c] for c, m in assignment.items()) / total
    return acc, tpot


@pytest.fixture()
def aime():
    return models_from_stats(_raw(TWO_CARD["aime"]))


@pytest.fixture()
def teleqna():
    return models_from_stats(_raw(TWO_CARD["teleqna"]))


@pytest.fixture()
def aime_1x():
    return models_from_stats(_raw(ONE_CARD["aime"]))


@pytest.fixture()
def teleqna_1x():
    return models_from_stats(_raw(ONE_CARD["teleqna"]))


class TestNormalizedCosts:
    def test_two_model_pool_spans_zero_to_one(self, aime):
        models, _ = aime
        costs = normalized_costs(models)
        assert costs[V] == 0.0
        assert costs[Q] == 1.0


class TestCrossovers:
    @staticmethod
    def _gaps() -> dict[str, float]:
        """Each cluster's error gap, VibeThinker minus Qwen3-30B, from the JSON."""
        errors = {m: s["errors"] for m, s in _raw(TWO_CARD["aime"])["models"].items()}
        return {c: errors[V][c] - errors[Q][c] for c in errors[V]}

    def test_aime_closed_form(self, aime):
        """K = 2: min-max sends the pair to {0, 1}, so each crossover is the
        cluster's error gap and the cost term cancels."""
        models, _ = aime
        assert crossover_candidates(models) == pytest.approx(sorted(self._gaps().values()), abs=1e-9)

    def test_aime_regions_hand_clusters_over_in_gap_order(self, aime):
        """Raising lambda moves clusters to the cheaper model one at a time,
        smallest error gap first, giving one region per cluster plus one."""
        models, _ = aime
        regions = routing_regions(models)
        order = sorted(self._gaps(), key=self._gaps().get)
        assert len(regions) == len(order) + 1
        for i, region in enumerate(regions):
            moved = set(order[:i])
            assert region.assignment == {c: V if c in moved else Q for c in order}
        assert math.isinf(regions[-1].lam_max)


class TestSystemMetrics:
    def test_aime_all_strong_row(self, aime):
        """lambda = 0 sends every cluster to Qwen3-30B."""
        models, sizes = aime
        everything_strong = {c: Q for c in sizes}
        assert assign(models, 0.0) == everything_strong
        acc, tpot = system_metrics(models, everything_strong, sizes)
        expected = _weighted(_raw(TWO_CARD["aime"]), everything_strong)
        assert (acc, tpot) == pytest.approx(expected, rel=1e-12)

    def test_aime_routed_row(self, aime):
        models, sizes = aime
        routed = _spec_assignment(AIME_ROUTING[0])
        acc, tpot = system_metrics(models, routed, sizes)
        assert (acc, tpot) == pytest.approx(_weighted(_raw(TWO_CARD["aime"]), routed), rel=1e-12)

    def test_eta_is_none_for_baseline(self, aime):
        models, sizes = aime
        assert eta(models, assign(models, 0.0), sizes) is None

    def test_eta_is_points_given_up_per_ms_saved(self, aime):
        models, sizes = aime
        raw = _raw(TWO_CARD["aime"])
        routed = _spec_assignment(AIME_ROUTING[0])
        acc0, tpot0 = _weighted(raw, {c: Q for c in sizes})
        acc, tpot = _weighted(raw, routed)
        assert eta(models, routed, sizes) == pytest.approx((acc0 - acc) * 100 / (tpot0 - tpot))


class TestLambdaSelection:
    def test_aime_budget_reproduces_the_shipped_routing(self, aime):
        models, sizes = aime
        spec, budget = AIME_ROUTING
        selection = select_lambda(models, sizes, budget_ms=budget)
        assert selection.region.assignment == _spec_assignment(spec)
        assert selection.region.lam_min <= selection.lambda_star < selection.region.lam_max
        assert selection.tpot_ms <= budget

    def test_infeasible_budget_raises(self, aime):
        models, sizes = aime
        with pytest.raises(ValueError, match="No routing strategy"):
            select_lambda(models, sizes, budget_ms=1.0)


class TestRepresentativeLambda:
    """lambda* only names a region, but it must name its own region: any value
    reported for a region has to reproduce that region's assignment."""

    # Three models over two clusters, chosen so the crossovers land on round
    # numbers (0.1, 0.2, 0.3, 0.5). Measured data rarely does, which is why the
    # naming bug below stayed hidden.
    POOL = [
        ModelStats(name="Small", tpot_ms=10.0, errors={"easy": 0.20, "hard": 0.60}),
        ModelStats(name="Mid", tpot_ms=30.0, errors={"easy": 0.10, "hard": 0.35}),
        ModelStats(name="Big", tpot_ms=50.0, errors={"easy": 0.05, "hard": 0.20}),
    ]

    def test_every_region_is_named_by_a_lambda_that_routes_the_same(self):
        for region in routing_regions(self.POOL):
            lam = region.representative_lambda
            assert region.lam_min <= lam < region.lam_max
            assert assign(self.POOL, lam) == region.assignment

    def test_round_upper_bound_is_excluded(self):
        """[0.1, 0.2) must not be named 0.2: rounding used to push the candidate
        onto the exclusive bound, naming the next region's lambda."""
        assert _nice_lambda(0.1, 0.2) < 0.2

    def test_prefers_the_fewest_decimals_that_fit(self):
        assert _nice_lambda(0.3, 0.5) == 0.4

    @pytest.mark.parametrize(
        "lo, hi, expected",
        [(0.313, 0.467, 0.4), (1.972, 6.157, 6.1)],
    )
    def test_published_telemath_operating_points_are_unchanged(self, lo, hi, expected):
        """The TPOT and E2EL operating points recorded in the results.

        Boundaries are the size-weighted ones; the earlier pair (0.314, 0.456)
        and (2.046, 6.537) came from the unweighted cost scalar and named 6.5
        for the E2EL point rather than 6.1. Source:
        `ref/results/telemath/lambda_sweep_full.md`.
        """
        assert _nice_lambda(lo, hi) == expected


class TestTeleQnA:
    def test_pareto_pruning(self, teleqna):
        """Gemma4-E2B is dominated by Qwen3-4B. Gemma4-E4B survives: it is
        dearer than Qwen3-4B but more accurate on C0 by more than the tolerance."""
        raw = _raw(TWO_CARD["teleqna"])
        tol = raw["error_tol"]
        models, _ = teleqna
        by = {m.name: m for m in models}
        q4, e2b, e4b = by["Qwen3-4B"], by["Gemma4-E2B"], by["Gemma4-E4B"]
        err = {m: s["errors"] for m, s in raw["models"].items()}
        assert q4.tpot_ms < e2b.tpot_ms
        assert all(err["Qwen3-4B"][c] < err["Gemma4-E2B"][c] - tol for c in err["Qwen3-4B"])
        assert err["Gemma4-E4B"]["0"] < err["Qwen3-4B"]["0"] - tol
        assert q4.tpot_ms < e4b.tpot_ms

        efficient, dominated = pareto_prune(models, tol)
        assert [m.name for m in dominated] == ["Gemma4-E2B"]
        assert "Gemma4-E4B" in {m.name for m in efficient}

    def test_budget_reproduces_the_shipped_routing(self, teleqna):
        models, sizes = teleqna
        spec, budget = TELEQNA_ROUTING
        efficient, _ = pareto_prune(models)
        selection = select_lambda(efficient, sizes, budget_ms=budget)
        assert selection.region.assignment == _spec_assignment(spec)
        assert select_lambda(models, sizes, budget_ms=budget).region.assignment == _spec_assignment(spec)

    def test_dominated_models_never_selected(self, teleqna):
        models, _ = teleqna
        _, dominated = pareto_prune(models)
        for region in routing_regions(models):
            assert not {m.name for m in dominated} & set(region.assignment.values())


class TestCascadeSystemMetrics:
    def test_no_escalation_matches_stage1(self):
        """With no escalations the cascade collapses to Stage 1 TPOT exactly."""
        def model(name, tpot):
            return ModelStats(name=name, tpot_ms=tpot, errors={"0": 0.3, "1": 0.2},
                              cluster_tpot_ms={"0": tpot, "1": tpot + 1.0},
                              e2el_ms=tpot * 100, cluster_output_tokens={"0": 40.0, "1": 60.0})
        models = [model("small", 9.0), model("large", 20.0)]
        assignment, sizes = {"0": "small", "1": "large"}, {"0": 590, "1": 410}
        _, stage1_tpot = system_metrics(models, assignment, sizes)
        tpot, _ = cascade_system_metrics(models, assignment, sizes, escalations={})
        assert tpot == pytest.approx(stage1_tpot)


class TestCascadeSystemMetricsNTier:
    """The N-tier generalisation; 2-tier ``cascade_system_metrics`` delegates to
    it, which ``test_two_tier_wrapper_equals_ntier`` checks."""

    def _models(self):
        # per-cluster tpot / e2el / output length for a single cluster "0"
        eff = ModelStats(name="eff", tpot_ms=10.0, errors={"0": 0.5},
                         cluster_tpot_ms={"0": 10.0}, e2el_ms=100.0,
                         cluster_e2el_ms={"0": 100.0}, cluster_output_tokens={"0": 50.0})
        mid = ModelStats(name="mid", tpot_ms=20.0, errors={"0": 0.3},
                         cluster_tpot_ms={"0": 20.0}, e2el_ms=300.0,
                         cluster_e2el_ms={"0": 300.0}, cluster_output_tokens={"0": 100.0})
        strong = ModelStats(name="strong", tpot_ms=30.0, errors={"0": 0.1},
                            cluster_tpot_ms={"0": 30.0}, e2el_ms=600.0,
                            cluster_e2el_ms={"0": 600.0}, cluster_output_tokens={"0": 200.0})
        return [eff, mid, strong]

    def test_three_tier_hand_computed(self):
        # reach [10,4,2]: 10 run eff, 4 escalate to mid, 2 further to strong.
        # E2EL = 10*100 + 4*300 + 2*600 = 3400 -> /10 = 340
        # TPOT: t0 6*(500/50)=60 ; t1 2*(2500/100)=50 ; t2 2*(8500/200)=85 -> 195/10 = 19.5
        cascades = {"0": [("eff", 10), ("mid", 4), ("strong", 2)]}
        tpot, e2el = cascade_system_metrics_ntier(self._models(), cascades, {"0": 10})
        assert e2el == pytest.approx(340.0)
        assert tpot == pytest.approx(19.5)

    def test_single_tier_is_direct_assignment(self):
        cascades = {"0": [("mid", 10)]}
        tpot, e2el = cascade_system_metrics_ntier(self._models(), cascades, {"0": 10})
        assert (tpot, e2el) == pytest.approx((20.0, 300.0))

    def test_two_tier_wrapper_equals_ntier(self):
        models = self._models()
        sizes = {"0": 10}
        direct = cascade_system_metrics_ntier(
            models, {"0": [("eff", 10), ("strong", 4)]}, sizes
        )
        wrapped = cascade_system_metrics(
            models, {"0": "eff"}, sizes, {"0": ("strong", 4)}
        )
        assert wrapped == pytest.approx(direct)

    def test_rejects_increasing_reach(self):
        with pytest.raises(ValueError, match="non-increasing"):
            cascade_system_metrics_ntier(
                self._models(), {"0": [("eff", 10), ("mid", 12)]}, {"0": 10}
            )

    def test_rejects_base_reach_mismatch(self):
        with pytest.raises(ValueError, match="cluster size"):
            cascade_system_metrics_ntier(
                self._models(), {"0": [("eff", 8), ("mid", 4)]}, {"0": 10}
            )


class TestClusterCascadeAccuracy:
    def test_per_query_composition(self):
        # accept -> keep weak; escalate -> take strong. FP (escalated-correct) and
        # FN (accepted-wrong) both handled by taking the actual per-query outcome.
        weak = [True, True, False, False]
        strong = [False, False, True, False]
        escalate = [False, False, True, True]
        # q0,q1 accepted+weak-correct; q2 escalated+strong-correct; q3 escalated+strong-wrong
        assert cluster_cascade_accuracy(weak, strong, escalate) == pytest.approx(3 / 4)

    def test_no_escalation_equals_weak(self):
        weak = [True, False, True]
        assert cluster_cascade_accuracy(weak, [False, False, False], [False, False, False]) == pytest.approx(2 / 3)

    def test_misaligned_lengths_raise(self):
        with pytest.raises(ValueError, match="align"):
            cluster_cascade_accuracy([True], [True, False], [False, False])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cluster_cascade_accuracy([], [], [])


class TestCascadeSystemAccuracy:
    """Stage 1+2 system accuracy: a gated cluster contributes its cascade
    accuracy, every other cluster its routed model's accuracy, weighted by size."""

    WEAK = ModelStats(name="weak", tpot_ms=5.0, errors={"0": 0.30, "1": 0.10, "2": 0.30})
    STRONG = ModelStats(name="strong", tpot_ms=12.0, errors={"0": 0.10, "1": 0.05, "2": 0.20})
    ASSIGNMENT = {"0": "strong", "1": "weak", "2": "strong"}
    SIZES = {"0": 10, "1": 20, "2": 10}

    def test_a_gated_cluster_takes_its_cascade_accuracy(self):
        # 10 x 0.90 + 20 x 0.95 + 10 x 0.80 = 36 of 40
        acc = cascade_system_accuracy([self.WEAK, self.STRONG], self.ASSIGNMENT, self.SIZES,
                                      cascade_accuracy={"1": 0.95})
        assert acc == pytest.approx(36 / 40)

    def test_no_cascade_matches_stage1_accuracy(self):
        stage1_acc, _ = system_metrics([self.WEAK, self.STRONG], self.ASSIGNMENT, self.SIZES)
        acc = cascade_system_accuracy([self.WEAK, self.STRONG], self.ASSIGNMENT, self.SIZES,
                                      cascade_accuracy={})
        assert acc == pytest.approx(stage1_acc)


# ---------------------------------------------------------------------------
# Cost-scalar weighting.
#
# models_from_stats collapses per-cluster cost into the one scalar Eq. 2
# normalises. Every fixture above uses a single cluster or equal-sized ones,
# where weighted and unweighted agree, so none of them can detect a regression
# here. These use deliberately unequal clusters.
# ---------------------------------------------------------------------------

_UNEQUAL = {
    "cluster_sizes": {"0": 90, "1": 10},
    "models": {
        # cheap in the big cluster, dear in the small one: the two conventions
        # disagree by a wide margin
        "skewed": {"errors": {"0": 0.2, "1": 0.2},
                   "cluster_tpot_ms": {"0": 10.0, "1": 100.0},
                   "cluster_e2el_ms": {"0": 1000.0, "1": 9000.0}},
        "flat":   {"errors": {"0": 0.3, "1": 0.3},
                   "cluster_tpot_ms": {"0": 20.0, "1": 20.0},
                   "cluster_e2el_ms": {"0": 2000.0, "1": 2000.0}},
    },
}


def test_cost_scalar_is_size_weighted_by_default():
    models, _ = models_from_stats(_UNEQUAL)
    by = {m.name: m for m in models}
    # (90*10 + 10*100) / 100 = 19.0, against an unweighted (10+100)/2 = 55.0
    assert by["skewed"].tpot_ms == pytest.approx(19.0)
    assert by["flat"].tpot_ms == pytest.approx(20.0)


def test_unweighted_opt_out_reproduces_the_old_scalar():
    models, _ = models_from_stats(_UNEQUAL, cost_weighting="unweighted")
    by = {m.name: m for m in models}
    assert by["skewed"].tpot_ms == pytest.approx(55.0)
    assert by["flat"].tpot_ms == pytest.approx(20.0)


def test_weighting_can_flip_which_model_is_cheaper():
    """The whole point: on unequal clusters the conventions can disagree."""
    w, _ = models_from_stats(_UNEQUAL)
    u, _ = models_from_stats(_UNEQUAL, cost_weighting="unweighted")
    wby = {m.name: m.tpot_ms for m in w}
    uby = {m.name: m.tpot_ms for m in u}
    assert wby["skewed"] < wby["flat"], "size-weighted: skewed is cheaper"
    assert uby["skewed"] > uby["flat"], "unweighted: skewed looks dearer"


def test_e2el_scalar_is_weighted_too():
    models, _ = models_from_stats(_UNEQUAL, cost_metric="e2el")
    by = {m.name: m for m in models}
    # (90*1000 + 10*9000) / 100 = 1800.0, against unweighted 5000.0
    assert by["skewed"].e2el_ms == pytest.approx(1800.0)


def test_equal_clusters_are_unaffected_by_the_convention():
    equal = {
        "cluster_sizes": {"0": 50, "1": 50},
        "models": {"m": {"errors": {"0": 0.1, "1": 0.2},
                         "cluster_tpot_ms": {"0": 10.0, "1": 30.0}}},
    }
    w, _ = models_from_stats(equal)
    u, _ = models_from_stats(equal, cost_weighting="unweighted")
    assert w[0].tpot_ms == pytest.approx(u[0].tpot_ms) == pytest.approx(20.0)


def test_stored_scalar_still_wins_over_per_cluster():
    """A stats file that stores tpot_ms is untouched by either convention."""
    stored = {
        "cluster_sizes": {"0": 90, "1": 10},
        "models": {"m": {"tpot_ms": 42.0, "errors": {"0": 0.1, "1": 0.2},
                         "cluster_tpot_ms": {"0": 10.0, "1": 100.0}}},
    }
    for weighting in ("size", "unweighted"):
        models, _ = models_from_stats(stored, cost_weighting=weighting)
        assert models[0].tpot_ms == pytest.approx(42.0)


class TestSelectionMargins:
    """`selection_margins` reports on the choice `assign` makes; it never alters it."""

    POOL = [
        ModelStats(name="cheap", tpot_ms=10.0, errors={"0": 0.30, "1": 0.30}),
        ModelStats(name="dear", tpot_ms=20.0, errors={"0": 0.10, "1": 0.295}),
    ]
    SIZES = {"0": 100.0, "1": 100.0}

    def test_it_agrees_with_assign(self):
        for lam in (0.0, 0.05, 0.15, 0.3):
            chosen = {c: m.chosen for c, m in
                      selection_margins(self.POOL, lam, self.SIZES).items()}
            assert chosen == assign(self.POOL, lam)

    def test_a_wide_margin_is_resolvable(self):
        """C0: 0.20 of error separates the pair, far above 1/100."""
        m = selection_margins(self.POOL, 0.0, self.SIZES)["0"]
        assert m.chosen == "dear" and m.resolvable
        assert m.ratio > 10

    def test_a_margin_below_one_question_is_not(self):
        """C1: the pair differ by 0.005, half the 1/100 a cluster resolves.

        Deliberately above DEFAULT_ERROR_TOL, so this exercises the diagnostic
        rather than the tie-break: a margin the data cannot support, which the
        tolerance is nevertheless too small to treat as a tie.
        """
        m = selection_margins(self.POOL, 0.0, self.SIZES)["1"]
        assert m.chosen == "dear" and m.runner_up == "cheap"
        assert m.gap == pytest.approx(0.005)
        assert not m.resolvable
        assert m.ratio == pytest.approx(0.5)

    def test_granularity_follows_cluster_size(self):
        small = selection_margins(self.POOL, 0.0, {"0": 10.0, "1": 10.0})["1"]
        large = selection_margins(self.POOL, 0.0, {"0": 1000.0, "1": 1000.0})["1"]
        assert small.granularity == pytest.approx(0.1)
        assert large.granularity == pytest.approx(0.001)
        assert not small.resolvable and large.resolvable

    def test_it_still_agrees_when_the_tolerance_fires(self):
        """A gap inside the tolerance flips `assign` to the cheaper model, and
        the diagnostic must follow rather than report the score winner."""
        pool = [
            ModelStats(name="cheap", tpot_ms=10.0, errors={"0": 0.231}),
            ModelStats(name="dear", tpot_ms=20.0, errors={"0": 0.230}),
        ]
        assert assign(pool, 0.0) == {"0": "cheap"}
        assert assign(pool, 0.0, error_tol=0.0) == {"0": "dear"}
        m = selection_margins(pool, 0.0, {"0": 100.0})["0"]
        assert m.chosen == "cheap" and m.runner_up == "dear"
        assert not m.resolvable

    def test_a_missing_cluster_size_is_never_called_resolvable(self):
        m = selection_margins(self.POOL, 0.0, {})["1"]
        assert math.isinf(m.granularity) and not m.resolvable


class TestErrorTolerance:
    """`error_tol` treats near-equal per-cluster errors as indistinguishable and
    lets cost decide. It never overrides a difference larger than itself."""

    @staticmethod
    def _pair(e_cheap: float, e_dear: float) -> list[ModelStats]:
        return [
            ModelStats(name="cheap", tpot_ms=10.0, errors={"0": e_cheap}),
            ModelStats(name="dear", tpot_ms=20.0, errors={"0": e_dear}),
        ]

    def test_a_gap_inside_the_tolerance_falls_to_cost(self):
        pool = self._pair(0.231, 0.230)
        assert assign(pool, 0.0, error_tol=0.001) == {"0": "cheap"}
        assert assign(pool, 0.0, error_tol=0.0) == {"0": "dear"}

    def test_the_float_boundary_is_guarded(self):
        """0.231 - 0.230 evaluates to 1.0000000000000009e-3, so a bare `<=`
        would refuse a gap that is 0.001 by construction. Four decades of it."""
        for a, b in ((0.231, 0.230), (0.331, 0.330), (0.431, 0.430), (0.531, 0.530)):
            assert (a - b) > 0.001, "the premise: naive comparison fails here"
            assert assign(self._pair(a, b), 0.0, error_tol=0.001) == {"0": "cheap"}

    def test_a_gap_outside_the_tolerance_is_respected(self):
        pool = self._pair(0.240, 0.230)
        assert assign(pool, 0.0, error_tol=0.001) == {"0": "dear"}

    def test_it_only_ever_moves_the_choice_towards_the_cheaper_model(self):
        pool = self._pair(0.230, 0.231)          # cheap is also more accurate
        assert assign(pool, 0.0, error_tol=0.001) == {"0": "cheap"}
        assert assign(pool, 0.0, error_tol=0.0) == {"0": "cheap"}

    def test_the_tolerated_set_is_anchored_not_chained(self):
        """A chain of within-tolerance steps must not walk the choice away from
        the argmin: `far` is 0.001 from `mid` but 0.002 from the best error."""
        pool = [
            ModelStats(name="far", tpot_ms=1.0, errors={"0": 0.232}),
            ModelStats(name="mid", tpot_ms=10.0, errors={"0": 0.231}),
            ModelStats(name="best", tpot_ms=20.0, errors={"0": 0.230}),
        ]
        assert assign(pool, 0.0, error_tol=0.001) == {"0": "mid"}

    def test_it_introduces_no_new_region_boundaries(self):
        """Errors do not depend on lambda, so the tolerated set can only change
        where the argmin already changes: every tolerant boundary is one the
        exact sweep already had. It can still *remove* one, by merging away a
        region that existed only to hold a within-tolerance preference.
        """
        pool = [
            ModelStats(name="cheap", tpot_ms=10.0, errors={"0": 0.30, "1": 0.231}),
            ModelStats(name="dear", tpot_ms=20.0, errors={"0": 0.10, "1": 0.230}),
        ]
        exact = [r.lam_min for r in routing_regions(pool, error_tol=0.0)]
        tolerant = [r.lam_min for r in routing_regions(pool, error_tol=0.001)]
        assert set(tolerant) <= set(exact)
        # here it does remove one: the exact sweep opens with a [0, 0.001)
        # region where `dear` takes C1 on a 0.001 error advantage
        assert exact == [0.0, 0.001, 0.2] and tolerant == [0.0, 0.2]

    def test_domination_is_never_mutual(self):
        """The tolerance applies to both halves of the test, so two models
        within it of each other cannot each dominate the other."""
        a = ModelStats(name="a", tpot_ms=10.0, errors={"0": 0.231})
        b = ModelStats(name="b", tpot_ms=10.0, errors={"0": 0.230})
        assert not (dominates(a, b, 0.001) and dominates(b, a, 0.001))

    def test_a_within_tolerance_advantage_does_not_rescue_a_dominated_model(self):
        cheap_good = ModelStats(name="cheap_good", tpot_ms=10.0, errors={"0": 0.231})
        dear_equal = ModelStats(name="dear_equal", tpot_ms=20.0, errors={"0": 0.230})
        assert dominates(cheap_good, dear_equal, 0.001)
        assert not dominates(cheap_good, dear_equal, 0.0)

    def test_the_default_is_the_published_constant(self):
        assert DEFAULT_ERROR_TOL == 0.001
        pool = self._pair(0.231, 0.230)
        assert assign(pool, 0.0) == assign(pool, 0.0, error_tol=DEFAULT_ERROR_TOL)

    def test_it_is_read_from_config_and_validated(self):
        assert error_tol_from_stats({}) == DEFAULT_ERROR_TOL
        assert error_tol_from_stats({"error_tol": None}) == DEFAULT_ERROR_TOL
        assert error_tol_from_stats({"error_tol": 0.0}) == 0.0
        assert error_tol_from_stats({"error_tol": 0.004}) == 0.004
        with pytest.raises(ValueError):
            error_tol_from_stats({"error_tol": -0.001})

    def test_the_shipped_configs_are_unchanged_by_the_tolerance(self):
        """The published claim: on every pool in this repo the tolerance selects
        exactly what an exact comparison selects."""
        root = Path(__file__).resolve().parents[1]
        for name in (*TWO_CARD.values(), *ONE_CARD.values()):
            stats = json.loads((root / "configs" / name).read_text())
            assert stats["error_tol"] == DEFAULT_ERROR_TOL, name
            for metric in ("tpot", "e2el"):
                try:
                    models, sizes = models_from_stats(stats, metric)
                except (KeyError, ValueError):
                    continue
                exact_eff, _ = pareto_prune(models, 0.0)
                tol_eff, _ = pareto_prune(models, DEFAULT_ERROR_TOL)
                assert [m.name for m in exact_eff] == [m.name for m in tol_eff], (name, metric)
                assert ([(r.lam_min, r.assignment) for r in routing_regions(models, 0.0)]
                        == [(r.lam_min, r.assignment)
                            for r in routing_regions(models, DEFAULT_ERROR_TOL)]), (name, metric)


class TestOneCardBasis:
    """The 1 x A100 measurements, which are a different pool.

    Same models, same questions, same answers: only the serving configuration
    changes, and it changes enough that a test written for one basis says
    nothing about the other. Pinning both is the point -- it is what stops a
    config being swapped underneath a claim.
    """

    def test_aime_crossovers_are_the_error_gaps(self, aime_1x):
        models, _ = aime_1x
        # with two models min-max sends the pair to {0, 1}, so each crossover is
        # the error gap and the cost term cancels
        by = {m.name: m for m in models}
        for cluster, expected in (("1", 0.0607), ("0", 0.0742), ("2", 0.1031)):
            gap = by[V].errors[cluster] - by[Q].errors[cluster]
            assert gap == pytest.approx(expected, abs=5e-4), cluster

    def test_aime_budget_30ms_selects_the_reported_routing(self, aime_1x):
        models, sizes = aime_1x
        chosen = None
        for region in routing_regions(models):
            _, tpot = system_metrics(models, region.assignment, sizes)
            if tpot <= 30.0:
                chosen = region
                break
        assert chosen is not None
        assert chosen.assignment == {"0": Q, "1": V, "2": Q}
        assert chosen.lam_min == pytest.approx(0.0607, abs=5e-4)

    def test_nothing_is_pruned_on_one_card(self, teleqna_1x):
        """The sharpest difference between the two bases.

        On two cards Gemma4-E2B is dominated and the sweep has three regions. On one card the whole pool is on the frontier and it has
        six, so a config without its basis in the name cannot be read safely.
        """
        models, _ = teleqna_1x
        kept, dropped = pareto_prune(models)
        assert [m.name for m in dropped] == []
        assert len(kept) == 4
        assert len(routing_regions(models)) == 6

    def test_the_two_bases_really_do_disagree(self, teleqna, teleqna_1x):
        two_card, _ = teleqna
        one_card, _ = teleqna_1x
        _, dropped_2x = pareto_prune(two_card)
        _, dropped_1x = pareto_prune(one_card)
        assert {m.name for m in dropped_2x} == {"Gemma4-E2B"}
        assert len(routing_regions(two_card)) == 3
        assert not dropped_1x


class TestTeleMath:
    """The TeleMath operating points `REPRODUCE.md` quotes, on the reported basis.

    Nine models, k = 4 clusters, 1 x A100 at concurrency 32 under vLLM 0.19.0.
    Every value below is printed by `cre fit` on the shipped config, and each
    budget is the one the paper reports under that cost term.
    """

    CONFIG = "telemath_stats_1xA100_Sep2026.json"
    Q30 = "Qwen3-30B-A3B-Thinking-2507"
    E2B = "Gemma4-E2B"
    G26 = "Gemma4-26B-A4B"

    def _fit(self, metric, budget, rule="model"):
        stats = json.loads((CONFIGS / self.CONFIG).read_text())
        models, sizes = models_from_stats(stats, metric)
        return select_lambda(models, sizes, budget, error_tol_from_stats(stats), rule)

    @pytest.mark.parametrize("metric, budget, rule, route, acc", [
        ("tpot", 20.0, "model", (Q30, E2B, Q30, "Gemma4-E4B-think"), 0.647),
        ("tpot", 25.0, "model", (Q30, E2B, Q30, "Gemma4-26B-A4B-think"), 0.691),
        ("e2el", 25000.0, "model", (G26, E2B, G26, G26), 0.650),
        ("tpot", 20.0, "cluster", (Q30, E2B, G26, Q30), 0.672),
    ])
    def test_reported_operating_points(self, metric, budget, rule, route, acc):
        sel = self._fit(metric, budget, rule)
        assert tuple(sel.region.assignment[c] for c in "0123") == route
        assert sel.accuracy == pytest.approx(acc, abs=5e-4)

    def test_cluster_sizes(self):
        stats = json.loads((CONFIGS / self.CONFIG).read_text())
        assert stats["cluster_sizes"] == {"0": 98, "1": 98, "2": 51, "3": 52}

    def test_one_model_is_dominated_under_tpot(self):
        stats = json.loads((CONFIGS / self.CONFIG).read_text())
        models, _ = models_from_stats(stats, "tpot")
        _, dropped = pareto_prune(models)
        assert [m.name for m in dropped] == [self.G26]


class TestTeleMathTwoCards(TestTeleMath):
    """The same pool on 2 x A100, tensor parallel 2, fitted on its own costs.

    At the one-card budgets the two-card pool can afford far stronger models:
    under TPOT the most accurate routing already costs less than 20 ms, and under
    E2EL the budget selects Gemma4-26B for every cluster.
    """

    CONFIG = "telemath_stats_2xA100_Sep2026.json"
    G26T = "Gemma4-26B-A4B-think"

    @pytest.mark.parametrize("metric, budget, rule, route, acc", [
        ("tpot", 20.0, "model", (TestTeleMath.Q30, TestTeleMath.Q30,
                                 "Gemma4-26B-A4B-think", "Gemma4-26B-A4B-think"), 0.740),
        ("tpot", 20.0, "cluster", (TestTeleMath.Q30, TestTeleMath.Q30,
                                   "Gemma4-26B-A4B-think", "Gemma4-26B-A4B-think"), 0.740),
        ("e2el", 25000.0, "model", (TestTeleMath.G26,) * 4, 0.660),
    ])
    def test_reported_operating_points(self, metric, budget, rule, route, acc):
        super().test_reported_operating_points(metric, budget, rule, route, acc)

    def test_one_model_is_dominated_under_tpot(self):
        stats = json.loads((CONFIGS / self.CONFIG).read_text())
        models, _ = models_from_stats(stats, "tpot")
        _, dropped = pareto_prune(models)
        assert [m.name for m in dropped] == ["Gemma4-E4B"]


class TestTwoCardsAimeTeleQnA:
    """AIME and TeleQnA on 2 x A100, each fitted on its own two-card costs.

    The one-card budgets stop binding on two cards (30 ms buys Qwen3-30B on
    every AIME cluster, 20 ms buys Gemma4-26B on both TeleQnA clusters), so
    the two-card systems use their own budgets, 20 ms and 15 ms.
    """

    def _fit(self, config, budget):
        stats = json.loads((CONFIGS / config).read_text())
        models, sizes = models_from_stats(stats, "tpot")
        return select_lambda(models, sizes, budget, error_tol_from_stats(stats))

    def test_aime_20ms(self):
        sel = self._fit("aime_stats_2xA100_Sep2026.json", 20.0)
        assert sel.region.assignment == {"0": Q, "1": V, "2": Q}
        assert sel.region.lam_min == pytest.approx(0.057, abs=5e-4)
        assert sel.accuracy == pytest.approx(0.918, abs=5e-4)

    def test_aime_30ms_is_always_strong(self):
        sel = self._fit("aime_stats_2xA100_Sep2026.json", 30.0)
        assert set(sel.region.assignment.values()) == {Q}

    def test_teleqna_15ms(self):
        sel = self._fit("teleqna_stats_2xA100_Sep2026.json", 15.0)
        assert sel.region.assignment == {"0": "Qwen3-4B", "1": "Gemma4-26B"}
        assert sel.accuracy == pytest.approx(0.719, abs=5e-4)

    def test_teleqna_20ms_is_always_strong(self):
        sel = self._fit("teleqna_stats_2xA100_Sep2026.json", 20.0)
        assert set(sel.region.assignment.values()) == {"Gemma4-26B"}

    def test_teleqna_prunes_e2b_only(self):
        stats = json.loads((CONFIGS / "teleqna_stats_2xA100_Sep2026.json").read_text())
        models, _ = models_from_stats(stats, "tpot")
        _, dropped = pareto_prune(models)
        assert [m.name for m in dropped] == ["Gemma4-E2B"]
