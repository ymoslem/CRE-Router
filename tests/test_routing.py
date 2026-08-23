"""Routing math verified against the numbers published in the paper.

AIME fixtures come from Table `aime_training` (per-cluster error and TPOT on
AIME 1983-2023); expected outputs come from Sec. 6.1 and Table `aime_lambda`.
TeleQnA fixtures come from Table `teleqna_training`.
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


@pytest.fixture()
def aime():
    stats = json.loads((CONFIGS / "aime_stats.json").read_text())
    return models_from_stats(stats)


@pytest.fixture()
def teleqna():
    stats = json.loads((CONFIGS / "teleqna_stats.json").read_text())
    return models_from_stats(stats)


class TestNormalizedCosts:
    def test_two_model_pool_spans_zero_to_one(self, aime):
        models, _ = aime
        costs = normalized_costs(models)
        assert costs[V] == 0.0
        assert costs[Q] == 1.0


class TestCrossovers:
    def test_aime_closed_form(self, aime):
        """Paper Sec. 6.1: lambda_0=0.067, lambda_1=0.052, lambda_2=0.099."""
        models, _ = aime
        assert crossover_candidates(models) == pytest.approx([0.052, 0.067, 0.099], abs=1e-9)

    def test_aime_four_regions_match_table(self, aime):
        """Paper Table `aime_lambda`: assignments per routing region."""
        models, _ = aime
        regions = routing_regions(models)
        assert len(regions) == 4
        assert regions[0].assignment == {"0": Q, "1": Q, "2": Q}
        assert regions[1].assignment == {"0": Q, "1": V, "2": Q}
        assert regions[2].assignment == {"0": V, "1": V, "2": Q}
        assert regions[3].assignment == {"0": V, "1": V, "2": V}
        assert math.isinf(regions[3].lam_max)


class TestSystemMetrics:
    def test_aime_baseline_row(self, aime):
        """Table `aime_lambda`, lambda=0: 94.4% accuracy at 24.8 ms."""
        models, sizes = aime
        acc, tpot = system_metrics(models, assign(models, 0.0), sizes)
        assert acc * 100 == pytest.approx(94.4, abs=0.05)
        assert tpot == pytest.approx(24.8, abs=0.05)

    def test_aime_lambda_star_row(self, aime):
        """Table `aime_lambda`, lambda=0.06: 92.1% accuracy at 18.4 ms."""
        models, sizes = aime
        acc, tpot = system_metrics(models, assign(models, 0.06), sizes)
        assert acc * 100 == pytest.approx(92.1, abs=0.05)
        assert tpot == pytest.approx(18.4, abs=0.06)

    def test_eta_is_none_for_baseline(self, aime):
        models, sizes = aime
        assert eta(models, assign(models, 0.0), sizes) is None

    def test_eta_at_lambda_star(self, aime):
        """The paper reports eta=0.36 from table-rounded accuracy and TPOT;
        the unrounded computation gives 0.354."""
        models, sizes = aime
        assert eta(models, assign(models, 0.06), sizes) == pytest.approx(0.354, abs=0.005)


class TestLambdaSelection:
    def test_aime_budget_20ms_selects_006(self, aime):
        """Paper Sec. 6.1: B=20 ms selects lambda*=0.06, C1 to VibeThinker."""
        models, sizes = aime
        selection = select_lambda(models, sizes, budget_ms=20.0)
        assert selection.lambda_star == pytest.approx(0.06)
        assert selection.region.assignment == {"0": Q, "1": V, "2": Q}
        assert selection.tpot_ms <= 20.0

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
    def test_pareto_pruning_matches_table(self, teleqna):
        """Table `teleqna_training`: G-E2B and G-E4B are dominated."""
        models, _ = teleqna
        efficient, dominated = pareto_prune(models)
        assert sorted(m.name for m in dominated) == ["Gemma4-E2B", "Gemma4-E4B"]
        assert sorted(m.name for m in efficient) == ["Gemma4-26B", "Qwen3-4B"]

    def test_surviving_pool_crossovers(self, teleqna):
        """K=2 closed form on the surviving pool. C0 crossover 0.066,
        C1 crossover 0.075, hence lambda*=0.07 selects Q3-4B/G-26B."""
        models, sizes = teleqna
        efficient, _ = pareto_prune(models)
        assert crossover_candidates(efficient) == pytest.approx([0.066, 0.075], abs=1e-9)

        selection = select_lambda(efficient, sizes, budget_ms=20.0)
        assert selection.lambda_star == pytest.approx(0.07)
        assert selection.region.assignment == {"0": "Qwen3-4B", "1": "Gemma4-26B"}

    def test_dominated_models_never_selected(self, teleqna):
        models, sizes = teleqna
        for region in routing_regions(models):
            chosen = set(region.assignment.values())
            assert "Gemma4-E2B" not in chosen
            assert "Gemma4-E4B" not in chosen


def _load_cascade(name: str):
    """Load a checked-in cascade config the way ``cre cascade`` does."""
    stats = json.loads((CONFIGS / name).read_text())
    models, cluster_sizes = models_from_stats(stats)
    assignment = {str(k): str(v) for k, v in stats["assignment"].items()}
    escalations = {
        str(k): (str(v[0]), float(v[1])) for k, v in stats.get("escalations", {}).items()
    }
    return models, assignment, cluster_sizes, escalations


class TestCascadeSystemMetrics:
    """Stage 1+2 system latency from the checked-in test-split cascade configs,
    verified against the paper's Tables `aime_test` and `teleqna_test`. The
    composed values are 9.75 / 23.65 ms; the paper reports 9.7 / 23.8 ms."""

    def test_aime_stage1plus2_latency(self):
        models, assignment, sizes, escalations = _load_cascade("aime_cascade_test.json")
        tpot, e2el = cascade_system_metrics(models, assignment, sizes, escalations)
        assert tpot == pytest.approx(9.7, abs=0.1)   # paper Table aime_test
        assert e2el == pytest.approx(156300, rel=1e-3)

    def test_teleqna_stage1plus2_latency(self):
        models, assignment, sizes, escalations = _load_cascade("teleqna_cascade_test.json")
        tpot, e2el = cascade_system_metrics(models, assignment, sizes, escalations)
        assert tpot == pytest.approx(23.8, abs=0.2)  # paper Table teleqna_test
        assert e2el == pytest.approx(1127, rel=1e-3)

    def test_no_escalation_matches_stage1(self):
        """With no escalations the cascade collapses to Stage 1 TPOT exactly."""
        models, assignment, sizes, _ = _load_cascade("teleqna_cascade_test.json")
        _, stage1_tpot = system_metrics(models, assignment, sizes)
        tpot, _ = cascade_system_metrics(models, assignment, sizes, escalations={})
        assert tpot == pytest.approx(stage1_tpot)


class TestCascadeSystemMetricsNTier:
    """The N-tier generalisation; 2-tier ``cascade_system_metrics`` delegates to
    it, so the AIME/TeleQnA tests above are the N=2 regression."""

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
    """Stage 1+2 system accuracy composition, verified against the paper's
    combined-cascade slides (AIME 88.4%, TeleQnA 74.3%)."""

    def test_aime_stage1plus2_accuracy(self):
        # AIME test: C0->Q3, C1->V (cascade to Q3), C2->Q3. Q3 test errors give
        # 0.867/0.980/0.829; C1's cascade accuracy is 0.96.
        v = ModelStats(name=V, tpot_ms=4.8, errors={"0": 0.311, "1": 0.100, "2": 0.291})
        q = ModelStats(name=Q, tpot_ms=11.8, errors={"0": 0.133, "1": 0.020, "2": 0.171})
        assignment = {"0": Q, "1": V, "2": Q}
        sizes = {"0": 9, "1": 10, "2": 11}
        acc = cascade_system_accuracy([v, q], assignment, sizes, cascade_accuracy={"1": 0.96})
        assert acc == pytest.approx(0.884, abs=0.001)

    def test_teleqna_stage1plus2_accuracy(self):
        # TeleQnA test: C0->Q-4B (cascade to G-26B, acc 0.740), C1->G-26B direct.
        q4b = ModelStats(name="Q-4B", tpot_ms=15.1, errors={"0": 0.311, "1": 0.360})
        g26 = ModelStats(name="G-26B", tpot_ms=24.5, errors={"0": 0.223, "1": 0.254})
        assignment = {"0": "Q-4B", "1": "G-26B"}
        sizes = {"0": 590, "1": 410}
        acc = cascade_system_accuracy([q4b, g26], assignment, sizes, cascade_accuracy={"0": 0.740})
        assert acc == pytest.approx(0.743, abs=0.001)

    def test_no_cascade_matches_stage1_accuracy(self):
        # With no escalated clusters, system accuracy == Stage 1 accuracy.
        q4b = ModelStats(name="Q-4B", tpot_ms=15.1, errors={"0": 0.311, "1": 0.360})
        g26 = ModelStats(name="G-26B", tpot_ms=24.5, errors={"0": 0.223, "1": 0.254})
        assignment = {"0": "Q-4B", "1": "G-26B"}
        sizes = {"0": 590, "1": 410}
        stage1_acc, _ = system_metrics([q4b, g26], assignment, sizes)
        acc = cascade_system_accuracy([q4b, g26], assignment, sizes, cascade_accuracy={})
        assert acc == pytest.approx(stage1_acc)

    def _cascade_acc_from_config(self, name: str) -> float:
        stats = json.loads((CONFIGS / name).read_text())
        models, sizes = models_from_stats(stats)
        assignment = {str(k): str(v) for k, v in stats["assignment"].items()}
        cascade_accuracy = {str(k): float(v) for k, v in stats.get("cascade_accuracy", {}).items()}
        return cascade_system_accuracy(models, assignment, sizes, cascade_accuracy)

    def test_aime_config_reproduces_884(self):
        assert self._cascade_acc_from_config("aime_cascade_test.json") == pytest.approx(0.884, abs=0.001)

    def test_teleqna_config_reproduces_743(self):
        assert self._cascade_acc_from_config("teleqna_cascade_test.json") == pytest.approx(0.743, abs=0.001)


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
        for name in ("aime_stats.json", "teleqna_stats.json"):
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
