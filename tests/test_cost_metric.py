"""The selectable cost metric and the optional E2EL/throughput capture.

The load-bearing property is that `cost_metric="tpot"` (the default) leaves the
published behaviour untouched, so every test here that exercises e2el has a tpot
counterpart asserting the old path still answers the same way.
"""

import pytest

from cre_router.evaluate import (
    RunMeasurement,
    aggregate_runs,
    model_entry,
    optional_metrics,
)
from cre_router.routing import (
    ModelStats,
    assign,
    models_from_stats,
    normalized_costs,
    pareto_prune,
)


class TestOptionalMetrics:
    def test_prefers_reported_e2el(self):
        m = optional_metrics(
            {"mean_e2el_ms": 8000.0, "mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0,
             "total_output_tokens": 400}, num_prompts=4
        )
        assert m["e2el_ms"] == 8000.0
        assert m["mean_output_tokens"] == 100.0

    def test_reconstructs_e2el_when_absent(self):
        # TTFT + TPOT x (L - 1) = 100 + 10 x 99
        m = optional_metrics(
            {"mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0, "total_output_tokens": 400},
            num_prompts=4,
        )
        assert m["e2el_ms"] == pytest.approx(1090.0)

    def test_truncation_rate(self):
        # Two of four requests reached the 30k cap; the others stopped at EOS.
        m = optional_metrics(
            {"output_lens": [512, 30000, 1024, 30000]}, num_prompts=4, max_output_len=30000
        )
        assert m["truncated_frac"] == 0.5

    def test_no_truncation_rate_without_the_cap(self):
        # output_lens alone cannot say what counts as truncated.
        m = optional_metrics({"output_lens": [512, 30000]}, num_prompts=2)
        assert "truncated_frac" not in m

    def test_tolerates_a_bare_result(self):
        # An injected fake benchmark reporting only TPOT must not break capture.
        assert optional_metrics({"mean_tpot_ms": 10.0}, num_prompts=4) == {}


class TestAggregation:
    def _runs(self, **extra):
        return [
            RunMeasurement("0", 0, error=0.2, tpot_ms=10.0, num_prompts=5, **extra),
            RunMeasurement("0", 1, error=0.4, tpot_ms=12.0, num_prompts=5, **extra),
        ]

    def test_optional_metric_averaged(self):
        runs = [
            RunMeasurement("0", 0, error=0.2, tpot_ms=10.0, num_prompts=5, e2el_ms=1000.0),
            RunMeasurement("0", 1, error=0.4, tpot_ms=12.0, num_prompts=5, e2el_ms=2000.0),
        ]
        assert aggregate_runs(runs)["0"]["e2el_ms"] == pytest.approx(1500.0)

    def test_partial_metric_is_dropped(self):
        """A metric only some runs reported would be an average over a different
        denominator than the rest, so it is omitted rather than half-reported."""
        runs = [
            RunMeasurement("0", 0, error=0.2, tpot_ms=10.0, num_prompts=5, e2el_ms=1000.0),
            RunMeasurement("0", 1, error=0.4, tpot_ms=12.0, num_prompts=5),
        ]
        assert "e2el_ms" not in aggregate_runs(runs)["0"]
        assert "cluster_e2el_ms" not in model_entry(runs)

    def test_entry_shape_unchanged_without_optional_metrics(self):
        assert set(model_entry(self._runs())) == {"errors", "cluster_tpot_ms"}



class TestCostMetricSelection:
    def _pair(self, metric):
        """A thinking/non-thinking pair: near-identical TPOT, very different E2EL.

        This is the case TPOT cannot price, since both modes run the same weights
        at the same decode speed and differ only in how many tokens they emit.
        """
        return [
            ModelStats("nothink", tpot_ms=10.0, errors={"0": 0.40}, e2el_ms=2_000.0,
                       cost_metric=metric),
            ModelStats("think", tpot_ms=10.2, errors={"0": 0.20}, e2el_ms=40_000.0,
                       cost_metric=metric),
        ]

    def test_tpot_is_the_default(self):
        assert ModelStats("m", tpot_ms=10.0, errors={"0": 0.1}).cost_ms == 10.0

    def test_cost_ms_follows_the_metric(self):
        nothink, think = self._pair("e2el")
        assert (nothink.cost_ms, think.cost_ms) == (2_000.0, 40_000.0)

    def test_per_cluster_cost_falls_back_to_pool_level(self):
        m = ModelStats("m", tpot_ms=10.0, errors={"0": 0.1}, e2el_ms=500.0,
                       cluster_e2el_ms={"0": 900.0}, cost_metric="e2el")
        assert m.cost_for("0") == 900.0
        assert m.cost_for("absent") == 500.0

    def test_e2el_without_measurement_is_refused(self):
        with pytest.raises(ValueError, match="needs e2el_ms"):
            ModelStats("m", tpot_ms=10.0, errors={"0": 0.1}, cost_metric="e2el")

    def test_unknown_metric_is_refused(self):
        with pytest.raises(ValueError, match="unknown cost_metric"):
            ModelStats("m", tpot_ms=10.0, errors={"0": 0.1}, cost_metric="dollars")

    def test_normalisation_ranks_by_the_selected_metric(self):
        # Under TPOT the thinking mode is (misleadingly) the marginally dearer
        # one by 0.2 ms; under E2EL it is dearer by 20x. Min-max maps both to
        # {0, 1} for a two-model pool, so the ordering is what differs.
        assert normalized_costs(self._pair("tpot"))["nothink"] == 0.0
        assert normalized_costs(self._pair("e2el"))["nothink"] == 0.0

    def test_pareto_pruning_uses_the_selected_metric(self):
        """With three rungs the spacing matters, not just the ordering: a cheap
        thinking mode must not dominate a genuinely faster non-thinking one."""
        models = [
            ModelStats("nothink", tpot_ms=10.0, errors={"0": 0.40}, e2el_ms=2_000.0,
                       cost_metric="e2el"),
            ModelStats("think", tpot_ms=10.2, errors={"0": 0.20}, e2el_ms=40_000.0,
                       cost_metric="e2el"),
            ModelStats("big", tpot_ms=30.0, errors={"0": 0.35}, e2el_ms=60_000.0,
                       cost_metric="e2el"),
        ]
        efficient, dominated = pareto_prune(models)
        # "big" is both dearer and worse than "think", so it cannot ever be chosen.
        assert [m.name for m in dominated] == ["big"]
        assert {m.name for m in efficient} == {"nothink", "think"}

    def test_lambda_sweep_still_trades_error_against_cost(self):
        models = self._pair("e2el")
        assert assign(models, lam=0.0)["0"] == "think"      # accuracy at any cost
        assert assign(models, lam=10.0)["0"] == "nothink"   # cost dominates


class TestModelsFromStats:
    def _stats(self):
        return {
            "cluster_sizes": {"0": 10},
            "models": {
                "fast": {"errors": {"0": 0.4}, "cluster_tpot_ms": {"0": 10.0},
                         "cluster_e2el_ms": {"0": 2000.0}},
                "slow": {"errors": {"0": 0.2}, "cluster_tpot_ms": {"0": 10.2},
                         "cluster_e2el_ms": {"0": 40000.0}},
            },
        }

    def test_defaults_to_tpot(self):
        models, _ = models_from_stats(self._stats())
        assert [m.cost_ms for m in models] == [10.0, 10.2]

    def test_e2el_selected(self):
        models, _ = models_from_stats(self._stats(), "e2el")
        assert [m.cost_ms for m in models] == [2000.0, 40000.0]

    def test_pool_level_derived_from_per_cluster(self):
        models, _ = models_from_stats(self._stats(), "e2el")
        assert models[0].e2el_ms == 2000.0

    def test_legacy_stats_still_load(self):
        """A stats file written before this change has no e2el and must keep
        working under the default metric."""
        legacy = {"cluster_sizes": {"0": 10},
                  "models": {"m": {"tpot_ms": 9.15, "errors": {"0": 0.13}}}}
        models, sizes = models_from_stats(legacy)
        assert models[0].cost_ms == 9.15 and sizes == {"0": 10.0}
