"""Selectable cost conditioning: whether Eq. 2 prices a model or a (model, cluster).

The load-bearing property is that ``cost_conditioning="model"`` (the default) is the
published rule and leaves every existing result untouched, so ``"cluster"`` is an
option rather than a replacement. The second property is a theorem rather than a
measurement: tightening the rule can only enlarge the surviving pool.
"""

import random

import pytest

from cre_router.routing import (
    DEFAULT_COST_CONDITIONING,
    ModelStats,
    assign,
    cost_table,
    dominates,
    models_from_stats,
    normalized_costs,
    pareto_prune,
    routing_regions,
)


def stats(models, sizes=None):
    return {"cluster_sizes": sizes or {"0": 1, "1": 1}, "models": models}


# A pool with a genuine cost inversion: "swing" is cheaper on average than
# "flat" (3 against 4) yet dearer in cluster 1 (5 against 4).
INVERTED = stats({
    "swing": {"errors": {"0": 0.40, "1": 0.40}, "cluster_tpot_ms": {"0": 1.0, "1": 5.0}},
    "flat": {"errors": {"0": 0.40, "1": 0.40}, "cluster_tpot_ms": {"0": 4.0, "1": 4.0}},
})

# The same pool with no inversion: "cheap" is cheaper in every cluster.
NO_INVERSION = stats({
    "cheap": {"errors": {"0": 0.50, "1": 0.55}, "cluster_tpot_ms": {"0": 1.0, "1": 2.0}},
    "dear": {"errors": {"0": 0.30, "1": 0.32}, "cluster_tpot_ms": {"0": 8.0, "1": 9.0}},
})


class TestDefaultIsUnchanged:
    def test_default_is_per_model(self):
        assert DEFAULT_COST_CONDITIONING == "model"

    def test_matches_normalized_costs(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        flat = normalized_costs(models)
        table = cost_table(models)
        for m in models:
            assert set(table[m.name].values()) == {flat[m.name]}

    def test_unknown_conditioning_is_refused(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        with pytest.raises(ValueError, match="unknown cost_conditioning"):
            cost_table(models, "per-query")


class TestPerClusterNeedsMeasurements:
    def test_missing_per_cluster_cost_raises(self):
        # cost_for would silently fall back to the pool scalar, which would look
        # like per-cluster routing while being per-model.
        models = [ModelStats(name="only-scalar", tpot_ms=5.0, errors={"0": 0.4, "1": 0.4})]
        with pytest.raises(ValueError, match="needs a measured cost"):
            cost_table(models, "cluster")

    def test_the_message_names_the_gap(self):
        models = [
            ModelStats(name="partial", tpot_ms=5.0, errors={"0": 0.4, "1": 0.4},
                       cluster_tpot_ms={"0": 5.0}),
        ]
        with pytest.raises(ValueError, match=r"partial lacks \['1'\]"):
            cost_table(models, "cluster")


class TestNormalisation:
    def test_cluster_range_spans_the_whole_matrix(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        table = cost_table(models, "cluster")
        flat = [v for row in table.values() for v in row.values()]
        assert min(flat) == 0.0 and max(flat) == 1.0

    def test_cheapest_cell_is_zero_not_the_cheapest_model(self):
        # swing at cluster 0 costs 1.0, the cheapest cell in the pool.
        models, _ = models_from_stats(INVERTED, "tpot")
        assert cost_table(models, "cluster")["swing"]["0"] == 0.0


class TestInversionChangesTheRouting:
    def test_per_model_prices_both_clusters_alike(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        table = cost_table(models, "model")
        assert table["swing"]["0"] == table["swing"]["1"]

    def test_per_cluster_separates_them(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        table = cost_table(models, "cluster")
        assert table["swing"]["0"] < table["swing"]["1"]

    def test_assignment_differs_where_the_pool_inverts(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        # Errors are equal, so cost alone decides: per-model always picks the
        # cheaper average, per-cluster picks the cheaper cell.
        per_model = assign(models, 1.0, 0.0, "model")
        per_cluster = assign(models, 1.0, 0.0, "cluster")
        assert per_model == {"0": "swing", "1": "swing"}
        assert per_cluster == {"0": "swing", "1": "flat"}

    def test_no_inversion_still_moves_the_boundary(self):
        """Absence of an inversion does not make the two rules agree pointwise.

        With no inversion the cost ordering inside each cluster is identical, so
        neither rule can prefer a different model for cost reasons alone. But the
        two normalise over different ranges -- the pool's scalars against the whole
        (model, cluster) matrix -- so the same error gap is bought at a different
        lambda and the crossover moves. Here "cheap" and "dear" swap at lambda 0.2
        under one rule and not the other, though "cheap" is cheaper in both
        clusters. This is why an unchanged routing has to be checked, not assumed.
        """
        models, _ = models_from_stats(NO_INVERSION, "tpot")
        assert assign(models, 0.2, 0.0, "model") == {"0": "cheap", "1": "dear"}
        assert assign(models, 0.2, 0.0, "cluster") == {"0": "dear", "1": "dear"}

    def test_no_inversion_preserves_the_reachable_frontier(self):
        # The boundaries move but the sequence of routings the sweep can reach is
        # the same, which is what makes a budgeted re-fit land in the same place.
        models, _ = models_from_stats(NO_INVERSION, "tpot")

        def frontier(conditioning):
            out = []
            for region in routing_regions(models, 0.0, conditioning):
                if not out or out[-1] != region.assignment:
                    out.append(region.assignment)
            return out

        assert frontier("model") == frontier("cluster")


class TestPruningCanOnlyGrow:
    def test_per_cluster_domination_implies_per_model(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        swing, flat = models[0], models[1]
        # swing is dearer in cluster 1, so it cannot dominate under "cluster",
        # but its average is lower, so it does under "model".
        assert dominates(swing, flat, 0.0, "model")
        assert not dominates(swing, flat, 0.0, "cluster")

    def test_survivors_never_shrink_on_random_pools(self):
        rng = random.Random(0)
        for _ in range(200):
            pool = {}
            for i in range(rng.randint(2, 5)):
                pool[f"m{i}"] = {
                    "errors": {c: round(rng.uniform(0.1, 0.9), 3) for c in "012"},
                    "cluster_tpot_ms": {c: round(rng.uniform(1.0, 40.0), 3) for c in "012"},
                }
            models, _ = models_from_stats(stats(pool, {c: 1 for c in "012"}), "tpot")
            by_model = {m.name for m in pareto_prune(models, 0.0, "model")[0]}
            by_cluster = {m.name for m in pareto_prune(models, 0.0, "cluster")[0]}
            assert by_model <= by_cluster


class TestRegionsStayCoherent:
    def test_regions_partition_lambda_under_both_rules(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        for conditioning in ("model", "cluster"):
            regions = routing_regions(models, 0.0, conditioning)
            assert regions[0].lam_min == 0.0
            for lo, hi in zip(regions, regions[1:]):
                assert lo.lam_max == hi.lam_min

    def test_each_region_reproduces_its_own_assignment(self):
        models, _ = models_from_stats(INVERTED, "tpot")
        for region in routing_regions(models, 0.0, "cluster"):
            probe = region.representative_lambda
            assert assign(models, probe, 0.0, "cluster") == region.assignment
