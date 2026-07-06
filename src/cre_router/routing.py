"""Stage 1: cost-aware, cluster-level routing.

Implements the routing score (paper Eq. 1-2), Pareto pruning (Sec. 4.3),
crossover points and routing regions (Eq. 3, Sec. 4.4), lambda* selection
under a TPOT budget (Eq. 4), and the efficiency metric eta (Eq. 5).

All functions operate on plain ``ModelStats`` records holding per-cluster
error rates and TPOT measured once, offline, on the training corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ModelStats:
    """Statistics for one candidate model, measured on the training corpus.

    ``tpot_ms`` is the pool-level average Time Per Output Token used in the
    normalised cost term. ``cluster_tpot_ms`` optionally refines system-TPOT
    estimates with per-cluster measurements; when absent, ``tpot_ms`` is used
    for every cluster.
    """

    name: str
    tpot_ms: float
    errors: dict[str, float]
    cluster_tpot_ms: dict[str, float] = field(default_factory=dict)

    def tpot_for(self, cluster: str) -> float:
        return self.cluster_tpot_ms.get(cluster, self.tpot_ms)


@dataclass
class Region:
    """A maximal lambda interval over which the routing assignment is constant."""

    lam_min: float
    lam_max: float  # math.inf for the last region
    assignment: dict[str, str]  # cluster id -> model name

    @property
    def interval_str(self) -> str:
        """The lambda range this region spans, e.g. ``[0.052, 0.067)`` or
        ``[0.099, inf)``. This is the region's real, deterministic extent;
        the routing is constant across it, so it is what a paper table reports."""
        hi = "inf" if math.isinf(self.lam_max) else f"{self.lam_max:.3f}"
        return f"[{self.lam_min:.3f}, {hi})"

    @property
    def representative_lambda(self) -> float:
        """A single clean lambda inside the region, used to name the selected
        operating point (lambda*). Any value in the region yields the same
        routing; this picks the fewest-decimals round one."""
        if math.isinf(self.lam_max):
            return _round_above(self.lam_min)
        return _nice_lambda(self.lam_min, self.lam_max)


def _nice_lambda(lo: float, hi: float) -> float:
    """A human-friendly lambda strictly inside (lo, hi): the fewest-decimals
    round number if one fits, otherwise the midpoint."""
    for decimals in range(1, 12):
        step = 10**-decimals
        candidate = math.floor(hi / step) * step
        if lo < candidate < hi:
            return round(candidate, decimals)
    return (lo + hi) / 2


def _round_above(x: float) -> float:
    """The fewest-decimals round number strictly greater than x (used to name
    the open-ended top region, e.g. 0.099 -> 0.1)."""
    for decimals in range(1, 12):
        step = 10**-decimals
        candidate = round((math.floor(x / step) + 1) * step, decimals)
        if candidate > x:
            return candidate
    return x


def clusters_of(models: list[ModelStats]) -> list[str]:
    clusters: set[str] = set()
    for m in models:
        clusters.update(m.errors)
    return sorted(clusters)


def normalized_costs(models: list[ModelStats]) -> dict[str, float]:
    """Eq. 2: min-max normalise pool TPOT so the fastest model costs 0 and
    the slowest costs 1."""
    lo = min(m.tpot_ms for m in models)
    hi = max(m.tpot_ms for m in models)
    if hi == lo:
        return {m.name: 0.0 for m in models}
    return {m.name: (m.tpot_ms - lo) / (hi - lo) for m in models}


def assign(models: list[ModelStats], lam: float) -> dict[str, str]:
    """Eq. 1: route each cluster to argmin_m Error(m, c) + lambda * Cost_norm(m),
    breaking ties in favour of the faster model."""
    costs = normalized_costs(models)
    table = {}
    for c in clusters_of(models):
        best = min(models, key=lambda m: (m.errors[c] + lam * costs[m.name], m.tpot_ms))
        table[c] = best.name
    return table


def dominates(a: ModelStats, b: ModelStats) -> bool:
    """True if ``a`` Pareto-dominates ``b``: no worse on TPOT and on every
    cluster's error, and strictly better on at least one of those."""
    if a.tpot_ms > b.tpot_ms:
        return False
    if any(a.errors[c] > b.errors[c] for c in b.errors):
        return False
    return a.tpot_ms < b.tpot_ms or any(a.errors[c] < b.errors[c] for c in b.errors)


def pareto_prune(models: list[ModelStats]) -> tuple[list[ModelStats], list[ModelStats]]:
    """Split the pool into (Pareto-efficient, dominated). Dominated models can
    never be selected by the routing score for any lambda."""
    efficient, dominated = [], []
    for m in models:
        if any(dominates(other, m) for other in models if other.name != m.name):
            dominated.append(m)
        else:
            efficient.append(m)
    return efficient, dominated


def crossover_candidates(models: list[ModelStats]) -> list[float]:
    """Candidate region boundaries: for each cluster and model pair, the lambda
    at which their scores are equal. For K=2 this reduces to the closed form
    of Eq. 3, lambda_c = Error(m_fast, c) - Error(m_strong, c)."""
    costs = normalized_costs(models)
    lams: set[float] = set()
    for c in clusters_of(models):
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                dcost = costs[b.name] - costs[a.name]
                if dcost == 0:
                    continue
                lam = (a.errors[c] - b.errors[c]) / dcost
                if lam > 0:
                    lams.add(round(lam, 12))
    return sorted(lams)


def routing_regions(models: list[ModelStats]) -> list[Region]:
    """Sweep lambda from 0 upward and return the maximal regions with constant
    cluster-to-model assignment. Candidate boundaries that do not change the
    argmin (possible for K > 2) are merged away."""
    edges = [0.0] + crossover_candidates(models)
    regions: list[Region] = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else math.inf
        probe = (lo + hi) / 2 if not math.isinf(hi) else lo + max(lo * 0.5, 0.05)
        table = assign(models, probe)
        if regions and regions[-1].assignment == table:
            regions[-1] = Region(regions[-1].lam_min, hi, table)
        else:
            regions.append(Region(lo, hi, table))
    return regions


def system_metrics(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
) -> tuple[float, float]:
    """System accuracy and average TPOT under a routing assignment, weighted
    by cluster size."""
    by_name = {m.name: m for m in models}
    total = sum(cluster_sizes.values())
    acc = sum(
        cluster_sizes[c] * (1.0 - by_name[name].errors[c]) for c, name in assignment.items()
    ) / total
    tpot = sum(
        cluster_sizes[c] * by_name[name].tpot_for(c) for c, name in assignment.items()
    ) / total
    return acc, tpot


def eta(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
) -> float | None:
    """Eq. 5: accuracy given up (percentage points) per millisecond of TPOT
    saved, relative to the lambda=0 baseline. Lower is better; None when the
    assignment equals the baseline."""
    acc0, tpot0 = system_metrics(models, assign(models, 0.0), cluster_sizes)
    acc, tpot = system_metrics(models, assignment, cluster_sizes)
    if math.isclose(tpot0, tpot):
        return None
    return (acc0 - acc) * 100.0 / (tpot0 - tpot)


@dataclass
class Selection:
    """Result of budgeted lambda* selection."""

    lambda_star: float
    region: Region
    accuracy: float
    tpot_ms: float
    eta: float | None


def select_lambda(
    models: list[ModelStats],
    cluster_sizes: dict[str, int | float],
    budget_ms: float,
) -> Selection:
    """Eq. 4: among all routing regions, pick the one with the highest training
    accuracy whose system TPOT satisfies the budget; ties favour lower TPOT."""
    best: tuple[float, float, Region] | None = None
    for region in routing_regions(models):
        acc, tpot = system_metrics(models, region.assignment, cluster_sizes)
        if tpot > budget_ms:
            continue
        if best is None or (acc, -tpot) > (best[0], -best[1]):
            best = (acc, tpot, region)
    if best is None:
        fastest = min(
            system_metrics(models, r.assignment, cluster_sizes)[1]
            for r in routing_regions(models)
        )
        raise ValueError(
            f"No routing strategy satisfies budget {budget_ms} ms "
            f"(fastest achievable system TPOT is {fastest:.1f} ms)."
        )
    acc, tpot, region = best
    return Selection(
        lambda_star=region.representative_lambda,
        region=region,
        accuracy=acc,
        tpot_ms=tpot,
        eta=eta(models, region.assignment, cluster_sizes),
    )


def models_from_stats(stats: dict) -> tuple[list[ModelStats], dict[str, float]]:
    """Build ``ModelStats`` from a stats dict (see configs/*_stats.json).

    Expected shape::

        {
          "cluster_sizes": {"0": 194, "1": 405, "2": 322},
          "models": {
            "name": {
              "tpot_ms": 9.15,                # optional if cluster_tpot_ms given
              "errors": {"0": 0.130, ...},
              "cluster_tpot_ms": {"0": 9.282, ...}   # optional
            }
          }
        }
    """
    models = []
    for name, spec in stats["models"].items():
        cluster_tpot = {str(k): float(v) for k, v in spec.get("cluster_tpot_ms", {}).items()}
        tpot = spec.get("tpot_ms")
        if tpot is None:
            if not cluster_tpot:
                raise ValueError(f"Model {name!r} needs tpot_ms or cluster_tpot_ms")
            tpot = sum(cluster_tpot.values()) / len(cluster_tpot)
        models.append(
            ModelStats(
                name=name,
                tpot_ms=float(tpot),
                errors={str(k): float(v) for k, v in spec["errors"].items()},
                cluster_tpot_ms=cluster_tpot,
            )
        )
    cluster_sizes = {str(k): float(v) for k, v in stats["cluster_sizes"].items()}
    return models, cluster_sizes
