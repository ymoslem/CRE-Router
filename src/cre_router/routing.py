"""Stage 1: cost-aware, cluster-level routing.

Implements the routing score (Eq. 1-2), Pareto pruning, crossover points and
routing regions (Eq. 3), lambda* selection under a budget (Eq. 4), the
efficiency metric eta (Eq. 5), and the near-tie diagnostics.

All functions operate on plain ``ModelStats`` records holding per-cluster
error rates and TPOT measured once, offline, on the training corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Two per-cluster error rates closer than this are treated as indistinguishable
#: and the choice between those models falls to cost alone. Absolute, in error
#: units: 0.001 is 0.1 percentage points, one unit of the precision the tables
#: print. It is a robustness guard, not a tuned parameter -- on every pool
#: published here it selects exactly what an exact comparison selects.
DEFAULT_ERROR_TOL = 0.001

#: How Eq. 2 prices a model. ``"model"`` is the published rule, one scalar per
#: model; ``"cluster"`` prices each (model, cluster) pair on its own measurement.
#: The default reproduces every published result unchanged.
COST_CONDITIONING = ("model", "cluster")
DEFAULT_COST_CONDITIONING = "model"

#: Guard against binary floating point placing a gap that is ``tol`` by
#: construction just above it: 0.231 - 0.230 evaluates to 1.0000000000000009e-3.
_TOL_EPS = 1e-9


@dataclass
class ModelStats:
    """Statistics for one candidate model, measured on the training corpus.

    ``tpot_ms`` is the pool-level average Time Per Output Token. ``cluster_tpot_ms``
    optionally refines system estimates with per-cluster measurements; when
    absent, the pool-level value is used for every cluster.

    ``cost_metric`` selects which measurement the routing arithmetic treats as
    Cost. ``"tpot"`` is the default and reproduces the published results exactly.
    ``"e2el"`` uses end-to-end request latency, which is required whenever pool
    members differ in output *length* rather than decode speed (a thinking and a
    non-thinking mode of one model have near-identical TPOT). E2EL generalises
    TPOT: since ``E2EL = TTFT + TPOT x L``, uniform L and TTFT make it affine in
    TPOT, and min-max normalisation is affine-invariant, so the two agree.
    """

    name: str
    tpot_ms: float
    errors: dict[str, float]
    cluster_tpot_ms: dict[str, float] = field(default_factory=dict)
    e2el_ms: float | None = None
    cluster_e2el_ms: dict[str, float] = field(default_factory=dict)
    cluster_output_tokens: dict[str, float] = field(default_factory=dict)
    cost_metric: str = "tpot"

    def __post_init__(self) -> None:
        if self.cost_metric not in ("tpot", "e2el"):
            raise ValueError(f"unknown cost_metric {self.cost_metric!r}")
        if self.cost_metric == "e2el" and self.e2el_ms is None:
            raise ValueError(
                f"{self.name}: cost_metric 'e2el' needs e2el_ms, which this stats "
                "entry does not have; re-measure, or fit with --cost-metric tpot"
            )

    @property
    def cost_ms(self) -> float:
        """The pool-level cost under the selected metric."""
        return self.tpot_ms if self.cost_metric == "tpot" else float(self.e2el_ms)

    def cost_for(self, cluster: str) -> float:
        """The per-cluster cost under the selected metric."""
        if self.cost_metric == "tpot":
            return self.cluster_tpot_ms.get(cluster, self.tpot_ms)
        return self.cluster_e2el_ms.get(cluster, float(self.e2el_ms))

    @property
    def measured_clusters(self) -> set[str]:
        """Clusters holding a measured per-cluster cost under the selected metric.

        ``cost_for`` falls back to the pool-level scalar outside this set. That is
        the right behaviour for reporting, but it would silently turn per-cluster
        scoring back into per-model scoring, so anything that needs genuinely
        per-cluster costs checks this first.
        """
        source = self.cluster_tpot_ms if self.cost_metric == "tpot" else self.cluster_e2el_ms
        return set(source)

    def tpot_for(self, cluster: str) -> float:
        return self.cluster_tpot_ms.get(cluster, self.tpot_ms)

    def e2el_for(self, cluster: str) -> float:
        """Per-cluster E2EL, independent of the selected cost metric."""
        if cluster in self.cluster_e2el_ms:
            return self.cluster_e2el_ms[cluster]
        if self.e2el_ms is None:
            raise ValueError(f"{self.name}: E2EL requested but none measured")
        return float(self.e2el_ms)


@dataclass
class Region:
    """A maximal lambda interval over which the routing assignment is constant."""

    lam_min: float
    lam_max: float  # math.inf for the last region
    assignment: dict[str, str]  # cluster id -> model name
    #: Set only on a routing no lambda reaches: the two clusters whose crossover
    #: order was treated as tied, with the later one switched first. Its lambda
    #: range is then the tie window, not a range that selects it.
    swapped: tuple[str, str] | None = None

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
    round number if one fits, otherwise the midpoint.

    The bounds are tested on the rounded value, which is the one returned:
    rounding a candidate that sits a hair below ``hi`` can push it onto the
    exclusive upper bound, naming a lambda that routes as the next region.
    Each decimal level tries the two highest multiples that could fit, so an
    interval whose upper bound is itself round, such as [0.3, 0.5), is still
    named with one decimal rather than falling through to a longer one.
    """
    for decimals in range(1, 12):
        step = 10**-decimals
        highest = math.floor(hi / step)
        for multiple in (highest, highest - 1):
            candidate = round(multiple * step, decimals)
            if lo < candidate < hi:
                return candidate
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
    """Eq. 2: min-max normalise the pool's cost metric so the cheapest model
    costs 0 and the most expensive costs 1."""
    lo = min(m.cost_ms for m in models)
    hi = max(m.cost_ms for m in models)
    if hi == lo:
        return {m.name: 0.0 for m in models}
    return {m.name: (m.cost_ms - lo) / (hi - lo) for m in models}


def cost_table(
    models: list[ModelStats], cost_conditioning: str = DEFAULT_COST_CONDITIONING
) -> dict[str, dict[str, float]]:
    """Eq. 2, keyed ``[model][cluster]`` so callers never branch on the rule.

    ``"model"`` is the published rule: one size-weighted scalar per model, so
    every cluster of a given model maps to the same normalised cost. Error is
    conditioned on the cluster but cost is not.

    ``"cluster"`` prices each (model, cluster) pair on its own measurement. A
    model that is cheap on short queries and dear on long ones is then priced
    as such in each cluster rather than averaged across them. This is available
    only because clustering happens offline: a per-query router cannot know a
    query's cost before serving it, but a per-cluster router can measure it.

    Both rules min-max over whatever they price, so the cheapest entry is 0 and
    the dearest is 1. Under ``"cluster"`` that range is taken over the whole
    (model, cluster) matrix rather than per cluster, which keeps one cost axis
    and so one meaning for lambda across clusters.
    """
    if cost_conditioning not in COST_CONDITIONING:
        raise ValueError(
            f"unknown cost_conditioning {cost_conditioning!r}; "
            f"expected one of {COST_CONDITIONING}"
        )
    clusters = clusters_of(models)
    if cost_conditioning == "model":
        raw = {m.name: {c: m.cost_ms for c in clusters} for m in models}
    else:
        missing = {m.name: sorted(set(clusters) - m.measured_clusters)
                   for m in models if not set(clusters) <= m.measured_clusters}
        if missing:
            detail = "; ".join(f"{n} lacks {cs}" for n, cs in sorted(missing.items()))
            raise ValueError(
                "cost_conditioning 'cluster' needs a measured cost for every "
                f"(model, cluster) pair, but {detail}. Re-measure per cluster, or "
                "fit with cost_conditioning 'model'."
            )
        raw = {m.name: {c: m.cost_for(c) for c in clusters} for m in models}

    values = [v for row in raw.values() for v in row.values()]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {name: {c: 0.0 for c in row} for name, row in raw.items()}
    return {name: {c: (v - lo) / (hi - lo) for c, v in row.items()}
            for name, row in raw.items()}


def assign(
    models: list[ModelStats],
    lam: float,
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> dict[str, str]:
    """Eq. 1: route each cluster to argmin_m Error(m, c) + lambda * Cost_norm(m),
    breaking ties in favour of the faster model.

    ``error_tol`` widens that tie-break: models whose error is within
    ``error_tol`` of the argmin's are treated as indistinguishable on accuracy,
    and the cheapest of them is chosen. Pass 0.0 for an exact comparison.

    The tolerated set is anchored on the argmin, never chained pairwise, so a
    run of near-ties cannot walk the choice away from the best model. Because
    errors do not depend on lambda, that set can only change where the argmin
    already changes, so the tolerance adds no region boundary. It can remove
    one, by merging away a region that held only a within-tolerance preference.
    """
    costs = cost_table(models, cost_conditioning)
    table = {}
    for c in clusters_of(models):
        best = min(models, key=lambda m: (m.errors[c] + lam * costs[m.name][c],
                                          costs[m.name][c]))
        tied = [m for m in models
                if abs(m.errors[c] - best.errors[c]) <= error_tol + _TOL_EPS]
        table[c] = min(tied, key=lambda m: (costs[m.name][c], m.errors[c], m.name)).name
    return table


@dataclass(frozen=True)
class Margin:
    """How far the chosen model beat the runner-up in one cluster.

    ``gap`` is the difference in decision score. ``granularity`` is ``1/n`` for
    that cluster: a cluster of n questions cannot resolve accuracy any finer, so
    a gap below it means the two models are not distinguished by the data even
    though the rule still picks one.
    """

    cluster: str
    chosen: str
    runner_up: str
    gap: float
    granularity: float

    @property
    def resolvable(self) -> bool:
        """True when the margin exceeds what the cluster can measure."""
        return self.gap >= self.granularity

    @property
    def ratio(self) -> float:
        """Gap in units of granularity. Below 1.0 the choice rests on noise."""
        return self.gap / self.granularity if self.granularity else math.inf


def selection_margins(
    models: list[ModelStats],
    lam: float,
    cluster_sizes: dict[str, float],
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> dict[str, Margin]:
    """Per cluster, how decisively ``assign`` preferred the winner at ``lam``.

    Purely diagnostic: it reports on the same choice ``assign`` makes and never
    alters it. Use it to see whether an operating point rests on a real margin
    or on a difference the test set is too small to measure.

    ``cluster_sizes`` is the second value returned by :func:`models_from_stats`.
    """
    costs = cost_table(models, cost_conditioning)
    chosen = assign(models, lam, error_tol, cost_conditioning)
    out: dict[str, Margin] = {}
    for c in clusters_of(models):
        ranked = sorted(models, key=lambda m: (m.errors[c] + lam * costs[m.name][c],
                                               costs[m.name][c]))
        n = float(cluster_sizes.get(c, 0) or 0)
        # Report on whatever `assign` chose, so the two can never disagree. Where
        # the tolerance overrode the score order, `gap` comes out negative, which
        # reads correctly: the data did not support preferring the cheaper model,
        # it only failed to rule it out.
        best = next(m for m in models if m.name == chosen[c])
        second = next(m for m in ranked if m.name != best.name)
        out[c] = Margin(
            cluster=c,
            chosen=best.name,
            runner_up=second.name,
            gap=((second.errors[c] + lam * costs[second.name][c])
                 - (best.errors[c] + lam * costs[best.name][c])),
            granularity=(1.0 / n) if n else math.inf,
        )
    return out


@dataclass(frozen=True)
class Crossover:
    """One region boundary: the lambda at which ``cluster`` moves to a cheaper model.

    ``se`` is the standard error of ``lam`` implied by the two error rates and
    the cluster's size, treating the two models' errors as independent binomial
    estimates. They answer the same questions, so this overstates the noise and
    errs towards calling a crossover order unresolved.
    """

    lam: float
    cluster: str
    from_model: str
    to_model: str
    se: float


@dataclass(frozen=True)
class CrossoverGap:
    """Two consecutive crossovers in different clusters, and whether the data
    can order them.

    The region between them is the only routing that switches ``lower.cluster``
    without ``upper.cluster``. When ``gap`` is within the noise of the two
    crossovers, the training data does not determine which cluster should
    switch first, and the routing that switches ``upper.cluster`` first is as
    well supported but reached by no lambda.
    """

    lower: Crossover
    upper: Crossover

    @property
    def gap(self) -> float:
        return self.upper.lam - self.lower.lam

    @property
    def se(self) -> float:
        return math.hypot(self.lower.se, self.upper.se)

    @property
    def resolvable(self) -> bool:
        """True when the gap exceeds twice its standard error."""
        return self.gap >= 2.0 * self.se

    @property
    def ratio(self) -> float:
        """Gap in standard errors. Below 2.0 the crossover order is unresolved."""
        return self.gap / self.se if self.se else math.inf


def crossover_gaps(
    models: list[ModelStats],
    cluster_sizes: dict[str, float],
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> list[CrossoverGap]:
    """Every pair of consecutive crossovers in different clusters along the sweep.

    Purely diagnostic, like :func:`selection_margins`: it reads the regions
    :func:`routing_regions` returns and never alters them. ``error_tol``
    compares two models within one cluster; this compares the switching points
    of two clusters, which no tolerance in :func:`assign` covers.

    Only boundaries where a single cluster changes are paired, and only adjacent
    ones, so three crossovers within noise of each other yield two pairs rather
    than every order of the three.
    """
    costs = cost_table(models, cost_conditioning)
    by_name = {m.name: m for m in models}
    regions = routing_regions(models, error_tol, cost_conditioning)
    bounds: list[Crossover | None] = []
    for before, after in zip(regions, regions[1:]):
        changed = [c for c in before.assignment if before.assignment[c] != after.assignment[c]]
        if len(changed) != 1:
            bounds.append(None)
            continue
        c = changed[0]
        a, b = by_name[before.assignment[c]], by_name[after.assignment[c]]
        n = float(cluster_sizes.get(c, 0) or 0)
        dcost = abs(costs[a.name][c] - costs[b.name][c])
        var = sum(e * (1.0 - e) for e in (a.errors[c], b.errors[c]))
        se = math.sqrt(var / n) / dcost if n and dcost else math.inf
        bounds.append(Crossover(after.lam_min, c, a.name, b.name, se))
    return [CrossoverGap(lo, hi) for lo, hi in zip(bounds, bounds[1:])
            if lo is not None and hi is not None and lo.cluster != hi.cluster]


def is_tied(gap: CrossoverGap, crossover_tol: float | str) -> bool:
    """Whether ``crossover_tol`` treats the two crossovers of ``gap`` as tied."""
    if crossover_tol == "auto":
        return not gap.resolvable
    tol = float(crossover_tol)
    return tol > 0.0 and gap.gap <= tol + _TOL_EPS


def swapped_candidates(
    models: list[ModelStats],
    cluster_sizes: dict[str, float],
    crossover_tol: float | str,
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> list[Region]:
    """The routings a tied crossover order adds: the later cluster switched first.

    ``crossover_tol`` is the largest gap between two crossovers treated as a tie,
    in lambda units (error units when the pair's normalised costs are 0 and 1),
    or ``"auto"`` to tie every gap the data cannot resolve. 0 adds nothing.
    """
    if crossover_tol != "auto" and float(crossover_tol) <= 0.0:
        return []
    regions = routing_regions(models, error_tol, cost_conditioning)
    out = []
    for g in crossover_gaps(models, cluster_sizes, error_tol, cost_conditioning):
        if not is_tied(g, crossover_tol):
            continue
        before = next(r for r in regions if r.lam_max == g.lower.lam)
        table = dict(before.assignment)
        table[g.upper.cluster] = g.upper.to_model
        if table not in [r.assignment for r in regions + out]:
            out.append(Region(g.lower.lam, g.upper.lam, table,
                              swapped=(g.lower.cluster, g.upper.cluster)))
    return out


def dominates(
    a: ModelStats,
    b: ModelStats,
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> bool:
    """True if ``a`` Pareto-dominates ``b``: no worse on cost and on every
    cluster's error, and strictly better on at least one of those.

    ``error_tol`` applies to both halves of that test, so an error advantage
    smaller than the tolerance neither rescues ``b`` nor, on its own, condemns
    it. Applying it to only one half would let two models dominate each other.

    ``cost_conditioning`` decides what "no worse on cost" compares. Under
    ``"model"`` it is the pool-level scalar, an average over clusters. Under
    ``"cluster"`` it must hold in every cluster, which is the stronger
    requirement: a model cheaper on average but dearer somewhere can still be
    the cheapest choice there, so averaging would prune a model that the
    per-cluster rule would select. Domination under ``"cluster"`` therefore
    implies domination under ``"model"``, never the reverse, and the surviving
    pool can only grow when the rule is tightened.
    """
    tol = error_tol + _TOL_EPS
    if cost_conditioning == "cluster":
        cheaper = [a.cost_for(c) < b.cost_for(c) for c in b.errors]
        if any(a.cost_for(c) > b.cost_for(c) for c in b.errors):
            return False
    else:
        cheaper = [a.cost_ms < b.cost_ms]
        if a.cost_ms > b.cost_ms:
            return False
    if any(a.errors[c] > b.errors[c] + tol for c in b.errors):
        return False
    return any(cheaper) or any(a.errors[c] < b.errors[c] - tol for c in b.errors)


def pareto_prune(
    models: list[ModelStats],
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> tuple[list[ModelStats], list[ModelStats]]:
    """Split the pool into (Pareto-efficient, dominated). Dominated models can
    never be selected by the routing score for any lambda.

    Tightening ``cost_conditioning`` to ``"cluster"`` can only move models from
    dominated to efficient, never the other way. See :func:`dominates`.
    """
    efficient, dominated = [], []
    for m in models:
        if any(dominates(other, m, error_tol, cost_conditioning)
               for other in models if other.name != m.name):
            dominated.append(m)
        else:
            efficient.append(m)
    return efficient, dominated


def crossover_candidates(
    models: list[ModelStats], cost_conditioning: str = DEFAULT_COST_CONDITIONING
) -> list[float]:
    """Candidate region boundaries: for each cluster and model pair, the lambda
    at which their scores are equal. For K=2 this reduces to the closed form
    of Eq. 3, lambda_c = Error(m_fast, c) - Error(m_strong, c)."""
    costs = cost_table(models, cost_conditioning)
    lams: set[float] = set()
    for c in clusters_of(models):
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                dcost = costs[b.name][c] - costs[a.name][c]
                if dcost == 0:
                    continue
                lam = (a.errors[c] - b.errors[c]) / dcost
                if lam > 0:
                    lams.add(round(lam, 12))
    return sorted(lams)


def routing_regions(
    models: list[ModelStats],
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> list[Region]:
    """Sweep lambda from 0 upward and return the maximal regions with constant
    cluster-to-model assignment. Candidate boundaries that do not change the
    argmin (possible for K > 2) are merged away.

    ``crossover_candidates`` needs no tolerance of its own: the tolerated set
    that ``assign`` uses cannot change anywhere the argmin does not, so every
    boundary it needs is already a candidate. A candidate that only separated
    two within-tolerance choices no longer changes the assignment, and is
    merged away here like any other unused one.
    """
    edges = [0.0] + crossover_candidates(models, cost_conditioning)
    regions: list[Region] = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else math.inf
        probe = (lo + hi) / 2 if not math.isinf(hi) else lo + max(lo * 0.5, 0.05)
        table = assign(models, probe, error_tol, cost_conditioning)
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
        cluster_sizes[c] * by_name[name].cost_for(c) for c, name in assignment.items()
    ) / total
    return acc, tpot


@dataclass(frozen=True)
class Stage2Plan:
    """Where Stage 2 escalates, and which clusters it gates.

    ``target`` is the model every rejected answer is escalated to, or None when
    the routing has nothing to escalate to. ``gated`` maps each gated cluster to
    the model that serves it at Stage 1, the model whose answers its quality
    estimator is trained on.
    """

    target: str | None
    gated: dict[str, str]

    @property
    def estimators(self) -> dict[str, list[str]]:
        """Gated model -> the clusters its estimator gates."""
        out: dict[str, list[str]] = {}
        for cluster, model in sorted(self.gated.items()):
            out.setdefault(model, []).append(cluster)
        return out


def stage2_plan(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
    escalate_to: str | None = None,
) -> Stage2Plan:
    """The escalation target and the gated clusters of a routing.

    The target is the most accurate model the routing uses, by size-weighted
    accuracy on the training data, ties going to the cheaper model. Stage 2 gates
    every cluster whose Stage 1 model costs less than the target under the fitted
    cost metric: escalating from a model to one that costs no less is the only
    direction that can pay. ``escalate_to`` overrides the target, for a routing
    that uses a single model and so has nothing of its own to escalate to.
    """
    by_name = {m.name: m for m in models}
    total = sum(cluster_sizes.values())

    def accuracy(m: ModelStats) -> float:
        return sum(n * (1.0 - m.errors[c]) for c, n in cluster_sizes.items()) / total

    if escalate_to is not None:
        if escalate_to not in by_name:
            raise ValueError(f"escalation target {escalate_to!r} is not in the stats")
        target = by_name[escalate_to]
    else:
        used = [by_name[name] for name in sorted(set(assignment.values()))]
        target = max(used, key=lambda m: (accuracy(m), -m.cost_ms))
        if len(used) == 1:
            return Stage2Plan(target=None, gated={})
    gated = {c: name for c, name in assignment.items()
             if by_name[name].cost_ms < target.cost_ms}
    return Stage2Plan(target=target.name if gated else None, gated=gated)


def eta(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
) -> float | None:
    """Eq. 5: accuracy given up (percentage points) per millisecond of TPOT
    saved, relative to the lambda=0 baseline. Lower is better; None when the
    assignment equals the baseline."""
    acc0, tpot0 = system_metrics(
        models, assign(models, 0.0, error_tol, cost_conditioning), cluster_sizes)
    acc, tpot = system_metrics(models, assignment, cluster_sizes)
    if math.isclose(tpot0, tpot):
        return None
    return (acc0 - acc) * 100.0 / (tpot0 - tpot)


@dataclass
class Selection:
    """Result of budgeted lambda* selection."""

    #: None when the selected routing is a swapped candidate no lambda reaches.
    lambda_star: float | None
    region: Region
    accuracy: float
    tpot_ms: float
    eta: float | None


def select_lambda(
    models: list[ModelStats],
    cluster_sizes: dict[str, int | float],
    budget_ms: float,
    error_tol: float = DEFAULT_ERROR_TOL,
    cost_conditioning: str = DEFAULT_COST_CONDITIONING,
    crossover_tol: float | str = 0.0,
) -> Selection:
    """Eq. 4: among all routing regions, pick the one with the highest training
    accuracy whose system TPOT satisfies the budget; ties favour lower TPOT.

    ``crossover_tol`` adds, as further candidates, the routings that tied
    crossover orders make available (see :func:`swapped_candidates`). Its
    default of 0 adds none.
    """
    best: tuple[float, float, Region] | None = None
    candidates = (routing_regions(models, error_tol, cost_conditioning)
                  + swapped_candidates(models, cluster_sizes, crossover_tol, error_tol,
                                       cost_conditioning))
    for region in candidates:
        acc, tpot = system_metrics(models, region.assignment, cluster_sizes)
        if tpot > budget_ms:
            continue
        if best is None or (acc, -tpot) > (best[0], -best[1]):
            best = (acc, tpot, region)
    if best is None:
        fastest = min(
            system_metrics(models, r.assignment, cluster_sizes)[1]
            for r in routing_regions(models, error_tol, cost_conditioning)
        )
        metric = models[0].cost_metric.upper() if models else "cost"
        raise ValueError(
            f"No routing strategy satisfies budget {budget_ms} ms "
            f"(fastest achievable system {metric} is {fastest:.1f} ms). "
            f"Note the budget is in {metric} units: a per-token TPOT budget is "
            f"orders of magnitude smaller than a per-request E2EL one."
        )
    acc, tpot, region = best
    return Selection(
        lambda_star=None if region.swapped else region.representative_lambda,
        region=region,
        accuracy=acc,
        tpot_ms=tpot,
        eta=eta(models, region.assignment, cluster_sizes, error_tol, cost_conditioning),
    )


def error_tol_from_stats(stats: dict) -> float:
    """Read the pool's error tolerance from a stats dict, or the default.

    Kept as config rather than a literal in the argmin so that setting it to 0.0
    reproduces an exact comparison without touching the code, which is what the
    regression gate exercises.
    """
    value = stats.get("error_tol", DEFAULT_ERROR_TOL)
    if value is None:
        return DEFAULT_ERROR_TOL
    value = float(value)
    if value < 0.0:
        raise ValueError(f"error_tol must be non-negative, got {value}")
    return value


def models_from_stats(
    stats: dict, cost_metric: str = "tpot", cost_weighting: str = "size"
) -> tuple[list[ModelStats], dict[str, float]]:
    """Build ``ModelStats`` from a stats dict (see configs/*_stats.json).

    Expected shape::

        {
          "cluster_sizes": {"0": 194, "1": 405, "2": 322},
          "models": {
            "name": {
              "tpot_ms": 9.15,                # optional if cluster_tpot_ms given
              "errors": {"0": 0.130, ...},
              "cluster_tpot_ms": {"0": 9.282, ...},  # optional
              "cluster_e2el_ms": {"0": 8123.4, ...}  # optional, needed for e2el
            }
          }
        }

    ``cost_metric`` defaults to ``"tpot"``, under which this reads exactly the
    fields it always has and reproduces published results unchanged.
    """
    sizes = {str(k): float(v) for k, v in stats.get("cluster_sizes", {}).items()}

    def _collapse(per_cluster: dict[str, float]) -> float:
        """Collapse per-cluster cost to the one scalar Eq. 2 normalises.

        Weighted by cluster size, so the result is the expected cost of a query
        if the whole corpus were served by this model. An unweighted mean would
        give a 51-query cluster the same say as a 98-query one, which
        corresponds to no real workload. Falls back to the unweighted mean when
        cluster sizes are unavailable.
        """
        if cost_weighting == "unweighted" or not sizes:
            return sum(per_cluster.values()) / len(per_cluster)
        total = sum(sizes[c] for c in per_cluster)
        return sum(sizes[c] * v for c, v in per_cluster.items()) / total

    models = []
    for name, spec in stats["models"].items():
        cluster_tpot = {str(k): float(v) for k, v in spec.get("cluster_tpot_ms", {}).items()}
        tpot = spec.get("tpot_ms")
        if tpot is None:
            if not cluster_tpot:
                raise ValueError(f"Model {name!r} needs tpot_ms or cluster_tpot_ms")
            tpot = _collapse(cluster_tpot)

        cluster_e2el = {str(k): float(v) for k, v in spec.get("cluster_e2el_ms", {}).items()}
        e2el = spec.get("e2el_ms")
        if e2el is None and cluster_e2el:
            e2el = _collapse(cluster_e2el)

        cluster_tokens = {
            str(k): float(v) for k, v in spec.get("cluster_output_tokens", {}).items()
        }

        models.append(
            ModelStats(
                name=name,
                tpot_ms=float(tpot),
                errors={str(k): float(v) for k, v in spec["errors"].items()},
                cluster_tpot_ms=cluster_tpot,
                e2el_ms=float(e2el) if e2el is not None else None,
                cluster_e2el_ms=cluster_e2el,
                cluster_output_tokens=cluster_tokens,
                cost_metric=cost_metric,
            )
        )
    cluster_sizes = {str(k): float(v) for k, v in stats["cluster_sizes"].items()}
    return models, cluster_sizes
