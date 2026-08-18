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

    def tpot_for(self, cluster: str) -> float:
        return self.cluster_tpot_ms.get(cluster, self.tpot_ms)

    def e2el_for(self, cluster: str) -> float:
        """Per-cluster E2EL, independent of the selected cost metric."""
        if cluster in self.cluster_e2el_ms:
            return self.cluster_e2el_ms[cluster]
        if self.e2el_ms is None:
            raise ValueError(f"{self.name}: E2EL requested but none measured")
        return float(self.e2el_ms)

    def output_tokens_for(self, cluster: str) -> float:
        """Mean output length on a cluster; needed to charge an escalated
        query's discarded efficient pass per delivered token."""
        length = self.cluster_output_tokens.get(cluster)
        if length is None:
            raise ValueError(
                f"{self.name}: cascade TPOT needs cluster_output_tokens[{cluster!r}]"
            )
        return length


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


def assign(models: list[ModelStats], lam: float) -> dict[str, str]:
    """Eq. 1: route each cluster to argmin_m Error(m, c) + lambda * Cost_norm(m),
    breaking ties in favour of the faster model."""
    costs = normalized_costs(models)
    table = {}
    for c in clusters_of(models):
        best = min(models, key=lambda m: (m.errors[c] + lam * costs[m.name], m.cost_ms))
        table[c] = best.name
    return table


def dominates(a: ModelStats, b: ModelStats) -> bool:
    """True if ``a`` Pareto-dominates ``b``: no worse on cost and on every
    cluster's error, and strictly better on at least one of those."""
    if a.cost_ms > b.cost_ms:
        return False
    if any(a.errors[c] > b.errors[c] for c in b.errors):
        return False
    return a.cost_ms < b.cost_ms or any(a.errors[c] < b.errors[c] for c in b.errors)


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
        cluster_sizes[c] * by_name[name].cost_for(c) for c, name in assignment.items()
    ) / total
    return acc, tpot


def cascade_system_metrics_ntier(
    models: list[ModelStats],
    cascades: dict[str, list[tuple[str, int | float]]],
    cluster_sizes: dict[str, int | float],
) -> tuple[float, float]:
    """System TPOT and E2EL for an N-tier cascade (>= 1 tier per cluster).

    ``cascades`` maps each cluster to its escalation chain, ordered from the
    Stage 1 (base) model up to the strongest tier:
    ``[(model_0, reach_0), (model_1, reach_1), ...]``. ``reach_i`` is the number
    of the cluster's queries that *execute* tier ``i`` -- ``reach_0`` equals the
    cluster size (every query runs the base model), and the sequence is
    non-increasing as the QE gate at each tier accepts some outputs and escalates
    the rest. A single-element chain is a direct (un-gated) assignment. Two-tier
    ``cascade_system_metrics`` is the special case ``[(eff, size), (strong, count)]``.

    Both metrics are query-weighted, matching Stage 1 and vLLM's per-request Mean
    TPOT. A query delivered at tier ``k`` has executed tiers ``0..k``:

    - **E2EL** is the plain sum of every executed pass, so each query reaching
      tier ``i`` pays ``E2EL_i`` in full: ``E2EL = sum_i reach_i * E2EL_i``.
    - **TPOT** charges all executed passes to the delivered tokens ``L_k``:
      ``(sum_{i<=k} TPOT_i * L_i) / L_k``, generalising the two-tier rule that
      amortises the discarded efficient generation over the delivered answer.

    Returns ``(tpot_ms, e2el_ms)``.
    """
    by_name = {m.name: m for m in models}
    total = sum(cluster_sizes.values())
    tpot_sum = 0.0
    e2el_sum = 0.0
    for cluster, chain in cascades.items():
        if not chain:
            raise ValueError(f"cluster {cluster!r} has an empty cascade chain")
        size = cluster_sizes[cluster]
        names = [t[0] for t in chain]
        reach = [t[1] for t in chain]
        if not math.isclose(reach[0], size):
            raise ValueError(
                f"cluster {cluster!r}: base tier reach {reach[0]} must equal cluster size {size}"
            )
        if any(reach[i + 1] > reach[i] for i in range(len(reach) - 1)):
            raise ValueError(f"cluster {cluster!r}: tier reach must be non-increasing, got {reach}")
        tpots = [by_name[n].tpot_for(cluster) for n in names]
        e2els = [by_name[n].e2el_for(cluster) for n in names]
        lengths = [by_name[n].output_tokens_for(cluster) for n in names]
        cumulative_decode = 0.0
        for k in range(len(names)):
            e2el_sum += reach[k] * e2els[k]
            cumulative_decode += tpots[k] * lengths[k]
            reach_next = reach[k + 1] if k + 1 < len(names) else 0
            delivered_k = reach[k] - reach_next
            tpot_sum += delivered_k * (cumulative_decode / lengths[k])
    return tpot_sum / total, e2el_sum / total


def cascade_system_metrics(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
    escalations: dict[str, tuple[str, float]],
) -> tuple[float, float]:
    """System TPOT and E2EL for a two-tier Stage 1+2 cascade.

    ``assignment`` is the Stage 1 cluster-to-model map. ``escalations`` maps a
    cluster to ``(strong_model, count)``: that many of the cluster's queries run
    the Stage 1 (efficient) model, are judged low-quality by the QE classifier,
    and are then re-run on ``strong_model``. Clusters absent from ``escalations``
    route entirely to their Stage 1 model, so ``system_metrics`` is the special
    case with no escalations.

    A thin wrapper over :func:`cascade_system_metrics_ntier` that builds a
    one- or two-tier chain per cluster. Reproduces the paper's Stage 1+2 latency,
    9.7 ms on AIME and 23.8 ms on TeleQnA.
    """
    cascades: dict[str, list[tuple[str, int | float]]] = {}
    for cluster, name in assignment.items():
        size = cluster_sizes[cluster]
        if cluster in escalations:
            strong_name, count = escalations[cluster]
            cascades[cluster] = [(name, size), (strong_name, count)]
        else:
            cascades[cluster] = [(name, size)]
    return cascade_system_metrics_ntier(models, cascades, cluster_sizes)


def cluster_cascade_accuracy(
    weak_correct: list[bool],
    strong_correct: list[bool],
    escalate: list[bool],
) -> float:
    """Per-cluster accuracy of the Stage 2 cascade, composed per query.

    For each query the QE classifier either accepts the efficient model's output
    (keep ``weak_correct``) or escalates it to the strong model (take
    ``strong_correct``). The three lists are aligned per query on the same
    cluster. This is the exact composition the paper reports: a wrongly escalated
    correct answer (false positive) still gets whatever the strong model returns,
    and a wrongly accepted wrong answer (false negative) stays wrong.
    """
    n = len(weak_correct)
    if not (len(strong_correct) == len(escalate) == n):
        raise ValueError("weak_correct, strong_correct, escalate must align per query")
    if n == 0:
        raise ValueError("cannot compute cascade accuracy over an empty cluster")
    correct = sum(
        (s if esc else w) for w, s, esc in zip(weak_correct, strong_correct, escalate)
    )
    return correct / n


def cascade_system_accuracy(
    models: list[ModelStats],
    assignment: dict[str, str],
    cluster_sizes: dict[str, int | float],
    cascade_accuracy: dict[str, float],
) -> float:
    """System accuracy for the full Stage 1+2 cascade, query-weighted.

    A cluster listed in ``cascade_accuracy`` contributes that measured cascade
    accuracy (efficient outputs gated by the QE classifier, rejects escalated to
    the strong model -- see ``cluster_cascade_accuracy``). A cluster absent from
    it routes entirely to its Stage 1 model and contributes ``1 - error`` for the
    assigned model, so ``system_metrics``' accuracy is the no-escalation special
    case. Reproduces the paper's Stage 1+2 accuracy, 88.4% on AIME and 74.3% on
    TeleQnA.
    """
    by_name = {m.name: m for m in models}
    total = sum(cluster_sizes.values())
    acc = 0.0
    for cluster, name in assignment.items():
        size = cluster_sizes[cluster]
        if cluster in cascade_accuracy:
            acc += size * cascade_accuracy[cluster]
        else:
            acc += size * (1.0 - by_name[name].errors[cluster])
    return acc / total


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
        metric = models[0].cost_metric.upper() if models else "cost"
        raise ValueError(
            f"No routing strategy satisfies budget {budget_ms} ms "
            f"(fastest achievable system {metric} is {fastest:.1f} ms). "
            f"Note the budget is in {metric} units: a per-token TPOT budget is "
            f"orders of magnitude smaller than a per-request E2EL one."
        )
    acc, tpot, region = best
    return Selection(
        lambda_star=region.representative_lambda,
        region=region,
        accuracy=acc,
        tpot_ms=tpot,
        eta=eta(models, region.assignment, cluster_sizes),
    )


def models_from_stats(
    stats: dict, cost_metric: str = "tpot"
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
    models = []
    for name, spec in stats["models"].items():
        cluster_tpot = {str(k): float(v) for k, v in spec.get("cluster_tpot_ms", {}).items()}
        tpot = spec.get("tpot_ms")
        if tpot is None:
            if not cluster_tpot:
                raise ValueError(f"Model {name!r} needs tpot_ms or cluster_tpot_ms")
            tpot = sum(cluster_tpot.values()) / len(cluster_tpot)

        cluster_e2el = {str(k): float(v) for k, v in spec.get("cluster_e2el_ms", {}).items()}
        e2el = spec.get("e2el_ms")
        if e2el is None and cluster_e2el:
            e2el = sum(cluster_e2el.values()) / len(cluster_e2el)

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
