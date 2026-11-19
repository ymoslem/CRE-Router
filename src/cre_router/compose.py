"""Compose a routed system's Stage 1 and Stage 1 + 2 cost and accuracy from captures.

This is how every Stage 1 + 2 number in the paper is produced. Each cluster is
served by its Stage 1 model; in a gated cluster the quality estimator's accept
probability for each (question, run) decides whether that answer is escalated,
and the escalated questions of each run were served again, as their own batch,
by the strong model. Every cost is therefore taken from the batch that was
really served: a question the cascade keeps is priced on its Stage 1 capture,
an escalated one pays for both passes, the strong one priced on its escalation
capture. The accounting of one request across two passes is
:func:`cre_router.cascade.per_request_metrics`; this module supplies what it
consumes.

With five efficient runs and five strong runs a system measurement exists for
every (efficient run r, strong run s) pair, 25 in all, and the reported figure
is their mean. A question the cascade never escalates is priced on the ``s``
axis, the axis a Stage 1 baseline resamples; a gated cluster's efficient pass is
priced on ``r``, because the escalation decision was made from that run's
output. The two indices are not symmetric and must not be made so.

That is the ``per_cluster`` basis. On the ``per_model`` basis (``Routing.basis``)
every model instead served exactly the traffic the routing sends it as one
batch: a model routed several clusters served them together, and the strong
model served its own clusters together with each run's escalations. Each
request is then priced from that one batch, and a capture that does not hold
exactly the model's routed traffic is refused (``_compose_per_model``).

Three things are refused rather than guessed:

- A gated question with no accept probability. Treating it as accepted would
  silently change which questions escalate.
- Accept probabilities scored on a different sampling run from the capture they
  gate. Question ids and run numbers line up between any two captures of the
  same model, so only output length can tell them apart, and it must agree.
- An escalation capture that does not hold exactly the questions its run
  escalates. A run that escalates nobody needs no capture, and has none.

Usage::

    from cre_router.captures import load_capture
    from cre_router.compose import Routing, compose
    routing = Routing.from_json("configs/routings/telemath_tpot_b20_1xA100.json")
    result = compose(routing, lambda tag: load_capture(tag, "captures/"), probs)
    result.stage1plus2      # (accuracy, TPOT ms, E2EL s)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from cre_router.cascade import per_request_metrics

__all__ = ["Routing", "Composition", "compose", "load_accept_probs"]

#: Share of (question, run) rows whose output length must match between the
#: accept probabilities and the capture they gate. The reported compositions
#: were checked at this threshold; every one agrees on every row.
PAIRING_AGREEMENT = 0.99


@dataclass(frozen=True)
class Routing:
    """A routed system: which model serves each cluster, and which are gated.

    ``tiers`` names each model's Stage 1 capture; ``assign`` maps cluster to
    model name; ``gated`` maps a gated cluster to the stem of its escalation
    captures, run ``r`` being ``<stem>_r<r>``, and clusters whose escalations
    were served together share a stem; ``strong`` is the model those
    escalations were served by, for the record.

    ``basis`` says how the system was served. ``"per_cluster"`` (the default):
    every model served each cluster as its own batch, and each run's escalations
    as their own batch. ``"per_model"``: every model served exactly the traffic
    the routing sends it, as one batch. Then ``tiers`` still fixes the clusters,
    and prices a model only where it serves a single cluster; ``served`` names
    the one-batch Stage 1 capture of every model that serves several clusters;
    and ``strong_batch`` is the stem of the strong model's one-batch captures,
    run ``r`` being ``<stem>_r<r>``, each holding the strong model's own Stage 1
    clusters and run ``r``'s escalations together. Every gated cluster then maps
    to ``strong_batch``.
    """

    tiers: Mapping[str, str]
    assign: Mapping[str, str]
    gated: Mapping[str, str] = field(default_factory=dict)
    strong: str | None = None
    tau: float = 0.5
    runs: int = 5
    name: str = ""
    basis: str = "per_cluster"
    served: Mapping[str, str] = field(default_factory=dict)
    strong_batch: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Routing":
        d = json.loads(Path(path).read_text())
        keys = {"tiers", "assign", "gated", "strong", "tau", "runs", "name",
                "basis", "served", "strong_batch"}
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass(frozen=True)
class Composition:
    """Means over the 25 (efficient run, strong run) measurements, and the grids."""

    stage1: tuple[float, float, float]
    stage1plus2: tuple[float, float, float]
    escalated_per_run: dict[str, list[int]]
    #: Question ids in the order the grids stack them, cluster by cluster.
    qids: list[str]
    #: (accuracy, TPOT, E2EL) grids, each (question, efficient run, strong run).
    grid_stage1: tuple[np.ndarray, np.ndarray, np.ndarray]
    grid_stage1plus2: tuple[np.ndarray, np.ndarray, np.ndarray]
    #: Stage 1 laid on Stage 1 + 2's run axes, for what Stage 2 adds: in a gated
    #: cluster the efficient pass is priced on the efficient run ``r`` its
    #: escalation was decided from, as in ``grid_stage1plus2``; elsewhere on ``s``,
    #: as in ``grid_stage1``. ``grid_stage1plus2 - grid_stage1_paired`` is then
    #: exactly zero for every answer Stage 2 keeps, and the difference comes only
    #: from the escalated ones. ``grid_stage1`` prices every cluster on ``s``, so
    #: subtracting it compares a gated question's efficient run ``r`` with a
    #: different run ``s`` and adds their run-to-run spread to the difference.
    #: Same mean as ``grid_stage1``; only the pairing of cells differs.
    grid_stage1_paired: tuple[np.ndarray, np.ndarray, np.ndarray]


def load_accept_probs(path: str | Path, source_file: str | None = None
                      ) -> dict[tuple[str, int], dict]:
    """(qid, run) -> {"p_accept", "num_tokens"} from the estimator's output.

    ``path`` is either a JSONL dump, or a parquet config of the released scores
    dataset, ``ymoslem/cre-decision-scores``; a parquet file holds
    several dumps, so ``source_file`` picks one (reading parquet needs the
    ``data`` extra).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
        if source_file is None:
            raise ValueError(f"{path.name} holds several dumps; pass source_file")
        rows = [r for r in rows if r.get("source_file") == source_file]
        if not rows:
            raise ValueError(f"{path.name}: no rows from {source_file}")
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(str(r["qid"]), int(r["run"])): {"p_accept": float(r["p_accept"]),
                                              "num_tokens": r.get("num_tokens")}
            for r in rows}


def _clusters(routing: Routing, captures: Mapping[str, Mapping]) -> dict[str, str]:
    """qid -> cluster, read from the Stage 1 captures, which must all agree."""
    qid2c: dict[str, str] = {}
    for name, cap in captures.items():
        for (q, _run), row in cap.items():
            c = str(row["cluster"])
            if qid2c.setdefault(q, c) != c:
                raise ValueError(f"{routing.tiers[name]} puts question {q} in cluster "
                                 f"{c}, another capture in {qid2c[q]}")
    missing = set(routing.assign) - set(qid2c.values())
    if missing:
        raise ValueError(f"the captures hold no question in cluster(s) {sorted(missing)}")
    return qid2c


def _grids(cap: Mapping, qs: list[str], runs: int, keep=None) -> dict[str, np.ndarray]:
    out = {f: np.zeros((len(qs), runs)) for f in ("correct", "e2el", "tokens", "tpot")}
    for i, q in enumerate(qs):
        if keep is not None and q not in keep:
            continue
        for r in range(runs):
            d = cap[(q, r)]
            out["correct"][i, r] = d["correct"]
            out["e2el"][i, r] = d["e2el_s"]
            out["tokens"][i, r] = d["tokens"]
            out["tpot"][i, r] = d["tpot_ms"]
    return out


def _check_pairing(probs, cap, qs, what: str) -> None:
    seen = hit = 0
    for q in qs:
        for (qq, r), row in ((k, v) for k, v in cap.items() if k[0] == q):
            p = probs.get((q, r))
            if p is None or p["num_tokens"] is None:
                continue
            seen += 1
            hit += p["num_tokens"] == row["tokens"]
    if seen == 0:
        raise ValueError(f"{what}: no accept probability carries an output length to "
                         f"check against the capture it gates")
    if hit / seen < PAIRING_AGREEMENT:
        raise ValueError(f"{what}: only {hit} of {seen} accept probabilities agree with "
                         f"the capture on output length, so they scored a different "
                         f"sampling run and cannot gate it")


def compose(routing: Routing, load: Callable[[str], Mapping],
            probs: Mapping[tuple[str, int], Mapping]) -> Composition:
    """Stage 1 and Stage 1 + 2 of ``routing``, priced on the batches actually served.

    ``load`` returns one capture as ``cre_router.captures.load_capture`` does,
    ``{(qid, run): {correct, tokens, tpot_ms, e2el_s, cluster, ...}}``; ``probs``
    maps (qid, run) to the estimator's ``p_accept`` and the ``num_tokens`` of the
    answer it scored.
    """
    if routing.basis == "per_model":
        return _compose_per_model(routing, load, probs)
    if routing.basis != "per_cluster":
        raise ValueError(f"unknown basis {routing.basis!r}")
    if routing.served or routing.strong_batch:
        raise ValueError("served and strong_batch belong to the per_model basis")
    runs = routing.runs
    used = sorted(set(routing.assign.values()))
    caps = {name: load(routing.tiers[name]) for name in used}
    qid2c = _clusters(routing, caps)
    by_cluster = {c: sorted((q for q, cc in qid2c.items() if cc == c),
                            key=lambda q: (len(q), q))
                  for c in sorted(routing.assign)}

    # Which (question, run) each gated cluster escalates, from the estimator.
    esc_sets: dict[str, list[set[str]]] = {}
    for c in routing.gated:
        qs = by_cluster[c]
        _check_pairing(probs, caps[routing.assign[c]], qs, f"cluster {c}, {routing.assign[c]}")
        sets = []
        for r in range(runs):
            missing = [q for q in qs if (q, r) not in probs]
            if missing:
                raise ValueError(f"cluster {c} run {r}: {len(missing)} gated questions "
                                 f"have no accept probability, e.g. {missing[0]}")
            sets.append({q for q in qs if probs[(q, r)]["p_accept"] < routing.tau})
        esc_sets[c] = sets

    # Each run's escalations were served as one batch per stem. Several gated
    # clusters may share a stem, so the batch must hold exactly their union.
    served: dict[tuple[str, int], Mapping] = {}
    for stem in sorted(set(routing.gated.values())):
        members = [c for c, s in routing.gated.items() if s == stem]
        for r in range(runs):
            want = set().union(*(esc_sets[c][r] for c in members))
            if not want:
                continue
            cap = load(f"{stem}_r{r}")
            got = {q for q, _ in cap}
            if got != want:
                raise ValueError(f"{stem}_r{r} holds {len(got)} questions but run {r} "
                                 f"escalates {len(want)} in cluster(s) {', '.join(members)}")
            served[(stem, r)] = cap

    s1_parts, s1p_parts, s12_parts, escalated = [], [], [], {}
    for c, qs in by_cluster.items():
        tier = _grids(caps[routing.assign[c]], qs, runs)
        off = np.zeros((len(qs), runs), dtype=bool)
        s1_parts.append(per_request_metrics(off, ~off, tier, tier))
        if c not in routing.gated:
            s1p_parts.append(per_request_metrics(off, ~off, tier, tier))
            s12_parts.append(per_request_metrics(off, ~off, tier, tier))
            continue
        # The same Stage 1 answers, priced on the efficient run axis r.
        s1p_parts.append(per_request_metrics(~off, off, tier, [tier] * runs))
        sets, stem = esc_sets[c], routing.gated[c]
        strong = [_grids(served[(stem, r)], qs, runs, keep=sets[r]) if sets[r] else None
                  for r in range(runs)]
        esc = np.array([[q in sets[r] for r in range(runs)] for q in qs])
        if all(x is None for x in strong):
            # Nobody escalated in any run, so no strong capture exists to size the
            # strong-run axis. The masks never read it; the tier's own grid stands
            # in, keeping the efficient pass priced on the run it was decided from.
            strong = [tier] * runs
        s12_parts.append(per_request_metrics(~off, esc, tier, strong))
        escalated[c] = [len(x) for x in sets]

    return _finish(s1_parts, s1p_parts, s12_parts, escalated, by_cluster, runs)


def _finish(s1_parts, s1p_parts, s12_parts, escalated, by_cluster, runs) -> Composition:
    def stack(parts):
        return tuple(np.concatenate([p[m] for p in parts], axis=0) for m in range(3))

    (a1, e1, t1), (a2, e2, t2) = stack(s1_parts), stack(s12_parts)
    a1p, e1p, t1p = stack(s1p_parts)

    def mean25(acc, tpot, e2el):
        pairs = [(acc[:, r, s].mean(), tpot[:, r, s].mean(), e2el[:, r, s].mean())
                 for r in range(runs) for s in range(runs)]
        return tuple(float(np.mean([p[i] for p in pairs])) for i in range(3))

    return Composition(
        stage1=mean25(a1, t1, e1), stage1plus2=mean25(a2, t2, e2),
        escalated_per_run=escalated,
        qids=[q for c in by_cluster for q in by_cluster[c]],
        grid_stage1=(a1, t1, e1), grid_stage1plus2=(a2, t2, e2),
        grid_stage1_paired=(a1p, t1p, e1p))


def _check_one_batch(cap: Mapping, tag: str, runs: int, stage1: set[str],
                     stage2: set[str] = frozenset(), eff_run: int | None = None) -> None:
    """Refuse a one-batch capture that is not exactly the traffic it must hold.

    It must have been served as one batch (every row carries the same batch
    label), in every run, holding exactly ``stage1`` as Stage 1 rows and exactly
    ``stage2`` as Stage 2 rows escalated from efficient run ``eff_run``.
    """
    labels = {row.get("batch") for row in cap.values()}
    if None in labels or len(labels) != 1:
        raise ValueError(f"{tag}: not served as one batch (batch labels {sorted(map(str, labels))})")
    for r in range(runs):
        got1 = {q for (q, rr), row in cap.items() if rr == r and row.get("stage") == 1}
        got2 = {q for (q, rr), row in cap.items() if rr == r and row.get("stage") == 2}
        if got1 != stage1:
            raise ValueError(f"{tag} run {r}: {len(got1)} Stage 1 rows, the routing sends "
                             f"this model {len(stage1)}; differ on {sorted(got1 ^ stage1)[:3]}")
        if got2 != set(stage2):
            raise ValueError(f"{tag} run {r}: {len(got2)} escalated rows, the routing "
                             f"escalates {len(stage2)}; differ on {sorted(got2 ^ set(stage2))[:3]}")
        bad = [q for (q, rr), row in cap.items()
               if rr == r and row.get("stage") == 2 and row.get("eff_run") != eff_run]
        if bad:
            raise ValueError(f"{tag} run {r}: escalations from another efficient run, e.g. {bad[0]}")
    extra = {rr for _, rr in cap} - set(range(runs))
    if extra:
        raise ValueError(f"{tag}: runs {sorted(extra)} beyond the {runs} expected")


def _compose_per_model(routing: Routing, load: Callable[[str], Mapping],
                       probs: Mapping[tuple[str, int], Mapping]) -> Composition:
    """Stage 1 and Stage 1 + 2 when every model served exactly its routed traffic.

    Each request is priced from the one batch its model served in this system:

    - a model routed one cluster, on that cluster's batch in its ``tiers`` capture;
    - a model routed several clusters, on its one-batch capture in ``served``;
    - in Stage 1 + 2, the strong model's own clusters and every escalation, on
      ``<strong_batch>_r<r>``, the batch it served while run ``r``'s escalations
      arrived. Its own clusters are therefore priced on (r, s) in Stage 1 + 2:
      escalations share its server and change what its other requests cost.

    The escalation decisions must have scored the answers priced here, so a gated
    cluster whose model serves several clusters needs accept probabilities for
    that model's one-batch answers; the output-length check refuses any others.
    """
    runs = routing.runs
    used = sorted(set(routing.assign.values()))
    tiers = {name: load(routing.tiers[name]) for name in used}
    qid2c = _clusters(routing, tiers)
    by_cluster = {c: sorted((q for q, cc in qid2c.items() if cc == c),
                            key=lambda q: (len(q), q))
                  for c in sorted(routing.assign)}
    clusters_of = {m: sorted(c for c, mm in routing.assign.items() if mm == m) for m in used}
    traffic = {m: set().union(*(by_cluster[c] for c in cs)) for m, cs in clusters_of.items()}

    if set(routing.served) - set(used):
        raise ValueError(f"served names models the routing does not use: "
                         f"{sorted(set(routing.served) - set(used))}")
    stage1_cap: dict[str, Mapping] = {}
    for m in used:
        if len(clusters_of[m]) == 1:
            if m in routing.served:
                raise ValueError(f"{m} serves one cluster, so its cluster batch is its whole "
                                 f"traffic; a served capture for it would be a second source")
            stage1_cap[m] = {k: v for k, v in tiers[m].items() if k[0] in traffic[m]}
            continue
        if m not in routing.served:
            raise ValueError(f"{m} serves clusters {', '.join(clusters_of[m])}; on the per_model "
                             f"basis it needs one batch holding all of them (served)")
        cap = load(routing.served[m])
        _check_one_batch(cap, routing.served[m], runs, traffic[m])
        stage1_cap[m] = cap

    if routing.gated:
        if not routing.strong_batch or set(routing.gated.values()) != {routing.strong_batch}:
            raise ValueError("on the per_model basis every gated cluster maps to strong_batch")
        if routing.strong in {routing.assign[c] for c in routing.gated}:
            raise ValueError("the strong model cannot be the efficient model of a gated cluster")
    elif routing.strong_batch:
        raise ValueError("strong_batch given but no cluster is gated")

    esc_sets: dict[str, list[set[str]]] = {}
    for c in routing.gated:
        qs = by_cluster[c]
        _check_pairing(probs, stage1_cap[routing.assign[c]], qs,
                       f"cluster {c}, {routing.assign[c]}")
        sets = []
        for r in range(runs):
            missing = [q for q in qs if (q, r) not in probs]
            if missing:
                raise ValueError(f"cluster {c} run {r}: {len(missing)} gated questions "
                                 f"have no accept probability, e.g. {missing[0]}")
            sets.append({q for q in qs if probs[(q, r)]["p_accept"] < routing.tau})
        esc_sets[c] = sets

    strong_own = traffic.get(routing.strong, set()) if routing.gated else set()
    strong_r: list[Mapping | None] = [None] * runs
    for r in range(runs):
        want2 = set().union(*(esc_sets[c][r] for c in routing.gated)) if routing.gated else set()
        if not want2 and not strong_own:
            continue
        tag = f"{routing.strong_batch}_r{r}"
        cap = load(tag)
        _check_one_batch(cap, tag, runs, strong_own, want2, eff_run=r)
        strong_r[r] = cap

    s1_parts, s1p_parts, s12_parts, escalated = [], [], [], {}
    for c, qs in by_cluster.items():
        m = routing.assign[c]
        tier = _grids(stage1_cap[m], qs, runs)
        off = np.zeros((len(qs), runs), dtype=bool)
        s1_parts.append(per_request_metrics(off, ~off, tier, tier))
        if c in routing.gated:
            s1p_parts.append(per_request_metrics(~off, off, tier, [tier] * runs))
            sets = esc_sets[c]
            strong = [_grids(strong_r[r], qs, runs, keep=sets[r]) if sets[r] else tier
                      for r in range(runs)]
            esc = np.array([[q in sets[r] for r in range(runs)] for q in qs])
            s12_parts.append(per_request_metrics(~off, esc, tier, strong))
            escalated[c] = [len(x) for x in sets]
        elif m == routing.strong and routing.gated:
            # The strong model's own cluster in Stage 1 + 2: answered on the batch
            # it shared with run r's escalations, so priced on (r, s).
            own = [_grids(strong_r[r], qs, runs) for r in range(runs)]
            s1p_parts.append(per_request_metrics(off, ~off, tier, tier))
            s12_parts.append(per_request_metrics(off, ~off, own[0], own))
        else:
            s1p_parts.append(per_request_metrics(off, ~off, tier, tier))
            s12_parts.append(per_request_metrics(off, ~off, tier, tier))

    return _finish(s1_parts, s1p_parts, s12_parts, escalated, by_cluster, runs)
