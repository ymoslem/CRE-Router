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
    """

    tiers: Mapping[str, str]
    assign: Mapping[str, str]
    gated: Mapping[str, str] = field(default_factory=dict)
    strong: str | None = None
    tau: float = 0.5
    runs: int = 5
    name: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "Routing":
        d = json.loads(Path(path).read_text())
        keys = {"tiers", "assign", "gated", "strong", "tau", "runs", "name"}
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


def load_accept_probs(path: str | Path, source_file: str | None = None
                      ) -> dict[tuple[str, int], dict]:
    """(qid, run) -> {"p_accept", "num_tokens"} from the estimator's output.

    ``path`` is either a JSONL dump, or a parquet config of the released scores
    dataset, ``ymoslem/cluster-route-escalate-scores``; a parquet file holds
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

    s1_parts, s12_parts, escalated = [], [], {}
    for c, qs in by_cluster.items():
        tier = _grids(caps[routing.assign[c]], qs, runs)
        off = np.zeros((len(qs), runs), dtype=bool)
        s1_parts.append(per_request_metrics(off, ~off, tier, tier))
        if c not in routing.gated:
            s12_parts.append(per_request_metrics(off, ~off, tier, tier))
            continue
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

    def stack(parts):
        return tuple(np.concatenate([p[m] for p in parts], axis=0) for m in range(3))

    (a1, e1, t1), (a2, e2, t2) = stack(s1_parts), stack(s12_parts)

    def mean25(acc, tpot, e2el):
        pairs = [(acc[:, r, s].mean(), tpot[:, r, s].mean(), e2el[:, r, s].mean())
                 for r in range(runs) for s in range(runs)]
        return tuple(float(np.mean([p[i] for p in pairs])) for i in range(3))

    return Composition(
        stage1=mean25(a1, t1, e1), stage1plus2=mean25(a2, t2, e2),
        escalated_per_run=escalated,
        qids=[q for c in by_cluster for q in by_cluster[c]],
        grid_stage1=(a1, t1, e1), grid_stage1plus2=(a2, t2, e2))
