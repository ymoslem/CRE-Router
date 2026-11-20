"""Build the per-cluster stats file ``cre fit`` reads, from saved captures.

``cre evaluate`` writes a model's stats as it serves it. This rebuilds the same
file afterwards from the captures, so a pool can be reassembled from released
data and regraded under the current grader without serving anything again. It
goes through the same path ``cre evaluate`` does: one
:class:`~cre_router.evaluate.RunMeasurement` per (cluster, run), averaged by
:func:`~cre_router.evaluate.model_entry`. The only difference is where the
numbers come from:

- **error** is regraded from each answer by the task's own grader, through
  :mod:`cre_router.captures`, never read from a stored verdict;
- **TPOT, TTFT and E2EL** are the batch's own means as the run recorded them.
  Where per-request detail exists they are also its average, which
  :mod:`cre_router.captures` checks; runs that kept only batch means are read
  with ``require_cost=False``, so both kinds of capture give the same file.
- A capture served as **one batch** (``cre evaluate --one-batch``) holds every
  cluster in a single batch, so its batch means describe the whole set. Each
  cluster's cost is then the mean of its own requests' per-request costs, the
  same statistic the batch mean is over all requests, taken under the load of
  the whole set. This needs the per-request detail.
- Because a one-batch capture prices every request on its own, the same capture
  serves **any clustering** of its questions: pass ``clusters`` (question id to
  cluster) and each request is counted under that cluster instead of the one it
  was served with. This is how the no-clustering ablation (every question in one
  cluster) and other values of k are priced without serving anything again.
  Captures served one batch per cluster cannot be relabelled, since their costs
  are batch means of the clusters they were served in.

A pool spec names each model's capture::

    {"name": "...", "_note": "...", "error_tol": 0.001,
     "models": {"Gemma4-E2B": "tm_train_e2b_nothink_regen", ...}}

Usage::

    cre stats --pool configs/pools/telemath_1xA100_Sep2026.json --dataset DIR --out stats.json
    cre stats --pool POOL --captures DIR --clusters train_k2.jsonl --out stats_k2.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

from cre_router.evaluate import SCORER_VERSION, RunMeasurement, model_entry

__all__ = ["measurements", "build_stats", "relabel", "load_clusters"]


def relabel(capture: Mapping[tuple[str, int], Mapping],
            clusters: Mapping[str, str]) -> dict[tuple[str, int], dict]:
    """The capture with each request counted under ``clusters[qid]``.

    Only a one-batch capture can be relabelled: its costs are per request, so a
    request's cost does not depend on which cluster it is counted under.
    """
    out = {}
    for (q, run), row in capture.items():
        if row.get("batch") is None:
            raise ValueError(f"question {q}: only a capture served as one batch can be "
                             f"priced under another clustering")
        if q not in clusters:
            raise ValueError(f"question {q} has no cluster in the given clustering")
        out[(q, run)] = {**row, "cluster": str(clusters[q])}
    extra = set(clusters) - {q for q, _ in capture}
    if extra:
        raise ValueError(f"the clustering names {len(extra)} questions the capture does not hold")
    return out


def load_clusters(path: str | Path) -> dict[str, str]:
    """Question id to cluster, from a JSONL dataset with ``cluster`` on every row."""
    from cre_router.evaluate import question_id

    out: dict[str, str] = {}
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if line.strip():
            row = json.loads(line)
            if "cluster" not in row:
                raise ValueError(f"{path} row {i}: no cluster")
            q = question_id(row)
            if q in out:
                raise ValueError(f"{path}: question {q} appears twice")
            out[q] = str(row["cluster"])
    return out


def measurements(capture: Mapping[tuple[str, int], Mapping]) -> list[RunMeasurement]:
    """One measurement per (cluster, run) of a capture, error regraded."""
    groups: dict[tuple[str, int], list[Mapping]] = {}
    for (_q, run), row in capture.items():
        groups.setdefault((str(row["cluster"]), int(run)), []).append(row)
    out = []
    for (cluster, run), rows in sorted(groups.items()):
        if any(r.get("batch") is not None for r in rows):
            out.append(_from_one_batch(cluster, run, rows))
            continue
        batch = rows[0]
        for key in ("batch_tpot_ms", "batch_ttft_ms", "batch_e2el_ms"):
            if any(r.get(key) != batch.get(key) for r in rows):
                raise ValueError(f"cluster {cluster} run {run}: rows disagree on {key}, so "
                                 f"they were not served as one batch")
        if batch.get("batch_tpot_ms") is None:
            raise ValueError(f"cluster {cluster} run {run}: no batch means to take cost from")
        out.append(RunMeasurement(
            cluster=cluster, run=run,
            error=1.0 - sum(bool(r["correct"]) for r in rows) / len(rows),
            tpot_ms=float(batch["batch_tpot_ms"]), num_prompts=len(rows),
            ttft_ms=batch.get("batch_ttft_ms"), e2el_ms=batch.get("batch_e2el_ms")))
    return out


def _from_one_batch(cluster: str, run: int, rows: list[Mapping]) -> RunMeasurement:
    """A cluster's measurement from its own requests within a one-batch capture."""
    if any(r.get("batch") is None for r in rows):
        raise ValueError(f"cluster {cluster} run {run}: some rows come from one batch and "
                         f"some do not")
    if any(r.get("stage", 1) != 1 for r in rows):
        raise ValueError(f"cluster {cluster} run {run}: escalated rows in a batch read for "
                         f"per-cluster statistics")
    if any(r.get(k) is None for r in rows for k in ("tpot_ms", "ttft_s", "e2el_s")):
        raise ValueError(f"cluster {cluster} run {run}: a one-batch capture needs per-request "
                         f"detail to price each cluster")
    n = len(rows)
    return RunMeasurement(
        cluster=cluster, run=run,
        error=1.0 - sum(bool(r["correct"]) for r in rows) / n,
        tpot_ms=sum(float(r["tpot_ms"]) for r in rows) / n, num_prompts=n,
        ttft_ms=sum(float(r["ttft_s"]) for r in rows) / n * 1000,
        e2el_ms=sum(float(r["e2el_s"]) for r in rows) / n * 1000)


def build_stats(pool: Mapping, load: Callable[[str], Mapping],
                clusters: Mapping[str, str] | None = None) -> dict:
    """The stats dict for every model in ``pool``, from its capture.

    ``load(tag)`` returns a capture with batch means attached, as
    ``cre_router.captures.load_capture(..., require_cost=False)`` and
    ``cre_router.captures.dataset_loader`` do. Every model must cover the same
    questions in the same clusters, or the file would mix populations.
    ``clusters`` prices one-batch captures under another clustering (see
    :func:`relabel`).
    """
    sizes, models = None, {}
    for name, tag in pool["models"].items():
        cap = load(tag)
        if clusters is not None:
            cap = relabel(cap, clusters)
        runs = {r for _, r in cap}
        n = {}
        for (q, _r), row in cap.items():
            n.setdefault(str(row["cluster"]), set()).add(q)
        n = {c: len(qs) for c, qs in sorted(n.items())}
        if sizes is None:
            sizes = n
        elif n != sizes:
            raise ValueError(f"{tag}: cluster sizes {n} differ from the pool's {sizes}")
        if any(len([k for k in cap if k[1] == r]) != sum(n.values()) for r in runs):
            raise ValueError(f"{tag}: not every run answered every question")
        models[name] = model_entry(measurements(cap))
        if pool.get("vllm_version"):
            # The serving stack, stated per model as `cre evaluate` records it,
            # because a pool can mix stacks and one value at the top would hide it.
            models[name]["vllm_version"] = pool["vllm_version"]
    out = {k: v for k, v in pool.items() if k not in ("models", "name")}
    out.update({"cluster_sizes": sizes, "models": models, "scorer_version": SCORER_VERSION,
                "built_from": dict(pool["models"])})
    return out


def load_pool(path: str | Path) -> dict:
    pool = json.loads(Path(path).read_text())
    if not pool.get("models"):
        raise ValueError(f"{path}: a pool spec needs a models map of name to capture tag")
    return pool
