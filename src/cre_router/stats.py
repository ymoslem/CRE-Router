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

A pool spec names each model's capture::

    {"name": "...", "_note": "...", "error_tol": 0.001,
     "models": {"Gemma4-E2B": "tm_train_e2b_nothink_regen", ...}}

Usage::

    cre stats --pool configs/pools/telemath_1xA100_Sep2026.json --dataset DIR --out stats.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

from cre_router.evaluate import SCORER_VERSION, RunMeasurement, model_entry

__all__ = ["measurements", "build_stats"]


def measurements(capture: Mapping[tuple[str, int], Mapping]) -> list[RunMeasurement]:
    """One measurement per (cluster, run) of a capture, error regraded."""
    groups: dict[tuple[str, int], list[Mapping]] = {}
    for (_q, run), row in capture.items():
        groups.setdefault((str(row["cluster"]), int(run)), []).append(row)
    out = []
    for (cluster, run), rows in sorted(groups.items()):
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


def build_stats(pool: Mapping, load: Callable[[str], Mapping]) -> dict:
    """The stats dict for every model in ``pool``, from its capture.

    ``load(tag)`` returns a capture with batch means attached, as
    ``cre_router.captures.load_capture(..., require_cost=False)`` and
    ``cre_router.captures.dataset_loader`` do. Every model must cover the same
    questions in the same clusters, or the file would mix populations.
    """
    sizes, models = None, {}
    for name, tag in pool["models"].items():
        cap = load(tag)
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
