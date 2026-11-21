"""Batch files that serve a routing as it is deployed: each model one batch.

On the ``per_model`` basis (:class:`cre_router.compose.Routing`) every model
serves exactly the traffic the routing sends it, as one batch:

- **Stage 1**: each model the routing uses gets one batch holding every question
  of the clusters routed to it, served with ``cre evaluate --one-batch``.
- **Stage 1 + 2**: for each run ``r`` of the efficient models, the strong model
  gets one batch holding its own Stage 1 clusters together with run ``r``'s
  escalations, the gated questions whose answer the estimator rejected
  (:func:`cre_router.compose.escalates`, the serving classifier's rule). The escalation decisions must score the efficient
  models' Stage 1 one-batch answers, so these batches are built only after those
  answers exist and have been scored.

Rows keep their true ``cluster``, a ``stage`` of 1 or 2 and, for Stage 2, the
``eff_run`` they escalate from (:func:`cre_router.evaluate.check_one_batch`).
Each batch is put in a fixed random order, so clusters and stages are
interleaved as in a live workload rather than served one after another.

:func:`write_plan` writes the batches, a manifest with each file's SHA-256, and
the routing file that :func:`cre_router.compose.compose` reads once the batches
have been served under the tags it names.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Mapping

from cre_router.compose import escalates
from cre_router.evaluate import check_one_batch, question_id

__all__ = ["stage1_batches", "strong_batch", "write_plan", "slug"]


def slug(name: str) -> str:
    """A model name as a tag component: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _row(item: Mapping, stage: int, eff_run: int | None = None) -> dict:
    row = {"id": question_id(item), "prompt": item["prompt"], "answer": item["answer"],
           "cluster": str(item["cluster"]), "stage": stage}
    if "question" in item:
        row["question"] = item["question"]
    if eff_run is not None:
        row["eff_run"] = eff_run
    return row


def _shuffled(rows: list[dict], seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["stage"], r["id"]))
    random.Random(seed).shuffle(rows)
    return rows


def _traffic(assign: Mapping[str, str], model: str, dataset: list[Mapping]) -> list[Mapping]:
    clusters = {c for c, m in assign.items() if m == model}
    rows = [item for item in dataset if str(item["cluster"]) in clusters]
    found = {str(item["cluster"]) for item in rows}
    if found != clusters:
        raise ValueError(f"{model}: its dataset holds no question in cluster(s) "
                         f"{sorted(clusters - found)}")
    return rows


def stage1_batches(assign: Mapping[str, str], datasets: Mapping[str, list[Mapping]],
                   seed: int = 0) -> dict[str, list[dict]]:
    """Model -> its Stage 1 batch: every question of the clusters routed to it.

    ``datasets`` gives each model its own copy of the questions, since prompts
    can be rendered per model (with or without reasoning, for example).
    """
    out = {}
    for model in sorted(set(assign.values())):
        if model not in datasets:
            raise ValueError(f"no dataset for {model}")
        rows = [_row(item, 1) for item in _traffic(assign, model, datasets[model])]
        check_one_batch(rows)
        out[model] = _shuffled(rows, seed)
    return out


def strong_batch(assign: Mapping[str, str], strong: str, gated: list[str],
                 datasets: Mapping[str, list[Mapping]],
                 probs: Mapping[tuple[str, int], Mapping], run: int, tau: float,
                 seed: int = 0) -> list[dict]:
    """The strong model's Stage 1 + 2 batch for efficient run ``run``.

    Its own Stage 1 clusters, plus every gated question whose run-``run`` answer
    is escalated (:func:`cre_router.compose.escalates`), posed to the strong model
    with its own prompt.
    """
    if strong in {assign[c] for c in gated}:
        raise ValueError("the strong model cannot be the efficient model of a gated cluster")
    if strong not in datasets:
        raise ValueError(f"no dataset for the strong model {strong}")
    by_qid = {question_id(item): item for item in datasets[strong]}
    rows = ([_row(item, 1) for item in _traffic(assign, strong, datasets[strong])]
            if strong in assign.values() else [])
    for c in gated:
        for item in datasets[strong]:
            if str(item["cluster"]) != c:
                continue
            q = question_id(item)
            if (q, run) not in probs:
                raise ValueError(f"cluster {c} run {run}: question {q} has no accept probability")
            if escalates(probs[(q, run)]["p_accept"], tau):
                rows.append(_row(by_qid[q], 2, eff_run=run))
    check_one_batch(rows)
    return _shuffled(rows, seed)


def _write(out_dir: Path, files: Mapping[str, list[dict]]) -> list[dict]:
    manifest = []
    for tag, rows in files.items():
        path = out_dir / f"{tag}.jsonl"
        if path.exists():
            raise FileExistsError(f"{path} exists; batches are written once")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest.append({"tag": tag, "file": path.name, "rows": len(rows),
                         "stage2_rows": sum(r["stage"] == 2 for r in rows),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return manifest


def _write_json(path: Path, payload) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    path.write_text(json.dumps(payload, indent=1) + "\n")


def write_plan(out_dir: str | Path, prefix: str, name: str, assign: Mapping[str, str],
               datasets: Mapping[str, list[Mapping]], *, strong: str | None = None,
               gated: list[str] | None = None,
               probs: Mapping[tuple[str, int], Mapping] | None = None,
               tau: float = 0.5, runs: int = 5, seed: int = 0) -> dict:
    """Write a routing's batches, their manifest and, when complete, its routing file.

    Tags are ``<prefix>_s1_<name>_<model>`` for Stage 1 and
    ``<prefix>_s12_<name>_<strong>_r<r>`` for the strong model's Stage 1 + 2
    batches. Without ``probs`` this writes the Stage 1 batches
    (``<name>_s1_manifest.json``). With ``probs``, the scores of the efficient
    models' Stage 1 answers, it writes the strong model's batches
    (``<name>_s12_manifest.json``) after checking that the Stage 1 batches on
    disk are the ones this plan builds. The ``per_model`` routing file
    (``<name>_routing.json``) is written by the call that completes the plan: the
    first for a routing that gates nothing, the second otherwise. Refuses to
    overwrite any file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gated = sorted(gated or [])
    if gated and strong is None:
        raise ValueError("a gated routing needs its strong model")
    s1 = {m: f"{prefix}_s1_{name}_{slug(m)}" for m in sorted(set(assign.values()))}
    batches = {s1[m]: rows for m, rows in stage1_batches(assign, datasets, seed).items()}
    multi = {m for m in s1 if sum(v == m for v in assign.values()) > 1}
    routing = {"name": name, "basis": "per_model", "assign": dict(assign),
               "tiers": dict(s1), "served": {m: s1[m] for m in sorted(multi)},
               "tau": tau, "runs": runs}
    if probs is None:
        _write_json(out_dir / f"{name}_s1_manifest.json", _write(out_dir, batches))
        if not gated:
            _write_json(out_dir / f"{name}_routing.json", routing)
        return routing
    if not gated:
        raise ValueError("scores given for a routing that gates nothing")
    for tag, rows in batches.items():
        path = out_dir / f"{tag}.jsonl"
        want = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if not path.exists() or path.read_text() != want:
            raise ValueError(f"{path} is missing or differs from this plan's Stage 1 batch; "
                             f"write the Stage 1 batches first, from the same inputs")
    stem = f"{prefix}_s12_{name}_{slug(strong)}"
    files = {f"{stem}_r{r}": strong_batch(assign, strong, gated, datasets, probs, r, tau, seed)
             for r in range(runs)}
    _write_json(out_dir / f"{name}_s12_manifest.json", _write(out_dir, files))
    routing.update({"strong": strong, "gated": {c: stem for c in gated}, "strong_batch": stem})
    _write_json(out_dir / f"{name}_routing.json", routing)
    return routing
