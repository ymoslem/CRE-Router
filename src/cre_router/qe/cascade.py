"""Stage 1+2 cascade evaluation.

Runs the QE classifier over an efficient model's per-cluster generations,
escalates the rejected outputs to the strong model, and composes the per-cluster
cascade accuracy and escalation counts that ``cre cascade`` consumes
(``routing.cascade_system_accuracy`` / ``cascade_system_metrics``).

The composition is split so the arithmetic is testable without a GPU:
``compose_cascade`` is a pure function over already-made accept/route decisions,
and ``run_qe`` is the thin wrapper that produces those decisions from a trained
classifier. Requires the ``qe`` extra only for ``run_qe``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def strong_correct_by_qid(outcomes: list[dict]) -> dict[str, float]:
    """Mean correctness per question id from the strong model's records.

    Accepts either the strong model's ``*_outcomes.jsonl`` or its
    ``*_generations.jsonl`` (both carry ``qid`` and ``correct``). When the strong
    model was run several times, the per-qid mean is its expected correctness on
    that query, which is what an escalation to it earns.
    """
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for row in outcomes:
        qid = str(row["qid"])
        total[qid] += 1.0 if row["correct"] else 0.0
        count[qid] += 1
    return {qid: total[qid] / count[qid] for qid in total}


def compose_cascade(
    generations: list[dict],
    escalate: list[bool],
    strong_correct: dict[str, float],
) -> dict[str, dict]:
    """Per-cluster cascade accuracy and average escalations-per-run.

    ``generations`` are the efficient model's per-question rows (``qid``,
    ``cluster``, ``run``, ``correct``); ``escalate[i]`` is the QE decision to
    escalate row ``i`` (True = route to the strong model). For each row the
    system answer is the strong model's (paired by ``qid``) when escalated, else
    the efficient model's own correctness.

    Returns ``{cluster: {"cascade_accuracy", "escalations", "n"}}`` where
    ``escalations`` is the mean number of escalated queries per run, the ``count``
    that ``cascade_system_metrics`` charges (``direct = size - count``).
    """
    if len(escalate) != len(generations):
        raise ValueError(
            f"escalate ({len(escalate)}) must align with generations ({len(generations)})"
        )
    correct_sum: dict[str, float] = defaultdict(float)
    esc_count: dict[str, int] = defaultdict(int)
    n_rows: dict[str, int] = defaultdict(int)
    runs: dict[str, set] = defaultdict(set)
    for row, esc in zip(generations, escalate):
        cluster = str(row["cluster"])
        n_rows[cluster] += 1
        runs[cluster].add(row["run"])
        if esc:
            qid = str(row["qid"])
            if qid not in strong_correct:
                raise KeyError(
                    f"no strong-model outcome for escalated qid {qid!r}; the strong "
                    f"model must be evaluated on every query it can be escalated -- run "
                    f"it on cluster {cluster!r} (its outcomes/generations feed strong_correct)"
                )
            esc_count[cluster] += 1
            correct_sum[cluster] += strong_correct[qid]
        else:
            correct_sum[cluster] += 1.0 if row["correct"] else 0.0
    report: dict[str, dict] = {}
    for cluster in n_rows:
        n_runs = len(runs[cluster]) or 1
        report[cluster] = {
            "cascade_accuracy": correct_sum[cluster] / n_rows[cluster],
            "escalations": esc_count[cluster] / n_runs,
            "n": n_rows[cluster],
        }
    return report


def run_qe(classifier, generations: list[dict], batch_size: int = 32) -> list[bool]:
    """Return the escalate decision (True = route) per generation.

    ``classifier`` needs a ``predict_batch(list[(question, output, num_tokens)])``
    returning objects with an ``.accept`` flag (``QEClassifier`` or a test stub).
    """
    escalate: list[bool] = []
    for start in range(0, len(generations), batch_size):
        chunk = generations[start : start + batch_size]
        items = [
            (row.get("question", row.get("prompt", "")), row["full_output"], row["num_tokens"])
            for row in chunk
        ]
        escalate.extend(not d.accept for d in classifier.predict_batch(items))
    return escalate


def write_cascade_stats(out_path: str | Path, strong_model: str, report: dict[str, dict]) -> None:
    """Merge ``escalations`` and ``cascade_accuracy`` into a cascade stats JSON.

    ``out_path`` must already hold the routing side (``assignment``,
    ``cluster_sizes``, ``models``) as produced for ``cre cascade``; this adds the
    Stage 2 fields per cluster in ``report`` and leaves the rest untouched.
    """
    path = Path(out_path)
    stats = json.loads(path.read_text())
    stats.setdefault("escalations", {})
    stats.setdefault("cascade_accuracy", {})
    for cluster, r in report.items():
        stats["escalations"][cluster] = [strong_model, r["escalations"]]
        stats["cascade_accuracy"][cluster] = r["cascade_accuracy"]
    path.write_text(json.dumps(stats, indent=2) + "\n")
