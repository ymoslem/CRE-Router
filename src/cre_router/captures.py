"""Read a saved capture back: graded answers joined to their measured cost.

``cre evaluate`` leaves two files per run. The generations JSONL holds one row
per question, and the detailed benchmark dump holds one record per request,
named by :func:`cre_router.evaluate._detail_name`. Analysis needs them joined,
and the join is the delicate part, so it lives here rather than in each script
that wants it.

Three rules, each of which has cost a result before:

- **Grade, never read.** ``correct`` in a saved file is whatever grader was live
  when the job ran. Every row is regraded with the parser and matcher of the
  task that produced it, taken from the generations filename, and the stored
  verdict is kept as ``correct_as_run``. Grading one benchmark by another's
  rules is not hypothetical: TeleMath's matcher accepts anything within 1% of
  the gold, which scores 384 as 385 on AIME, and reads a TeleQnA answer of
  "0, 1, 2, 3, 4" as the number 0.
- **The detail file has no question id.** Request *i* of a (cluster, run) is
  generations row *i* of that (cluster, run). That is asserted request by
  request on output length before any cost is attached, because a cost
  attributed to the wrong question is worse than no cost at all.
- **Cost is taken as measured.** TPOT is the serving stack's own per-token time,
  the mean inter-token latency; E2EL is the time to first token plus the decode.
  Neither is reconstructed from the other.

Usage::

    from cre_router.captures import load_capture
    cap = load_capture("tm_test_e2b_nothink", captures_dir="captures/")
    cap[("q17", 0)]["tpot_ms"]

``expect`` raises unless the capture holds exactly the questions and runs asked
for, which is how a partially served capture is refused rather than averaged.

Some runs kept only the batch means, one row per (cluster, run), and never wrote
per-request detail. They are refused by default, because a missing detail file
is usually a mistake. Pass ``require_cost=False`` to read them: every row then
carries ``batch_ttft_ms``, ``batch_tpot_ms`` and ``batch_e2el_ms``, the means of
the batch it was served in, and the per-request fields are ``None``. That is
what a Stage 1 fit reads, and making the caller ask for it keeps the weaker
basis visible at the call site.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cre_router.evaluate import TASKS, Task, score_generations

__all__ = ["CaptureError", "task_of", "generations_path", "load_generations",
           "batch_means", "load_capture", "dataset_loader"]

#: Model-id prefixes that start the model segment of a results directory name.
#: A capture directory is ``results_<tag>_<model with / replaced by _>``, so the
#: vendor pins where the tag ends: without it, ``fc_ours_eff`` also matches
#: ``fc_ours_eff_memm``.
VENDORS = ("google_", "Qwen_", "WeiboAI_", "nvidia_", "mistralai_", "bottlecapai_",
           "meta-llama_", "microsoft_", "deepseek-ai_")


class CaptureError(RuntimeError):
    """A capture is missing, incomplete, or does not match its detail files."""


@dataclass(frozen=True)
class Request:
    """One served request: what was asked, what came back, what it cost."""

    qid: str
    cluster: str
    run: int
    correct: bool
    correct_as_run: bool
    tokens: int
    ttft_s: float
    tpot_ms: float
    e2el_s: float

    #: Means of the batch this request was served in, as the run recorded them.
    batch: dict | None = None

    def as_dict(self) -> dict:
        row = {"correct": self.correct, "correct_as_run": self.correct_as_run,
               "tokens": self.tokens, "ttft_s": self.ttft_s,
               "tpot_ms": self.tpot_ms, "e2el_s": self.e2el_s,
               "cluster": self.cluster}
        row.update(self.batch or {})
        return row


def task_of(generations: str | Path) -> Task:
    """The task whose grader produced this file, from its name.

    The harness writes ``<task>_<model>_<timestamp>_generations.jsonl``. Longest
    match first, so ``telemath_nothink`` is not read as ``telemath``.
    """
    name = Path(generations).name
    for task in sorted(TASKS, key=len, reverse=True):
        if name.startswith(task + "_"):
            return TASKS[task]
    raise CaptureError(
        f"{name}: no task in the filename, so no grader is fixed. Rename the "
        f"file or add the task to TASKS rather than grading it by guesswork.")


def generations_path(tag: str, captures_dir: str | Path) -> Path:
    """The one generations file of this capture, with the tag boundary pinned."""
    hits = [g for g in glob.glob(str(Path(captures_dir) / f"results_{tag}_*"
                                     / "*_generations.jsonl"))
            if Path(g).parent.name[len(f"results_{tag}_"):].startswith(VENDORS)]
    if len(hits) != 1:
        raise CaptureError(f"{tag}: expected one generations file, found {len(hits)}")
    return Path(hits[0])


def load_generations(tag: str, captures_dir: str | Path) -> tuple[list[dict], dict]:
    """Rows of one capture with ``correct`` regraded, plus a grading report."""
    path = generations_path(tag, captures_dir)
    task = task_of(path)
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise CaptureError(f"{tag}: the generations file is empty")
    _, correct = score_generations([r["full_output"] for r in rows],
                                   [r.get("ground_truth_answer") for r in rows],
                                   task.parse, task.match)
    flips = 0
    for row, now in zip(rows, correct):
        row["correct_as_run"] = bool(row.get("correct"))
        row["correct"] = bool(now)
        flips += row["correct_as_run"] != row["correct"]
    return rows, {"task": task.name, "rows": len(rows), "regraded_flips": flips}


def batch_means(tag: str, captures_dir: str | Path) -> dict[tuple[str, int], dict]:
    """The run's own per-(cluster, run) means, keyed as the generations are.

    The harness writes one row per batch beside the generations. These are the
    means a Stage 1 fit reads, and where per-request detail exists they are a
    second check on the join: the requests must average to them.
    """
    path = generations_path(tag, captures_dir)
    hits = [p for p in path.parent.glob("*.jsonl")
            if not p.name.endswith(("_generations.jsonl", "_outcomes.jsonl"))]
    if len(hits) != 1:
        raise CaptureError(f"{tag}: expected one batch-means file, found {len(hits)}")
    out = {}
    for line in hits[0].open():
        if line.strip():
            r = json.loads(line)
            out[(str(r["cluster"]), int(r["run"]))] = {
                "batch_ttft_ms": r.get("ttft_ms"), "batch_tpot_ms": r.get("tpot_ms"),
                "batch_e2el_ms": r.get("e2el_ms"), "num_prompts": r.get("num_prompts")}
    if not out:
        raise CaptureError(f"{tag}: the batch-means file is empty")
    return out


def _detail_path(tag: str, captures_dir: Path, cluster: str, run: int) -> Path:
    hits = glob.glob(str(captures_dir / f"detail_{tag}" / f"*_c{cluster}_r{run}.json"))
    if len(hits) != 1:
        raise CaptureError(f"{tag}: expected one detail file for cluster {cluster} "
                           f"run {run}, found {len(hits)}; this capture cannot "
                           f"carry a cost")
    return Path(hits[0])


def load_capture(
    tag: str,
    captures_dir: str | Path,
    expect_questions: int | None = None,
    expect_runs: Iterable[int] | None = None,
    require_cost: bool = True,
) -> dict[tuple[str, int], dict]:
    """Per-(qid, run) grade and measured cost for one capture.

    Returns ``{(qid, run): {correct, correct_as_run, tokens, ttft_s, tpot_ms,
    e2el_s, cluster, batch_ttft_ms, batch_tpot_ms, batch_e2el_ms}}``. Raises
    :class:`CaptureError` rather than returning a partial result. With
    ``require_cost=False`` a capture that kept only batch means is read, and its
    per-request fields are ``None``.
    """
    captures_dir = Path(captures_dir)
    rows, _ = load_generations(tag, captures_dir)
    batches = batch_means(tag, captures_dir)

    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["cluster"]), int(row["run"])), []).append(row)

    out: dict[tuple[str, int], dict] = {}
    for (cluster, run), group in sorted(grouped.items()):
        if (cluster, run) not in batches:
            raise CaptureError(f"{tag}: no batch means for cluster {cluster} run {run}")
        batch = {k: v for k, v in batches[(cluster, run)].items() if k != "num_prompts"}
        served = batches[(cluster, run)].get("num_prompts")
        if served is not None and served != len(group):
            raise CaptureError(f"{tag} c{cluster} r{run}: the batch served {served} "
                               f"prompts, the generations hold {len(group)}")
        if not require_cost and not glob.glob(
                str(captures_dir / f"detail_{tag}" / f"*_c{cluster}_r{run}.json")):
            for row in group:
                out[(str(row["qid"]), run)] = Request(
                    qid=str(row["qid"]), cluster=str(cluster), run=run,
                    correct=bool(row["correct"]),
                    correct_as_run=bool(row["correct_as_run"]),
                    tokens=int(row["num_tokens"]), ttft_s=None, tpot_ms=None,
                    e2el_s=None, batch=batch).as_dict()
            continue
        detail = json.loads(_detail_path(tag, captures_dir, cluster, run).read_text())
        if len(detail["ttfts"]) != len(group):
            raise CaptureError(f"{tag} c{cluster} r{run}: {len(detail['ttfts'])} "
                               f"requests against {len(group)} generations")
        for i, row in enumerate(group):
            tokens = detail["output_lens"][i]
            if tokens != row["num_tokens"]:
                raise CaptureError(
                    f"{tag} c{cluster} r{run} request {i}: detail output_len "
                    f"{tokens} != generation num_tokens {row['num_tokens']}. The "
                    f"positional join does not hold, so this detail file was not "
                    f"produced by these generations.")
            itl = detail["itls"][i]
            out[(str(row["qid"]), run)] = Request(
                qid=str(row["qid"]), cluster=str(cluster), run=run,
                correct=bool(row["correct"]), correct_as_run=bool(row["correct_as_run"]),
                tokens=tokens, ttft_s=detail["ttfts"][i],
                tpot_ms=(sum(itl) / (tokens - 1) * 1000) if tokens > 1 else 0.0,
                e2el_s=detail["ttfts"][i] + sum(itl), batch=batch,
            ).as_dict()

    questions = {q for q, _ in out}
    runs = {r for _, r in out}
    if expect_questions is not None and len(questions) != expect_questions:
        raise CaptureError(f"{tag}: {len(questions)} questions, expected "
                           f"{expect_questions}")
    if expect_runs is not None and runs != set(expect_runs):
        raise CaptureError(f"{tag}: runs {sorted(runs)}, expected "
                           f"{sorted(set(expect_runs))}")
    return out


def dataset_loader(dataset_dir: str | Path):
    """A ``load(tag)`` that reads captures from the released captures dataset.

    ``dataset_dir`` is a download of ``ymoslem/cluster-route-escalate-captures``.
    Its rows were built by this module's rules (graded by the task, joined on
    output length) and hold the same fields ``load_capture`` returns, so a
    composition gives the same numbers from either source. Needs ``pyarrow``,
    installed by the ``data`` extra.
    """
    import pyarrow.parquet as pq

    cols = ["capture", "qid", "run", "cluster", "correct", "correct_as_run",
            "output_tokens", "ttft_s", "tpot_ms", "e2el_s"]
    rows: dict[str, dict] = {}
    for f in sorted(Path(dataset_dir).glob("data/*/*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            rows.setdefault(r["capture"], {})[(str(r["qid"]), int(r["run"]))] = {
                "correct": bool(r["correct"]), "correct_as_run": bool(r["correct_as_run"]),
                "tokens": r["output_tokens"], "ttft_s": r["ttft_s"],
                "tpot_ms": r["tpot_ms"], "e2el_s": r["e2el_s"],
                "cluster": str(r["cluster"])}
    if not rows:
        raise CaptureError(f"{dataset_dir}: no parquet shards under data/")

    def load(tag: str) -> dict[tuple[str, int], dict]:
        if tag not in rows:
            raise CaptureError(f"{tag}: not in the captures dataset")
        cap = rows[tag]
        if any(v["tpot_ms"] is None for v in cap.values()):
            raise CaptureError(f"{tag}: kept only batch means, so it cannot carry a "
                               f"per-request cost")
        return cap

    return load

