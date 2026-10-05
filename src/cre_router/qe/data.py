"""The QE dataset ``cre qe-train`` reads, built from ``cre evaluate`` generations.

Each generation row (written by ``cre evaluate --save-generations``) already
carries everything the QE classifier needs, ``prompt``, ``full_output``,
``num_tokens`` and ``correct``, so this only relabels it into the schema
``cre qe-train`` expects: ``decision_label`` is 1 (accept) when the gated model
was correct, else 0 (escalate). ``build`` writes ``train.jsonl`` and
``test.jsonl`` into a directory that ``cre qe-train --dataset <dir>`` loads
directly. ``cre qe-data`` is its command line.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from cre_router.evaluate import SCORER_VERSION

_WARNED: list[int] = []


def qe_row(gen: dict, task: str | None = None) -> dict:
    """One generation row -> one QE example (columns match ymoslem/*-router).

    ``task`` names an entry in ``cre_router.evaluate.TASKS``. When given, the
    label and the extracted answer are recomputed from ``full_output`` with the
    current grader instead of being copied from the generation log. Pass it
    whenever the log predates a grader change, so a stale verdict cannot become
    a training label. Without it the stored fields are used unchanged.
    """
    full_output = gen["full_output"]
    answer = gen.get("answer")
    if task is not None:
        from cre_router.evaluate import TASKS

        spec = TASKS[task]
        answer = spec.parse(full_output)
        match = spec.match or (lambda p, g: p is not None and p == g)
        correct = bool(match(answer, gen.get("ground_truth_answer")))
    else:
        # No task given, so the stored verdict is copied through. This is the
        # hole that produced the first TeleMath QE classifiers: they were built
        # from logs graded before the 2026-08-13 fixes, so every training label
        # was the old grader's. Copying is still allowed, because AIME and
        # TeleQnA logs are unaffected and re-parsing them needs no task, but it
        # is never silent: a log that names a scorer older than the current one
        # is refused outright.
        stored_version = gen.get("scorer_version")
        if stored_version is not None and stored_version < SCORER_VERSION:
            raise ValueError(
                f"generation was graded by scorer_version {stored_version}, current "
                f"is {SCORER_VERSION}. Pass task=... so the label is recomputed from "
                f"full_output; copying it would train on a stale verdict."
            )
        if not _WARNED:
            _WARNED.append(1)
            warnings.warn(
                "qe_row(task=None): copying the stored `correct` field. Pass task= "
                "to regrade from full_output.", RuntimeWarning, stacklevel=2)
        correct = bool(gen["correct"])
    return {
        "question": gen.get("question", gen.get("prompt", "")),
        "prompt": gen.get("prompt", ""),
        "ground_truth_answer": gen.get("ground_truth_answer"),
        "full_output": full_output,
        "answer": answer,
        "accuracy": float(correct),
        "num_words": len(full_output.split()),
        "num_tokens": gen["num_tokens"],
        "score": float(correct),
        "decision_label": 1 if correct else 0,
        "decision_str": "accept" if correct else "route",
        "cluster": gen.get("cluster"),
        "qid": gen.get("qid"),
        "run": gen.get("run"),
    }


def to_qe_rows(generations: list[dict], task: str | None = None) -> list[dict]:
    """Convert generation rows to QE examples, pooling multiple files/models."""
    return [qe_row(g, task=task) for g in generations]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(train_files: list[str], test_files: list[str], out_dir: str,
          task: str | None = None) -> dict[str, int]:
    """Write ``{out_dir}/train.jsonl`` and ``test.jsonl``; return split sizes.

    ``task`` regrades every answer from ``full_output``, as in ``qe_row``.
    """
    out = Path(out_dir)
    sizes = {}
    for split, files in (("train", train_files), ("test", test_files)):
        rows: list[dict] = []
        dropped = 0
        for f in files:
            gens = _read_jsonl(Path(f))
            # num_tokens feeds the QE input verbatim; a null (a generations file
            # written without output_lens) would render the string "None", so drop
            # those rows rather than poison the dataset.
            kept = [g for g in gens if g.get("num_tokens") is not None]
            dropped += len(gens) - len(kept)
            rows.extend(to_qe_rows(kept, task=task))
        if dropped:
            print(f"WARNING: dropped {dropped} {split} row(s) with null num_tokens")
        _write_jsonl(rows, out / f"{split}.jsonl")
        sizes[split] = len(rows)
    return sizes
