#!/usr/bin/env python
"""Turn m-a-p/SuperGPQA into router-ready train and test JSONL.

SuperGPQA is 26,529 graduate-level multiple-choice questions across 13
disciplines, 72 fields and 285 subfields. It is the only benchmark in this
project that contains both reasoning-heavy and knowledge-only queries in one
dataset, labelled by the authors via ``is_calculation``, which is what makes it
useful for a router that has to tell the two apart.

The full dataset is far larger than the cascade needs. This script draws a
stratified subset, splits it into disjoint train and test halves, and writes
each split twice:

    <out>_train.jsonl           prompts as-is, for thinking-mode members
    <out>_train_nothink.jsonl   same prompts with the /no_think switch prepended

Two files rather than one because a mixed pool serves some members in thinking
mode and some not, and the switch has to travel with the prompt. This follows
the paper's own layout, where telecom_test.jsonl and telecom_test_nothink.jsonl
differ only by that prefix. Pair them with the matching task, ``supergpqa`` or
``supergpqa_nothink``, so each arm also gets its own sampling parameters.

Sampling is proportional to discipline, not even across disciplines. SuperGPQA
is naturally skewed, from Science at 9,838 questions down to Sociology at 143,
and flattening that would invent a dataset the authors did not build. It would
also distort the calculation balance, since ``is_calculation`` is largely a
discipline effect: Science is 66.3% calculation and Engineering 55.4%, while
Law, Education and Philosophy are all under 1%.

Usage:

    python data/prep_supergpqa.py --train-size 3000 --test-size 1000 \\
        --out data/supergpqa

Only aggregate statistics are ever committed, never the rows themselves.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJ"

NO_THINK = "/no_think"

PROMPT = (
    "{question}\n\n{options}\n\n"
    "Answer with the single letter of the correct option, "
    "written as \\boxed{{X}}."
)


def build_prompt(row: dict) -> str:
    """Render one question with its options lettered A, B, C, ..."""
    options = "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(row["options"])
    )
    return PROMPT.format(question=row["question"], options=options)


def gold_index(row: dict) -> int:
    """Map the gold letter to a 0-based index, matching parse_supergpqa_answer."""
    letter = str(row["answer_letter"]).strip().upper()
    if letter not in LETTERS[: len(row["options"])]:
        raise ValueError(
            f"gold letter {letter!r} is outside the {len(row['options'])} options "
            f"for question {row['uuid']}"
        )
    return ord(letter) - ord("A")


def proportional_sample(dataset, n: int, seed: int) -> list[dict]:
    """Draw ``n`` rows, allocating each discipline its share of the whole.

    Largest-remainder allocation, so the rounding leftovers go to the
    disciplines that lost the most to truncation rather than to whichever
    happens to be first.
    """
    by_discipline: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        by_discipline[row["discipline"]].append(row)

    total = sum(len(v) for v in by_discipline.values())
    exact = {k: len(v) * n / total for k, v in by_discipline.items()}
    quota = {k: int(v) for k, v in exact.items()}

    remaining = n - sum(quota.values())
    for key in sorted(exact, key=lambda k: exact[k] - quota[k], reverse=True):
        if remaining <= 0:
            break
        if quota[key] < len(by_discipline[key]):
            quota[key] += 1
            remaining -= 1

    rng = random.Random(seed)
    sample: list[dict] = []
    for key in sorted(by_discipline):
        take = min(quota[key], len(by_discipline[key]))
        sample.extend(rng.sample(by_discipline[key], take))
    rng.shuffle(sample)
    return sample


def write_split(rows: list[dict], path: Path, no_think: bool,
                cluster_by: str | None = None) -> None:
    """Write one JSONL split. ``no_think`` prepends the suppression switch.

    Metadata travels with each row for later analysis only. Nothing downstream
    routes on it: the cascade clusters on query embeddings and never reads these
    fields. ``difficulty`` in particular must not become an input, since
    SuperGPQA derives it partly from LLM response accuracy and it is therefore
    not independent of what we are measuring.

    ``cluster_by`` names a metadata field to write into the ``cluster`` column,
    which makes `cre evaluate` report error and cost broken down by that field
    instead of by a learned cluster. Setting it to ``is_calculation`` turns the
    real measurement path into the pool-comparison tool, so no separate probe
    script is needed. This is a diagnostic use of `cre evaluate`, not routing:
    the cascade proper clusters on embeddings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = build_prompt(row)
            if no_think:
                prompt = f"{NO_THINK} {prompt}"
            record = {
                        "prompt": prompt,
                        "answer": gold_index(row),
                        "uuid": row["uuid"],
                        "discipline": row["discipline"],
                        "field": row["field"],
                        "difficulty": row["difficulty"],
                        "is_calculation": row["is_calculation"],
                        "n_options": len(row["options"]),
            }
            if cluster_by:
                record["cluster"] = str(record[cluster_by])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def describe(rows: list[dict], label: str) -> None:
    calc = sum(1 for r in rows if r["is_calculation"])
    print(f"\n{label}: {len(rows)} rows")
    print(f"  is_calculation True {calc} ({calc / len(rows):.1%})")
    print(f"  difficulty {dict(Counter(r['difficulty'] for r in rows))}")
    top = Counter(r["discipline"] for r in rows).most_common(5)
    print(f"  top disciplines {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/supergpqa",
                        help="path prefix; _train.jsonl etc. are appended")
    parser.add_argument("--uuids-from", default=None,
                        help="JSONL of rows carrying a uuid; restrict to exactly "
                             "those questions instead of sampling. Use to rebuild "
                             "an earlier subset so results stay paired.")
    parser.add_argument("--cluster-by", default=None,
                        choices=("is_calculation", "difficulty", "discipline"),
                        help="write this metadata field into the 'cluster' column "
                             "so `cre evaluate` reports error and cost per group. "
                             "Diagnostic only; the cascade clusters on embeddings.")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = list(load_dataset("m-a-p/SuperGPQA")["train"])
    print(f"loaded {len(dataset)} SuperGPQA rows")

    out = Path(args.out)

    if args.uuids_from:
        wanted = {json.loads(line)["uuid"] for line in open(args.uuids_from)}
        rows = [r for r in dataset if r["uuid"] in wanted]
        print(f"restricted to {len(rows)} of {len(wanted)} requested uuids")
        write_split(rows, out.with_name(f"{out.name}.jsonl"),
                    no_think=False, cluster_by=args.cluster_by)
        write_split(rows, out.with_name(f"{out.name}_nothink.jsonl"),
                    no_think=True, cluster_by=args.cluster_by)
        describe(rows, out.name)
        print(f"\nwrote {out.name}[_nothink].jsonl next to {out.parent}")
        return

    # Draw train and test in one pass so they cannot overlap, then split. Drawing
    # them separately would risk the same question landing in both.
    pooled = proportional_sample(dataset, args.train_size + args.test_size, args.seed)
    train, test = pooled[: args.train_size], pooled[args.train_size :]

    for rows, split in ((train, "train"), (test, "test")):
        write_split(rows, out.with_name(f"{out.name}_{split}.jsonl"),
                    no_think=False, cluster_by=args.cluster_by)
        write_split(rows, out.with_name(f"{out.name}_{split}_nothink.jsonl"),
                    no_think=True, cluster_by=args.cluster_by)
        describe(rows, split)

    overlap = {r["uuid"] for r in train} & {r["uuid"] for r in test}
    print(f"\ntrain/test uuid overlap: {len(overlap)} (must be 0)")
    print(f"wrote {out.name}_[train|test][_nothink].jsonl next to {out.parent}")


if __name__ == "__main__":
    main()
