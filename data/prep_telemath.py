#!/usr/bin/env python
"""Prepare TeleMath (netop/TeleMath) for the CRE pipeline.

Downloads the 500-item TeleMath benchmark, telecom mathematical problems with
numerical answers, appends a numerical-answer prompt suffix, and writes a
stratified-by-category train/test split (approx 300/200). TeleMath ships a
single ``test`` split with no train partition, so the split is created here and
pinned by seed.

TeleMath is a gated dataset; set ``HF_TOKEN`` in the environment before running.

Usage:
    HF_TOKEN=... python data/prep_telemath.py --out data/telemath
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# The questions already name the quantity and its unit, so the suffix only fixes
# the output format: a single numerical answer, in the stated unit, in a box.
PROMPT_SUFFIX = (
    "\n\nSolve the problem and give the final numerical answer, in the unit "
    "stated in the question, inside \\boxed{}."
)


def build_prompt(row: dict) -> str:
    return row["question"].strip() + PROMPT_SUFFIX


def stratified_split(rows: list[dict], train_size: int, seed: int) -> tuple[list, list]:
    """Split rows into train/test, stratified by category so every topic is
    represented on both sides in proportion to its frequency."""
    frac = train_size / len(rows)
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for _, items in sorted(by_cat.items()):
        items = items[:]
        rng.shuffle(items)
        n_train = round(len(items) * frac)
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    return train, test


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            record = {
                "id": row["id"],
                "prompt": build_prompt(row),
                "answer": row["answer"],
                "category": row["category"],
                "difficulty": row["difficulty"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def describe(rows: list[dict], label: str) -> None:
    cats = Counter(r["category"] for r in rows)
    print(f"{label}: {len(rows)} rows | {dict(cats.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default="data/telemath", help="output path prefix")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test",
                        help="TeleMath ships a single 'test' split")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = list(load_dataset("netop/TeleMath")[args.split])
    rows = [dict(row, id=i) for i, row in enumerate(dataset)]

    train, test = stratified_split(rows, args.train_size, args.seed)
    out = Path(args.out)
    write_split(train, out.with_name(f"{out.name}_train.jsonl"))
    write_split(test, out.with_name(f"{out.name}_test.jsonl"))

    describe(train, "train")
    describe(test, "test")
    print(f"-> {out}_train.jsonl / {out}_test.jsonl")


if __name__ == "__main__":
    main()
