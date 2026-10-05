"""Build a QE training dataset from ``*_generations.jsonl`` files.

The logic lives in ``cre_router.qe.data``; ``cre qe-data`` is the command line.
This script is kept for the ``--push-to-hub`` option.

Usage:
    python data/prep_qe.py \
        --train tm_train_instruct_nothink_r5_..._generations.jsonl \
        --test  tm_test_instruct_nothink_r5_..._generations.jsonl \
        --out data/telemath_router
    cre qe-train --dataset data/telemath_router --max-length 4096 --output-dir ./qe-telemath
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cre_router.qe.data import build, qe_row, to_qe_rows  # noqa: F401

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--train", nargs="+", required=True, help="generations JSONL file(s) for the train split")
    parser.add_argument("--test", nargs="+", required=True, help="generations JSONL file(s) for the test split")
    parser.add_argument("--out", required=True, help="output directory for train.jsonl / test.jsonl")
    parser.add_argument("--push-to-hub", default=None, help="also push the DatasetDict to this HF hub id")
    parser.add_argument("--hub-private", action="store_true")
    args = parser.parse_args(argv)

    sizes = build(args.train, args.test, args.out)
    print(f"Wrote {args.out}/train.jsonl ({sizes['train']}) and test.jsonl ({sizes['test']})")
    label_pos = sum(
        1 for line in open(Path(args.out) / "train.jsonl") if json.loads(line)["decision_label"] == 1
    )
    print(f"Train accept/route balance: {label_pos} accept / {sizes['train'] - label_pos} route")

    if args.push_to_hub:
        from datasets import load_dataset

        ds = load_dataset("json", data_files={
            "train": str(Path(args.out) / "train.jsonl"),
            "test": str(Path(args.out) / "test.jsonl"),
        })
        ds.push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"Pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
