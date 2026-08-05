"""Build a QE training dataset from ``*_generations.jsonl`` files.

Each generation row (written by ``cre evaluate --save-generations``) already
carries everything the QE classifier needs -- ``prompt``, ``full_output``,
``num_tokens`` and ``correct`` -- so this converter only relabels it into the
schema ``cre qe-train`` expects: ``decision_label`` is 1 (accept) when the
efficient model was correct, else 0 (route/escalate). It writes ``train.jsonl``
and ``test.jsonl`` into an output directory that ``cre qe-train --dataset <dir>``
loads directly, no Hugging Face Hub round-trip needed.

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


def qe_row(gen: dict) -> dict:
    """One generation row -> one QE example (columns match ymoslem/*-router)."""
    correct = bool(gen["correct"])
    full_output = gen["full_output"]
    return {
        "question": gen.get("question", gen.get("prompt", "")),
        "prompt": gen.get("prompt", ""),
        "ground_truth_answer": gen.get("ground_truth_answer"),
        "full_output": full_output,
        "answer": gen.get("answer"),
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


def to_qe_rows(generations: list[dict]) -> list[dict]:
    """Convert generation rows to QE examples, pooling multiple files/models."""
    return [qe_row(g) for g in generations]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(train_files: list[str], test_files: list[str], out_dir: str) -> dict[str, int]:
    """Write ``{out_dir}/train.jsonl`` and ``test.jsonl``; return split sizes."""
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
            rows.extend(to_qe_rows(kept))
        if dropped:
            print(f"WARNING: dropped {dropped} {split} row(s) with null num_tokens")
        _write_jsonl(rows, out / f"{split}.jsonl")
        sizes[split] = len(rows)
    return sizes


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
