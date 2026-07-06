#!/usr/bin/env python
"""Download the paper's datasets from the HuggingFace Hub as JSONL files.

Usage:
    python data/download.py --dataset ymoslem/AIME-router
    python data/download.py --dataset ymoslem/AIME-router --split train --output data/aime_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", required=True, help="HF dataset id")
    parser.add_argument("--split", default=None, help="one split; default: all splits")
    parser.add_argument("--output", default=None, help="output path (single split only)")
    parser.add_argument("--output-dir", default="data", help="output directory (all splits)")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset)
    short = args.dataset.split("/")[-1]

    splits = [args.split] if args.split else list(dataset.keys())
    for split in splits:
        if args.output and args.split:
            path = Path(args.output)
        else:
            path = Path(args.output_dir) / f"{short}_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset[split].to_json(str(path), lines=True, force_ascii=False)
        print(f"{args.dataset}:{split} -> {path} ({len(dataset[split])} rows)")


if __name__ == "__main__":
    main()
