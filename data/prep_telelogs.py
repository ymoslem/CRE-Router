#!/usr/bin/env python
"""Turn the gated netop/TeleLogs `troubleshooting` split into router-ready JSONL.

TeleLogs ships two string columns: `question`, a self-contained RCA prompt that
already lists the eight candidate root causes and the drive-test data and asks
for the cause number in ``\\boxed{}``; and `answer`, a single label ``C1``..``C8``.
The router expects a `prompt` field and an integer `answer`, so this maps
`question` -> `prompt` and ``C<n>`` -> the integer ``<n>``.

TeleLogs is gated on the Hub; download it under your own HF auth first, e.g.

    python data/download.py --dataset netop/TeleLogs --config troubleshooting

then convert each split:

    python data/prep_telelogs.py --input data/TeleLogs_train.jsonl \\
        --output data/telelogs_train.jsonl

The dataset is not redistributed here, so its rows are never committed; only
aggregate stats (configs/telelogs_stats.json) are.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LABEL = re.compile(r"C?\s*(\d+)")


def parse_label(answer: object) -> int:
    """Map a TeleLogs gold label (``"C3"`` or ``"3"``) to the integer 3."""
    match = _LABEL.search(str(answer).strip())
    if not match:
        raise ValueError(f"cannot parse a root-cause label from {answer!r}")
    n = int(match.group(1))
    if not 1 <= n <= 8:
        raise ValueError(f"root-cause label {n} is out of range 1-8 (from {answer!r})")
    return n


def telelogs_record(row: dict) -> dict:
    """Convert one raw TeleLogs row to a router-ready {prompt, answer} record."""
    return {"prompt": row["question"], "answer": parse_label(row["answer"])}


def convert(rows: list[dict]) -> list[dict]:
    return [telelogs_record(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", required=True, help="verbatim TeleLogs JSONL (question, answer)")
    parser.add_argument("--output", required=True, help="router-ready JSONL (prompt, integer answer)")
    args = parser.parse_args()

    in_path = Path(args.input)
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    records = convert(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{in_path} -> {out_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
