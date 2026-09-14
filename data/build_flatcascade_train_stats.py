"""Build the two-model flat-cascade Stage 1 config from the TRAIN whole captures.

The flat-cascade ablation uses a deliberately separate two-model pool, Gemma4-E2B
efficient and Gemma4-26B-A4B strong, served on the whole-dataset basis: a flat
cascade has no clusters to serve within, so every model is run over the undivided
split and carries one cost scalar rather than four.

Stage 1 is fit on train and applied to test, so this reads the train captures only.
Per-cluster error rates come from the train generations relabelled to their clusters
by qid; the cost is the uniform whole-dataset value per model.

Error rates are recomputed from ``full_output`` with the current grader rather than
read from the saved ``correct`` field, which carries whatever grader was live when
the job ran. Cost fields are read as measured.

Then fit:
    cre fit --stats configs/telemath_flatcascade_train_stats.json \
            --cost-metric {tpot,e2el} --budget B
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cre-router" / "src"))

from cre_router.evaluate import numeric_match, parse_telemath_answer  # noqa: E402

# Name -> capture tag. Two models only; this is not a subset of the nine-model pool.
TAGS = {
    "E2B-nothink": "tm_train_e2b_nothink_whole",
    "26B-nothink": "tm_train_26b_nothink_whole",
}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def regrade(row: dict) -> bool:
    """Grade one saved generation with the current grader, ignoring the stored value."""
    return bool(numeric_match(parse_telemath_answer(row["full_output"]),
                              row.get("ground_truth_answer")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default=str(ROOT / "grove-backup" / "cre-eval"),
                    help="directory holding results_<tag>_*/ and <tag>_*_stats.json")
    ap.add_argument("--clustered", default=str(ROOT / "grove-backup" / "cre-router"
                                               / "data" / "telemath_train_clustered.jsonl"),
                    help="train split carrying the cluster label per query id")
    ap.add_argument("--out", default=str(ROOT / "cre-router" / "configs"
                                         / "telemath_flatcascade_train_stats.json"))
    ap.add_argument("--error-tol", type=float, default=0.001)
    args = ap.parse_args()

    cmap = {str(r["id"]): str(r["cluster"]) for r in read_jsonl(args.clustered)}
    sizes: dict[str, int] = {}
    for c in cmap.values():
        sizes[c] = sizes.get(c, 0) + 1

    models: dict[str, dict] = {}
    for name, tag in TAGS.items():
        gens = sorted(glob.glob(f"{args.captures}/results_{tag}_*/*_generations.jsonl"))
        stats = sorted(glob.glob(f"{args.captures}/{tag}_*_stats.json"))
        if not gens:
            sys.exit(f"missing generations for {name} ({tag}); errors cannot be regraded")
        if not stats:
            sys.exit(f"missing stats for {name} ({tag}); cost is unavailable")
        rows = read_jsonl(gens[0])
        (_, m), = json.loads(Path(stats[0]).read_text())["models"].items()

        total: dict[str, int] = {}
        wrong: dict[str, int] = {}
        stale = 0
        for g in rows:
            c = cmap.get(str(g["qid"]))
            if c is None:
                continue
            total[c] = total.get(c, 0) + 1
            correct = regrade(g)
            if bool(g.get("correct")) != correct:
                stale += 1
            if not correct:
                wrong[c] = wrong.get(c, 0) + 1

        errors = {c: wrong.get(c, 0) / total[c] for c in sorted(total)}
        e2el = m["cluster_e2el_ms"]["0"]
        tpot = m["cluster_tpot_ms"]["0"]
        tokens = m["cluster_mean_output_tokens"]["0"]
        models[name] = {
            "errors": errors,
            "cluster_e2el_ms": {c: e2el for c in errors},
            "cluster_tpot_ms": {c: tpot for c in errors},
            "cluster_output_tokens": {c: tokens for c in errors},
        }
        acc = 1 - sum(wrong.values()) / sum(total.values())
        print(f"{name:14s} n={sum(total.values()):5d} acc={acc:.4f} "
              f"E2EL={e2el/1000:6.2f}s TPOT={tpot:6.2f}ms  "
              f"regrade changed {stale} of {len(rows)} rows")

    out = {
        "_note": ("Two-model flat-cascade ablation pool, TRAIN split, whole-dataset "
                  "serving basis. Errors regraded from full_output; cost is the "
                  "uniform whole-dataset value per model. Built by "
                  "data/build_flatcascade_train_stats.py."),
        "cluster_sizes": {c: sizes[c] for c in sorted(sizes)},
        "error_tol": args.error_tol,
        "models": models,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    print(f"cluster_sizes {out['cluster_sizes']}  N={sum(sizes.values())}")


if __name__ == "__main__":
    main()
