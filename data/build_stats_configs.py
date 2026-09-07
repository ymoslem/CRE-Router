"""Regenerate the shipped Stage 1 stats configs from the 1 x A100 captures.

The configs a reader runs `cre fit` on decide which routing they reproduce, so
they have to name their basis. Four of the five shipped ones were built on the
submitted 2 x A100 / vLLM 0.17.0 configuration and carried no marker, which meant
REPRODUCE.md quietly walked a reader to the submitted numbers rather than the
reported ones. `teleqna_stats.json` was the clearest case: it prunes Gemma4-E2B
and Gemma4-E4B, true on two cards and false on one, where nothing is dominated
and the sweep has six regions rather than three.

Every config carries its basis in the filename and there is no basis-less
default, so a reader cannot pick one up without knowing what produced it:

    <pool>_stats_1xA100_Sep2026.json  the reported basis
    <pool>_stats_2xA100_Jun2026.json  the preprint's, kept reproducible

The archive is named for the **paper**, June 2026, not for the measurement, so
one date covers all four files; the true capture dates go inside. A second
2 x A100 basis is being measured now and will land as `*_2xA100_Sep2026.json`.
Every name is basis plus measurement month: two files never differ only in
hardware when they also differ in date and serving stack.

Errors are regraded from `full_output` rather than read from a stored verdict,
and per-cluster TPOT comes from the vLLM per-request record.

Usage:
    python cre-router/data/build_stats_configs.py [--write]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref" / "results"))
sys.path.insert(0, str(ROOT / "cre-router" / "src"))

import bench_captures as B  # noqa: E402

CONFIGS = ROOT / "cre-router" / "configs"
CSV = ROOT / "ref" / "results" / "csv"
ARCHIVE = "2xA100_Jun2026"      # the preprint, not the measurement date
REPORTED = "1xA100_Sep2026"

POOLS = {
    "aime": {
        "csv": "AIME-Clusters-Train",
        "measured": "2026-03-08 to 2026-03-10",
        "clusters": ("0", "1", "2"),
        "models": {"VibeThinker-1.5B": "aime1g_train_vibe",
                   "Qwen3-30B-A3B": "aime1g_train_q30b"},
        "csv_stem": {"VibeThinker-1.5B": "VibeThinker",
                     "Qwen3-30B-A3B": "Qwen3-30B-A3B-Thinking-2507-FP8"},
        "note": ("Per-cluster training statistics for the AIME 1983-2023 pool, "
                 "measured on 1 x A100 at concurrency 32 under vLLM 0.19.0, the "
                 "basis the paper reports. Five runs, errors regraded from "
                 "full_output. Reproduce lambda* with: "
                 "cre fit --stats configs/aime_stats.json --budget 30"),
    },
    "teleqna": {
        "csv": "TeleQnA-Clusters-Train",
        "measured": "2026-04-11 to 2026-04-13",
        "clusters": ("0", "1"),
        "models": {"Qwen3-4B": "tq1g_train_q4bi", "Gemma4-E2B": "tq1g_train_e2b",
                   "Gemma4-E4B": "tq1g_train_e4b", "Gemma4-26B": "tq1g_train_26b"},
        "csv_stem": {"Qwen3-4B": "Qwen3-4B-Instruct-2507", "Gemma4-E2B": "gemma-4-E2B-it",
                     "Gemma4-E4B": "gemma-4-E4B-it", "Gemma4-26B": "gemma-4-26B-A4B-it"},
        "note": ("Per-cluster training statistics for the TeleQnA four-model pool, "
                 "measured on 1 x A100 at concurrency 32 under vLLM 0.19.0, the "
                 "basis the paper reports. NOTHING is Pareto-dominated here and "
                 "the sweep has six regions; on the submitted 2 x A100 "
                 "configuration Gemma4-E2B and Gemma4-E4B were pruned and it had "
                 "three. Reproduce lambda* with: "
                 "cre fit --stats configs/teleqna_stats.json --budget 20"),
    },
}


def csv_versions(split_dir: str) -> dict[str, list[str]]:
    """Per-model vLLM version from the archived CSVs.

    Per model, not per file: the submitted TeleQnA pool served Qwen3-4B-Instruct
    under 0.17.0 and the three Gemmas under 0.19.0, so one value at the top of
    the config would be false for three of its four rows.
    """
    out = {}
    for f in sorted((CSV / split_dir).glob("*.csv")):
        for line in f.open():
            if line.lower().startswith("vllm version,"):
                out[f.stem] = sorted({c.strip() for c in line.split(",")[1:] if c.strip()})
                break
    return out


def build(pool: str, spec: dict) -> dict:
    clusters = spec["clusters"]
    sizes, models = None, {}
    for name, tag in spec["models"].items():
        cap = B.load_capture(tag)
        by = {c: [v for v in cap.values() if v["cluster"] == c] for c in clusters}
        n = {c: len({q for (q, _), v in cap.items() if v["cluster"] == c}) for c in clusters}
        if sizes is None:
            sizes = n
        elif sizes != n:
            raise SystemExit(f"{tag}: cluster sizes {n} disagree with {sizes}")
        models[name] = {
            "errors": {c: round(1 - statistics.fmean(v["correct"] for v in by[c]), 4)
                       for c in clusters},
            "cluster_tpot_ms": {c: round(statistics.fmean(v["tpot_ms"] for v in by[c]), 3)
                                for c in clusters},
        }
    for m in models.values():
        m["vllm_version"] = "0.19.0"       # audited across all 132 reported captures
    return {"_note": spec["note"], "basis": "1 x A100 SXM 80 GB, concurrency 32",
            "vllm_version": "0.19.0", "cluster_sizes": sizes,
            "error_tol": 0.001, "models": models}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for pool, spec in POOLS.items():
        new = build(pool, spec)
        current = CONFIGS / f"{pool}_stats.json"
        old = json.loads(current.read_text())
        print(f"== {pool}")
        for name in new["models"]:
            o = old["models"].get(name, {}).get("cluster_tpot_ms", {})
            n = new["models"][name]["cluster_tpot_ms"]
            print(f"   {name:18} 2x {list(o.values())}  ->  1x {list(n.values())}")
        if not args.write:
            continue

        # archive every config on the submitted basis, stats and cascade alike,
        # stamping each model with the version that actually served it
        versions = csv_versions(spec["csv"])
        for stem in (f"{pool}_stats", f"{pool}_cascade_test"):
            src = CONFIGS / f"{stem}.json"
            if not src.exists():
                continue
            d = json.loads(src.read_text())
            d["basis"] = "2 x A100 SXM 80 GB, concurrency 32, the June 2026 preprint"
            d["measured"] = spec["measured"]
            d["vllm_version"] = sorted({v for vs in versions.values() for v in vs})
            # explicit map, never a fuzzy match: a near-match here would stamp
            # the wrong serving version onto a model and look plausible
            for name, m in d.get("models", {}).items():
                csv_key = spec["csv_stem"].get(name)
                if csv_key is None or csv_key not in versions:
                    raise SystemExit(f"{csv_key or name!r}: no CSV to read a version "
                                     f"from; fix csv_stem rather than guessing")
                hit = versions[csv_key]
                m["vllm_version"] = hit[0] if len(hit) == 1 else hit
            d["_note"] = ("SUBMITTED basis, kept so the preprint stays reproducible. "
                          "Do not use it for the reported numbers. " + d.get("_note", ""))
            (CONFIGS / f"{stem}_{ARCHIVE}.json").write_text(json.dumps(d, indent=2) + "\n")
            src.unlink()
            print(f"   {src.name} -> {stem}_{ARCHIVE}.json")

        out = CONFIGS / f"{pool}_stats_{REPORTED}.json"
        out.write_text(json.dumps(new, indent=2) + "\n")
        print(f"   wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
