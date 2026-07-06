"""Measure per-cluster accuracy and TPOT for each candidate model.

This is the measurement step of Stage 1: its output feeds the routing. Each
model is run over the clustered training queries, and for every cluster we
record the error rate and the mean Time Per Output Token (TPOT). Results are
aggregated into the stats JSON consumed by ``cre fit``.

Measurement uses vLLM's own benchmark (``vllm bench serve``) as the engine,
the same tool that produced the paper's reported numbers, so TPOT is measured
consistently and with all of vLLM's serving features (warmup, TTFT/TPOT/ITL
split, concurrency control). It runs against an already-running vLLM server
and requires the ``eval`` extra (vllm).

The pure functions here (answer parsing, accuracy scoring, per-cluster
aggregation, stats assembly) are unit-tested without a GPU. The vLLM engine
in ``run_vllm_benchmark`` is the part to verify on a GPU box; it is isolated
so nothing else depends on vLLM being importable.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def split_thinking(text: str) -> str:
    """Return the post-reasoning content, dropping a leading <think>...</think>
    block when present."""
    end = text.find("</think>")
    return text[end + len("</think>") :].strip() if end != -1 else text.strip()


def parse_aime_answer(text: str) -> int | None:
    """Extract an AIME answer (integer 0-999) from a completion.

    Preference order: a \\boxed{n}, then an explicit ``Answer: n``, then the
    last short integer in the content. Mirrors the reference evaluation.
    """
    content = split_thinking(text)
    boxed = re.findall(r"\\boxed\{(\d+)\}", content)
    if boxed:
        return int(boxed[-1])
    labelled = re.search(r"Answer:\s*(\d+)", content)
    if labelled:
        return int(labelled.group(1))
    numbers = [n for n in re.findall(r"\b(\d+)\b", content) if len(n) <= 4]
    return int(numbers[-1]) if numbers else None


def parse_teleqna_answer(text: str) -> int | None:
    """Extract a TeleQnA multiple-choice index from a completion.

    The prompt asks for ``Answer: <choice_number_only>``; fall back to the
    last standalone integer.
    """
    content = split_thinking(text)
    labelled = re.search(r"Answer:\s*(\d+)", content)
    if labelled:
        return int(labelled.group(1))
    numbers = re.findall(r"\b(\d+)\b", content)
    return int(numbers[-1]) if numbers else None


def answers_match(predicted: int | None, gold: Any) -> bool:
    if predicted is None:
        return False
    try:
        return int(predicted) == int(gold)
    except (TypeError, ValueError):
        return str(predicted).strip() == str(gold).strip()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """Per-dataset evaluation settings following the paper's Appendix A.

    Sampling follows the model recommendations: thinking-mode models (AIME)
    use temperature 0.6 / top-p 0.95, direct-answer models (TeleQnA) use
    0.7 / 0.8; both use top-k 20 and min-p 0.
    """

    name: str
    parse: Callable[[str], int | None]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    max_tokens: int


TASKS: dict[str, Task] = {
    "aime": Task(
        name="aime",
        parse=parse_aime_answer,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=40960,
    ),
    "teleqna": Task(
        name="teleqna",
        parse=parse_teleqna_answer,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=1024,
    ),
}


# ---------------------------------------------------------------------------
# Accuracy scoring (pure)
# ---------------------------------------------------------------------------


def score_generations(
    generated_texts: list[str], gold_answers: list[Any], parse: Callable[[str], int | None]
) -> tuple[float, list[bool]]:
    """Return (error_rate, per-item correctness) for one benchmark run.

    ``generated_texts`` are assumed aligned with ``gold_answers`` (the vLLM
    benchmark preserves dataset order with shuffling disabled).
    """
    if len(generated_texts) != len(gold_answers):
        raise ValueError(
            f"{len(generated_texts)} generations vs {len(gold_answers)} gold answers"
        )
    correct = [answers_match(parse(t), g) for t, g in zip(generated_texts, gold_answers)]
    error = 1.0 - sum(correct) / len(correct) if correct else 1.0
    return error, correct


# ---------------------------------------------------------------------------
# Per-cluster orchestration
# ---------------------------------------------------------------------------


def split_by_cluster(dataset: list[dict]) -> dict[str, list[dict]]:
    """Group dataset rows by their ``cluster`` field."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        groups[str(row["cluster"])].append(row)
    return dict(sorted(groups.items()))


def cluster_sizes(dataset: list[dict]) -> dict[str, int]:
    return dict(Counter(str(row["cluster"]) for row in dataset))


@dataclass
class RunMeasurement:
    """One (cluster, run) benchmark outcome, kept for provenance."""

    cluster: str
    run: int
    error: float
    tpot_ms: float
    num_prompts: int


def aggregate_runs(measurements: list[RunMeasurement]) -> dict[str, dict[str, float]]:
    """Average error and TPOT across runs, per cluster."""
    errors: dict[str, list[float]] = defaultdict(list)
    tpots: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        errors[m.cluster].append(m.error)
        tpots[m.cluster].append(m.tpot_ms)
    return {
        c: {"error": mean(errors[c]), "tpot_ms": mean(tpots[c])} for c in sorted(errors)
    }


def model_entry(measurements: list[RunMeasurement]) -> dict:
    """The per-model block for a stats file: per-cluster error and TPOT."""
    agg = aggregate_runs(measurements)
    return {
        "errors": {c: round(agg[c]["error"], 6) for c in agg},
        "cluster_tpot_ms": {c: round(agg[c]["tpot_ms"], 6) for c in agg},
    }


def merge_model_into_stats(
    stats_path: str | Path, model_name: str, entry: dict, sizes: dict[str, int]
) -> None:
    """Add or update one model's entry in a stats JSON, preserving other
    models. Cluster sizes are (re)written from the evaluated dataset."""
    path = Path(stats_path)
    stats = json.loads(path.read_text()) if path.exists() else {}
    stats.setdefault("cluster_sizes", {})
    stats.setdefault("models", {})
    stats["cluster_sizes"] = {str(k): int(v) for k, v in sorted(sizes.items())}
    stats["models"][model_name] = entry
    path.write_text(json.dumps(stats, indent=2) + "\n")


def save_raw_measurements(measurements: list[RunMeasurement], path: str | Path) -> None:
    """Write per-(cluster, run) measurements to a JSONL provenance file so every
    number in the stats can be traced back to a benchmark run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for m in sorted(measurements, key=lambda m: (m.cluster, m.run)):
            f.write(
                json.dumps(
                    {
                        "cluster": m.cluster,
                        "run": m.run,
                        "error": round(m.error, 6),
                        "tpot_ms": round(m.tpot_ms, 6),
                        "num_prompts": m.num_prompts,
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# vLLM benchmark engine (verify on GPU; nothing else imports vllm)
# ---------------------------------------------------------------------------


def run_vllm_benchmark(
    dataset_path: str | Path,
    model: str,
    task: Task,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_concurrency: int = 32,
    seed: int = 0,
    download_dir: str | None = None,
) -> dict:
    """Run ``vllm bench serve`` against a running vLLM server and return its
    result dict (which includes ``generated_texts`` and ``mean_tpot_ms``).

    The server for ``model`` must already be serving at ``host:port`` (start it
    with ``vllm serve <model> --host <host> --port <port>``). Number of prompts
    is inferred from the dataset file; shuffling is disabled so the returned
    generations stay aligned with the dataset order.
    """
    import argparse

    from vllm.benchmarks.serve import add_cli_args
    from vllm.benchmarks.serve import main as benchmark_main

    num_prompts = sum(1 for line in Path(dataset_path).read_text().splitlines() if line.strip())

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    for action in parser._actions:  # defaults only; nothing is required here
        action.required = False
    args = parser.parse_args([])

    args.backend = "vllm"
    args.endpoint = "/v1/completions"
    args.host = host
    args.port = port
    args.model = model
    args.dataset_name = "custom"
    args.dataset_path = str(dataset_path)
    args.num_prompts = num_prompts
    args.max_concurrency = max_concurrency
    args.seed = seed
    args.temperature = task.temperature
    args.top_p = task.top_p
    args.top_k = task.top_k
    args.min_p = task.min_p
    args.custom_output_len = task.max_tokens
    args.disable_shuffle = True
    args.no_oversample = True
    args.request_rate = float("inf")
    args.burstiness = 1.0
    args.save_result = False
    # Keep the per-request fields (generated_texts, errors) in the returned
    # dict; without this vLLM strips them for a summary-only result.
    args.save_detailed = True
    if download_dir is not None:
        args.download_dir = download_dir

    return benchmark_main(args)


def evaluate_model(
    dataset: list[dict],
    model: str,
    task: Task,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    runs: int = 5,
    max_concurrency: int = 32,
    base_seed: int = 0,
    workdir: str | Path = "results/splits",
    download_dir: str | None = None,
    benchmark: Callable[..., dict] | None = None,
) -> list[RunMeasurement]:
    """Benchmark ``model`` on each cluster for ``runs`` repetitions.

    Each dataset row needs ``prompt``, ``answer``, and ``cluster``. Returns the
    per-(cluster, run) measurements; aggregate them with ``model_entry``.
    ``benchmark`` defaults to ``run_vllm_benchmark`` and is injected in tests.
    """
    run_benchmark = benchmark or run_vllm_benchmark
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    measurements: list[RunMeasurement] = []
    for cluster, items in split_by_cluster(dataset).items():
        split_path = workdir / f"{task.name}_cluster_{cluster}.jsonl"
        with split_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        gold = [item["answer"] for item in items]

        for run in range(runs):
            result = run_benchmark(
                split_path,
                model,
                task,
                host=host,
                port=port,
                max_concurrency=max_concurrency,
                seed=base_seed + run,
                download_dir=download_dir,
            )
            error, _ = score_generations(result["generated_texts"], gold, task.parse)
            tpot_ms = result.get("mean_tpot_ms")
            if tpot_ms is None:
                raise ValueError("benchmark result is missing 'mean_tpot_ms'")
            measurements.append(
                RunMeasurement(
                    cluster=cluster,
                    run=run,
                    error=error,
                    tpot_ms=float(tpot_ms),
                    num_prompts=len(items),
                )
            )
    return measurements
