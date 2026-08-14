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

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from cre_router.textutils import split_thinking

# Identifies the grading rules that produced a result file. Bump this whenever a
# parser or a match rule changes what counts as correct, so saved artifacts
# declare their own provenance and a reader never has to infer it from a
# timestamp. Stamped into stats files and generation rows.
#
#   1  original rules
#   2  2026-08-13: TeleMath parser fix (fractions, units and comma separators
#      inside \boxed{}, double-boxed answers, exponent leakage) and
#      numeric_match abs_tol 1e-9 -> 0
SCORER_VERSION = 2

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


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


# TeleMath gold answers are short numbers (at most 17 characters across the
# 500-question dataset) and a well-formed completion states the answer at the
# very end. Degenerate or truncated generations, however, can run to hundreds of
# kilobytes; searching all of it made the number pattern below backtrack
# quadratically and stall for hours. Restricting the search to a tail window
# keeps every pattern linear in the window, and a real answer (a few characters,
# at the end) is never clipped.
_TELEMATH_TAIL_CHARS = 4000
# One integer run with an optional fraction, or a leading-dot decimal, plus an
# optional exponent. Unlike ``\d*\.?\d+`` this has no two adjacent
# variable-length digit runs, so it cannot backtrack catastrophically.
_TELEMATH_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
# A LaTeX fraction (``\frac{7}{6}`` or ``\dfrac``/``\tfrac``), a bare ``a/b``,
# or a plain number, in that priority. Each numerator/denominator/number is
# the bounded ``_TELEMATH_NUMBER`` above, so this has the same no-backtracking
# guarantee.
_TELEMATH_VALUE = (
    rf"(?:(?P<fsign>[-+])?\\[dt]?frac\{{\s*(?P<fn>{_TELEMATH_NUMBER})\s*\}}\{{\s*(?P<fd>{_TELEMATH_NUMBER})\s*\}}"
    rf"|(?P<rn>{_TELEMATH_NUMBER})\s*/\s*(?P<rd>{_TELEMATH_NUMBER})"
    rf"|(?P<num>{_TELEMATH_NUMBER}))"
)
_TELEMATH_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
# A value is only trusted with trailing content after it (e.g. a unit) when
# that content is a harmless label rather than more math -- otherwise
# ``\boxed{2e^{-2}}`` (meaning 2 times e to the minus 2) would truncate to 2.
_TELEMATH_SAFE_TRAILER = re.compile(r"^\s*(\\text\{|$)")
# A superscript exponent, braced or bare. Bounded repetition keeps this linear.
_TELEMATH_SUPERSCRIPT = re.compile(r"\^\s*\{[^{}]{0,20}\}|\^\s*-?\d+(?:\.\d+)?")


def _telemath_value(match: re.Match) -> float | None:
    if match.group("fn") is not None:
        num, den = float(match.group("fn")), float(match.group("fd"))
        if match.group("fsign") == "-":
            num = -num
    elif match.group("rn") is not None:
        num, den = float(match.group("rn")), float(match.group("rd"))
    else:
        return float(match.group("num"))
    return num / den if den != 0 else None


def _telemath_value_at_start(s: str) -> float | None:
    """A fraction or number from the start of ``s``, ignoring a trailing
    unit label, or None if nothing trustworthy is at the start."""
    s = s.lstrip()
    match = re.match(_TELEMATH_VALUE, s)
    if match and _TELEMATH_SAFE_TRAILER.match(s[match.end():]):
        try:
            return _telemath_value(match)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def parse_telemath_answer(text: str) -> float | None:
    """Extract a TeleMath numerical answer (a float) from a completion.

    TeleMath answers are numerical quantities, often long decimals, LaTeX
    fractions, or scientific notation (e.g. 233.333333333333, 7.2e-05,
    \\frac{7}{6}, -62.0854). The final value is taken from a ``\\boxed{}``
    when present, then an explicit ``Answer:``, then the last value anywhere
    in the text. LaTeX scientific notation (``7.2 \\times 10^{-5}``) and
    comma thousands separators (``1,382,400``) are normalised before
    matching. A fraction inside ``\\boxed{}``/``Answer:`` is evaluated
    (``\\frac{7}{6}`` -> 1.1667); a bare unit label after the value is
    ignored (``\\boxed{0.2 \\text{ packets/s}}`` -> 0.2), but anything else
    trailing it is treated as more math and the match is rejected rather than
    silently truncated (``\\boxed{2e^{-2}}`` is not truncated to 2).
    """
    content = split_thinking(text)[-_TELEMATH_TAIL_CHARS:]
    content = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", ""), content)
    content = re.sub(
        r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\\times\s*10\^\{?(-?\d+)\}?",
        r"\1e\2",
        content,
    )

    # A model sometimes boxes the same answer twice, a clean decimal first and
    # a symbolic restatement last (``\boxed{1.732}`` ... ``\boxed{\sqrt{3}}``).
    # Scan boxed occurrences from last to first so an unparseable final box
    # falls back to an earlier clean one, rather than to the noisier tiers
    # below.
    for candidate in reversed(_TELEMATH_BOXED.findall(content)):
        value = _telemath_value_at_start(candidate)
        if value is not None:
            return value

    labelled = re.findall(rf"\*{{0,2}}Answer\*{{0,2}}\s*[:：]\s*(.{{0,80}})", content)
    if labelled:
        value = _telemath_value_at_start(labelled[-1])
        if value is not None:
            return value

    # Last resort: the last value anywhere. Blank out superscript exponents
    # first, or ``2e^{-2}`` and ``10^{-0.3}`` contribute their exponent as a
    # standalone candidate and the scan ends on it.
    content = _TELEMATH_SUPERSCRIPT.sub(" ", content)
    last = None
    for match in re.finditer(_TELEMATH_VALUE, content):
        try:
            value = _telemath_value(match)
        except (ValueError, ZeroDivisionError):
            continue
        if value is not None:
            last = value
    return last


def answers_match(predicted: int | None, gold: Any) -> bool:
    if predicted is None:
        return False
    try:
        return int(predicted) == int(gold)
    except (TypeError, ValueError):
        return str(predicted).strip() == str(gold).strip()


def numeric_match(predicted: float | None, gold: Any, rel_tol: float = 1e-2) -> bool:
    """Correct if the predicted value is within a relative tolerance of the gold.

    Uses ``math.isclose`` with a 1% relative tolerance, which accepts the same
    quantity reported at different rounding (233.33 vs 233.333333) and rejects
    genuinely different values. TeleMath publishes no tolerance of its own, so
    1% is our stated choice; it sits inside the range over which model ranking
    and cascade break-even are both invariant (see
    ``ref/results/telemath/tolerance_decision.md``).

    The comparison is purely relative. An absolute floor cannot be used here
    because some gold answers are themselves smaller than any plausible floor
    (down to 1e-10), so a floor would accept zero as correct for them. A gold
    of exactly zero still matches a predicted zero, since ``math.isclose``
    compares equal values as close at any tolerance.
    """
    if predicted is None:
        return False
    try:
        return math.isclose(float(predicted), float(gold), rel_tol=rel_tol, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


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
    # Gemma 4's thinking switch (`enable_thinking`) is a chat-template kwarg,
    # not a text-level prefix like Qwen's /no_think, and vLLM's own bench-serve
    # dataset loader never forwards template kwargs (confirmed by reading
    # vllm/benchmarks/datasets.py CustomDataset.sample: it calls
    # apply_chat_template with a fixed argument list). So for this family the
    # dataset prep step renders the template itself and stores the finished
    # text as the prompt; this flag tells the benchmark not to template it
    # again on top.
    pre_rendered: bool = False
    # How a parsed answer is compared to the gold label. Defaults to exact
    # integer/string equality (`answers_match`); numerical-answer tasks such as
    # TeleMath set this to a tolerance-based comparison instead.
    match: Callable[[Any, Any], bool] | None = None


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
    # TeleMath: telecom mathematical problems with numerical (float) answers,
    # scored by relative tolerance rather than exact match. Two arms, since the
    # pool mixes thinking and non-thinking models and each has its own
    # recommended sampling: `telemath` for thinking models, `telemath_nothink`
    # for instruct models.
    "telemath": Task(
        name="telemath",
        parse=parse_telemath_answer,
        match=numeric_match,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=40960,
    ),
    "telemath_nothink": Task(
        name="telemath_nothink",
        parse=parse_telemath_answer,
        match=numeric_match,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=16384,
    ),
    # Gemma 4 with thinking enabled. Same reasoning sampling as `telemath`, but
    # the thinking switch is a chat-template kwarg baked into the prompt text by
    # data/prep_gemma4_thinking.py, so the prompts are pre_rendered and served
    # verbatim rather than templated again by the benchmark.
    "telemath_gemma4": Task(
        name="telemath_gemma4",
        parse=parse_telemath_answer,
        match=numeric_match,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=40960,
        pre_rendered=True,
    ),
}


# ---------------------------------------------------------------------------
# Accuracy scoring (pure)
# ---------------------------------------------------------------------------


def score_generations(
    generated_texts: list[str],
    gold_answers: list[Any],
    parse: Callable[[str], Any],
    match: Callable[[Any, Any], bool] | None = None,
) -> tuple[float, list[bool]]:
    """Return (error_rate, per-item correctness) for one benchmark run.

    ``generated_texts`` are assumed aligned with ``gold_answers`` (the vLLM
    benchmark preserves dataset order with shuffling disabled). ``match``
    defaults to exact equality; numerical tasks pass a tolerance comparison.
    """
    if len(generated_texts) != len(gold_answers):
        raise ValueError(
            f"{len(generated_texts)} generations vs {len(gold_answers)} gold answers"
        )
    matcher = match or answers_match
    correct = [matcher(parse(t), g) for t, g in zip(generated_texts, gold_answers)]
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


# Optional per-run metrics. TPOT alone cannot price a thinking/non-thinking pool,
# because both modes decode at the same speed and differ only in how many tokens
# they emit; E2EL (= TTFT + TPOT x output length) captures that, and unlike
# request throughput it is per-request, so it adds up correctly across cascade
# rungs. All are optional so an injected benchmark may supply only TPOT.
OPTIONAL_METRICS = (
    "ttft_ms",
    "e2el_ms",
    "request_throughput",
    "mean_output_tokens",
    "truncated_frac",
)


@dataclass
class RunMeasurement:
    """One (cluster, run) benchmark outcome, kept for provenance."""

    cluster: str
    run: int
    error: float
    tpot_ms: float
    num_prompts: int
    ttft_ms: float | None = None
    e2el_ms: float | None = None
    request_throughput: float | None = None
    mean_output_tokens: float | None = None
    truncated_frac: float | None = None


def optional_metrics(
    result: dict, num_prompts: int, max_output_len: int | None = None
) -> dict[str, float]:
    """Pull the optional cost metrics out of a benchmark result, tolerating any
    that this vLLM version (or an injected fake) does not report.

    ``mean_e2el_ms`` is preferred when present; otherwise it is reconstructed as
    ``TTFT + TPOT x (output length - 1)``, which is how it is recoverable from
    the summary block of runs that predate this capture.

    ``max_output_len`` is the task's token cap; when given alongside the
    per-request ``output_lens``, the fraction of requests that hit the cap is
    reported as ``truncated_frac``.
    """
    metrics: dict[str, float] = {}

    output_tokens = result.get("total_output_tokens")
    mean_output = float(output_tokens) / num_prompts if output_tokens and num_prompts else None
    if mean_output is not None:
        metrics["mean_output_tokens"] = mean_output

    ttft = result.get("mean_ttft_ms")
    if ttft is not None:
        metrics["ttft_ms"] = float(ttft)

    e2el = result.get("mean_e2el_ms")
    if e2el is None:
        ttft, tpot = result.get("mean_ttft_ms"), result.get("mean_tpot_ms")
        if ttft is not None and tpot is not None and mean_output:
            e2el = float(ttft) + float(tpot) * max(mean_output - 1.0, 0.0)
    if e2el is not None:
        metrics["e2el_ms"] = float(e2el)

    throughput = result.get("request_throughput")
    if throughput is not None:
        metrics["request_throughput"] = float(throughput)

    # A capped generation corrupts accuracy and understates cost at the same
    # time, so the truncation rate has to travel with the numbers. vLLM's serve
    # benchmark does not report a finish reason, but it does return per-request
    # output_lens; a request that reached the cap was cut off. EOS-terminated
    # requests stop below the cap, so equality with it is the truncation test.
    output_lens = result.get("output_lens")
    if output_lens and max_output_len:
        truncated = sum(1 for length in output_lens if length >= max_output_len)
        metrics["truncated_frac"] = truncated / len(output_lens)

    return metrics


def aggregate_runs(measurements: list[RunMeasurement]) -> dict[str, dict[str, float]]:
    """Average error, TPOT and any available optional metric across runs, per
    cluster. An optional metric is reported only when every run supplied it."""
    errors: dict[str, list[float]] = defaultdict(list)
    tpots: dict[str, list[float]] = defaultdict(list)
    extra: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        errors[m.cluster].append(m.error)
        tpots[m.cluster].append(m.tpot_ms)
        for metric in OPTIONAL_METRICS:
            value = getattr(m, metric)
            if value is not None:
                extra[m.cluster][metric].append(float(value))

    agg = {}
    for c in sorted(errors):
        entry = {"error": mean(errors[c]), "tpot_ms": mean(tpots[c])}
        for metric, values in extra[c].items():
            if len(values) == len(errors[c]):
                entry[metric] = mean(values)
        agg[c] = entry
    return agg


def model_entry(measurements: list[RunMeasurement]) -> dict:
    """The per-model block for a stats file: per-cluster error and TPOT, plus any
    optional metric that every run reported."""
    agg = aggregate_runs(measurements)
    entry = {
        "errors": {c: round(agg[c]["error"], 6) for c in agg},
        "cluster_tpot_ms": {c: round(agg[c]["tpot_ms"], 6) for c in agg},
    }
    for metric in OPTIONAL_METRICS:
        present = {c: agg[c][metric] for c in agg if metric in agg[c]}
        if len(present) == len(agg):
            entry[f"cluster_{metric}"] = {c: round(v, 6) for c, v in present.items()}
    return entry


def merge_model_into_stats(
    stats_path: str | Path, model_name: str, entry: dict, sizes: dict[str, int]
) -> None:
    """Add or update one model's entry in a stats JSON, preserving other
    models. Cluster sizes are (re)written from the evaluated dataset.

    The file records ``scorer_version``, so a reader can tell which grader
    produced its per-cluster error rates instead of guessing from the file's
    timestamp. A file without the key predates the stamp and should be treated
    as ungraded by the current rules.
    """
    path = Path(stats_path)
    stats = json.loads(path.read_text()) if path.exists() else {}
    stats.setdefault("cluster_sizes", {})
    stats.setdefault("models", {})
    stats["cluster_sizes"] = {str(k): int(v) for k, v in sorted(sizes.items())}
    stats["models"][model_name] = entry
    stats["scorer_version"] = SCORER_VERSION
    path.write_text(json.dumps(stats, indent=2) + "\n")


def save_raw_measurements(measurements: list[RunMeasurement], path: str | Path) -> None:
    """Write per-(cluster, run) measurements to a JSONL provenance file so every
    number in the stats can be traced back to a benchmark run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for m in sorted(measurements, key=lambda m: (m.cluster, m.run)):
            record = {
                "cluster": m.cluster,
                "run": m.run,
                "error": round(m.error, 6),
                "tpot_ms": round(m.tpot_ms, 6),
                "num_prompts": m.num_prompts,
            }
            for metric in OPTIONAL_METRICS:
                value = getattr(m, metric)
                if value is not None:
                    record[metric] = round(float(value), 6)
            f.write(json.dumps(record) + "\n")


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
    # Prompts for a pre_rendered task already carry the fully-templated text
    # (chat-template kwargs like Gemma 4's enable_thinking baked in), so the
    # loader must serve them verbatim rather than templating a second time.
    args.skip_chat_template = task.pre_rendered
    if task.pre_rendered:
        # vLLM's completions endpoint decodes with skip_special_tokens=True by
        # default, which silently removes Gemma 4's <|channel> marker (a
        # registered special token) from the returned text while leaving the
        # ordinary word "thought" that follows it untouched -- confirmed by a
        # direct A/B request against a live google/gemma-4-E2B-it server.
        # Qwen's <think>/</think> are not special tokens and are
        # unaffected either way, so this is only needed for pre_rendered tasks.
        args.extra_body = {"skip_special_tokens": False}
    args.disable_shuffle = True
    args.no_oversample = True
    args.request_rate = float("inf")
    args.burstiness = 1.0
    # vLLM reports only the metrics named here; its generative default is
    # "ttft,tpot,itl", which drops e2el entirely (see benchmarks/serve.py,
    # process_one_metric). Ask for it explicitly so `--cost-metric e2el` reads a
    # measured value instead of falling back to reconstructing it from means.
    args.percentile_metrics = "ttft,tpot,itl,e2el"
    args.save_result = False
    # Keep the per-request fields (generated_texts, errors) in the returned
    # dict; without this vLLM strips them for a summary-only result.
    args.save_detailed = True
    if download_dir is not None:
        args.download_dir = download_dir

    return benchmark_main(args)


def question_id(item: dict) -> str:
    """A stable per-question key for joining outcomes across models/runs.

    Uses the dataset's ``id`` when present (preserved through clustering and
    Gemma pre-rendering), otherwise a hash of the raw prompt. It must identify
    the same question identically across every model, so downstream analysis can
    pair per-question correctness (confidence intervals, McNemar significance)."""
    qid = item.get("id")
    if qid is not None:
        return str(qid)
    return hashlib.md5(item["prompt"].encode("utf-8")).hexdigest()[:12]


# Chat-template control tokens, across the families we serve. A pre_rendered
# prompt has been through the tokenizer's template and carries at least one.
_TEMPLATE_MARKER = re.compile(
    r"<\|[^|>]{1,40}\|?>?"      # Gemma 4 <|turn>, GPT-style <|im_start|>
    r"|<bos>|<s>"               # SentencePiece BOS
    r"|<start_of_turn>"         # earlier Gemma
    r"|\[INST\]"                # Llama 2 / Mistral
    r"|<｜[^｜]{1,40}｜>"  # DeepSeek full-width bars
)


def check_pre_rendered(dataset: list[dict], task: Task, sample: int = 32) -> None:
    """Fail fast when a pre_rendered task is handed un-templated prompts.

    A pre_rendered task serves ``prompt`` verbatim, with the chat template
    already baked in by ``data/prep_gemma4_thinking.py``. Passing the raw
    dataset instead is not an error the server reports: it answers happily, but
    the model never receives the tokens that open and close its thinking
    section, so it never emits the matching stop token and generates until it
    hits ``max_tokens``. A run that hit this burned three and a half GPU-hours
    and returned 52% truncated output at 0.5% accuracy, which is only
    recognisable after the fact.

    Only ``telemath_gemma4`` is pre_rendered today, so in practice this guards
    the Gemma thinking runs, but it keys off the task flag rather than the model
    name so any future pre-rendered task is covered too.
    """
    if not task.pre_rendered:
        return
    prompts = [row.get("prompt", "") for row in dataset[:sample]]
    if not prompts:
        return
    bad = sum(1 for p in prompts if not _TEMPLATE_MARKER.search(p))
    if bad:
        raise ValueError(
            f"task {task.name!r} is pre_rendered, but {bad} of {len(prompts)} sampled "
            f"prompts carry no chat-template markers. The raw dataset was almost "
            f"certainly passed instead of the pre-rendered one; serving it would "
            f"generate to max_tokens without terminating. Use the output of "
            f"data/prep_gemma4_thinking.py, e.g. "
            f"telemath_train_gemma_<model>_think.jsonl.\n"
            f"  first offending prompt: {next(p for p in prompts if not _TEMPLATE_MARKER.search(p))[:120]!r}"
        )


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
    outcomes_out: str | Path | None = None,
    generations_out: str | Path | None = None,
) -> list[RunMeasurement]:
    """Benchmark ``model`` on each cluster for ``runs`` repetitions.

    Each dataset row needs ``prompt``, ``answer``, and ``cluster``. Returns the
    per-(cluster, run) measurements; aggregate them with ``model_entry``.
    ``benchmark`` defaults to ``run_vllm_benchmark`` and is injected in tests.

    When ``outcomes_out`` is given, writes one JSONL row per (question, run) with
    ``qid``, ``cluster``, ``run``, ``correct`` and ``output_len``. This is the
    per-question record needed for confidence intervals and paired significance
    tests; it is irrecoverable once discarded, so capture it during the run.

    When ``generations_out`` is given, writes one JSONL row per (question, run)
    additionally carrying the model's ``full_output`` text and ``num_tokens``.
    This is the training data for the Stage 2 accept/escalate QE classifier
    (``question [SEP] full_output [SEP] num_tokens`` -> correct), and like the
    generations themselves it is irrecoverable once the run ends.
    """
    run_benchmark = benchmark or run_vllm_benchmark
    check_pre_rendered(dataset, task)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    measurements: list[RunMeasurement] = []
    outcomes: list[dict] | None = [] if outcomes_out is not None else None
    generations: list[dict] | None = [] if generations_out is not None else None
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
            error, correct = score_generations(
                result["generated_texts"], gold, task.parse, task.match
            )
            output_lens = result.get("output_lens") or [None] * len(items)
            if outcomes is not None:
                for item, ok, olen in zip(items, correct, output_lens):
                    outcomes.append(
                        {
                            "qid": question_id(item),
                            "cluster": cluster,
                            "run": run,
                            "correct": bool(ok),
                            "output_len": olen,
                            # An outcomes file keeps no generated text, so its
                            # verdicts can never be rechecked on their own. The
                            # stamp is the only way a reader can tell whether
                            # they were graded by the current rules.
                            "scorer_version": SCORER_VERSION,
                        }
                    )
            if generations is not None:
                texts = result["generated_texts"]
                for item, ok, olen, text in zip(items, correct, output_lens, texts):
                    generations.append(
                        {
                            "qid": question_id(item),
                            "cluster": cluster,
                            "run": run,
                            # ``question`` is the raw query the QE classifier and
                            # humans read; ``prompt`` is the exact served input
                            # (chat-templated for pre_rendered tasks). They differ
                            # only for pre_rendered models, where prep stores the
                            # original question under ``question``.
                            "question": item.get("question", item.get("prompt", "")),
                            "prompt": item.get("prompt", ""),
                            "ground_truth_answer": item["answer"],
                            "answer": task.parse(text),
                            "full_output": text,
                            "num_tokens": olen,
                            "correct": bool(ok),
                            "scorer_version": SCORER_VERSION,
                        }
                    )
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
                    **optional_metrics(result, len(items), task.max_tokens),
                )
            )
    if outcomes is not None:
        out_path = Path(outcomes_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")
    if generations is not None:
        gen_path = Path(generations_out)
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with gen_path.open("w") as f:
            for g in generations:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return measurements
