"""Per-request Stage 1 + 2 accounting, priced on measured latencies.

:func:`~cre_router.routing.cascade_system_metrics` answers the aggregate
question: given each model's mean cost and how many queries escalate, what does
the system cost? That is enough to choose an operating point. It is not enough
to *report* one, because a mean over the whole pool cannot say what an escalated
query actually waited, and cannot carry a confidence interval.

This module does the per-request version. Every query is priced on the tiers it
really ran, from the vLLM per-request record, and the result is a grid rather
than a scalar so a paired bootstrap has something to resample.

Two ideas do the work.

**Two masks, not one.** ``runs_eff`` says whether the efficient tier generated at
all; ``uses_strong`` says whether the answer returned came from the strong tier.
A cluster routed straight to the strong model never runs the efficient one, so
charging every query for an efficient pass overstates the bill. That mistake
inflated a Stage 1 row by 5 ms and 90 s and looked entirely plausible next to the
real figure.

**One strong capture per efficient run.** The queries that escalate depend on
what the efficient tier produced, so each of its runs yields a different
escalation set, and a set has to be *served as its own batch* to be priced: a
per-request latency only means anything together with the batch it was served in.
``strong_by_run`` therefore takes one measured capture per efficient run. Passing
a single capture is allowed and means the strong tier was measured once over
every query, which is the weaker whole-cluster approximation.

**A run that escalates nothing is still a run.** At a tight threshold some runs
gate no query at all. That run is not missing data and must not be dropped: it is
a real observation whose cost is the efficient tier alone, and dropping it would
bias the mean towards the runs that did escalate. Pass ``None`` or an empty
mapping for it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["per_request_metrics"]

# What a capture must provide per (question, run). `tpot` is vLLM's own
# per-token time, (e2el - ttft) / (tokens - 1); it is NOT derivable from the
# other three, because it excludes the wait before the first token.
FIELDS = ("correct", "e2el", "tokens", "tpot")


def _stack(strong_by_run: Any, n_q: int, n_eff: int) -> dict[str, np.ndarray]:
    """(question, efficient run, strong run) arrays for the strong tier.

    A single mapping is broadcast across efficient runs. A sequence supplies one
    capture per efficient run, and a ``None`` or empty entry means that run
    escalated nothing, which is left as zeros: the masks never read it.
    """
    if isinstance(strong_by_run, Mapping):
        runs = [strong_by_run] * n_eff
    elif isinstance(strong_by_run, Sequence):
        runs = list(strong_by_run)
    else:
        raise TypeError("strong_by_run must be a capture mapping or a sequence of them")
    if len(runs) != n_eff:
        raise ValueError(f"strong_by_run has {len(runs)} entries for {n_eff} efficient runs")

    n_s = next((np.asarray(r["correct"]).shape[1] for r in runs if r), None)
    if n_s is None:                       # no run escalated anything, anywhere
        n_s = 1
    out = {f: np.zeros((n_q, n_eff, n_s), dtype=float) for f in FIELDS}
    for i, run in enumerate(runs):
        if not run:                       # this run escalated nothing
            continue
        for f in FIELDS:
            a = np.asarray(run[f], dtype=float)
            if a.shape != (n_q, n_s):
                raise ValueError(f"strong run {i} field {f!r} has shape {a.shape}, "
                                 f"expected {(n_q, n_s)}")
            out[f][:, i, :] = a
    return out


def per_request_metrics(
    runs_eff: np.ndarray,
    uses_strong: np.ndarray,
    efficient: Mapping[str, Any],
    strong_by_run: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correctness, E2EL and TPOT on a (question, eff run, strong run) grid.

    ``runs_eff`` and ``uses_strong`` are boolean, indexed by (question,
    efficient run). ``efficient`` and each entry of ``strong_by_run`` map
    ``correct`` / ``e2el`` / ``tokens`` to (question, run) arrays.

    E2EL is what the user waits: the efficient pass when it happened, plus the
    strong pass when it happened, time before the first token included.

    TPOT is the serving stack's own per-token time, taken from the capture and
    never reconstructed. vLLM defines it as ``(e2el - ttft) / (tokens - 1)``, so
    a pass spends exactly ``tpot * (tokens - 1)`` decoding. A query answered by
    one tier therefore reports that tier's measured TPOT unchanged.

    An escalated query has no single measured TPOT, because it is two requests.
    Its passes are combined over the tokens the user actually received::

        (tpot_eff * (n_eff - 1) + tpot_strong * (n_strong - 1)) / (n_strong - 1)

    The efficient pass's decode time is charged in full, exactly as E2EL adds
    both passes' wall time. However, its tokens are not counted, because they
    were discarded and never reached the user. A wasted pass therefore costs
    time and earns no tokens to spread that time over.

    Total wait over tokens delivered is a different quantity again: it folds in
    the time before the first token, which TPOT excludes by definition. The two
    agree to a rounding error on long generations and differ by 11% on short
    answers, so the measured value is used throughout.
    """
    runs_eff = np.asarray(runs_eff, dtype=bool)
    uses_strong = np.asarray(uses_strong, dtype=bool)
    if runs_eff.shape != uses_strong.shape:
        raise ValueError(f"masks disagree: {runs_eff.shape} against {uses_strong.shape}")
    n_q, n_eff = runs_eff.shape

    eff = {f: np.asarray(efficient[f], dtype=float) for f in FIELDS}
    for f, a in eff.items():
        if a.shape != (n_q, n_eff):
            raise ValueError(f"efficient field {f!r} has shape {a.shape}, "
                             f"expected {(n_q, n_eff)}")
    strong = _stack(strong_by_run, n_q, n_eff)

    re_, us = runs_eff[:, :, None], uses_strong[:, :, None]
    correct = np.where(us, strong["correct"], eff["correct"][:, :, None])
    e2el = (np.where(re_, eff["e2el"][:, :, None], 0.0)
            + np.where(us, strong["e2el"], 0.0))
    delivered = np.where(us, strong["tokens"], eff["tokens"][:, :, None])
    # TPOT is a time BETWEEN tokens, so a one-token answer has none to report and
    # neither does the stack. Refusing is right: silently treating it as zero
    # would pull the mean down by exactly the requests we know least about.
    if np.any(delivered < 2):
        raise ValueError("a delivered answer has fewer than 2 tokens, so it has "
                         "no TPOT; drop those requests or price them another way")
    decode = (np.where(re_, (eff["tpot"] * (eff["tokens"] - 1.0))[:, :, None], 0.0)
              + np.where(us, strong["tpot"] * (strong["tokens"] - 1.0), 0.0))
    return correct, e2el, decode / (delivered - 1.0)
