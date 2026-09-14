"""FrugalGPT's LLM cascade: a learned model list with learned per-stage thresholds.

Chen, Zaharia and Zou, *FrugalGPT: How to Use Large Language Models While
Reducing Cost and Improving Performance*, TMLR 2024. Reference implementation at
https://github.com/stanford-futuredata/FrugalGPT (Apache-2.0).

A query goes to the first model in an ordered list ``L``. A scorer rates that
answer; if the score clears that stage's threshold the answer is returned,
otherwise the query goes to the next model. The last model always answers. The
method learns both which models are in the list, in which order, and the
threshold vector, by maximising accuracy subject to a mean-cost budget.

**This follows the released code, which differs from the paper in two places.**

*The scorer is a binary classifier, not a regression.* The paper describes "a
simple regression model" and "a DistilBERT tailored to regression", but
``scoring.py`` builds ``DistilBertForSequenceClassification``, loads integer
labels, and ``get_score`` returns ``softmax(logits)[1]``. So the score is an
accept probability from a two-class head. This module takes those probabilities
as input and is agnostic to how they were produced.

*The list search enumerates, it does not prune.* The paper says the optimizer
"prunes the search space of L by ignoring any list of LLMs with small answer
disagreement". ``llmchain.py`` calls ``itertools.permutations(service_ids, ell)``
and evaluates every one. This module enumerates.

Two further details of the released code that the paper does not state, both
reproduced here: thresholds are searched in *quantile* space rather than
directly, and are constrained non-decreasing; and the cascade is fit on the
training split.

Cost is whatever the caller measures. The paper uses dollars because it targets
commercial APIs, but its constraint is linear in prompt and output tokens with
per-model coefficients, so a per-request latency serves the same role.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brute, fmin

__all__ = ["CascadeFit", "fit_cascade", "apply_cascade", "optimise_thresholds"]

# Their grid resolution per threshold dimension, `Ns=40` in optimizer.py.
GRID = 40
# Their quantile search box, `(1e-5, 1-1e-5)`.
QUANTILE_LO, QUANTILE_HI = 1e-5, 1.0 - 1e-5
# Their penalty for an infeasible point.
INFEASIBLE = 10000.0


@dataclass(frozen=True)
class CascadeFit:
    """One fitted cascade: which models, in what order, with what thresholds."""

    models: tuple[int, ...]
    """Column indices into the candidate pool, in the order they are queried."""
    thresholds: np.ndarray
    """One per stage. The last is 1.0, so the final model always answers."""
    quantiles: np.ndarray
    """The search-space point the thresholds were derived from, one per stage
    but the last. Kept because a threshold is only meaningful beside the
    distribution it was taken from."""
    accuracy: float
    """Mean correctness on the split it was fit to."""
    cost: float
    """Mean per-query cost on that split, which the budget bounds."""

    def names(self, pool: Sequence[str]) -> tuple[str, ...]:
        return tuple(pool[i] for i in self.models)


def _first_acceptance(d_mat: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Which stage answers each query: the first whose distance clears its bar.

    ``d_mat`` is distance, ``1 - score``, so a *small* value is a confident
    answer and the test is ``d < threshold``. Later stages are masked off once a
    stage has accepted, which is what makes this a cascade rather than a vote.
    """
    accept = d_mat < thresholds
    # Keep the first True in each row, clear the rest.
    first = np.zeros_like(accept)
    taken = np.zeros(accept.shape[0], dtype=bool)
    for j in range(accept.shape[1]):
        here = accept[:, j] & ~taken
        first[:, j] = here
        taken |= here
    return first


def _thresholds_from_quantiles(d_mat: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Map a point in quantile space to thresholds, on the surviving queries.

    Each stage's threshold is a quantile of the distances of the queries that
    actually reach it, not of all queries. That is why the search is done in
    quantile space: a raw threshold means something different at each stage,
    where a quantile means the same thing everywhere.
    """
    n_stages = d_mat.shape[1]
    thresholds = np.zeros(n_stages)
    thresholds[-1] = 1.0                      # the last model always answers
    surviving = np.ones(d_mat.shape[0], dtype=bool)
    for i in range(n_stages - 1):
        q = float(np.clip(quantiles[i], 0.0, 1.0))
        thresholds[i] = np.quantile(d_mat[surviving, i], 1.0 - q)
        surviving = surviving & (d_mat[:, i] >= thresholds[i])
        if not surviving.any():               # nothing escalates past here
            surviving = np.ones(d_mat.shape[0], dtype=bool)
    return thresholds


def optimise_thresholds(
    L_mat: np.ndarray, C_mat: np.ndarray, d_mat: np.ndarray, budget: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Best thresholds for one fixed model list, under a mean-cost budget.

    ``L_mat`` is per-query correctness at each stage, ``C_mat`` the *cumulative*
    cost of having reached and run that stage, ``d_mat`` the distance
    ``1 - score``. All are (queries, stages).

    Returns mean accuracy, the thresholds, and the quantiles they came from.
    Accuracy is ``-inf`` when even the first model alone exceeds the budget,
    which is their "Base API too expensive, skip".
    """
    n, n_stages = L_mat.shape
    if C_mat[:, 0].mean() > budget:
        return -np.inf, np.zeros(n_stages), np.zeros(max(n_stages - 1, 1))

    def objective(quantiles: np.ndarray) -> float:
        quantiles = np.atleast_1d(quantiles)
        if np.any(np.diff(quantiles) < 0):    # thresholds must not tighten
            return INFEASIBLE
        thresholds = _thresholds_from_quantiles(d_mat, quantiles)
        answered = _first_acceptance(d_mat, thresholds)
        if (answered * C_mat).sum() > budget * n:
            return INFEASIBLE
        return -(answered * L_mat).sum()

    if n_stages == 1:                          # nothing to search
        return float(L_mat[:, 0].mean()), np.ones(1), np.zeros(1)

    def evaluate(quantiles):
        quantiles = np.atleast_1d(quantiles)
        thresholds = _thresholds_from_quantiles(d_mat, quantiles)
        answered = _first_acceptance(d_mat, thresholds)
        feasible = (answered * C_mat).sum() <= budget * n
        return feasible, float((answered * L_mat).sum() / n), thresholds, quantiles

    ranges = [(QUANTILE_LO, QUANTILE_HI)] * (n_stages - 1)
    point, _, grid, values = brute(
        objective, ranges, full_output=True, finish=fmin, Ns=GRID)

    # `finish=fmin` polishes off the grid and can step outside the feasible box,
    # where the objective is a flat penalty with nothing for a local search to
    # follow. Their code returns that point regardless. Fall back to the best
    # feasible point the grid already evaluated, so an over-budget polish costs
    # a little accuracy instead of discarding an entire model list.
    feasible, acc, thresholds, quantiles = evaluate(point)
    if feasible:
        return acc, thresholds, quantiles

    values = np.atleast_1d(values)
    flat = np.reshape(grid, (len(ranges), -1))
    for idx in np.argsort(values, axis=None):
        if values.flat[idx] >= INFEASIBLE:
            break                              # sorted, so the rest are worse
        feasible, acc, thresholds, quantiles = evaluate(flat[:, idx])
        if feasible:
            return acc, thresholds, quantiles
    return -np.inf, np.zeros(n_stages), np.zeros(max(n_stages - 1, 1))


def _stage_matrices(
    correct: np.ndarray, cost: np.ndarray, score: np.ndarray, order: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool-wide arrays reduced to one ordered candidate list."""
    L_mat = correct[:, order]
    C_mat = np.cumsum(cost[:, order], axis=1)   # a query pays every stage it ran
    d_mat = 1.0 - score[:, order]
    return L_mat, C_mat, d_mat


def fit_cascade(
    correct: np.ndarray,
    cost: np.ndarray,
    score: np.ndarray,
    budget: float,
    depth: int = 3,
) -> CascadeFit:
    """Search ordered model lists of length ``depth`` and fit each one's thresholds.

    ``correct``, ``cost`` and ``score`` are (queries, pool) arrays: correctness
    in {0, 1}, measured per-query cost, and the stage scorer's accept
    probability for that model's answer to that query.

    Every ordered list is evaluated, per the released code. ``depth`` is fixed
    rather than searched, matching their experiments, which use a cascade length
    of 3 over a pool of 12.
    """
    for name, arr in (("correct", correct), ("cost", cost), ("score", score)):
        if arr.shape != correct.shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {correct.shape}")
    n_pool = correct.shape[1]
    if not 1 <= depth <= n_pool:
        raise ValueError(f"depth {depth} outside 1..{n_pool} for this pool")

    best: CascadeFit | None = None
    for order in itertools.permutations(range(n_pool), depth):
        L_mat, C_mat, d_mat = _stage_matrices(correct, cost, score, order)
        acc, thresholds, quantiles = optimise_thresholds(L_mat, C_mat, d_mat, budget)
        if acc == -np.inf:
            continue
        answered = _first_acceptance(d_mat, thresholds)
        spend = float((answered * C_mat).sum() / correct.shape[0])
        if best is None or acc > best.accuracy:
            best = CascadeFit(order, thresholds, quantiles, acc, spend)
    if best is None:
        raise ValueError(
            f"no ordered list of {depth} model(s) meets a budget of {budget:g}; "
            f"the cheapest model alone averages {cost.mean(axis=0).min():g}"
        )
    return best


def apply_cascade(
    fit: CascadeFit, correct: np.ndarray, cost: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a fitted cascade on a split, returning per-query outcomes.

    Returns correctness, cost and the index of the answering stage, one entry
    per query. Thresholds are the fitted ones, applied unchanged: refitting them
    on the split being reported would be fitting on test.
    """
    L_mat, C_mat, d_mat = _stage_matrices(correct, cost, score, fit.models)
    answered = _first_acceptance(d_mat, fit.thresholds)
    if not answered.any(axis=1).all():
        raise ValueError("a query was answered by no stage; the last threshold "
                         "must be 1.0 so the final model always answers")
    stage = answered.argmax(axis=1)
    rows = np.arange(correct.shape[0])
    return L_mat[rows, stage], C_mat[rows, stage], stage
