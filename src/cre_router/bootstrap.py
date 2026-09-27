"""Confidence intervals for evaluations served as questions x runs.

Each system's outcomes form a grid of (question, efficient run, strong run): a
system with one stage has a single run axis, stored twice so every grid has the
same shape. A run is one batch served together, so a run's latency conditions
are shared by every question in it. The data are therefore **crossed**, not
nested, and the right resampling for crossed data draws the rows and the columns
independently, the pigeonhole bootstrap of Owen (2007). Resampling questions
alone, or averaging the runs away first, misses the run-level variation, which
matters most for TPOT and E2EL. The pigeonhole bootstrap is mildly conservative;
no bootstrap is exact for crossed data (McCullagh, 2000).

Two systems are compared on **paired differences per question** (Miller, 2024):
both answered the same questions, so the questions are resampled once for both.
Whether their runs are paired as well depends on whether they are the same
batches:

- :func:`gap`, for two **different** systems, ours against a baseline. Their
  runs are separate batches, so each system's run axes are resampled on their
  own. No run of one is tied to a run of the other.
- :func:`interval`, for **one** grid whose runs are shared, typically a
  difference computed cell by cell from the same batches: Stage 2 against
  Stage 1 of one system, or ours against always-strong priced on the very
  capture our system uses. There the pairing of runs is real and is kept.

:func:`questions_only` averages each question's runs first and resamples
questions alone. It is kept for comparison with the binomial interval, never
for a claim.

References: A. B. Owen, "The pigeonhole bootstrap", *Annals of Applied
Statistics* 1(2), 2007. P. McCullagh, "Resampling and exchangeable arrays",
*Bernoulli* 6, 2000. E. Miller, "Adding error bars to evals", arXiv:2411.00640,
2024.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DRAWS", "SEED", "gap", "interval", "questions_only"]

DRAWS = 20000
SEED = 0


def _check(grid: np.ndarray, name: str) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 3:
        raise ValueError(f"{name} must be (question, run, run), got shape {grid.shape}")
    return grid


def gap(a: np.ndarray, b: np.ndarray, draws: int = DRAWS, seed: int = SEED
        ) -> tuple[float, float, float]:
    """Mean of ``a`` minus ``b`` and its 95% interval, for two different systems.

    ``a`` and ``b`` are (question, run, run) grids over the same questions in
    the same order. Questions are resampled once for both; each system's two
    run axes are resampled independently of the other system's.
    """
    a, b = _check(a, "a"), _check(b, "b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"a has {a.shape[0]} questions, b has {b.shape[0]}")
    nq = a.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    # Drawn in blocks to bound memory. The block size fixes the random stream,
    # so it is part of what makes a published interval reproducible.
    block = 500
    for lo in range(0, draws, block):
        n = min(block, draws - lo)
        qi = rng.integers(0, nq, (n, nq))[:, :, None, None]
        ar = rng.integers(0, a.shape[1], (n, a.shape[1]))[:, None, :, None]
        as_ = rng.integers(0, a.shape[2], (n, a.shape[2]))[:, None, None, :]
        br = rng.integers(0, b.shape[1], (n, b.shape[1]))[:, None, :, None]
        bs = rng.integers(0, b.shape[2], (n, b.shape[2]))[:, None, None, :]
        boot[lo:lo + n] = (a[qi, ar, as_].mean(axis=(1, 2, 3))
                           - b[qi, br, bs].mean(axis=(1, 2, 3)))
    lo_, hi_ = np.percentile(boot, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo_), float(hi_)


def interval(grid: np.ndarray, draws: int = DRAWS, seed: int = SEED
             ) -> tuple[float, float, float]:
    """Mean of one grid and its 95% interval, its run axes shared by every cell.

    Pass a difference grid built cell by cell from the same batches to compare
    two things that share their runs; the pairing of runs is then kept.
    """
    grid = _check(grid, "grid")
    nq, nr, ns = grid.shape
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    block = 1000
    for lo in range(0, draws, block):
        n = min(block, draws - lo)
        qi = rng.integers(0, nq, (n, nq))
        ri = rng.integers(0, nr, (n, nr))
        si = rng.integers(0, ns, (n, ns))
        boot[lo:lo + n] = grid[qi[:, :, None, None], ri[:, None, :, None],
                               si[:, None, None, :]].mean(axis=(1, 2, 3))
    lo_, hi_ = np.quantile(boot, [0.025, 0.975])
    return float(grid.mean()), float(lo_), float(hi_)


def questions_only(per_question: np.ndarray, draws: int = DRAWS, seed: int = SEED
                   ) -> tuple[float, float, float]:
    """Mean and 95% interval resampling questions alone, runs already averaged.

    For comparison with the binomial interval only: it leaves out the run-level
    variation that :func:`gap` and :func:`interval` keep, so it is too narrow for
    a claim about a served system.
    """
    v = np.asarray(per_question, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"per_question must be one value per question, got {v.shape}")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    lo_, hi_ = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo_), float(hi_)
