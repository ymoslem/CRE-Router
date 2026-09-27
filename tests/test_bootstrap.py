"""The crossed (questions x runs) bootstrap: pairing, run resampling, refusals."""
from __future__ import annotations

import numpy as np
import pytest

from cre_router.bootstrap import gap, interval, questions_only

RNG = np.random.default_rng(0)


def test_identical_systems_have_a_zero_gap_and_a_zero_width_interval():
    a = RNG.random((50, 5, 5))
    m, lo, hi = interval(a - a)
    assert (m, lo, hi) == (0.0, 0.0, 0.0)


def test_a_constant_shift_is_recovered_exactly_when_runs_are_shared():
    a = RNG.random((50, 5, 5))
    m, lo, hi = interval((a + 0.1) - a)
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_questions_are_paired_in_gap():
    """A per-question offset common to both systems cancels, as pairing requires."""
    a = RNG.normal(0, 0.01, (200, 5, 5))
    b = RNG.normal(0, 0.01, (200, 5, 5))
    hard = RNG.normal(0, 1.0, 200)[:, None, None]   # large question difficulty
    _, lo, hi = gap(a + hard, b + hard)
    _, lo0, hi0 = gap(a, b)
    assert hi - lo == pytest.approx(hi0 - lo0)


def test_run_level_variation_widens_the_interval():
    """A batch effect shared by every question in a run is missed by questions alone."""
    base = RNG.normal(0, 0.05, (200, 5, 5))
    batch = RNG.normal(0, 1.0, (1, 1, 5))           # one shift per run, all questions
    grid = base + batch
    _, lo, hi = interval(grid)
    _, qlo, qhi = questions_only(grid.mean(axis=(1, 2)))
    assert hi - lo > 3 * (qhi - qlo)


def test_gap_resamples_each_systems_runs_on_their_own():
    """Different systems: shifting one system's runs changes its mean, not pairing."""
    a = RNG.random((80, 5, 5))
    b = a.copy()
    m, lo, hi = gap(a, b[:, ::-1, ::-1])            # same values, run order reversed
    assert m == pytest.approx(0.0, abs=1e-12)
    assert lo < 0 < hi


def test_results_are_reproducible_from_the_seed():
    a, b = RNG.random((30, 5, 5)), RNG.random((30, 5, 5))
    assert gap(a, b, seed=3) == gap(a, b, seed=3)
    assert interval(a, seed=3) == interval(a, seed=3)


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="question, run, run"):
        interval(np.zeros((10, 5)))
    with pytest.raises(ValueError, match="questions"):
        gap(np.zeros((10, 5, 5)), np.zeros((11, 5, 5)))
    with pytest.raises(ValueError, match="one value per question"):
        questions_only(np.zeros((10, 5)))
