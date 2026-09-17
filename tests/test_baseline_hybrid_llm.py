"""HybridLLM's label schemes must match the formulas in their released code."""
from __future__ import annotations

import numpy as np
import pytest

from cre_router.baselines.hybrid_llm import (
    TransformationFit,
    choose_t,
    det_labels,
    match_prob,
    prob_labels,
    transformation_grid,
)

GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.3, 2.0]


def _reference_match_prob(s, l, t):
    """Their `_match_prob`, transcribed, as an independent check."""
    s, l = np.asarray(s, float), np.asarray(l, float)
    return sum([sum(i >= l - t) / len(l) for i in s]) / len(s)


class TestMatchProb:
    def test_agrees_with_their_transcribed_formula(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.uniform(-5, 0, rng.integers(1, 7))
            l = rng.uniform(-5, 0, rng.integers(1, 7))
            t = float(rng.uniform(-1, 2))
            assert match_prob(s, l, t) == pytest.approx(_reference_match_prob(s, l, t))

    def test_is_the_full_cross_product_not_paired_samples(self):
        # Paired would give 0.5 here: (1>=0) and (0>=1) -> one of two.
        # The cross product is 3 of 4: (1,0) (1,1) (0,0) pass, (0,1) does not.
        s, l = [1.0, 0.0], [0.0, 1.0]
        assert match_prob(s, l) == pytest.approx(0.75)

    def test_relaxation_can_only_admit_more_pairs(self):
        rng = np.random.default_rng(1)
        s, l = rng.uniform(-5, 0, 6), rng.uniform(-5, 0, 6)
        vals = [match_prob(s, l, t) for t in (0.0, 0.5, 1.0, 2.0)]
        assert vals == sorted(vals)

    def test_empty_is_refused_rather_than_averaged(self):
        with pytest.raises(ValueError, match="at least one scored response"):
            match_prob([], [1.0])


class TestDetLabels:
    def test_reads_one_response_per_model(self):
        s = np.array([[1.0, 0.0], [0.0, 0.0]])
        l = np.array([[0.0, 1.0], [1.0, 1.0]])
        # index 0 only: small wins the first query, loses the second
        assert det_labels(s, l).tolist() == [1.0, 0.0]
        # index 1 flips the first
        assert det_labels(s, l, sample=1).tolist() == [0.0, 0.0]

    def test_threshold_relaxes_the_comparison(self):
        s, l = np.array([[0.4]]), np.array([[1.0]])
        assert det_labels(s, l, t=0.0).tolist() == [0.0]
        assert det_labels(s, l, t=0.7).tolist() == [1.0]


class TestSpread:
    def test_equation_3_matches_the_quadratic_form(self):
        from cre_router.baselines.hybrid_llm import _spread
        rng = np.random.default_rng(2)
        for _ in range(20):
            y = rng.uniform(0, 1, rng.integers(2, 40))
            want = float(np.abs(y[:, None] - y[None, :]).mean())
            assert _spread(y) == pytest.approx(want)

    def test_constant_labels_have_no_spread(self):
        from cre_router.baselines.hybrid_llm import _spread
        assert _spread(np.ones(10)) == pytest.approx(0.0)


class TestTransformationOnAContinuousMetric:
    """With a spectrum of quality, the relaxation does what it was written for."""

    def test_the_search_moves_off_zero_and_spreads_the_labels(self):
        rng = np.random.default_rng(3)
        n = 300
        # the large model is much stronger, their motivating case
        s = rng.normal(-4.5, 0.3, (n, 5))
        l = rng.normal(-3.0, 0.5, (n, 5))
        at_zero = prob_labels(s, l, 0.0)
        assert at_zero.mean() < 0.1, "setup should give almost no signal at t=0"
        fit = transformation_grid(s, l, np.linspace(0.0, 4.0, 41))
        assert fit.t > 0.0
        assert fit.labels.std() > at_zero.std()
        assert not fit.collapsed


class TestTransformationOnBinaryCorrectness:
    """Exact-match correctness has no spectrum, so the relaxation cannot act.

    This is the property the TeleMath measurement found; asserting it here
    fixes the behaviour in code rather than leaving it as an observation.
    """

    def test_labels_are_constant_below_one_and_saturate_at_one(self):
        rng = np.random.default_rng(4)
        s = rng.integers(0, 2, (200, 5)).astype(float)
        l = rng.integers(0, 2, (200, 5)).astype(float)
        base = prob_labels(s, l, 0.0)
        for t in (0.1, 0.25, 0.5, 0.75, 0.99):
            assert np.array_equal(prob_labels(s, l, t), base)
        for t in (1.0, 1.5, 3.0):
            assert np.array_equal(prob_labels(s, l, t), np.ones(200))

    def test_the_search_returns_the_t_equals_zero_region(self):
        rng = np.random.default_rng(5)
        s = rng.integers(0, 2, (200, 5)).astype(float)
        l = rng.integers(0, 2, (200, 5)).astype(float)
        fit = transformation_grid(s, l, GRID)
        assert fit.t < 1.0
        assert np.array_equal(fit.labels, prob_labels(s, l, 0.0))
        assert fit.collapsed
        # every t >= 1 kills the signal outright
        assert fit.spreads[fit.grid >= 1.0].max() == pytest.approx(0.0)


class TestCollapsedIsNotNoSignal:
    """The two degenerate cases are different and must not share a flag.

    An earlier version compared spreads, so `collapsed` fed an empty slice to
    `allclose` and reported True whenever the search found no signal anywhere.
    """

    def test_no_signal_is_reported_separately(self):
        # every label identical, so nothing to separate at any t
        fit = transformation_grid(np.ones((20, 3)), np.ones((20, 3)), [0.0, 0.5, 1.0])
        assert fit.no_signal
        assert fit.spreads.max() == pytest.approx(0.0)

    def test_binary_correctness_collapses_but_has_signal(self):
        rng = np.random.default_rng(8)
        s = rng.integers(0, 2, (200, 5)).astype(float)
        l = rng.integers(0, 2, (200, 5)).astype(float)
        fit = transformation_grid(s, l, GRID)
        assert fit.collapsed
        assert not fit.no_signal

    def test_a_continuous_metric_neither_collapses_nor_lacks_signal(self):
        rng = np.random.default_rng(9)
        s = rng.normal(-4.5, 0.3, (300, 5))
        l = rng.normal(-3.0, 0.5, (300, 5))
        fit = transformation_grid(s, l, np.linspace(0.0, 4.0, 41))
        assert not fit.collapsed
        assert not fit.no_signal

    def test_collapsed_holds_when_the_grid_omits_zero(self):
        """`labels_at_zero` is computed, not looked up, so this still works."""
        rng = np.random.default_rng(10)
        s = rng.integers(0, 2, (100, 4)).astype(float)
        l = rng.integers(0, 2, (100, 4)).astype(float)
        fit = transformation_grid(s, l, [0.3, 0.6, 0.9])
        assert fit.collapsed
        assert np.array_equal(fit.labels_at_zero, prob_labels(s, l, 0.0))


class TestPlumbing:
    def test_choose_t_agrees_with_the_full_fit(self):
        rng = np.random.default_rng(6)
        s, l = rng.normal(-4, 1, (50, 3)), rng.normal(-3, 1, (50, 3))
        assert choose_t(s, l, GRID) == transformation_grid(s, l, GRID).t

    def test_mismatched_query_counts_are_refused(self):
        with pytest.raises(ValueError, match="queries against"):
            prob_labels(np.zeros((3, 2)), np.zeros((4, 2)))

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(ValueError, match="grid is empty"):
            transformation_grid(np.zeros((2, 2)), np.zeros((2, 2)), [])

    def test_the_fit_keeps_every_grid_point(self):
        rng = np.random.default_rng(7)
        s, l = rng.normal(-4, 1, (30, 3)), rng.normal(-3, 1, (30, 3))
        fit = transformation_grid(s, l, GRID)
        assert isinstance(fit, TransformationFit)
        assert fit.grid.size == len(GRID) == fit.spreads.size


class TestRouterScoresLoading:
    """Scoring must load the router the way it was trained.

    Nothing runs through the model here: `from_pretrained` is replaced, so the
    test inspects only how it was called and where the model was sent.
    """

    def test_attention_kernel_and_device_match_training(self, monkeypatch):
        # transformers raises ImportError, not ModuleNotFoundError, when its own
        # dependency pins are unmet, and since pytest 8.2 `importorskip` lets
        # that through. An environment whose huggingface-hub sits outside the pin
        # says nothing about the kwargs this test checks, so skip on either.
        try:
            import torch
            import transformers
            auto_model = transformers.AutoModelForSequenceClassification
            auto_tok = transformers.AutoTokenizer
        except ImportError as exc:
            pytest.skip(f"torch or transformers cannot load here: {exc}")
        from cre_router.baselines.hybrid_llm import RouterConfig, router_scores

        calls = {}

        class _Model:
            def to(self, device):
                calls["device"] = device
                return self

            def eval(self):
                return self

        def fake_model(path, **kw):
            calls["kw"] = kw
            return _Model()

        monkeypatch.setattr(auto_model, "from_pretrained", staticmethod(fake_model))
        monkeypatch.setattr(auto_tok, "from_pretrained",
                            staticmethod(lambda path, **kw: None))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        router_scores("unused", [], config=RouterConfig())
        assert calls["kw"]["attn_implementation"] == RouterConfig().attn_implementation
        assert calls["device"] == "cuda"
