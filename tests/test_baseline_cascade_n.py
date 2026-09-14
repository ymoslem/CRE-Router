"""The N-stage baseline accounting must agree with the router's at depth 2."""
from __future__ import annotations

import numpy as np
import pytest

from cre_router.baselines.cascade_n import n_stage_metrics
from cre_router.cascade import per_request_metrics

NQ, NR = 7, 5


def grids(rng):
    return {"correct": rng.integers(0, 2, (NQ, NR)).astype(float),
            "e2el": rng.uniform(1.0, 40.0, (NQ, NR)),
            "tokens": rng.integers(50, 4000, (NQ, NR)).astype(float),
            "tpot": rng.uniform(5.0, 45.0, (NQ, NR))}


class TestAgreesWithTheRouterAtDepthTwo:
    @pytest.mark.parametrize("seed", range(5))
    def test_same_metrics(self, seed):
        rng = np.random.default_rng(seed)
        eff, strong = grids(rng), [grids(rng) for _ in range(NR)]
        escalate = rng.integers(0, 2, (NQ, NR)).astype(bool)
        runs_eff = np.ones((NQ, NR), dtype=bool)

        want = per_request_metrics(runs_eff, escalate, eff, strong)
        got = n_stage_metrics(escalate.astype(int), eff, [strong])
        for a, b, name in zip(got, want, ("correct", "e2el", "tpot")):
            assert np.allclose(a, b), name


class TestTheTpotRule:
    def test_a_discarded_pass_costs_time_and_earns_no_tokens(self):
        # One question, one run each way. Stage 1 burns 1,001 tokens at 10 ms,
        # stage 2 delivers 101 tokens at 20 ms. Decode time is
        # 10*1000 + 20*100 = 12,000 ms over the 100 tokens the user received.
        eff = {"correct": np.zeros((1, 1)), "e2el": np.full((1, 1), 11.0),
               "tokens": np.full((1, 1), 1001.0), "tpot": np.full((1, 1), 10.0)}
        strong = [{"correct": np.ones((1, 1)), "e2el": np.full((1, 1), 3.0),
                   "tokens": np.full((1, 1), 101.0), "tpot": np.full((1, 1), 20.0)}]
        correct, e2el, tpot = n_stage_metrics(np.ones((1, 1), dtype=int), eff, [strong])
        assert correct[0, 0, 0] == 1.0
        assert e2el[0, 0, 0] == pytest.approx(14.0)
        assert tpot[0, 0, 0] == pytest.approx(120.0)
        # Summing the two TPOTs would give 30, which is the error v0.3.1 fixed.
        assert tpot[0, 0, 0] != pytest.approx(30.0)

    def test_an_unescalated_query_reports_its_own_measured_tpot(self):
        rng = np.random.default_rng(0)
        eff, strong = grids(rng), [grids(rng) for _ in range(NR)]
        _, _, tpot = n_stage_metrics(np.zeros((NQ, NR), dtype=int), eff, [strong])
        assert np.allclose(tpot, np.repeat(eff["tpot"][:, :, None], NR, axis=2))


class TestRejectsBadInput:
    def test_answering_stage_beyond_the_list(self):
        rng = np.random.default_rng(0)
        eff = grids(rng)
        with pytest.raises(ValueError, match="only 1 stages"):
            n_stage_metrics(np.ones((NQ, NR), dtype=int), eff, [])

    def test_missing_field(self):
        rng = np.random.default_rng(0)
        eff = grids(rng)
        del eff["tokens"]
        with pytest.raises(ValueError, match="missing 'tokens'"):
            n_stage_metrics(np.zeros((NQ, NR), dtype=int), eff, [])
