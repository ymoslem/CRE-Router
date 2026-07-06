"""End-to-end cascade behaviour with injected components (no GPUs, no servers)."""

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from cre_router.server.cascade_router import CascadeRouter, extract_query

WEAK, MID, STRONG = "weak-model", "mid-model", "strong-model"
LADDER = [WEAK, MID, STRONG]

# Two well-separated centroids; queries embed onto one or the other axis.
CENTROIDS = np.array([[1.0, 0.0], [0.0, 1.0]])
EMBEDDINGS = {"easy question": [0.9, 0.1], "hard question": [0.1, 0.9]}


def embed_fn(texts):
    return np.array([EMBEDDINGS[t] for t in texts])


@dataclass
class StubDecision:
    accept: bool
    p_accept: float = 0.5


class Backend:
    """Records calls and answers as a chat-completions dict."""

    def __init__(self):
        self.calls = []

    async def __call__(self, model, request):
        self.calls.append(model)
        return {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": f"answer from {model}"}}],
            "usage": {"completion_tokens": 42},
        }


def make_router(backend, routing_table=None, qe_predict_fns=None, escalation_order=LADDER):
    return CascadeRouter(
        centroids=CENTROIDS,
        routing_table=routing_table or {"0": WEAK, "1": STRONG},
        embed_fn=embed_fn,
        completion_fn=backend,
        escalation_order=escalation_order,
        qe_predict_fns=qe_predict_fns,
    )


def request(text):
    return {"messages": [{"role": "user", "content": text}]}


def always(decision):
    return lambda q, o, n: decision


class TestStage1:
    def test_routes_by_nearest_centroid(self):
        backend = Backend()
        router = make_router(backend)
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert (meta.cluster, meta.stage1_model) == (0, WEAK)
        _, meta = asyncio.run(router.acompletion(request("hard question")))
        assert (meta.cluster, meta.stage1_model) == (1, STRONG)
        assert backend.calls == [WEAK, STRONG]

    def test_unknown_cluster_in_table_rejected(self):
        with pytest.raises(ValueError, match="unknown clusters"):
            CascadeRouter(
                centroids=CENTROIDS,
                routing_table={"0": WEAK, "7": STRONG},
                embed_fn=embed_fn,
                completion_fn=Backend(),
            )

    def test_uncovered_cluster_rejected(self):
        with pytest.raises(ValueError, match="no entry for clusters"):
            CascadeRouter(
                centroids=CENTROIDS,
                routing_table={"0": WEAK},
                embed_fn=embed_fn,
                completion_fn=Backend(),
            )


class TestTwoTierCascade:
    def test_low_quality_output_escalates_one_step(self):
        backend = Backend()
        router = make_router(backend, qe_predict_fns={WEAK: always(StubDecision(accept=False))})
        response, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK, MID]  # one rung up the ladder
        assert meta.path == [WEAK, MID]
        assert meta.escalated and meta.final_model == MID

    def test_accepted_output_is_returned(self):
        backend = Backend()
        router = make_router(backend, qe_predict_fns={WEAK: always(StubDecision(accept=True))})
        response, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK]
        assert not meta.escalated and meta.path == [WEAK]

    def test_strongest_model_bypasses_qe(self):
        backend = Backend()
        calls = []

        def qe(q, o, n):
            calls.append(q)
            return StubDecision(accept=False)

        # STRONG is the top of the ladder, so even a classifier for it never fires.
        router = make_router(
            backend,
            routing_table={"0": WEAK, "1": STRONG},
            qe_predict_fns={WEAK: qe},
        )
        _, meta = asyncio.run(router.acompletion(request("hard question")))
        assert calls == []  # QE never ran for STRONG
        assert backend.calls == [STRONG] and not meta.escalated


class TestMultiLevelCascade:
    def test_rolls_through_ladder_until_accept(self):
        backend = Backend()
        # WEAK escalates, MID accepts -> stop at MID.
        router = make_router(
            backend,
            routing_table={"0": WEAK, "1": STRONG},
            qe_predict_fns={
                WEAK: always(StubDecision(accept=False, p_accept=0.1)),
                MID: always(StubDecision(accept=True, p_accept=0.9)),
            },
        )
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK, MID]
        assert meta.path == [WEAK, MID]
        assert meta.p_accept == 0.9  # last QE decision recorded

    def test_rolls_all_the_way_to_strongest(self):
        backend = Backend()
        # WEAK and MID both escalate -> climb to STRONG (top), then stop.
        router = make_router(
            backend,
            routing_table={"0": WEAK, "1": STRONG},
            qe_predict_fns={
                WEAK: always(StubDecision(accept=False)),
                MID: always(StubDecision(accept=False)),
            },
        )
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK, MID, STRONG]
        assert meta.path == [WEAK, MID, STRONG]
        assert meta.final_model == STRONG

    def test_starts_mid_ladder_and_escalates_once(self):
        backend = Backend()
        # Stage 1 routes cluster 0 to MID; MID escalates -> STRONG (top).
        router = make_router(
            backend,
            routing_table={"0": MID, "1": STRONG},
            qe_predict_fns={MID: always(StubDecision(accept=False))},
        )
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [MID, STRONG]
        assert meta.path == [MID, STRONG]


class TestLadderValidation:
    def test_classifier_for_strongest_rejected(self):
        with pytest.raises(ValueError, match="nothing to escalate to"):
            make_router(Backend(), qe_predict_fns={STRONG: always(StubDecision(accept=False))})

    def test_classifier_off_ladder_rejected(self):
        with pytest.raises(ValueError, match="not in escalation_order"):
            make_router(
                Backend(),
                escalation_order=[WEAK, MID],  # STRONG missing
                qe_predict_fns={STRONG: always(StubDecision(accept=False))},
            )

    def test_routed_model_off_ladder_rejected(self):
        with pytest.raises(ValueError, match="not in escalation_order"):
            make_router(
                Backend(),
                routing_table={"0": "unlisted", "1": STRONG},
                escalation_order=LADDER,
                qe_predict_fns={WEAK: always(StubDecision(accept=True))},
            )

    def test_routed_model_without_classifier_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            # MID is routed and non-top, but only WEAK has a classifier.
            make_router(
                Backend(),
                routing_table={"0": MID, "1": STRONG},
                qe_predict_fns={WEAK: always(StubDecision(accept=True))},
            )
        assert any("never escalate" in r.message for r in caplog.records)


class TestExtractQuery:
    def test_takes_last_user_message(self):
        messages = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        assert extract_query(messages) == "second"

    def test_joins_text_parts(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert extract_query(messages) == "part one part two"

    def test_no_user_message_raises(self):
        with pytest.raises(ValueError, match="no user message"):
            extract_query([{"role": "system", "content": "hi"}])

    def test_null_content_does_not_crash_qe(self):
        # A model returning content: null must not crash the QE step.
        class NullBackend(Backend):
            async def __call__(self, model, request):
                self.calls.append(model)
                return {
                    "model": model,
                    "choices": [{"message": {"role": "assistant", "content": None}}],
                    "usage": {"completion_tokens": 0},
                }

        backend = NullBackend()
        router = make_router(backend, qe_predict_fns={WEAK: always(StubDecision(accept=False))})
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert meta.path == [WEAK, MID]


class TestEscalationOrderDerivation:
    """The auto-derived ladder must match the lambda-table arithmetic: drop
    Pareto-dominated models, then order weak -> strong by TPOT."""

    def _artifacts(self):
        from cre_router.artifacts import RouterArtifacts

        # V is fast/weak; Q3.5 dominates Q3-30B (lower TPOT AND lower error).
        stats = {
            "cluster_sizes": {"0": 10},
            "models": {
                "V": {"tpot_ms": 9.0, "errors": {"0": 0.18}},
                "Q3.5": {"tpot_ms": 17.0, "errors": {"0": 0.05}},
                "Q3-30B": {"tpot_ms": 24.0, "errors": {"0": 0.08}},
            },
        }
        return RouterArtifacts(stats=stats)

    def test_derived_order_drops_dominated_and_sorts_weak_to_strong(self):
        from cre_router.server.cascade_router import _escalation_order

        config = {"models": {"V": {}, "Q3.5": {}, "Q3-30B": {}}, "qe": {"enabled": True}}
        # Q3-30B is dominated by Q3.5 -> excluded; ladder is V (weak) -> Q3.5 (strong).
        assert _escalation_order(config, self._artifacts()) == ["V", "Q3.5"]

    def test_derivation_restricted_to_served_models(self):
        from cre_router.server.cascade_router import _escalation_order

        config = {"models": {"V": {}, "Q3-30B": {}}, "qe": {"enabled": True}}
        # Only V and Q3-30B are served; neither dominates the other here.
        assert _escalation_order(config, self._artifacts()) == ["V", "Q3-30B"]

    def test_missing_stats_raises(self):
        from cre_router.artifacts import RouterArtifacts
        from cre_router.server.cascade_router import _escalation_order

        with pytest.raises(ValueError, match="no stats"):
            _escalation_order({"models": {"V": {}}, "qe": {"enabled": True}}, RouterArtifacts())
