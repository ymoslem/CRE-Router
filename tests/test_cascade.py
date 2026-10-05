"""End-to-end cascade behaviour with injected components (no GPUs, no servers)."""

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from cre_router.server.cascade_router import CascadeRouter, extract_query

WEAK, MID, STRONG = "weak-model", "mid-model", "strong-model"

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


def make_router(backend, routing_table=None, qe_predict_fns=None, target=STRONG, gated=None):
    table = routing_table or {"0": WEAK, "1": STRONG}
    return CascadeRouter(
        centroids=CENTROIDS,
        routing_table=table,
        embed_fn=embed_fn,
        completion_fn=backend,
        escalation_target=target,
        gated_clusters={"0": table["0"]} if gated is None else gated,
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


class TestStage2:
    def test_rejected_answer_escalates_to_the_target(self):
        backend = Backend()
        router = make_router(backend, qe_predict_fns={WEAK: always(StubDecision(accept=False))})
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK, STRONG]
        assert meta.path == [WEAK, STRONG]
        assert meta.escalated and meta.final_model == STRONG

    def test_accepted_answer_is_returned(self):
        backend = Backend()
        router = make_router(backend, qe_predict_fns={WEAK: always(StubDecision(accept=True, p_accept=0.9))})
        _, meta = asyncio.run(router.acompletion(request("easy question")))
        assert backend.calls == [WEAK]
        assert not meta.escalated and meta.p_accept == 0.9

    def test_escalation_goes_straight_to_the_target(self):
        """A mid-cost model is never an intermediate step: one escalation, to the target."""
        backend = Backend()
        router = make_router(
            backend,
            routing_table={"0": MID, "1": WEAK},
            gated={"0": MID, "1": WEAK},
            qe_predict_fns={MID: always(StubDecision(accept=False)),
                            WEAK: always(StubDecision(accept=False))},
        )
        _, meta = asyncio.run(router.acompletion(request("hard question")))
        assert backend.calls == [WEAK, STRONG]

    def test_ungated_cluster_skips_qe(self):
        backend = Backend()
        calls = []

        def qe(q, o, n):
            calls.append(q)
            return StubDecision(accept=False)

        # Cluster 1 is served by a model that costs more than the target in the
        # plan, so it is not gated, even though a classifier exists for WEAK.
        router = make_router(backend, routing_table={"0": WEAK, "1": MID},
                             qe_predict_fns={WEAK: qe})
        _, meta = asyncio.run(router.acompletion(request("hard question")))
        assert calls == [] and backend.calls == [MID] and not meta.escalated


class TestPlanValidation:
    def test_gated_cluster_must_match_the_routing(self):
        with pytest.raises(ValueError, match="gated on"):
            make_router(Backend(), gated={"0": MID})

    def test_gated_clusters_need_a_target(self):
        with pytest.raises(ValueError, match="need an escalation target"):
            make_router(Backend(), target=None)

    def test_classifier_for_an_ungated_model_rejected(self):
        with pytest.raises(ValueError, match="no gated cluster uses"):
            make_router(Backend(), qe_predict_fns={STRONG: always(StubDecision(accept=False))})

    def test_gated_model_without_classifier_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            make_router(Backend(), routing_table={"0": WEAK, "1": MID},
                        gated={"0": WEAK, "1": MID},
                        qe_predict_fns={WEAK: always(StubDecision(accept=True))})
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
        assert meta.path == [WEAK, STRONG]


class TestArtifacts:
    def test_artifacts_without_a_recorded_metric_default_to_tpot(self):
        """router.json written before the metric was recorded must still load."""
        from cre_router.artifacts import RouterArtifacts

        assert RouterArtifacts().cost_metric == "tpot"

    def test_plan_survives_a_save_load_round_trip(self, tmp_path):
        from cre_router.artifacts import RouterArtifacts

        RouterArtifacts(cost_metric="e2el", escalation_target=STRONG,
                        gated_clusters={"0": WEAK}).save(tmp_path)
        back = RouterArtifacts.load(tmp_path)
        assert (back.cost_metric, back.escalation_target, back.gated_clusters) == (
            "e2el", STRONG, {"0": WEAK})


class TestBackendPayload:
    def test_model_params_override_the_client(self):
        from cre_router.server.cascade_router import backend_payload

        spec = {"params": {"temperature": 0.6,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}
        req = {"model": "ignored", "messages": [], "temperature": 1.0,
               "extra_body": {"top_k": 20}}
        out = backend_payload(req, spec)
        assert "model" not in out and out["temperature"] == 0.6
        assert out["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}

    def test_no_params_passes_the_request_through(self):
        from cre_router.server.cascade_router import backend_payload

        assert backend_payload({"model": "m", "messages": [1]}, {}) == {"messages": [1]}

    def test_example_configs_name_known_models(self):
        """Each shipped config's models carry params only of a known shape."""
        from pathlib import Path

        import yaml

        for path in (Path(__file__).parents[1] / "src" / "cre_router" / "server").glob("example_config_*.yaml"):
            cfg = yaml.safe_load(path.read_text())
            for name, spec in cfg["models"].items():
                assert {"litellm_model", "api_base"} <= set(spec), (path.name, name)
                assert isinstance(spec.get("params", {}), dict), (path.name, name)
