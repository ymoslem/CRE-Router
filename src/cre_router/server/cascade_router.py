"""The two-stage cascade router used at serving time.

Stage 1: embed the incoming query, assign it to the nearest training
centroid, and send it to the model the offline routing table selected for
that cluster (paper Sec. 4).

Stage 2: a quality estimator reads the answer of each gated cluster's model and
escalates the answers it rejects to one strong model. The target and the gated
clusters are the Stage 2 plan ``cre fit`` stores in the artifacts
(``routing.stage2_plan``), never configured by hand: the target is the most
accurate model the routing uses, and every cluster served by a cheaper model is
gated. Each gated model has its own classifier, trained on its own answers.

Model calls go through ``litellm.Router`` against the vLLM servers in the
pool. All heavy components are injectable, which is also how the tests
exercise the cascade without GPUs or servers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import anyio
import numpy as np

from cre_router.artifacts import RouterArtifacts
from cre_router.clustering import assign_clusters

logger = logging.getLogger(__name__)


class SupportsAccept(Protocol):
    accept: bool
    p_accept: float


EmbedFn = Callable[[list[str]], np.ndarray]
CompletionFn = Callable[[str, dict], Awaitable[Any]]
QEPredictFn = Callable[[str, str, int], SupportsAccept]


@dataclass
class RouteMeta:
    """Per-request routing trace, surfaced as x-cre-* response headers."""

    cluster: int
    path: list[str]  # models tried, in order: [stage1_model, ...escalations]
    p_accept: float | None = None  # QE accept probability at the final QE step

    @property
    def stage1_model(self) -> str:
        return self.path[0]

    @property
    def final_model(self) -> str:
        return self.path[-1]

    @property
    def escalated(self) -> bool:
        return len(self.path) > 1


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_query(messages: list[dict]) -> str:
    """The routed query is the latest user message (text parts only)."""
    for message in reversed(messages):
        if _field(message, "role") == "user":
            content = _field(message, "content", "")
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
            return content or ""
    raise ValueError("request has no user message to route")


def _extract_output(response: Any) -> tuple[str, int]:
    """Return (content, output_token_count) from a chat-completions response,
    guarding against a null ``content`` and a missing/zero token count."""
    choices = _field(response, "choices") or []
    message = _field(choices[0], "message", {}) if choices else {}
    output = _field(message, "content", "") or ""
    usage = _field(response, "usage", {})
    completion_tokens = _field(usage, "completion_tokens", None)
    num_tokens = completion_tokens if completion_tokens is not None else len(output.split())
    return output, int(num_tokens)


class CascadeRouter:
    def __init__(
        self,
        *,
        centroids: np.ndarray,
        routing_table: dict[str, str],
        embed_fn: EmbedFn,
        completion_fn: CompletionFn,
        escalation_target: str | None = None,
        gated_clusters: dict[str, str] | None = None,
        qe_predict_fns: dict[str, QEPredictFn] | None = None,
    ):
        self.centroids = np.asarray(centroids)
        self.routing_table = {str(k): v for k, v in routing_table.items()}
        self.embed_fn = embed_fn
        self.completion_fn = completion_fn
        self.escalation_target = escalation_target
        self.gated_clusters = {str(k): v for k, v in (gated_clusters or {}).items()}
        self.qe_predict_fns = dict(qe_predict_fns or {})

        cluster_ids = {str(i) for i in range(len(self.centroids))}
        extra = set(self.routing_table) - cluster_ids
        if extra:
            raise ValueError(f"routing table references unknown clusters: {sorted(extra)}")
        uncovered = cluster_ids - set(self.routing_table)
        if uncovered:
            raise ValueError(
                f"routing table has no entry for clusters {sorted(uncovered)}; "
                f"the {len(self.centroids)} centroids and the routing table must "
                f"come from the same clustering run."
            )
        if self.gated_clusters and self.escalation_target is None:
            raise ValueError("gated clusters need an escalation target")
        for cluster, model in self.gated_clusters.items():
            if self.routing_table.get(cluster) != model:
                raise ValueError(
                    f"cluster {cluster} is gated on {model!r} but routed to "
                    f"{self.routing_table.get(cluster)!r}; the plan and the routing "
                    f"table must come from the same `cre fit`"
                )
        gated_models = set(self.gated_clusters.values())
        for model in self.qe_predict_fns:
            if model not in gated_models:
                raise ValueError(f"QE classifier for {model!r}, which no gated cluster uses")
        # A gated model with no classifier never escalates. Legal, since Stage 2
        # can be switched off, but when some classifiers are given it is a mistake.
        if self.qe_predict_fns:
            for model in sorted(gated_models - set(self.qe_predict_fns)):
                logger.warning(
                    "model %r serves gated clusters but has no QE classifier; "
                    "its answers will never escalate", model)

    def route_query(self, query: str) -> tuple[int, str]:
        """Stage 1: nearest centroid, then the offline cluster-to-model table."""
        embedding = self.embed_fn([query])
        cluster = int(assign_clusters(embedding, self.centroids)[0])
        return cluster, self.routing_table[str(cluster)]

    async def acompletion(self, request: dict) -> tuple[Any, RouteMeta]:
        query = extract_query(request["messages"])
        # Embedding and QE inference are synchronous CPU/GPU work; run them off
        # the event loop so concurrent requests are not serialized behind them.
        cluster, model = await anyio.to_thread.run_sync(self.route_query, query)
        response = await self.completion_fn(model, request)
        meta = RouteMeta(cluster=cluster, path=[model])

        # Stage 2: in a gated cluster, the estimator reads the answer, and a
        # rejected answer is escalated to the target.
        predict = self.qe_predict_fns.get(model)
        if str(cluster) in self.gated_clusters and predict is not None:
            output, num_tokens = _extract_output(response)
            decision = await anyio.to_thread.run_sync(predict, query, output, num_tokens)
            meta.p_accept = getattr(decision, "p_accept", None)
            if not decision.accept:
                response = await self.completion_fn(self.escalation_target, request)
                meta.path.append(self.escalation_target)
        return response, meta

    @classmethod
    def from_config(cls, config: dict | str | Path) -> "CascadeRouter":
        """Build a production router from a YAML config (see
        ``server/example_config_aime24.yaml``): artifacts dir + backend pool + QE."""
        if not isinstance(config, dict):
            import yaml

            config = yaml.safe_load(Path(config).read_text())

        artifacts = RouterArtifacts.load(config["artifacts_dir"])
        if artifacts.centroids is None:
            raise ValueError(f"{config['artifacts_dir']} has no centroids.npy; run `cre cluster`")
        if not artifacts.routing_table:
            raise ValueError(f"{config['artifacts_dir']} has no routing table; run `cre fit`")

        unknown = set(artifacts.routing_table.values()) - set(config["models"])
        if unknown:
            raise ValueError(f"routing table needs models missing from config: {sorted(unknown)}")

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(config.get("embedding_model", artifacts.embedding_model))

        def embed_fn(texts: list[str]) -> np.ndarray:
            return np.asarray(encoder.encode(texts))

        import litellm

        model_list = [
            {
                "model_name": name,
                "litellm_params": {
                    "model": spec["litellm_model"],
                    "api_base": spec.get("api_base"),
                    "api_key": spec.get("api_key", "EMPTY"),
                },
            }
            for name, spec in config["models"].items()
        ]
        litellm_router = litellm.Router(model_list=model_list)

        async def completion_fn(model: str, request: dict) -> Any:
            payload = {k: v for k, v in request.items() if k != "model"}
            return await litellm_router.acompletion(model=model, **payload)

        qe_predict_fns: dict[str, QEPredictFn] = {}
        qe_cfg = config.get("qe") or {}
        target, gated = artifacts.escalation_target, artifacts.gated_clusters
        if qe_cfg.get("enabled"):
            if not gated:
                raise ValueError(
                    f"{config['artifacts_dir']} gates no cluster, so QE has nothing to do. "
                    f"Either the routing has no Stage 2, or the artifacts predate the "
                    f"Stage 2 plan; run `cre fit --output {config['artifacts_dir']}` again")
            if target is not None and target not in config["models"]:
                raise ValueError(f"the escalation target {target!r} is missing from the "
                                 f"config's models")

            from cre_router.qe import QEClassifier

            for model_name, spec in (qe_cfg.get("classifiers") or {}).items():
                classifier = QEClassifier(
                    model_name=spec["checkpoint"],
                    base_tokenizer=spec.get("base_tokenizer"),
                    accept_threshold=spec.get("accept_threshold", 0.5),
                    max_length=spec.get("max_length", 4096),
                )
                qe_predict_fns[model_name] = classifier.predict

        return cls(
            centroids=artifacts.centroids,
            routing_table=artifacts.routing_table,
            embed_fn=embed_fn,
            completion_fn=completion_fn,
            escalation_target=target if qe_cfg.get("enabled") else None,
            gated_clusters=gated if qe_cfg.get("enabled") else {},
            qe_predict_fns=qe_predict_fns,
        )
