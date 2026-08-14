"""The two-stage cascade router used at serving time.

Stage 1: embed the incoming query, assign it to the nearest training
centroid, and send it to the model the offline routing table selected for
that cluster (paper Sec. 4).

Stage 2: a quality-estimation (QE) cascade over an ordered escalation ladder
(paper Sec. 5; the paper runs two-model pools, this generalises to more). The
ladder is always derived from the arithmetic, never configured by hand: it
follows the lambda table (as lambda rises the cheaper model wins), so the
Pareto-efficient models ordered by ascending cost give the escalation order,
under whichever cost metric ``cre fit`` used. Each model on the ladder (except
the strongest) has its own QE classifier trained on that model's outputs.
Starting from the model Stage 1 picked, the router runs the QE classifier for
the current model; on "escalate" it moves to the next stronger model and
repeats, stopping when a model's output is accepted or the top of the ladder
is reached. Outputs from a model with no classifier (e.g. the strongest) are
returned as-is.

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
        escalation_order: list[str] | None = None,
        qe_predict_fns: dict[str, QEPredictFn] | None = None,
    ):
        self.centroids = np.asarray(centroids)
        self.routing_table = {str(k): v for k, v in routing_table.items()}
        self.embed_fn = embed_fn
        self.completion_fn = completion_fn
        self.escalation_order = list(escalation_order or [])
        self.qe_predict_fns = dict(qe_predict_fns or {})
        self._rung = {model: i for i, model in enumerate(self.escalation_order)}

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
        # Every model a classifier can fire for must sit on the ladder with a
        # stronger model above it to escalate into.
        for model in self.qe_predict_fns:
            if model not in self._rung:
                raise ValueError(f"QE classifier for {model!r} but it is not in escalation_order")
            if self._rung[model] == len(self.escalation_order) - 1:
                raise ValueError(
                    f"QE classifier for {model!r} which is the strongest model; "
                    f"it has nothing to escalate to"
                )
        # Any model Stage 1 can route to should be on the ladder, otherwise a
        # cascade could never start from it.
        top_model = self.escalation_order[-1] if self.escalation_order else None
        for model in set(self.routing_table.values()):
            if self.qe_predict_fns and model not in self._rung:
                raise ValueError(
                    f"routing table uses {model!r} which is not in escalation_order"
                )
            # A routed, non-strongest model with no classifier is terminal: its
            # cluster's queries can never escalate. Legal, but usually a mistake.
            if self.qe_predict_fns and model != top_model and model not in self.qe_predict_fns:
                logger.warning(
                    "model %r is routed to but has no QE classifier and is not the "
                    "strongest model; its cluster(s) will never escalate",
                    model,
                )

    def route_query(self, query: str) -> tuple[int, str]:
        """Stage 1: nearest centroid, then the offline cluster-to-model table."""
        embedding = self.embed_fn([query])
        cluster = int(assign_clusters(embedding, self.centroids)[0])
        return cluster, self.routing_table[str(cluster)]

    def _stronger_model(self, model: str) -> str | None:
        """The next model up the ladder, or None if ``model`` is the top / off-ladder."""
        rung = self._rung.get(model)
        if rung is None or rung + 1 >= len(self.escalation_order):
            return None
        return self.escalation_order[rung + 1]

    async def acompletion(self, request: dict) -> tuple[Any, RouteMeta]:
        query = extract_query(request["messages"])
        # Embedding and QE inference are synchronous CPU/GPU work; run them off
        # the event loop so concurrent requests are not serialized behind them.
        cluster, model = await anyio.to_thread.run_sync(self.route_query, query)
        response = await self.completion_fn(model, request)
        meta = RouteMeta(cluster=cluster, path=[model])

        # Walk up the ladder: evaluate the current model's output and escalate
        # while a classifier says so and a stronger model exists.
        while True:
            current = meta.path[-1]
            predict = self.qe_predict_fns.get(current)
            stronger = self._stronger_model(current)
            if predict is None or stronger is None:
                break
            output, num_tokens = _extract_output(response)
            decision = await anyio.to_thread.run_sync(predict, query, output, num_tokens)
            meta.p_accept = getattr(decision, "p_accept", None)
            if decision.accept:
                break
            response = await self.completion_fn(stronger, request)
            meta.path.append(stronger)
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

        escalation_order: list[str] = []
        qe_predict_fns: dict[str, QEPredictFn] = {}
        qe_cfg = config.get("qe") or {}
        if qe_cfg.get("enabled"):
            escalation_order = _escalation_order(config, artifacts)

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
            escalation_order=escalation_order,
            qe_predict_fns=qe_predict_fns,
        )


def _escalation_order(config: dict, artifacts: RouterArtifacts) -> list[str]:
    """The weakest-to-strongest escalation ladder, derived purely from the
    arithmetic ``cre fit`` uses; it is never configured by hand.

    Build the pool from the fitted stats, drop Pareto-dominated models, and
    order the survivors by ascending cost. This is exactly the lambda-table
    capability order (as lambda rises the cheaper, weaker model wins), so the
    ladder cannot place a dominated model (costlier yet less accurate) as a
    stronger rung. Only served models (present in the config) are kept, since
    each rung must be callable.

    Cost means whichever metric ``cre fit`` used, read back from the artifacts.
    Deriving the ladder under TPOT while the table was fitted under E2EL breaks
    the correspondence, and on a pool mixing thinking and non-thinking members
    the two metrics disagree about both pruning and order.
    """
    from cre_router.routing import models_from_stats, pareto_prune

    if not (artifacts.stats or {}).get("models"):
        raise ValueError(
            "cannot derive the escalation ladder: the artifacts hold no stats. "
            "Run `cre fit --output <dir>` so the routing stats are stored."
        )
    all_models, _ = models_from_stats(artifacts.stats, artifacts.cost_metric)
    served = set(config["models"])
    pool = [m for m in all_models if m.name in served]
    efficient, _ = pareto_prune(pool)
    return [m.name for m in sorted(efficient, key=lambda m: m.cost_ms)]
