"""Implementation of Ding et al. (2024), the HybridLLM router.

"Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing", ICLR 2024. The
router makes one up-front choice between a small and a large model. Unlike a
cascade it makes a **single** LLM call per query: a learned score decides
between the two models before either runs, and there is no escalation. That
makes it the counterpart to our Stage 1, not to Stage 1 + 2.

The paper points at `github.com/m365-core/hybrid_llm_routing`, which is a
Microsoft Enterprise Managed Users tenant and returns 404 to everyone outside
it. The formulas here follow `microsoft/best-route-llm`, the same lead author's
public MIT repository, whose package is named `hybrid_llm/`; its
`pair_ranker/ranker.py` is where the label schemes are actually specified.

**What this module does and does not cover.** It builds the three label schemes
and the threshold search, which is the part that defines the method. Training
the encoder and scoring queries is ordinary sequence classification and lives
with the rest of our scorer code.

Quality gap, their Section 3::

    H(x) := q(S(x)) - q(L(x))

with ``q`` a response-quality metric, ``S`` the small model and ``L`` the large
one. A router ``r: X -> {0, 1}`` sends x to the small model when ``r(x) = 0``.

The three label schemes, their Equations 1, 2 and 4:

``det``
    ``y = 1[q(S) >= q(L) - t]`` from a **single** response per model. Their
    ``ranker.py`` reads sample index 0 for this, matching the paper's "single
    response per query", and applies ``self.t`` even though the paper's Eq 1
    carries no threshold.
``prob``
    ``y = Pr[H >= -t]``, estimated over responses. Their ``_match_prob`` is the
    **full cross-product**, not paired samples::

        sum([sum(i >= score_large - t) / len_l for i in score_small]) / len_s

    so with 5 runs per model it averages 25 pairs. The paper samples 10.
``trans``
    ``prob`` with ``t`` chosen by their Eq 3, which maximises the mean pairwise
    spread of the labels::

        t* = argmax_t (1/N^2) sum_{i,i'} |y(t)_i - y(t)_i'|

    solved by grid search. It exists because ``Pr[H >= 0]`` is near zero for
    almost every query when the large model is much stronger, which leaves the
    router almost no signal.

**A property worth knowing before using ``trans``.** The relaxation needs a
quality metric with a spectrum. Where ``q`` is binary correctness, ``y(t)`` is
constant for every ``0 <= t < 1`` and becomes all-ones at ``t >= 1``, where the
spread is exactly zero; the search then returns the ``t = 0`` region and ``trans``
coincides with ``prob``. That is a property of the evaluation metric rather than
of this code, so it is measured by the caller and reported, not special-cased
here. :func:`transformation_grid` returns the whole grid so the collapse is
visible rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "match_prob",
    "det_labels",
    "prob_labels",
    "transformation_grid",
    "choose_t",
    "TransformationFit",
    "RouterConfig",
    "train_router",
    "router_scores",
]


def match_prob(score_small: Sequence[float], score_large: Sequence[float],
               t: float = 0.0) -> float:
    """Their ``_match_prob``: the fraction of (small, large) response pairs in
    which the small model is within ``t`` of the large one.

    The average is over **every** pair, ``len_s * len_l`` of them, which is what
    their implementation does and is not what "sampling 10 responses from each
    model" on its own implies.
    """
    s = np.asarray(score_small, dtype=float)
    lg = np.asarray(score_large, dtype=float)
    if s.size == 0 or lg.size == 0:
        raise ValueError("both models need at least one scored response")
    return float(sum(np.sum(i >= lg - t) / lg.size for i in s) / s.size)


def det_labels(score_small: np.ndarray, score_large: np.ndarray,
               t: float = 0.0, sample: int = 0) -> np.ndarray:
    """Equation 1. One label per query from a single response per model.

    ``score_small`` and ``score_large`` are (queries, responses). ``sample``
    selects which response, index 0 as in their ``scores[:, :, :, 0]``.

    Returns 1.0 where the small model is good enough, so the label is the
    probability of **class 0**, their "route to the small model". Their code
    stores the complementary integer class; this returns the probability so all
    three schemes share one convention.
    """
    s = np.asarray(score_small, dtype=float)[:, sample]
    lg = np.asarray(score_large, dtype=float)[:, sample]
    return (s >= lg - t).astype(float)


def prob_labels(score_small: np.ndarray, score_large: np.ndarray,
                t: float = 0.0) -> np.ndarray:
    """Equation 2, and Equation 4 when ``t`` comes from :func:`choose_t`."""
    s = np.asarray(score_small, dtype=float)
    lg = np.asarray(score_large, dtype=float)
    if s.shape[0] != lg.shape[0]:
        raise ValueError(f"{s.shape[0]} queries against {lg.shape[0]}")
    return np.array([match_prob(s[i], lg[i], t) for i in range(s.shape[0])])


def _spread(y: np.ndarray) -> float:
    """Equation 3's objective, the mean pairwise absolute difference.

    Computed by sorting rather than as an N x N matrix: the same quantity, and
    N is the number of queries, so the quadratic form is wasteful and on a large
    split would dominate the search.
    """
    n = y.size
    if n < 2:
        return 0.0
    z = np.sort(y)
    w = np.arange(n, dtype=float)
    return float(2.0 * np.sum((2.0 * w - n + 1.0) * z) / (n * n))


@dataclass(frozen=True)
class TransformationFit:
    """The chosen relaxation and what the search saw, so it can be reported."""

    t: float
    labels: np.ndarray
    grid: np.ndarray
    spreads: np.ndarray
    labels_at_zero: np.ndarray

    @property
    def collapsed(self) -> bool:
        """True when the chosen relaxation gives exactly the labels of ``t = 0``.

        Then ``trans`` is ``prob`` and the relaxation had no room to act. Tested
        against the ``t = 0`` labels directly rather than by comparing spreads:
        two different label vectors can share a spread, and an earlier version
        compared spreads and reported a collapse whenever the search found no
        signal anywhere, which is a different situation entirely. See
        :attr:`no_signal` for that one.
        """
        return bool(np.array_equal(self.labels, self.labels_at_zero))

    @property
    def no_signal(self) -> bool:
        """True when no candidate ``t`` produced any spread at all.

        Every label is then identical, so the router has nothing to separate and
        the fit is not usable. Distinct from :attr:`collapsed`, which says the
        relaxation changed nothing while the labels still carry signal.
        """
        return bool(self.spreads.max() <= 0.0)


def transformation_grid(score_small: np.ndarray, score_large: np.ndarray,
                        grid: Sequence[float]) -> TransformationFit:
    """Equation 3 by grid search, keeping the whole grid.

    Their optimiser searches ``t`` by grid search and reports only the winner.
    Keeping every point costs nothing and is what lets a caller see a
    degenerate search instead of trusting a single returned number.
    """
    g = np.asarray(list(grid), dtype=float)
    if g.size == 0:
        raise ValueError("the grid is empty")
    labels = [prob_labels(score_small, score_large, float(t)) for t in g]
    spreads = np.array([_spread(y) for y in labels])
    best = int(np.argmax(spreads))
    # Computed rather than looked up: `collapsed` must mean something even when
    # the caller's grid does not contain 0.
    at_zero = prob_labels(score_small, score_large, 0.0)
    return TransformationFit(t=float(g[best]), labels=labels[best],
                             grid=g, spreads=spreads, labels_at_zero=at_zero)


def choose_t(score_small: np.ndarray, score_large: np.ndarray,
             grid: Sequence[float]) -> float:
    """Just the chosen relaxation, for callers that want nothing else."""
    return transformation_grid(score_small, score_large, grid).t


@dataclass(frozen=True)
class RouterConfig:
    """Training settings for the router encoder.

    Their `config.py` defaults: `microsoft/deberta-v3-large`, 5 epochs, and
    `source_max_length = 128`. Two are changed deliberately and both are
    recorded rather than silently adopted.

    ``base_model`` is ModernBERT-base, the same encoder our own quality
    estimator uses. Running DeBERTa for theirs and ModernBERT for ours would
    compare encoders rather than routing methods. Their own code makes the
    backbone a parameter, so this is a supported axis, not a departure.

    ``max_length`` is 256 rather than their 128. Their 128 was sized for
    MixInstruct instructions; a TeleMath question averages about 99 tokens and
    **16% of them exceed 128**, so their cap would truncate one question in six
    and the router would decide on a fragment. 256 clears the longest question
    in the split.
    """

    base_model: str = "answerdotai/ModernBERT-base"
    epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_length: int = 256
    val_fraction: float = 0.2
    seed: int = 2024
    attn_implementation: str = "eager"


def train_router(texts, labels, out_dir, config: RouterConfig | None = None):
    """Fit one router on query text with SOFT labels in [0, 1].

    ``labels[i]`` is the probability that the small model is good enough for
    query i, so it is the target for class 0. Their ``ranker.py`` forms
    ``[[p, 1 - p]]`` and takes ``F.cross_entropy`` against that distribution;
    this does the same, which is why the loss is supplied rather than left to
    the Trainer's default integer-label path.

    ``det`` labels are 0 or 1 and go through the same path unchanged, matching
    their code, where both variants share one loss with different targets.
    """
    cfg = config or RouterConfig()
    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts against {len(labels)} labels")
    lab = np.asarray(labels, dtype=float)
    if lab.size and (lab.min() < 0.0 or lab.max() > 1.0):
        raise ValueError("labels are probabilities and must lie in [0, 1]")

    from pathlib import Path as _Path

    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    out_dir = _Path(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tr_x, va_x, tr_y, va_y = train_test_split(
        list(texts), lab.tolist(), test_size=cfg.val_fraction,
        random_state=cfg.seed)

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.enc = tokenizer(xs, truncation=True, padding=True,
                                 max_length=cfg.max_length)
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            # float, so the collator keeps it as a soft target
            item["labels"] = torch.tensor(self.ys[i], dtype=torch.float)
            return item

    class _SoftTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            p = inputs.pop("labels")
            logits = model(**inputs).logits
            target = torch.stack([p, 1.0 - p], dim=-1)
            loss = F.cross_entropy(logits, target)
            return (loss, {"logits": logits}) if return_outputs else loss

    attn = cfg.attn_implementation if torch.cuda.is_available() else "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, attn_implementation=attn)
    trainer = _SoftTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=50,
            seed=cfg.seed,
            report_to=[],
            # Stated rather than inferred. With a custom loss the Trainer has to
            # be told which input is the target, or a float "labels" column can
            # be routed down the default integer-label path and the soft targets
            # silently become class indices.
            label_names=["labels"],
        ),
        train_dataset=_Dataset(tr_x, tr_y),
        eval_dataset=_Dataset(va_x, va_y),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def router_scores(model_dir, texts, config: RouterConfig | None = None,
                  batch_size: int = 64):
    """Router score per query: the probability of class 0, the small model.

    Their test-time rule routes a query to the small model when its score is
    ABOVE the threshold, so a high score means "the small model will do".
    """
    cfg = config or RouterConfig()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Loaded exactly as it was trained: same attention kernel, same precision,
    # on the GPU when there is one. Left to its defaults, ModernBERT selects
    # flash-attention whenever the package is installed, which needs half
    # precision on a CUDA device and fails on the float32 CPU model that a bare
    # `from_pretrained` returns.
    on_gpu = torch.cuda.is_available()
    attn = cfg.attn_implementation if on_gpu else "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), attn_implementation=attn)
    model.to("cuda" if on_gpu else "cpu")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), truncation=True,
                            padding=True, max_length=cfg.max_length,
                            return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 0].cpu().tolist())
    return np.asarray(out)
