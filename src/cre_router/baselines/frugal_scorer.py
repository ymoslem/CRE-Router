"""FrugalGPT's generation scorer: is this answer good enough to return?

The scoring function ``g(q, a)`` of Chen, Zaharia and Zou (TMLR 2024), following
their released ``src/FrugalGPT/scoring.py`` rather than the paper's description
of it. The paper calls it "a simple regression model"; the code builds a
two-class sequence classifier, trains on integer labels, and reads the score off
``softmax(logits)[1]``. That accept probability is what the cascade thresholds.

One scorer is trained per pool model, because a stage judges the answers of the
model at that stage. ``llmcascade.py`` keeps them in ``self.MyScores[service]``.

**What is deliberately not copied.** Their ``scorer_text`` reduces the input to
``"Q:" + text.split("Q:")[-1]``, which strips a few-shot prefix from their
prompt format. Our prompts carry no such prefix, so the faithful analogue is the
question and the answer alone, which is what :func:`scorer_input` builds. In
particular the answer's token count is *not* included: our own quality estimator
uses it as a feature, and handing it to the baseline would erase part of what
distinguishes the two methods.

**Backbone.** Their code uses ``distilbert-base-uncased``. The default here is
the same encoder our own estimator uses, so a difference between the two methods
is a difference of method rather than of model capacity. Pass ``base_model`` to
reproduce their exact backbone instead; the comparison is then confounded, which
is why it is not the default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["ScorerConfig", "scorer_input", "train_scorer", "score_texts"]

# Their encoder. Ours is the default; this is here for a fidelity check.
FRUGALGPT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"


@dataclass(frozen=True)
class ScorerConfig:
    """Training settings, defaulting to the values in their ``TrainingArguments``.

    ``val_fraction`` is their ``test_size=0.55``: the split is 45% train and 55%
    validation, which is unusual and is theirs, not a typo here. It costs real
    training data, so it is recorded rather than quietly improved.
    """

    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = 8
    batch_size: int = 8
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_length: int = 512
    val_fraction: float = 0.55
    seed: int = 2024
    # ModernBERT needs its attention kernel named explicitly; letting
    # transformers choose sends it down a Triton path that fails with
    # "Pointer argument cannot be accessed from Triton (cpu tensor?)".
    # Our own estimator passes the same, falling back to sdpa without CUDA.
    attn_implementation: str = "flash_attention_2"
    extra: dict = field(default_factory=dict)


def scorer_input(question: str, answer: str) -> str:
    """The text the scorer sees: the query and the answer it is judging.

    Their ``llmcascade.py`` forms ``query + " " + response`` and then trims the
    few-shot prefix. Ours have no prefix to trim, so this is the whole input.
    """
    return f"{question} {answer}"


def train_scorer(
    texts: Sequence[str],
    labels: Sequence[int],
    out_dir: str | Path,
    config: ScorerConfig | None = None,
):
    """Fit one stage's scorer on (query + answer) text with binary correctness.

    ``labels`` are 1 when the answer was correct and 0 otherwise, matching their
    integer labels. Returns the directory the model was written to.

    Imports of transformers and torch happen here rather than at module import,
    so the cascade optimiser stays usable on a machine with neither.
    """
    # Checked before the heavy imports, so a bad call fails immediately rather
    # than after torch and transformers have loaded.
    cfg = config or ScorerConfig()
    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts against {len(labels)} labels")
    if not set(int(v) for v in labels) <= {0, 1}:
        raise ValueError("labels must be 0 or 1; the head is two-class")

    import numpy as _np
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    import torch

    out_dir = Path(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tr_x, va_x, tr_y, va_y = train_test_split(
        list(texts), [int(v) for v in labels],
        test_size=cfg.val_fraction, random_state=cfg.seed,
    )

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.enc = tokenizer(xs, truncation=True, padding=True,
                                 max_length=cfg.max_length)
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    def accuracy(pred):
        return {"accuracy": float(
            (_np.argmax(pred.predictions, axis=-1) == pred.label_ids).mean())}

    attn = cfg.attn_implementation if torch.cuda.is_available() else "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, attn_implementation=attn)
    trainer = Trainer(
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
            load_best_model_at_end=True,
            # Keep one checkpoint, not one per epoch. `load_best_model_at_end`
            # needs a saved checkpoint to restore from, but the other seven are
            # dead weight: the final model is written separately by save_model.
            # Eight per tier filled a 500 GB quota and took three unrelated jobs
            # down with it.
            save_total_limit=1,
            seed=cfg.seed,
            **cfg.extra,
        ),
        train_dataset=_Dataset(tr_x, tr_y),
        eval_dataset=_Dataset(va_x, va_y),
        compute_metrics=accuracy,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def score_texts(
    model_dir: str | Path,
    texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Accept probability per text, their ``softmax(logits)[1]``.

    Returned in the order given, so the result can be reshaped straight into the
    ``score`` column the cascade optimiser expects for this stage.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    attn = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), attn_implementation=attn)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    out = np.empty(len(texts), dtype=float)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(chunk, truncation=True, padding=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            out[i:i + len(chunk)] = probs.cpu().numpy()
    return out
