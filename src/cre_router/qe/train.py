"""Fine-tune a binary accept/escalate QE classifier.

One config-driven entry point, with nothing specific to a benchmark. The
dataset is expected to provide the columns ``question``, ``full_output``,
``num_tokens``, and ``decision_label`` (label 2, "continue", is folded into 0,
"route"), as in the released HF datasets ``ymoslem/AIME-24-25-26-router``,
``ymoslem/TeleQnA-router-qwen4bi-261012``, and ``ymoslem/TeleMath-router``. ``--dataset``
also takes a local directory of ``train*.jsonl`` and ``test*.jsonl`` written
by ``data/prep_qe.py``, which is how a classifier for a new pool is trained.

``--dataset`` plus ``--train-split`` and ``--eval-split`` is the whole interface;
nothing here knows how a particular dataset is laid out. ``--eval-run`` is an
optional extra for a split that repeats its questions across runs, such as
``ymoslem/TeleQnA-router-qwen4bi-261012``, whose eval split holds five runs of the same 1,000
questions where one is enough for early stopping. It is ignored where there is
no ``run`` column.

Pass ``--train-split`` and ``--eval-split`` whenever the dataset carries more
than one pair. The defaults take the first split whose name starts with
"train" and with "test", which identifies the right data only when there is a
single pair, as in ``AIME-router`` and ``TeleQnA-router``. ``TeleMath-router``
instead holds a pair per efficient tier (``e2b``, ``e4b_think``, ``m26``,
``frugal_e2b``), and one classifier is trained per tier, so leaving the splits
to the defaults there would train on whichever tier the loader happened to
list first.

Usage (``python -m cre_router.qe.train`` takes the same arguments):
    cre qe-train --dataset ymoslem/AIME-router \
        --max-length 4096 --learning-rate 5e-5 --output-dir ./qe-aime
    cre qe-train --dataset ymoslem/TeleMath-router \
        --train-split train_e2b --eval-split test_e2b \
        --max-length 4096 --output-dir ./qe-telemath-e2b
"""

from __future__ import annotations

import argparse

from cre_router.qe.classifier import MAX_OUTPUT_WORDS, format_qe_input
from cre_router.textutils import HelpFormatter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cre qe-train", description="fine-tune a binary accept/escalate QE classifier (requires [qe])",
                                     formatter_class=HelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset with QE training data")
    parser.add_argument("--base-model", default="answerdotai/ModernBERT-base",
                        help="encoder to fine-tune")
    parser.add_argument("--train-split", default=None, help="default: first split starting with 'train'")
    parser.add_argument("--eval-split", default=None, help="default: first split starting with 'test'")
    parser.add_argument("--eval-run", type=int, default=None,
                        help="optional: evaluate on this run only, for a split that carries a "
                             "'run' column. Default is every row of the split")
    parser.add_argument("--output-dir", required=True, help="where to save the checkpoint")
    parser.add_argument("--max-length", type=int, default=4096, help="4096 for AIME and TeleMath, 512 for TeleQnA")
    parser.add_argument("--max-output-words", type=int, default=MAX_OUTPUT_WORDS,
                        help="keep only the last this-many words of each answer")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="5e-5 for AIME, 2e-5 for TeleQnA and TeleMath")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="effective training batch size, split over --grad-accum steps")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="gradient accumulation steps; each step takes batch-size / grad-accum "
                             "examples, so a 48 GB card can keep the effective batch of an 80 GB one")
    parser.add_argument("--early-stopping-patience", type=int, default=4,
                        help="epochs without improvement in eval macro F1 before stopping")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="attention kernel for ModernBERT; use 'sdpa' if flash-attn is unavailable",
    )
    parser.add_argument("--no-class-weights", action="store_true",
                        help="train without weighting the loss by class frequency")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache")
    parser.add_argument(
        "--report-to",
        default="none",
        help="training logger (default none; use 'tensorboard' if installed)",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="upload the checkpoint")
    parser.add_argument("--hub-private", action="store_true", help="make the upload private")
    parser.add_argument("--seed", type=int, default=0, help="training seed")
    parser.add_argument(
        "--cv-folds", type=int, default=0,
        help="choose the epoch count by k-fold cross-validation on the training split, "
             "folds split by --group-column, then retrain on the whole training split and "
             "stop at that epoch. No eval split is read. Default 0 keeps early stopping on "
             "--eval-split")
    parser.add_argument(
        "--group-column", default="question",
        help="with --cv-folds: rows sharing this value go to the same fold, so every "
             "answer to one question stays together")
    return parser


def group_folds(groups: list, k: int, seed: int) -> list[int]:
    """Fold index per row, every row of one group in the same fold.

    Groups are shuffled with ``seed`` and dealt round-robin, so fold sizes in groups
    differ by at most one. Rows of a group are never split across folds, which is
    what keeps the five answers to one question out of each other's validation fold.
    """
    import random

    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    unique = sorted(set(groups), key=str)
    if len(unique) < k:
        raise ValueError(f"{len(unique)} groups cannot fill {k} folds")
    random.Random(seed).shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return [fold_of[g] for g in groups]


def choose_epoch(per_fold: list[list[float]]) -> tuple[int, list[float]]:
    """The 1-based epoch with the best mean score across folds, and the means.

    ``per_fold[f][e]`` is fold ``f``'s validation macro-F1 after epoch ``e + 1``.
    Every fold must report the same number of epochs. Ties go to the earlier epoch.
    """
    if not per_fold or any(len(f) != len(per_fold[0]) for f in per_fold):
        raise ValueError("every fold must report a score for every epoch")
    n = len(per_fold[0])
    means = [sum(f[e] for f in per_fold) / len(per_fold) for e in range(n)]
    best = max(range(n), key=lambda e: (means[e], -e))
    return best + 1, means


def select_eval_run(eval_data, eval_run: int | None = None, split_name: str = "eval"):
    """Optionally restrict an eval split to a single run.

    ``eval_run`` is None by default, which evaluates on the whole split and leaves
    the tool ignorant of how a dataset is laid out. Pass a run when a split repeats
    its questions and the extra copies are not wanted: the eval split drives early
    stopping and best-checkpoint selection only, so ``ymoslem/TeleQnA-router``'s
    five runs of the same 1,000 questions add nothing there.

    A split with no ``run`` column is returned unchanged, which covers the released
    datasets that never had one and the local ``prep_qe.py`` directories.

    Returns the data to evaluate on and a line describing what was selected.
    """
    if eval_run is None or "run" not in eval_data.column_names:
        return eval_data, f"Eval split:  {split_name} ({len(eval_data)} examples)"
    available = sorted(set(eval_data["run"]))
    kept = eval_data.filter(lambda row: row["run"] == eval_run)
    if not len(kept):
        raise SystemExit(f"--eval-run {eval_run} matches no row in {split_name}; "
                         f"runs present: {available}")
    return kept, (f"Eval split:  {split_name} run {eval_run} "
                  f"({len(kept)} of {len(eval_data)} examples, runs {available})")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # A local directory of {train,test}.jsonl (from `python data/prep_qe.py`)
    # loads via the json builder; anything else is a Hub dataset id.
    from pathlib import Path as _Path

    if _Path(args.dataset).is_dir():
        # Only the split files themselves; a dir may also hold sidecar jsonl (e.g.
        # cascade_test_gens.jsonl) with a different schema that would break the
        # single-schema json builder if globbed in.
        data_files = {
            p.stem: str(p)
            for p in sorted(_Path(args.dataset).glob("*.jsonl"))
            if p.stem.startswith(("train", "test"))
        }
        if not data_files:
            raise SystemExit(f"{args.dataset} is a directory but has no train*/test* .jsonl splits")
        dataset = load_dataset("json", data_files=data_files, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)

    train_split = args.train_split or next(s for s in dataset if s.startswith("train"))
    print(f"Train split: {train_split} ({len(dataset[train_split])} examples)")

    def _binary(label):
        # decision_label 2 ("continue") folds into 0 ("route"); labels may be
        # float-formatted strings ("0.0") in the released datasets.
        value = int(float(label))
        return 0 if value == 2 else value

    def to_binary(batch):
        return {"labels": [_binary(l) for l in batch["decision_label"]]}

    def tokenize(batch):
        texts = [
            format_qe_input(q, o, t, tokenizer.sep_token, args.max_output_words)
            for q, o, t in zip(batch["question"], batch["full_output"], batch["num_tokens"])
        ]
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_length)
        tokens["labels"] = batch["labels"]
        return tokens

    columns = ["question", "full_output", "num_tokens", "labels"]

    def prepare(data):
        split = data.map(to_binary, batched=True)
        split = split.remove_columns([c for c in split.column_names if c not in columns])
        return split.map(tokenize, batched=True, remove_columns=columns)

    def weights_for(labels):
        if args.no_class_weights:
            return None
        w = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        return torch.tensor(w, dtype=torch.float32)

    # FlashAttention needs a GPU; fall back to sdpa on CPU regardless of the flag.
    attn = args.attn_implementation if torch.cuda.is_available() else "sdpa"

    def new_model():
        return AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=2, attn_implementation=attn, cache_dir=args.cache_dir
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

    def make_trainer(class_weights, output_dir, train_ds, eval_ds, select_best, callbacks):
        if args.batch_size % args.grad_accum:
            raise SystemExit(f"--batch-size {args.batch_size} is not divisible by --grad-accum {args.grad_accum}")
        micro = args.batch_size // args.grad_accum
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=args.grad_accum,
            per_device_eval_batch_size=micro,
            learning_rate=args.learning_rate,
            # The schedule always spans --epochs, also when training stops earlier,
            # so a stopped run and the cross-validation folds share one schedule.
            num_train_epochs=args.epochs,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=20,
            save_strategy="epoch" if select_best else "no",
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_total_limit=2 if select_best else None,
            load_best_model_at_end=select_best,
            metric_for_best_model="eval_f1_macro" if select_best else None,
            greater_is_better=True if select_best else None,
            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
            hub_private_repo=args.hub_private,
            seed=args.seed,
        )

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))(
                    outputs.logits.view(-1, model.config.num_labels), labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        trainer_cls = Trainer if class_weights is None else WeightedTrainer
        return trainer_cls(model=new_model(), args=training_args, train_dataset=train_ds,
                           eval_dataset=eval_ds, compute_metrics=compute_metrics,
                           callbacks=callbacks)

    train_labels = [_binary(l) for l in dataset[train_split]["decision_label"]]
    assert set(train_labels) <= {0, 1}, "labels must be binary after mapping"

    if args.cv_folds:
        _cross_validate(args, dataset[train_split], train_labels, prepare, weights_for,
                        make_trainer, tokenizer, TrainerCallback)
        return

    # The eval split only drives early stopping and best-checkpoint selection, so
    # one run per question is the convention: AIME-router and TeleMath-router bake
    # it into the data, while TeleQnA-router ships all five runs with `run` as a
    # column. Filtering here holds to that convention without a second copy of the
    # dataset, and repeated runs of the same questions would in any case add
    # nothing to a signal used only for choosing a checkpoint.
    eval_split = args.eval_split or next(s for s in dataset if s.startswith("test"))
    eval_data, message = select_eval_run(dataset[eval_split], args.eval_run, eval_split)
    print(message)
    splits = {train_split: prepare(dataset[train_split]), eval_split: prepare(eval_data)}

    class_weights = weights_for(train_labels)
    if class_weights is not None:
        print(f"Class weights (Route, Accept): {class_weights.numpy().round(3).tolist()}")

    trainer = make_trainer(class_weights, args.output_dir, splits[train_split], splits[eval_split],
                           True, [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
    trainer.train()

    predictions = trainer.predict(splits[eval_split])
    preds = np.argmax(predictions.predictions, axis=1)
    print("\n" + classification_report(
        predictions.label_ids, preds, target_names=["Route", "Accept"], zero_division=0
    ))

    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(
            dataset_tags=[args.dataset],
            language=["en"],
            finetuned_from=args.base_model,
            tags=["quality-estimation", "binary-classification", "llm-cascades"],
        )


def _cross_validate(args, raw, train_labels, prepare, weights_for, make_trainer, tokenizer,
                    TrainerCallback) -> None:
    """Choose the epoch count by k-fold cross-validation, then retrain on every row.

    Each fold trains for the full ``--epochs`` schedule with no early stopping and is
    scored on its held-out fold after every epoch. The chosen epoch is the one with
    the best macro-F1 averaged across folds. The final model trains on the whole
    training split under the same schedule and stops after that epoch. No test or
    eval split is read. Fold scores and the choice are written to ``cv.json``.
    """
    import json
    from pathlib import Path

    import torch

    if args.push_to_hub:
        raise SystemExit("--cv-folds trains one model per fold; push the final model separately")
    if args.group_column not in raw.column_names:
        raise SystemExit(f"--group-column {args.group_column!r} is not a column of the training split")
    groups = raw[args.group_column]
    fold = group_folds(groups, args.cv_folds, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for k in range(args.cv_folds):
        tr_idx = [i for i, f in enumerate(fold) if f != k]
        va_idx = [i for i, f in enumerate(fold) if f == k]
        print(f"Fold {k + 1}/{args.cv_folds}: {len(tr_idx)} training rows, {len(va_idx)} held out, "
              f"{len(set(groups[i] for i in va_idx))} held-out groups")
        trainer = make_trainer(weights_for([train_labels[i] for i in tr_idx]), str(out / f"fold{k}"),
                               prepare(raw.select(tr_idx)), prepare(raw.select(va_idx)), False, [])
        trainer.train()
        scores = [h["eval_f1_macro"] for h in trainer.state.log_history if "eval_f1_macro" in h]
        if len(scores) != args.epochs:
            raise SystemExit(f"fold {k}: {len(scores)} epoch scores, expected {args.epochs}")
        per_fold.append(scores)
        print(f"Fold {k + 1} macro-F1 by epoch: {[round(s, 4) for s in scores]}")
    epoch, means = choose_epoch(per_fold)
    print(f"Mean macro-F1 by epoch: {[round(m, 4) for m in means]}; chosen epoch {epoch}")

    class StopAfter(TrainerCallback):
        def on_epoch_end(self, train_args, state, control, **kwargs):
            if state.epoch is not None and state.epoch >= epoch - 1e-6:
                control.should_training_stop = True
            return control

    final = make_trainer(weights_for(train_labels), str(out / "final_run"), prepare(raw), None,
                         False, [StopAfter()])
    final.train()
    trained = round(final.state.epoch or 0)
    if trained != epoch:
        raise SystemExit(f"final model trained {trained} epochs, expected {epoch}")
    tokenizer.save_pretrained(args.output_dir)
    final.save_model(args.output_dir)
    (out / "cv.json").write_text(json.dumps({
        "dataset": args.dataset, "train_split": args.train_split, "group_column": args.group_column,
        "folds": args.cv_folds, "seed": args.seed, "epochs_schedule": args.epochs,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "per_fold_f1_macro": per_fold, "mean_f1_macro": means, "chosen_epoch": epoch,
        "rows": len(groups), "groups": len(set(groups)),
    }, indent=1))


if __name__ == "__main__":
    main()
