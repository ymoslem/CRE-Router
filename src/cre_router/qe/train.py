"""Fine-tune a binary accept/escalate QE classifier (paper Sec. 5.1, App. C).

Consolidates the AIME and TeleQnA training scripts into one config-driven
entry point. The dataset is expected to provide the columns ``question``,
``full_output``, ``num_tokens``, and ``decision_label`` (label 2, "continue",
is folded into 0, "route"), as in the released HF datasets
``ymoslem/AIME-router`` and ``ymoslem/TeleQnA-router``.

Usage (also exposed as ``cre qe-train``):
    python -m cre_router.qe.train --dataset ymoslem/AIME-router \
        --max-length 4096 --learning-rate 5e-5 --output-dir ./qe-aime
"""

from __future__ import annotations

import argparse

from cre_router.qe.classifier import MAX_OUTPUT_WORDS, format_qe_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", required=True, help="HF dataset with QE training data")
    parser.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--train-split", default=None, help="default: first split starting with 'train'")
    parser.add_argument("--eval-split", default=None, help="default: first split starting with 'test'")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096, help="4096 for AIME, 512 for TeleQnA")
    parser.add_argument("--max-output-words", type=int, default=MAX_OUTPUT_WORDS)
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="5e-5 AIME, 2e-5 TeleQnA")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="attention kernel for ModernBERT; use 'sdpa' if flash-attn is unavailable",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--report-to",
        default="none",
        help="training logger (default none; use 'tensorboard' if installed)",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


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
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # A local directory of {train,test}.jsonl (from `python -m cre_router.data.prep_qe`)
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
    eval_split = args.eval_split or next(s for s in dataset if s.startswith("test"))
    print(f"Train split: {train_split} ({len(dataset[train_split])} examples)")
    print(f"Eval split:  {eval_split} ({len(dataset[eval_split])} examples)")

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
    splits = {}
    for name in (train_split, eval_split):
        split = dataset[name].map(to_binary, batched=True)
        split = split.remove_columns([c for c in split.column_names if c not in columns])
        splits[name] = split.map(tokenize, batched=True, remove_columns=columns)

    train_labels = [_binary(l) for l in dataset[train_split]["decision_label"]]
    assert set(train_labels) <= {0, 1}, "labels must be binary after mapping"

    if args.no_class_weights:
        class_weights = None
    else:
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights (Route, Accept): {weights.round(3).tolist()}")

    # FlashAttention needs a GPU; fall back to sdpa on CPU regardless of the flag.
    attn = args.attn_implementation if torch.cuda.is_available() else "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
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
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=splits[train_split],
        eval_dataset=splits[eval_split],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

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


if __name__ == "__main__":
    main()
