"""Evaluate a trained QE classifier on its own (paper Appendix D).

Runs the accept/escalate classifier over a dataset split of model outputs and
reports the classification metrics (accuracy, macro-F1, per-class scores, and
the confusion matrix with false-positive / false-negative counts) used in the
paper's QE cascade appendices. Requires the ``qe`` extra.

Exposed as ``cre qe-eval``.
"""

from __future__ import annotations

import argparse

from cre_router.qe.classifier import ACCEPT, ROUTE


def to_binary_label(decision_label) -> int:
    """Fold the 3-way training label into binary: 2 ("continue") -> 0 ("route").

    Accepts ints, int-like strings, and float-formatted strings ("0.0"), which
    is how the released datasets store the label.
    """
    value = int(float(decision_label))
    return ROUTE if value == 2 else value


def qe_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Classification metrics for accept(1)/route(0) predictions.

    Confusion counts are reported from the escalation point of view, where the
    positive class is "route" (escalate): a false positive is a correct answer
    escalated unnecessarily, a false negative is a wrong answer wrongly accepted.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    n = len(y_true)
    accept_true = sum(1 for y in y_true if y == ACCEPT)
    # labels=[ACCEPT, ROUTE] -> rows/cols ordered (accept, route)
    (tp_accept, fn_accept), (fp_accept, tn_accept) = confusion_matrix(
        y_true, y_pred, labels=[ACCEPT, ROUTE]
    )
    # "route" as the positive (escalate) class
    routed_correct = tn_accept  # true route: was route, predicted route
    false_escalations = fn_accept  # was accept, predicted route (unnecessary escalation)
    missed_escalations = fp_accept  # was route, predicted accept (wrong answer accepted)
    return {
        "n": n,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_route": f1_score(y_true, y_pred, pos_label=ROUTE, zero_division=0),
        "f1_accept": f1_score(y_true, y_pred, pos_label=ACCEPT, zero_division=0),
        "n_accept_true": accept_true,
        "escalations": routed_correct + false_escalations,
        "true_escalations": routed_correct,
        "false_escalations": false_escalations,
        "missed_escalations": missed_escalations,
    }


def evaluate_split(classifier, dataset_split, batch_size: int = 32) -> tuple[list[int], list[int]]:
    """Run the classifier over a dataset split, returning (y_true, y_pred).

    Each row needs ``question``, ``full_output``, ``num_tokens``, and
    ``decision_label``. Prediction is batched to bound memory.
    """
    y_true = [to_binary_label(row) for row in dataset_split["decision_label"]]
    questions = dataset_split["question"]
    outputs = dataset_split["full_output"]
    tokens = dataset_split["num_tokens"]

    y_pred: list[int] = []
    for start in range(0, len(y_true), batch_size):
        items = list(
            zip(
                questions[start : start + batch_size],
                outputs[start : start + batch_size],
                tokens[start : start + batch_size],
            )
        )
        for decision in classifier.predict_batch(items):
            y_pred.append(ACCEPT if decision.accept else ROUTE)
    return y_true, y_pred


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--classifier", required=True, help="trained QE checkpoint")
    parser.add_argument("--dataset", required=True, help="HF dataset of model outputs")
    parser.add_argument("--split", required=True, help="dataset split to evaluate")
    parser.add_argument("--base-tokenizer", default=None,
                        help="defaults to the checkpoint itself, which ships its own tokenizer; "
                             "give a base model id only for a checkpoint saved without one")
    parser.add_argument("--max-length", type=int, default=4096, help="4096 AIME, 512 TeleQnA")
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    from datasets import load_dataset

    from cre_router.qe import QEClassifier

    classifier = QEClassifier(
        model_name=args.classifier,
        base_tokenizer=args.base_tokenizer,
        accept_threshold=args.accept_threshold,
        max_length=args.max_length,
    )
    split = load_dataset(args.dataset, cache_dir=args.cache_dir)[args.split]
    y_true, y_pred = evaluate_split(classifier, split, batch_size=args.batch_size)
    m = qe_metrics(y_true, y_pred)

    print(f"\nQE evaluation: {args.dataset}:{args.split} ({m['n']} examples)")
    print(f"  accuracy   {m['accuracy']:.4f}")
    print(f"  f1_macro   {m['f1_macro']:.4f}  (route {m['f1_route']:.4f}, accept {m['f1_accept']:.4f})")
    print(f"  escalations {m['true_escalations']}/{m['escalations']} correct "
          f"({m['false_escalations']} unnecessary), {m['missed_escalations']} wrong answers accepted")


if __name__ == "__main__":
    main()
