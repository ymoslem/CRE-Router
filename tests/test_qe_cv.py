"""Cross-validated epoch choice for `cre qe-train --cv-folds`."""
import pytest

from cre_router.qe.train import build_parser, choose_epoch, group_folds, out_of_fold_rows


def test_every_answer_to_a_question_shares_a_fold():
    groups = [f"q{i // 5}" for i in range(5 * 23)]  # 23 questions, five runs each
    fold = group_folds(groups, 5, seed=42)
    for q in set(groups):
        assert len({f for g, f in zip(groups, fold) if g == q}) == 1


def test_folds_are_balanced_in_groups_and_all_used():
    groups = [f"q{i // 5}" for i in range(5 * 23)]
    fold = group_folds(groups, 5, seed=42)
    per_fold = [len({g for g, f in zip(groups, fold) if f == k}) for k in range(5)]
    assert sorted(per_fold) == [4, 4, 5, 5, 5]


def test_folds_depend_on_the_seed_only():
    groups = [f"q{i // 5}" for i in range(50)]
    assert group_folds(groups, 3, seed=1) == group_folds(list(groups), 3, seed=1)
    assert group_folds(groups, 3, seed=1) != group_folds(groups, 3, seed=2)


def test_too_few_groups_or_folds_is_refused():
    with pytest.raises(ValueError):
        group_folds(["a", "a", "b"], 3, seed=0)
    with pytest.raises(ValueError):
        group_folds(["a", "b"], 1, seed=0)


def test_best_mean_epoch_is_chosen_and_ties_go_earlier():
    epoch, means = choose_epoch([[0.5, 0.7, 0.6], [0.6, 0.6, 0.7]])
    assert epoch == 2 and means == pytest.approx([0.55, 0.65, 0.65])


def test_folds_must_report_every_epoch():
    with pytest.raises(ValueError):
        choose_epoch([[0.5, 0.6], [0.5]])


def test_cv_flags_parse_and_default_off():
    args = build_parser().parse_args(["--dataset", "d", "--output-dir", "o"])
    assert args.cv_folds == 0 and args.group_column == "question"
    args = build_parser().parse_args(["--dataset", "d", "--output-dir", "o", "--cv-folds", "5"])
    assert args.cv_folds == 5


def test_grad_accum_defaults_to_one():
    args = build_parser().parse_args(["--dataset", "d", "--output-dir", "o"])
    assert args.grad_accum == 1 and args.batch_size == 64
    args = build_parser().parse_args(["--dataset", "d", "--output-dir", "o", "--grad-accum", "4"])
    assert args.grad_accum == 4


def test_out_of_fold_rows_take_the_chosen_epoch_for_each_held_out_answer():
    raw = [{"qid": f"q{i // 2}", "run": i % 2, "num_tokens": 10 + i, "cluster": "0"} for i in range(4)]
    rows = out_of_fold_rows(raw, [1, 3], [[0.1, 0.2], [0.8, 0.9]], epoch=2, fold=1)
    assert [(r["qid"], r["run"], r["p_accept"], r["num_tokens"]) for r in rows] == \
        [("q0", 1, 0.8, 11), ("q1", 1, 0.9, 13)]
    assert all(r["fold"] == 1 and r["epoch"] == 2 for r in rows)


def test_out_of_fold_rows_refuse_mismatched_or_unkeyed_rows():
    raw = [{"qid": "a", "run": 0, "num_tokens": 3}]
    with pytest.raises(ValueError, match="scores for"):
        out_of_fold_rows(raw, [0], [[0.1, 0.2]], epoch=1, fold=0)
    with pytest.raises(ValueError, match="no run"):
        out_of_fold_rows([{"qid": "a", "num_tokens": 3}], [0], [[0.5]], epoch=1, fold=0)
