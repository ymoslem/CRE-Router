"""Cross-validated epoch choice for `cre qe-train --cv-folds`."""
import pytest

from cre_router.qe.train import build_parser, choose_epoch, group_folds


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
