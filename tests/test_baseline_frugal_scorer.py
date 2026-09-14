"""FrugalGPT scorer baseline: the input it sees and the settings it inherits."""
import pytest

from cre_router.baselines.frugal_scorer import (
    DEFAULT_BASE_MODEL,
    FRUGALGPT_BASE_MODEL,
    ScorerConfig,
    scorer_input,
)


def test_the_scorer_sees_the_question_and_the_answer_only():
    """Not the token count. Our own estimator uses it; the baseline must not."""
    text = scorer_input("What is 2+2?", "The answer is 4.")
    assert text == "What is 2+2? The answer is 4."
    assert "num_tokens" not in text


def test_training_settings_are_the_ones_in_their_TrainingArguments():
    cfg = ScorerConfig()
    assert (cfg.epochs, cfg.batch_size, cfg.warmup_steps) == (8, 8, 500)
    assert (cfg.weight_decay, cfg.max_length, cfg.seed) == (0.01, 512, 2024)


def test_their_unusual_validation_split_is_kept_not_corrected():
    """test_size=0.55 in their code: 45% trains, 55% validates."""
    assert ScorerConfig().val_fraction == 0.55


def test_the_backbone_defaults_to_ours_so_the_comparison_is_of_methods():
    cfg = ScorerConfig()
    assert cfg.base_model == DEFAULT_BASE_MODEL
    assert cfg.base_model != FRUGALGPT_BASE_MODEL
    assert ScorerConfig(base_model=FRUGALGPT_BASE_MODEL).base_model == "distilbert-base-uncased"


def test_labels_must_be_binary():
    from cre_router.baselines.frugal_scorer import train_scorer
    with pytest.raises(ValueError, match="two-class"):
        train_scorer(["a", "b"], [0, 2], out_dir="/tmp/never-written")


def test_mismatched_lengths_raise_before_any_training_starts():
    from cre_router.baselines.frugal_scorer import train_scorer
    with pytest.raises(ValueError, match="against"):
        train_scorer(["a", "b", "c"], [1, 0], out_dir="/tmp/never-written")
