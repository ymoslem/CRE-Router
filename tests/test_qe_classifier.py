"""format_qe_input: think-stripping (all model families) + last-N-words
truncation. Pure string formatting, no torch/transformers needed."""

from cre_router.qe.classifier import format_qe_input


def test_strips_qwen_thinking():
    out = format_qe_input("q", "<think>long private reasoning</think>Final: 42", 123, "[SEP]")
    assert out == "q [SEP] Final: 42 [SEP] 123"
    assert "reasoning" not in out


def test_strips_gemma_channel():
    out = format_qe_input("q", "<|channel>thought\nscratch work<channel|>Answer 7", 5, "[SEP]")
    assert "thought" not in out and "scratch" not in out
    assert out == "q [SEP] Answer 7 [SEP] 5"


def test_idempotent_on_plain_output():
    # no reasoning marker -> unchanged (safe on already-de-thought data)
    out = format_qe_input("q", "just the answer", 9, "[SEP]")
    assert out == "q [SEP] just the answer [SEP] 9"


def test_num_tokens_is_the_full_length_not_the_stripped_text():
    # num_tokens reflects the whole generation (cost), even though the thinking
    # text is dropped from the classifier's view.
    out = format_qe_input("q", "<think>" + "x " * 50 + "</think>done", 999, "[SEP]")
    assert out.endswith("[SEP] 999")
    assert "x" not in out.split(" [SEP] ")[1]


def test_truncation_applies_after_stripping():
    answer = " ".join(str(i) for i in range(2000))  # 2000 words after the think block
    out = format_qe_input(
        "q", "<think>" + ("noise " * 5000) + "</think>" + answer, 42, "[SEP]", max_output_words=1000
    )
    body = out.split(" [SEP] ")[1]
    assert "noise" not in body            # reasoning gone
    assert len(body.split()) == 1000      # kept the last 1000 words
    assert body.split()[-1] == "1999"     # of the answer, not the thinking


def test_none_num_tokens_raises():
    import pytest
    with pytest.raises(ValueError, match="num_tokens is None"):
        format_qe_input("q", "answer", None, "[SEP]")
