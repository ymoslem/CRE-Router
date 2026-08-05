"""render_rows preserves the raw question when baking the chat template."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "prep_gemma4", Path(__file__).parent.parent / "data" / "prep_gemma4_thinking.py"
)
prep_gemma4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_gemma4)


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return f"<bos><start>{messages[0]['content']}<gen think={enable_thinking}>"


def test_render_preserves_raw_question_and_other_fields():
    rows = [{"id": "a", "prompt": "what is 2+2?", "answer": 4, "category": "math"}]
    out = prep_gemma4.render_rows(rows, _FakeTokenizer(), enable_thinking=True)
    assert out[0]["question"] == "what is 2+2?"                       # raw preserved
    assert out[0]["prompt"] == "<bos><start>what is 2+2?<gen think=True>"  # templated
    assert out[0]["answer"] == 4 and out[0]["category"] == "math"     # untouched


def test_render_keeps_explicit_question_if_present():
    rows = [{"prompt": "already raw", "question": "the real question", "answer": 1}]
    out = prep_gemma4.render_rows(rows, _FakeTokenizer(), enable_thinking=False)
    assert out[0]["question"] == "the real question"
