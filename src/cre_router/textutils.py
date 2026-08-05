"""Small, dependency-free text helpers shared across the pipeline."""

from __future__ import annotations


def split_thinking(text: str) -> str:
    """Return the post-reasoning content, dropping a leading reasoning block.

    Two marker styles are in use across the pool: Qwen's ``<think>...</think>``
    and Gemma 4's ``<|channel>thought\\n...\\n<channel|>`` (confirmed against
    google/gemma-4-E2B-it's own chat_template.jinja and model card).
    Gemma's E2B/E4B variants emit no channel markers at all when thinking is
    disabled, unlike its larger siblings and unlike Qwen, which always emits an
    empty <think></think>; checking for both end markers handles every case
    without needing to know which family produced the text. Idempotent: text with
    no reasoning marker is returned unchanged (stripped), so it is safe to apply
    to outputs that were already de-thought (e.g. the released router datasets).
    """
    for end_marker in ("</think>", "<channel|>"):
        end = text.find(end_marker)
        if end != -1:
            return text[end + len(end_marker) :].strip()
    return text.strip()
