#!/usr/bin/env python
"""Pre-render Gemma 4 chat-templated prompts with enable_thinking baked in.

Gemma 4's thinking switch is a chat-template kwarg (``enable_thinking``), not
a text-level prefix like Qwen3's ``/no_think``. Its own chat_template.jinja
(google/gemma-4-E2B-it) injects a ``<|think|>`` token at
the top of the system turn only when ``enable_thinking`` is true; the model
then opens its reply with ``<|channel>thought\\n...\\n<channel|>`` before the
final answer. The model card confirms the same mechanism and notes that the
E2B/E4B variants, unlike their larger siblings, emit no channel markers at
all when thinking is disabled.

vLLM's own ``vllm bench serve`` applies the chat template itself before
posting to ``/v1/completions`` (its ``CustomDataset.sample`` calls
``tokenizer.apply_chat_template`` with a fixed set of keyword arguments), and
that call never forwards a template kwarg such as ``enable_thinking``
(verified by reading ``vllm/benchmarks/datasets.py``). So the switch has to
be baked into the prompt text at prep time, here, with each row's fully
rendered text stored as ``prompt``; the ``telemath_gemma4`` task entry in
``evaluate.py`` sets ``pre_rendered=True`` so the benchmark passes
``--skip-chat-template`` and serves the text verbatim.

Usage:

    python data/prep_gemma4_thinking.py --in data/telemath_test.jsonl \\
        --out data/telemath_test_gemma --model google/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_rows(rows: list[dict], tokenizer, enable_thinking: bool) -> list[dict]:
    """Render each row's ``prompt`` through the model's own chat template.

    The raw question is preserved under ``question`` before ``prompt`` is
    overwritten with the templated text, so the QE dataset built from these
    runs' generations carries the plain query, not chat-template markers.
    """
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        rendered.append({**row, "question": row.get("question", row["prompt"]), "prompt": text})
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--in", dest="in_path", required=True,
                        help="an existing clustered prompts JSONL")
    parser.add_argument("--out", required=True,
                        help="path prefix; _think.jsonl and _nothink.jsonl are appended")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole file")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = [
        json.loads(line)
        for line in Path(args.in_path).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for enable_thinking, suffix in ((True, "_think"), (False, "_nothink")):
        rendered = render_rows(rows, tokenizer, enable_thinking)
        path = out.with_name(f"{out.name}{suffix}.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for row in rendered:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rendered)} rows to {path} (enable_thinking={enable_thinking})")


if __name__ == "__main__":
    main()
