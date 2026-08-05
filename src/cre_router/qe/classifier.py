"""Binary accept/escalate quality-estimation classifier (paper Sec. 5.1).

A fine-tuned ModernBERT sequence classifier inspects an efficient model's
completion and decides whether to accept it or escalate the query to a
stronger model. Requires the ``qe`` extra (torch + transformers).

Labels: 0 = escalate ("Route"), 1 = accept.
"""

from __future__ import annotations

from dataclasses import dataclass

from cre_router.textutils import split_thinking

ROUTE, ACCEPT = 0, 1
CLASS_NAMES = ("Route", "Accept")

# The completion is truncated to its last N words: the tail of a long
# chain-of-thought carries the answer and the model's final confidence.
MAX_OUTPUT_WORDS = 1000


def format_qe_input(
    question: str, output: str, num_tokens: int, sep_token: str, max_output_words: int = MAX_OUTPUT_WORDS
) -> str:
    """Training/inference input format: ``query [SEP] output [SEP] num_tokens``.

    The reasoning block is dropped first (``split_thinking``, all model families)
    so the classifier sees the delivered answer, not the chain of thought -- the
    thinking is not part of the output. ``num_tokens`` still reflects the *full*
    generation length (the real serving cost), giving the classifier direct
    access to output length as a proxy for model confidence. The output is then
    truncated to its last ``max_output_words`` words.
    """
    if num_tokens is None:
        raise ValueError(
            "num_tokens is None: the generation length is required for the QE input "
            "(a generations file written without output_lens). Build the dataset via "
            "prep_qe, which drops such rows, or re-run the benchmark so output_lens is present."
        )
    output = split_thinking(output)
    words = output.split()
    if len(words) > max_output_words:
        output = " ".join(words[-max_output_words:])
    return f"{question} {sep_token} {output} {sep_token} {num_tokens}"


@dataclass
class QEDecision:
    accept: bool
    p_accept: float
    p_route: float


class QEClassifier:
    """Wraps a fine-tuned binary QE checkpoint for cascade decisions.

    ``accept_threshold`` guards low-confidence accepts: even when the argmax
    is Accept, the query is escalated unless P(accept) >= threshold.
    """

    def __init__(
        self,
        model_name: str,
        base_tokenizer: str | None = None,
        accept_threshold: float = 0.5,
        max_length: int = 4096,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.accept_threshold = accept_threshold
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer or model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            dtype=getattr(torch, dtype) if self.device != "cpu" else torch.float32,
        )
        self.model.eval().to(self.device)

    def predict(self, question: str, output: str, num_tokens: int) -> QEDecision:
        return self.predict_batch([(question, output, num_tokens)])[0]

    def predict_batch(self, items: list[tuple[str, str, int]]) -> list[QEDecision]:
        torch = self._torch
        texts = [
            format_qe_input(q, o, n, sep_token=self.tokenizer.sep_token) for q, o, n in items
        ]
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        decisions = []
        for p in probs:
            accept = bool(p.argmax() == ACCEPT and p[ACCEPT] >= self.accept_threshold)
            decisions.append(QEDecision(accept=accept, p_accept=float(p[ACCEPT]), p_route=float(p[ROUTE])))
        return decisions
