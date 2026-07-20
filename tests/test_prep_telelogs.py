"""Field mapping and label normalization for the TeleLogs prep script.

The prep script lives in data/ (not the installed package), so it is loaded by
path. No network or HF auth is needed; these test the pure conversion.
"""

import importlib.util
from pathlib import Path

import pytest

_PREP = Path(__file__).resolve().parents[1] / "data" / "prep_telelogs.py"
_spec = importlib.util.spec_from_file_location("prep_telelogs", _PREP)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


class TestParseLabel:
    def test_c_prefixed(self):
        assert prep.parse_label("C3") == 3

    def test_bare_integer(self):
        assert prep.parse_label("7") == 7

    def test_strips_whitespace(self):
        assert prep.parse_label("  C1 ") == 1

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            prep.parse_label("C9")

    def test_rejects_unparseable(self):
        with pytest.raises(ValueError, match="cannot parse"):
            prep.parse_label("unknown")


class TestConvert:
    def test_maps_question_to_prompt_and_label_to_int(self):
        rows = [
            {"question": "Analyze ... \\boxed{}", "answer": "C1"},
            {"question": "Another RCA prompt", "answer": "C8"},
        ]
        out = prep.convert(rows)
        assert out == [
            {"prompt": "Analyze ... \\boxed{}", "answer": 1},
            {"prompt": "Another RCA prompt", "answer": 8},
        ]
