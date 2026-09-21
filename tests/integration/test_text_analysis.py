"""
Focused integration coverage for the (now archived) RoBERTa-based text
analyzer's public interface, analyze_text() -- moved to
temporary/models/roberta/text_analysis.py during the text-pipeline
restructuring (it's not part of the active ai/text/ pipeline, which does
not do emotion classification -- see ai/text/pipeline.py instead for the
active, currently-supported text entry point).

Kept here as coverage for the archived module in case it's reconnected in
a future phase; skipped entirely if it can't be imported.
"""
import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

_ARCHIVED_ROBERTA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "temporary", "models", "roberta"
)
sys.path.insert(0, _ARCHIVED_ROBERTA_DIR)

try:
    from text_analysis import analyze_text
except ImportError:
    pytest.skip(
        "Archived text_analysis module (temporary/models/roberta/) not importable.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.slow

REQUIRED_ANALYSIS_KEYS = {
    "language",
    "normalized_text",
    "translated_text",
    "emotion",
    "top_emotions",
    "metrics",
    "psychological_signals",
}


def test_english_text_returns_success_contract():
    result = analyze_text("I feel really anxious about my exams tomorrow, I can't sleep")

    assert result["module"] == "text"
    assert result["status"] == "success"
    assert result["input"]["text"]

    analysis = result["analysis"]
    assert REQUIRED_ANALYSIS_KEYS.issubset(analysis.keys())
    assert isinstance(analysis["emotion"]["label"], str) and analysis["emotion"]["label"]
    assert 0.0 <= analysis["emotion"]["confidence"] <= 1.0
    assert len(analysis["top_emotions"]) > 0
    for metric in ("intensity", "diversity", "complexity"):
        assert 0 <= analysis["metrics"][metric] <= 100

    # Internal-only fields must not leak into the public contract.
    assert "preprocessing_metadata" not in analysis
    assert "corrected_sentence" not in analysis
    assert "visualization_data" not in analysis


def test_realistic_patient_journal_entry():
    text = (
        "I have been feeling really low lately. I cant focus on anything and "
        "I feel like a burden to everyone. Nothing feels worth it anymore."
    )
    result = analyze_text(text)

    assert result["status"] == "success"
    analysis = result["analysis"]
    assert analysis["emotion"]["label"]
    # A visibly distressed entry should register on at least one signal.
    assert any(v > 0 for v in analysis["psychological_signals"].values())


def test_tanglish_text_does_not_crash_and_returns_contract():
    result = analyze_text("romba kavalaya iruku, enaku onnume purila")

    assert result["status"] == "success"
    assert REQUIRED_ANALYSIS_KEYS.issubset(result["analysis"].keys())


@pytest.mark.parametrize("bad_input", ["", "   ", None, 12345])
def test_invalid_input_returns_clean_error_contract(bad_input):
    result = analyze_text(bad_input)

    assert result["module"] == "text"
    assert result["status"] == "error"
    assert "type" in result["error"] and "message" in result["error"]
    assert "analysis" not in result


def test_result_is_json_serializable():
    import json

    result = analyze_text("This is a simple happy sentence, I feel great today.")
    serialized = json.dumps(result)
    assert json.loads(serialized)["status"] == "success"
