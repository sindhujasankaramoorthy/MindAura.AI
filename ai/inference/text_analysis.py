"""
Text Module — pipeline-friendly entry point for text/journal analysis.

This is a thin, stable wrapper around the existing preprocessing +
EmotionAnalyzer stack (ai/preprocessing/text_normalizer.py,
ai/inference/emotion_predict.py, ai/inference/psychological_signals.py).
It does not reimplement any of that logic — it only shapes the result into
a clean, integration-friendly JSON contract and adds input validation /
error handling, so FastAPI, the psychiatrist dashboard, or a future
fusion layer (text + voice + face) can call one function and always get a
predictable structure back, success or failure.

Usage:
    from ai.inference.text_analysis import analyze_text
    result = analyze_text("I feel really anxious about tomorrow")

CLI (manual smoke test):
    python -m ai.inference.text_analysis "some text"
"""

import json
import logging
import sys
from typing import Any, Dict, Optional

from ai.inference.emotion_predict import EmotionAnalyzer

logger = logging.getLogger(__name__)

# ── Config / constants (avoid magic numbers scattered through the module) ──
MODULE_NAME = "text"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
TOP_EMOTIONS_COUNT = 5

_analyzer: Optional[EmotionAnalyzer] = None


def _get_analyzer() -> EmotionAnalyzer:
    """Lazy singleton so TextNormalizer/EmotionAnalyzer's model loading only
    happens once per process, no matter how many times analyze_text() is called."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EmotionAnalyzer()
    return _analyzer


def _error_response(text: Any, error_type: str, message: str) -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "status": STATUS_ERROR,
        "input": {"text": text if isinstance(text, str) else str(text)},
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _build_success_response(raw_text: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shapes EmotionAnalyzer.process()'s full internal result into the stable,
    downstream-facing contract. Deliberately drops:
      - preprocessing_metadata (large per-token debug array, internal-only)
      - corrected_sentence (byte-for-byte duplicate of translated_text)
      - visualization_data (bar/radar chart data that's just top_emotions
        reshaped twice — a dashboard can derive it from top_emotions itself)
    """
    emotion_scores = raw["emotion_scores"]
    dominant_emotion = raw["dominant_emotion"]

    return {
        "module": MODULE_NAME,
        "status": STATUS_SUCCESS,
        "input": {
            "text": raw_text,
        },
        "analysis": {
            "language": raw["original_language"],
            "normalized_text": raw["processed_text"],
            "translated_text": raw["translated_text"],
            "emotion": {
                "label": dominant_emotion,
                "confidence": round(float(emotion_scores.get(dominant_emotion, 0.0)), 4),
                "narrative": raw["dominant_narrative"],
            },
            "top_emotions": raw["top_emotions"],
            "metrics": {
                "intensity": raw["emotional_intensity"],
                "diversity": raw["emotional_diversity"],
                "complexity": raw["emotional_complexity"],
            },
            "psychological_signals": raw["psychological_signals"],
        },
    }


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze a single piece of text (journal entry, blog, message) and return
    a structured, JSON-serializable dict. Never raises for expected failure
    modes (empty/invalid input, model errors) -- always returns a dict with
    a "status" field so callers (FastAPI, dashboard, tests) can branch on it
    instead of wrapping every call in try/except.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        logger.warning("analyze_text called with empty/invalid input.")
        return _error_response(text, "EmptyInputError", "Input text is empty or invalid.")

    try:
        analyzer = _get_analyzer()
        raw = analyzer.process(text)
    except Exception as e:
        logger.exception("Text analysis failed.")
        return _error_response(text, type(e).__name__, str(e))

    try:
        return _build_success_response(text, raw)
    except (KeyError, TypeError) as e:
        # The underlying pipeline returned something we didn't expect the
        # shape of -- surface it as a clean error rather than crashing the
        # caller or silently returning a half-built result.
        logger.exception("Unexpected result shape from EmotionAnalyzer.process().")
        return _error_response(text, "UnexpectedModelOutput", str(e))


if __name__ == "__main__":
    # force=True: emotion_predict.py's import already configured the root
    # logger at INFO, and basicConfig() only takes effect on its first call
    # per-process -- force it so the CLI summary stays concise.
    logging.basicConfig(level=logging.WARNING, force=True)

    input_text = " ".join(sys.argv[1:]).strip()
    if not input_text:
        input_text = input("Enter text to analyze: ").strip()

    result = analyze_text(input_text)

    if result["status"] == STATUS_SUCCESS:
        emotion = result["analysis"]["emotion"]
        print(f"Text analysis completed")
        print(f"Emotion: {emotion['label']} (confidence: {emotion['confidence']})")
    else:
        print(f"Text analysis failed: {result['error']['type']} - {result['error']['message']}")

    print("\nFull JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
