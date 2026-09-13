"""
Thin service layer wrapping ai.inference.emotion_predict.EmotionAnalyzer so
the FastAPI layer doesn't need to know about the ai/ package's internals.
"""
import logging
from typing import Any, Dict

from ai.inference.emotion_predict import EmotionAnalyzer
from ai.inference.qwen_reasoning import QwenReasoning

logger = logging.getLogger(__name__)

# Module-level lazy singleton: EmotionAnalyzer itself defers real model
# loading to first use (via ai.model_registry), so constructing it here is
# cheap — the first /journals/analyze request pays the real model-load cost,
# every request after that reuses the already-loaded models.
_analyzer: EmotionAnalyzer = None


def _get_analyzer() -> EmotionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EmotionAnalyzer()
    return _analyzer


def analyze(text: str, include_interpretation: bool = False) -> Dict[str, Any]:
    """
    Run the full emotion-analysis pipeline on a journal entry.

    include_interpretation=True additionally calls QwenReasoning, which
    requires a local Ollama instance running qwen3:14b and is slow — off by
    default so the basic analysis endpoint stays fast and doesn't hard-fail
    when Ollama isn't running.
    """
    analyzer = _get_analyzer()
    results = analyzer.process(text)

    response = {
        "dominant_emotion": results["dominant_emotion"],
        "emotion_scores": results["emotion_scores"],
        "psychological_signals": results["psychological_signals"],
    }

    if include_interpretation:
        try:
            reasoner = QwenReasoning()
            response["interpretation"] = reasoner.interpret(results)
        except Exception as e:
            logger.error(f"Qwen interpretation failed: {e}")
            response["interpretation"] = None

    return response
