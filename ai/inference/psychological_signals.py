import re
from typing import Any, Dict, Iterable, Mapping


SIGNAL_NAMES = (
    "mental_fatigue",
    "cognitive_overload",
    "restlessness",
    "emotional_conflict",
    "self_criticism",
    "social_withdrawal",
    "helplessness_language",
    "motivation_reduction",
)


TEXT_PATTERNS = {
    "mental_fatigue": (
        r"\btired\b", r"\bexhausted\b", r"\bdrained\b", r"\bfatigue(?:d)?\b",
        r"\bhead feels heavy\b", r"\bheart feels heavy\b", r"\bnot able to sleep\b",
        r"\bmentally disturbed\b",
    ),
    "cognitive_overload": (
        r"\boverthink(?:ing)?\b", r"\bthinking repeatedly\b", r"\bmind feels full\b",
        r"\bcannot concentrate\b", r"\bcannot focus\b", r"\bmind is not calm\b",
        r"\bpressure inside\b",
    ),
    "restlessness": (
        r"\brestless\b", r"\btense\b", r"\btension\b", r"\buneasy\b",
        r"\bnot calm\b", r"\bdisturbed\b",
    ),
    "emotional_conflict": (
        r"\bconfused\b", r"\bmixed feelings\b", r"\bone side\b", r"\bbut still\b",
        r"\bi don't know\b", r"\bdo not understand\b",
    ),
    "self_criticism": (
        r"\bmy fault\b", r"\bi am useless\b", r"\bi'm useless\b", r"\bnot good enough\b",
        r"\bi failed\b", r"\bi hate myself\b", r"\bblame myself\b",
    ),
    "social_withdrawal": (
        r"\balone\b", r"\blonely\b", r"\bavoid(?:ing)? people\b",
        r"\bnot feel like talking\b", r"\bdo not feel like talking\b",
        r"\bdo not feel like talking to anyone\b",
    ),
    "helplessness_language": (
        r"\bcannot handle it\b", r"\bcan't handle it\b", r"\bcannot do anything\b",
        r"\bnot able to do anything\b", r"\bhelpless\b", r"\bstuck\b",
        r"\bnothing works\b",
    ),
    "motivation_reduction": (
        r"\bno motivation\b", r"\bdo not feel like\b", r"\bdon't feel like\b",
        r"\bnot able to work\b", r"\bnot able to do anything\b",
        r"\bdo not feel like eating\b",
    ),
}


EMOTION_WEIGHTS = {
    "mental_fatigue": {"sadness": 0.18, "disappointment": 0.14, "neutral": 0.08, "grief": 0.12},
    "cognitive_overload": {"confusion": 0.24, "nervousness": 0.18, "fear": 0.12, "surprise": 0.06},
    "restlessness": {"nervousness": 0.28, "fear": 0.16, "annoyance": 0.10, "anger": 0.08},
    "emotional_conflict": {"confusion": 0.22, "realization": 0.10, "sadness": 0.08, "remorse": 0.08},
    "self_criticism": {"remorse": 0.24, "embarrassment": 0.18, "disapproval": 0.10, "sadness": 0.08},
    "social_withdrawal": {"sadness": 0.18, "grief": 0.12, "neutral": 0.08, "disappointment": 0.08},
    "helplessness_language": {"sadness": 0.20, "disappointment": 0.16, "fear": 0.10, "grief": 0.08},
    "motivation_reduction": {"sadness": 0.18, "disappointment": 0.16, "neutral": 0.10, "grief": 0.08},
}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _count_matches(text: str, patterns: Iterable[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def extract_text_signal_scores(*texts: str) -> Dict[str, float]:
    """
    Score descriptive psychological language from original and normalized text only.
    Scores are interpretation aids, not diagnostic labels or severity estimates.
    """
    combined_text = " ".join(text for text in texts if text).lower()
    scores: Dict[str, float] = {}

    for signal_name, patterns in TEXT_PATTERNS.items():
        matches = _count_matches(combined_text, patterns)
        scores[signal_name] = _clamp(matches * 0.28)

    return scores


def merge_emotion_signal_scores(
    text_signal_scores: Mapping[str, float],
    emotion_scores: Mapping[str, float],
) -> Dict[str, float]:
    """
    Blend text-derived indicators with GoEmotions scores using fixed transparent weights.
    """
    merged: Dict[str, float] = {}
    for signal_name in SIGNAL_NAMES:
        text_score = float(text_signal_scores.get(signal_name, 0.0))
        emotion_boost = 0.0
        for emotion, weight in EMOTION_WEIGHTS.get(signal_name, {}).items():
            emotion_boost += float(emotion_scores.get(emotion, 0.0)) * weight

        merged[signal_name] = round(_clamp(text_score + emotion_boost), 4)

    return merged


def extract_psychological_signals(
    text: str,
    emotion_scores: Mapping[str, float],
    processed_text: str = "",
    translated_text: str = "",
) -> Dict[str, float]:
    text_scores = extract_text_signal_scores(text, processed_text, translated_text)
    return merge_emotion_signal_scores(text_scores, emotion_scores)


class PsychologicalSignalExtractor:
    """
    Rule-based, non-diagnostic signal extractor for Qwen prompt enrichment.
    """

    def extract_text_scores(self, *texts: str) -> Dict[str, float]:
        return extract_text_signal_scores(*texts)

    def merge_with_emotions(
        self,
        text_signal_scores: Mapping[str, float],
        emotion_scores: Mapping[str, float],
    ) -> Dict[str, float]:
        return merge_emotion_signal_scores(text_signal_scores, emotion_scores)

    def extract(self, results: Dict[str, Any]) -> Dict[str, float]:
        return extract_psychological_signals(
            results.get("original_text", ""),
            results.get("emotion_scores", {}),
            results.get("processed_text", ""),
            results.get("translated_text", ""),
        )
