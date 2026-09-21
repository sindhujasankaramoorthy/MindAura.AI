"""
Decides whether "Tanglish fallback" applies: Tamil expressed in Latin
characters rather than Tamil script. Tanglish is NOT a language of its own
(see module docstring in rules.py) -- this just answers "was this fallback
mechanism used for this input", reusing the per-token language
classification the pipeline already computes rather than re-detecting.
"""
from typing import Iterable, Tuple


def is_tanglish_fallback(token_classifications: Iterable[Tuple[str, str]]) -> bool:
    """
    `token_classifications` is the (token, language_label) pairs already
    produced by ai.text.language.detector.LanguageDetector.detect() --
    True if any token was classified as "Tanglish".
    """
    return any(lang == "Tanglish" for _token, lang in token_classifications)
