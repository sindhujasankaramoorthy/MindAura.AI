"""
Shared thresholds for the Tanglish/English lexical classification logic
scattered across word_classifier.py, tanglish_patterns.py, and
language_boundary.py. Intentionally dependency-light (only wordfreq/rapidfuzz)
so it can be imported from any of those modules without creating cycles.

The three Zipf thresholds below gate three different decisions and are
deliberately NOT collapsed into one value:
- LOOSE   (word_classifier.py):    is this word English at all, for the
                                    purposes of a single-token classification?
- STANDARD (language_boundary.py): is this word English enough to count
                                    toward routing a whole sentence to the
                                    English pipeline?
- STRICT  (tanglish_patterns.py):  is this word unambiguously English enough
                                    to skip fuzzy-matching it against Tanglish
                                    vocabulary at all?
"""
from typing import Iterable, Optional, Tuple

from rapidfuzz import fuzz, process

ZIPF_ENGLISH_THRESHOLD_LOOSE = 2.5
ZIPF_ENGLISH_THRESHOLD_STANDARD = 3.0
ZIPF_ENGLISH_THRESHOLD_STRICT = 3.8

FUZZY_MATCH_THRESHOLD = 85.0


def fuzzy_match_tanglish(
    word: str,
    vocabulary: Iterable[str],
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> Optional[str]:
    """
    Best-effort fuzzy match of `word` against `vocabulary` using rapidfuzz's
    ratio scorer. Returns the matched vocabulary entry, or None if nothing
    clears `threshold`.
    """
    match: Optional[Tuple[str, float, object]] = process.extractOne(
        word, vocabulary, scorer=fuzz.ratio
    )
    if match and match[1] >= threshold:
        return match[0]
    return None
