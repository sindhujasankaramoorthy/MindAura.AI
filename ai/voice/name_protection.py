"""
Indian-name protection for the voice pipeline -- a dedicated stage, kept
separate from general text correction/translation on purpose (per the
project's own requirement) so a PERSON entity can never be silently
mangled by spelling correction, Tanglish normalization, or translation.

Reuses existing infrastructure rather than building new NER or a new
fuzzy-matching library:
  - PERSON detection: ai.text.entities.protector.NERProtection (GLiNER +
    BERT-NER fallback), already used by the text-journal pipeline.
  - Indian-name reference set: ai.text.entities.name_dictionary.COMMON_NAMES.
  - Fuzzy/phonetic similarity: rapidfuzz (already an installed dependency).

This stage only ever *proposes* a correction when the ASR output is a
high-confidence phonetic near-miss of a known Indian name. It never
guesses a name from nothing, and a low-confidence candidate is left
exactly as the ASR produced it -- per the "never blindly autocorrect a
detected PERSON entity" requirement.
"""
import re
from typing import Any, Dict, List, Tuple

from rapidfuzz import fuzz

from ai.text.entities.protector import NERProtection
from ai.text.entities.name_dictionary import COMMON_NAMES

_ner = NERProtection()

# Small, explicit substitution table for common Indian-name transliteration
# variance (e.g. "Sureesh"/"Suresh", "Preetha"/"Pritha"). Deliberately not a
# general English phonetic algorithm (Soundex/Metaphone) -- those model
# English sound patterns, not Indian-language ones -- and deliberately not
# a new dependency; this is a handful of substitution rules applied before
# the fuzzy-ratio comparison below.
_PHONETIC_SUBS: List[Tuple[str, str]] = [
    (r"ee", "i"), (r"oo", "u"), (r"sh", "s"), (r"ph", "f"),
    (r"kh", "k"), (r"th", "t"), (r"dh", "d"), (r"bh", "b"),
    (r"gh", "g"), (r"v", "w"), (r"([a-z])\1+", r"\1"),
]

# Below this, a candidate isn't a name correction at all -- just noise.
# Kept high deliberately: it's safer to leave a genuine ASR near-miss
# uncorrected than to "fix" an unrelated real word/name into the wrong
# one (see ai/voice/name_protection.py's test suite for the "Kumar" ->
# "Kumaran" false-positive this threshold was tuned against).
_CORRECTION_THRESHOLD = 85
_INITIAL_RE = re.compile(r"^[A-Za-z]\.?$")


def _phonetic_key(word: str) -> str:
    key = word.lower()
    for pattern, repl in _PHONETIC_SUBS:
        key = re.sub(pattern, repl, key)
    return key


def _best_dictionary_match(word: str) -> Tuple[str, float]:
    """Best (name, score 0-100) match for `word` against COMMON_NAMES,
    taking the better of plain and phonetic-key similarity."""
    target = word.lower()
    target_key = _phonetic_key(word)
    best_name, best_score = "", 0.0
    for name in COMMON_NAMES:
        score = max(fuzz.ratio(target, name), fuzz.ratio(target_key, _phonetic_key(name)))
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def _correct_token(word: str) -> Dict[str, Any]:
    """Corrects a single name token. Initials are passed through untouched
    -- there is nothing meaningful to fuzzy-match an initial against."""
    if _INITIAL_RE.match(word):
        return {"original": word, "normalized": word, "confidence": 1.0, "correction_applied": False}

    if word.lower() in COMMON_NAMES:
        return {"original": word, "normalized": word.title(), "confidence": 1.0, "correction_applied": False}

    best_name, score = _best_dictionary_match(word)
    if best_name and score >= _CORRECTION_THRESHOLD:
        return {
            "original": word,
            "normalized": best_name.title(),
            "confidence": round(score / 100.0, 2),
            "correction_applied": True,
        }

    # Below threshold: keep the ASR's own text rather than guess.
    return {
        "original": word,
        "normalized": word,
        "confidence": round(score / 100.0, 2) if best_name else 0.0,
        "correction_applied": False,
    }


def protect_names(text: str) -> Dict[str, Any]:
    """
    Detects PERSON entities in `text` and phonetically checks each one
    against the Indian-name reference set. Multi-word names (e.g. "Suresh
    Kumar") are corrected token-by-token and rejoined, so one wrong token
    doesn't block the rest of the name from matching.

    Returns:
      {
        "entities": [
          {"original": str, "entity_type": "PERSON", "normalized": str,
           "confidence": float, "correction_applied": bool},
          ...
        ],
        "corrected_text": str,  # `text` with only high-confidence
                                 # corrections substituted in; everything
                                 # else (including low-confidence PERSON
                                 # spans) is left exactly as given.
      }
    """
    entities = _ner.detect_entities(text)
    person_spans = [(s, e, w) for s, e, t, w in entities if t == "PERSON"]

    results: List[Dict[str, Any]] = []
    replacements: List[Tuple[int, int, str]] = []

    for start, end, phrase in person_spans:
        tokens = phrase.split()
        corrected_tokens = []
        confidences = []
        applied = False
        for tok in tokens:
            c = _correct_token(tok)
            corrected_tokens.append(c["normalized"])
            confidences.append(c["confidence"])
            applied = applied or c["correction_applied"]

        normalized_phrase = " ".join(corrected_tokens) if applied else phrase
        confidence = round(min(confidences), 2) if confidences else 0.0

        results.append({
            "original": phrase,
            "entity_type": "PERSON",
            "normalized": normalized_phrase,
            "confidence": confidence,
            "correction_applied": applied,
        })
        if applied:
            replacements.append((start, end, normalized_phrase))

    corrected_text = text
    for start, end, replacement in sorted(replacements, key=lambda r: r[0], reverse=True):
        corrected_text = corrected_text[:start] + replacement + corrected_text[end:]

    return {"entities": results, "corrected_text": corrected_text}
