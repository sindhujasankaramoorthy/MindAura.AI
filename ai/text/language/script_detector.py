"""
Unicode-script detection: what SCRIPT is this text written in, independent
of what LANGUAGE it is. This is what lets the pipeline tell the difference
between "Tamil written in Tamil script" and "Tamil written in Latin
characters" (Tanglish) -- the same distinction applies to Hindi/Devanagari
vs. romanized Hindi, etc.

Deliberately a general Unicode-block classifier, not a hardcoded word list.
"""
from typing import Dict

# Unicode block -> (script name, ISO language code most associated with it
# in this project's supported-language set). Order matters for scripts that
# share overlapping punctuation ranges -- checked most-specific first.
_SCRIPT_BLOCKS = [
    ("Tamil", "ta", 0x0B80, 0x0BFF),
    ("Devanagari", "hi", 0x0900, 0x097F),
    ("Telugu", "te", 0x0C00, 0x0C7F),
    ("Malayalam", "ml", 0x0D00, 0x0D7F),
    ("Kannada", "kn", 0x0C80, 0x0CFF),
    ("Bengali", "bn", 0x0980, 0x09FF),
    ("Gujarati", "gu", 0x0A80, 0x0AFF),
    ("Gurmukhi", "pa", 0x0A00, 0x0A7F),
    ("Arabic", "ur", 0x0600, 0x06FF),
]


def detect_script(text: str) -> str:
    """
    Returns the dominant script name found in `text`: one of the names in
    _SCRIPT_BLOCKS, or "Latin" if the text is primarily Latin-alphabet
    (covers English and Tanglish/romanized input alike -- distinguishing
    "Tamil-as-Tanglish" from "genuinely English" is the tanglish detector's
    job, not this one's).
    """
    counts: Dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        matched = False
        for name, _lang, lo, hi in _SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                matched = True
                break
        if not matched and ch.isalpha():
            counts["Latin"] = counts.get("Latin", 0) + 1

    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


def script_to_lang_hint(script: str) -> str:
    """The ISO code most associated with a given non-Latin script, if any."""
    for name, lang, _lo, _hi in _SCRIPT_BLOCKS:
        if name == script:
            return lang
    return None
