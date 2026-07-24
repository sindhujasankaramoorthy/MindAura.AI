import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Dictionary to normalize informal/shortened Tanglish spellings to canonical Tanglish.
# This should NOT translate to English; it only fixes spoken/shortened Tanglish
# so that IndicXlit can map them cleanly to Tamil script.
CANONICAL_TANGLISH_MAPPING: Dict[str, str] = {
    "mudla": "mudiyala",
    "enmaku": "enaku",
    "kastama": "kashtama",
    "kuda": "kooda",
    "purila": "puriyala",
    "pudikula": "pudikala",
    "pidikula": "pidikala",
    "bayama": "bayama",
}

def normalize_canonical_tanglish(word: str) -> str:
    """
    Normalizes a single Tanglish word to its canonical form if present in the mapping.
    Otherwise returns the word as-is.
    """
    word_lower = word.lower()
    
    if word_lower in CANONICAL_TANGLISH_MAPPING:
        normalized = CANONICAL_TANGLISH_MAPPING[word_lower]
        # Preserve original casing
        if word.istitle():
            return normalized.title()
        elif word.isupper():
            return normalized.upper()
        return normalized
        
    return word

def normalize_canonical_tanglish_sentence(text: str) -> str:
    """
    Normalizes a full sentence by applying canonical normalization word by word,
    preserving punctuation and spaces.
    """
    tokens = re.split(r'(\W+)', text)
    out_tokens = []
    for t in tokens:
        if not t.strip() or re.match(r'\W+', t):
            out_tokens.append(t)
        else:
            out_tokens.append(normalize_canonical_tanglish(t))
    return "".join(out_tokens)

