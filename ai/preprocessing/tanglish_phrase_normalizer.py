import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Emotion-domain focused phrase mapping layer
# This improves contextual Tanglish understanding before character transliteration.
PHRASE_MAPPINGS: Dict[str, str] = {
    r"\brendu perum\b": "iruvarum",
    r"\brendu peru\b": "iruvarum",
    r"\benaku\b": "enakku",
    r"\benmaku\b": "enakku",
    r"\bunaku\b": "unakku",
    r"\bpidikala\b": "pidikkala",
    r"\bpidikula\b": "pidikkala",
    r"\bpidikilla\b": "pidikkala",
    r"\bpudikala\b": "pidikkala",
    r"\bpudikula\b": "pidikkala",
    r"\bkastama\b": "kashtama",
    r"\bkastam\b": "kashtam",
    r"\bkuda\b": "kooda",
    r"\biruku\b": "irukku",
    r"\billa\b": "illai",
}

def normalize_tanglish_phrases(text: str) -> str:
    """
    Normalizes multi-word Tanglish phrases to their contextual/canonical forms.
    """
    for pattern, repl in PHRASE_MAPPINGS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text
