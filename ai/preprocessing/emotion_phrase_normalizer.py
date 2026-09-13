import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Emotion-domain focused phrase mapping layer
# This improves contextual Tanglish understanding before character transliteration.
PHRASE_MAPPINGS: Dict[str, str] = {
    # Negative preference
    r"\bpidikala\b": "pidikkala",
    r"\bpidikula\b": "pidikkala",
    r"\bpidikilla\b": "pidikkala",
    r"\bpudikala\b": "pidikkala",
    r"\bpudikula\b": "pidikkala",
    
    # Context phrases
    r"\brendu perum\b": "iruvarum",
    r"\brendu peru\b": "iruvarum",
    
    # Conflict phrases
    r"\bsandai podraanga\b": "sandai poduraanga",
    r"\bsandai podra\b": "sandai podura",
}

def normalize_emotion_phrases(text: str) -> str:
    """
    Normalizes only emotion-critical Tanglish phrases before IndicXlit.
    """
    for pattern, repl in PHRASE_MAPPINGS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text
