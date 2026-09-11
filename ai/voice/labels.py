"""
Unified label space for the multi-lingual SER model.

We train on 7 categorical emotions. Corpora that carry extra classes
(calm, boredom, enthusiasm, ...) map those to ``None`` and the sample is
dropped during :mod:`ai.voice.prepare`.

Rationale for dropping "calm": only RAVDESS carries it (192 clips) and it is
not clinically distinct enough from "neutral" to keep as its own class at this
data volume. Revisit once EmoTa / IEMOCAP land (Stage 3).
"""
from __future__ import annotations

UNIFIED_EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

LABEL2ID: dict[str, int] = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}
ID2LABEL: dict[int, str] = {i: e for e, i in LABEL2ID.items()}

# Raw corpus token (lower-cased) -> unified emotion, or None to drop the sample.
CORPUS_LABEL_MAPS: dict[str, dict[str, str | None]] = {
    "ravdess": {
        "neutral": "neutral",
        "calm": None,
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fearful": "fear",
        "disgust": "disgust",
        "surprised": "surprise",
    },
    "crema_d": {
        "ang": "angry",
        "dis": "disgust",
        "fea": "fear",
        "hap": "happy",
        "neu": "neutral",
        "sad": "sad",
    },
    "tess": {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "ps": "surprise",              # "pleasant surprise"
        "pleasant_surprise": "surprise",
        "sad": "sad",
    },
    "emodb": {
        "w": "angry",                  # Wut / Ärger
        "l": None,                     # Langeweile (boredom)
        "e": "disgust",                # Ekel
        "a": "fear",                   # Angst
        "f": "happy",                  # Freude
        "t": "sad",                    # Trauer
        "n": "neutral",
    },
    "urdu": {
        "angry": "angry",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
    },
    "shemo": {
        "a": "angry",
        "h": "happy",
        "n": "neutral",
        "s": "sad",
        "w": "surprise",
        "f": "fear",
    },
    "resd": {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "enthusiasm": None,
        "fear": "fear",
        "happiness": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "sad": "sad",
    },
    # EmoTa (Tamil) — 5 emotions, no disgust/surprise (learned from English side)
    "emota": {
        "anger": "angry",
        "angry": "angry",
        "ang": "angry",
        "happiness": "happy",
        "happy": "happy",
        "hap": "happy",
        "sadness": "sad",
        "sad": "sad",
        "fear": "fear",
        "fea": "fear",
        "neutral": "neutral",
        "neu": "neutral",
    },
    # SUBESCO (Bangla) — 7 emotions, exact match to the unified space
    "subesco": {
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "surprise": "surprise",
    },
    # MindAura's own Tamil corpus — acted, 5 emotions
    "mindaura_tamil": {
        "angry": "angry", "ang": "angry",
        "happy": "happy", "hap": "happy",
        "sad": "sad",
        "fear": "fear", "fea": "fear",
        "neutral": "neutral", "neu": "neutral",
    },
    # Kannada emotional speech — 6 emotions (no disgust), numeric codes 01-06
    "kannada": {
        "01": "angry",
        "02": "sad",
        "03": "surprise",
        "04": "happy",
        "05": "fear",
        "06": "neutral",
        "anger": "angry",
        "sadness": "sad",
        "surprise": "surprise",
        "happiness": "happy",
        "fear": "fear",
        "neutral": "neutral",
    },
}


def to_unified(corpus: str, raw_label: str) -> str | None:
    """Map a corpus-native label to the unified space. None -> drop sample."""
    table = CORPUS_LABEL_MAPS.get(corpus)
    if table is None:
        raise KeyError(f"no label map registered for corpus {corpus!r}")
    key = raw_label.strip().lower()
    if key not in table:
        raise KeyError(f"unmapped {corpus!r} label {raw_label!r}")
    return table[key]
