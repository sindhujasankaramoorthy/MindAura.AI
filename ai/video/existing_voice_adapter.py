"""
Thin adapter around the EXISTING voice model (ai/voice/voice_analysis.py,
analyze_voice()). Does not reimplement transcription or acoustic analysis --
it only reshapes analyze_voice()'s real output into the common MindAura
video-analysis schema, and honestly marks any requested field that model
does not actually produce as "not_available" rather than inventing it.

Per the raw-transcription requirement: analyze_voice()'s "raw_transcript" is
already the untranslated, native-script transcript (Whisper's own
transcribe task, not its translate task) -- exactly what's needed here.
Nothing is translated/normalized/Tanglish-processed at this stage; that
belongs to a later pipeline stage (ai/text/pipeline.py), not this one.
"""
import logging
from typing import Any, Dict, Optional

from ai.voice.voice_analysis import analyze_voice

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "not_available"


def analyze_audio(audio_path: str, language: Optional[str] = None, multilingual: bool = True) -> Dict[str, Any]:
    """
    Runs the existing voice model on `audio_path` and returns the common
    MindAura schema. `multilingual=True` by default since patient video
    audio can't be assumed to be a single known language ahead of time.

    Returns a dict shaped like:
      {
        "status": "ok" | "error",
        "transcription": {
          "raw_text": str,               # native-script, untranslated
          "normalized_text": str,        # raw_text with high-confidence Indian-name corrections applied
          "segments": [...],             # word-level timing (available)
          "language_detected": str,
          "languages_detected": [str],
          "language": {...},             # primary/secondary/code_switched/confidence
        },
        "entities": [...],               # PERSON entities found, with Indian-name correction info
        "acoustics": { ... },            # only fields the model actually computes
        "confidence": {...},             # real per-stage confidence where the model exposes one
        "duration_sec": float,
      }

    Translation to English is deliberately left out of this adapter's
    output -- raw_text/normalized_text stay native-script, since English
    translation is the job of ai/text/pipeline.py downstream, not this
    voice-specific stage (see module docstring).
    """
    try:
        result = analyze_voice(audio_path, language=language, multilingual=multilingual)
    except Exception as e:
        logger.exception("Existing voice model failed on %s", audio_path)
        return {
            "status": "error",
            "error": str(e),
            "transcription": {
                "raw_text": NOT_AVAILABLE,
                "normalized_text": NOT_AVAILABLE,
                "segments": [],
                "language_detected": NOT_AVAILABLE,
                "languages_detected": [],
                "language": NOT_AVAILABLE,
            },
            "entities": [],
            "acoustics": _empty_acoustics(),
            "confidence": {"language": NOT_AVAILABLE, "transcription": NOT_AVAILABLE},
            "duration_sec": NOT_AVAILABLE,
        }

    words = result.get("words", [])
    segments = [
        {"word": w["word"], "start_sec": round(w["start"], 3), "end_sec": round(w["end"], 3)}
        for w in words
    ]

    acoustic = result.get("acoustic_features", {})
    confidence = result.get("confidence", {})

    return {
        "status": "ok",
        "transcription": {
            "raw_text": result.get("raw_transcript", NOT_AVAILABLE),
            "normalized_text": result.get("normalized_transcription", NOT_AVAILABLE),
            "segments": segments,
            "language_detected": result.get("language_detected", NOT_AVAILABLE),
            "languages_detected": result.get("languages_detected", []),
            "language": result.get("language", NOT_AVAILABLE),
        },
        # PERSON entities from the transcript, with Indian-name protection's
        # correction info (see ai/voice/name_protection.py). Never a fake
        # emotion/diagnosis field -- just what was said and who it's about.
        "entities": result.get("entities", []),
        "acoustics": _map_acoustics(acoustic),
        "confidence": {
            "language": confidence.get("language", NOT_AVAILABLE),
            "transcription": confidence.get("transcription", NOT_AVAILABLE),
        },
        "duration_sec": result.get("duration_sec", NOT_AVAILABLE),
    }


def _map_acoustics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps the existing model's actual acoustic_features onto the requested
    schema fields (section 9 of the video-pipeline spec). Only fields the
    model genuinely computes get a real value; everything else the spec
    asked for but the model doesn't produce is explicitly "not_available" --
    no fabricated jitter/shimmer/pause-count/etc.
    """
    return {
        "pitch": {
            "f0_mean_hz": raw.get("pitch_mean_hz", NOT_AVAILABLE),
            "f0_min_hz": NOT_AVAILABLE,
            "f0_max_hz": NOT_AVAILABLE,
            "f0_std_hz": NOT_AVAILABLE,
            "f0_variability": raw.get("pitch_variability", NOT_AVAILABLE),
        },
        "speaking_rate": {
            "words_per_sec": raw.get("speaking_rate_wps", NOT_AVAILABLE),
            "articulation_rate": NOT_AVAILABLE,
        },
        "pauses": {
            "pause_ratio": raw.get("pause_ratio", NOT_AVAILABLE),
            "pause_count": NOT_AVAILABLE,
            "average_pause_duration_sec": NOT_AVAILABLE,
            "longest_pause_sec": NOT_AVAILABLE,
            "silence_ratio": raw.get("pause_ratio", NOT_AVAILABLE),
        },
        "energy": {
            "energy_mean": raw.get("energy_rms", NOT_AVAILABLE),
            "energy_variation": NOT_AVAILABLE,
        },
        "voice_quality": {
            "jitter": NOT_AVAILABLE,
            "shimmer": NOT_AVAILABLE,
        },
    }


def _empty_acoustics() -> Dict[str, Any]:
    return _map_acoustics({})
