"""
Standalone voice analysis: produces the two feature groups discussed, as one
JSON-able dict. Self-contained within ai/voice — loads its own Faster-Whisper
and NLLB-200 models directly rather than reaching into ai/model_registry.py
or ai/inference/*, and does NOT touch the emotion/SER/fusion/Qwen stack
(emotion_predict, fusion_engine, qwen_reasoning).

1. Acoustic features: pitch, speaking rate, pauses, energy, variability.
2. Language id -> raw transcript (native script) -> word-by-word gloss ->
   contextual (sentence-level) English translation.

CLI:
    python -m ai.voice.voice_analysis <path_to_audio.wav>
"""

import os
import sys
import json

import librosa

try:
    from ai.voice.voice_features import extract_voice_features
    from ai.voice.name_protection import protect_names
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from ai.voice.voice_features import extract_voice_features
    from ai.voice.name_protection import protect_names

# Faster-Whisper model size. "small" hallucinates badly on Tamil (verified
# against a real recording, see docs/voice_pipeline_notes.md) -- "medium"
# is the smallest step up with meaningfully better low-resource-language
# support. Overridable via env var so it can be tuned per-deployment
# without a code change.
WHISPER_MODEL_SIZE = os.environ.get("MINDAURA_WHISPER_MODEL", "medium")

# Short (Whisper) language code -> FLORES-200 code (NLLB's language space).
# Mirrors ai/inference/emotion_predict.py's NLLB_LANG_CODES, plus "ur" which
# that map lacks (added here using the same code IndicTrans2 uses for Urdu
# elsewhere in the repo, ai/preprocessing/advanced_correction.py).
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "ur": "urd_Arab",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
}

# Unicode block -> language code, used to tag which language each word/segment
# is actually in when a clip mixes languages (e.g. Tamil sentences with English
# words dropped in mid-sentence). Whisper's `multilingual=True` mode re-detects
# language internally per segment during decoding but doesn't expose that tag
# on the returned objects, so we recover it ourselves from the script the
# transcribed text came out in. This only distinguishes *script*, not the
# specific language within a script (e.g. any Latin-script word reads as "en").
_SCRIPT_RANGES = [
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi/Marathi)
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("ur", 0x0600, 0x06FF),  # Arabic script (Urdu/Arabic)
]


def _detect_script_lang(text):
    """Best-effort language guess from the Unicode script of `text`."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
        else:
            if ch.isalpha():
                counts["en"] = counts.get("en", 0) + 1
    if not counts:
        return "en"
    return max(counts, key=counts.get)


_model = None
_nllb_tokenizer = None
_nllb_model = None
_nllb_device = None


def _load_model():
    """Faster-Whisper (WHISPER_MODEL_SIZE, CPU, int8) — loaded once, cached here."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def _load_nllb():
    """NLLB-200-distilled-600M — loaded once, cached here."""
    global _nllb_tokenizer, _nllb_model, _nllb_device
    if _nllb_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "facebook/nllb-200-distilled-600M"
        _nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _nllb_device = 0 if torch.cuda.is_available() else -1
        if _nllb_device == 0:
            _nllb_model = _nllb_model.to("cuda")
    return _nllb_tokenizer, _nllb_model, _nllb_device


def _translate_text_to_english(text, lang_code):
    """Text-to-text translation of a single word/phrase via NLLB-200."""
    if not text or not text.strip():
        return ""

    tokenizer, model, device = _load_nllb()
    src = NLLB_LANG_CODES.get(lang_code)
    if src:
        tokenizer.src_lang = src

    inputs = tokenizer(text, return_tensors="pt")
    if device == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=64,
    )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def _avg_logprob_to_confidence(logprobs):
    """exp() of the mean segment avg_logprob -- a real signal straight from
    the model's own decoding, not an invented heuristic. Whisper doesn't
    expose a single first-class "confidence" score, so this is the
    closest honest proxy available."""
    import math

    if not logprobs:
        return None
    return round(math.exp(sum(logprobs) / len(logprobs)), 2)


def _transcribe_native(audio_path, language=None, multilingual=False):
    """
    Transcribe in the spoken language, with word-level timestamps.

    `language` (e.g. "ta" for Tamil) skips Whisper's auto-detect entirely.
    Pass it whenever the speaker's language is already known and the whole
    clip is that one language (e.g. Tamil data-collection sessions) —
    auto-detect can misidentify close language pairs like Urdu/Hindi,
    especially on the "small" model.

    `multilingual=True` is for the opposite case: a single clip that mixes
    languages mid-sentence (e.g. Tamil with English words dropped in).
    Whisper re-detects language per internal segment in this mode instead of
    locking the whole clip to one language. Mutually exclusive with
    `language` — don't force a language if you want it free to switch.
    Each returned word is also tagged with `lang` (a script-based guess, see
    `_detect_script_lang`), since Whisper doesn't expose its own per-segment
    language choice on the returned objects.
    """
    model = _load_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        language=language,
        multilingual=multilingual,
        # Built-in Silero VAD (already bundled with faster-whisper, no new
        # dependency) -- skips non-speech stretches instead of feeding them
        # to the decoder, which otherwise sometimes hallucinates words for
        # silence/background noise.
        vad_filter=True,
    )

    words = []
    text_parts = []
    logprobs = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        if segment.avg_logprob is not None:
            logprobs.append(segment.avg_logprob)
        for w in (segment.words or []):
            word_text = w.word.strip()
            words.append({
                "word": word_text,
                "start": w.start,
                "end": w.end,
                "lang": _detect_script_lang(word_text),
            })

    languages_detected = sorted({w["lang"] for w in words}) or [language or info.language]

    return {
        "language": language or info.language,
        "language_probability": getattr(info, "language_probability", None),
        "languages_detected": languages_detected,
        "raw_transcript": " ".join(p for p in text_parts if p).strip(),
        "words": words,
        # Real signal from the model, not fabricated: average of each
        # segment's own avg_logprob, converted from log-space to a
        # (0, 1] confidence-like value. None if there were no segments
        # (e.g. silence/no speech detected).
        "transcription_confidence": _avg_logprob_to_confidence(logprobs),
    }


def _translate_contextual(audio_path, language=None, multilingual=False):
    """
    Full speech-to-English translation, using sentence-level context.
    Whisper's translate task already maps arbitrary (including code-switched)
    speech straight to English in one pass, so `multilingual=True` here just
    lets it re-detect language per segment the same way transcription does.
    """
    model = _load_model()
    segments, _ = model.transcribe(
        audio_path, beam_size=5, task="translate", language=language, multilingual=multilingual,
        vad_filter=True,
    )
    segments = list(segments)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    confidence = _avg_logprob_to_confidence([s.avg_logprob for s in segments if s.avg_logprob is not None])
    return text, confidence


def _word_by_word_gloss(words):
    """
    Naive literal gloss: translate each native word individually as text, so
    the result is word-for-word rather than reflowed to fit English grammar
    (contextual_translation does that part, with full sentence context).

    Text-based on purpose — translating tiny per-word *audio* slices was tried
    first and Whisper hallucinated full sentences from clips that short.

    Each word carries its own `lang` tag (set in _transcribe_native), so in a
    code-switched clip an English word mixed into a Tamil sentence is passed
    through unchanged instead of being force-translated with the wrong source
    language, which would just mangle it.
    """
    if not words:
        return []

    gloss = []
    for w in words:
        if w.get("lang") == "en":
            gloss.append(w["word"])
            continue
        try:
            gloss.append(_translate_text_to_english(w["word"], w.get("lang")))
        except Exception:
            gloss.append("")
    return gloss


def _build_language_profile(primary, language_probability, languages_detected):
    """
    Reshapes real detection signal already computed elsewhere -- Whisper's
    own per-file language_probability, plus the script-based per-word tags
    from _transcribe_native -- into the requested primary/secondary/
    code_switched schema. Nothing here is invented: code_switched is
    exactly "more than one script/language actually found in the words",
    and confidence is exactly what Whisper itself reported (None if it
    wasn't computed, e.g. when `language` was forced and detection was
    skipped, rather than a fabricated 1.0).
    """
    secondary = [l for l in languages_detected if l != primary]
    return {
        "primary": primary,
        "secondary": secondary,
        "code_switched": len(languages_detected) > 1,
        "confidence": round(language_probability, 2) if language_probability is not None else None,
    }


def _acoustic_features(audio_path, word_count, duration):
    raw = extract_voice_features(audio_path)
    speaking_rate = (
        round(word_count / duration, 2) if duration > 0 and word_count else raw.get("speech_rate", 0.0)
    )
    return {
        "pitch_mean_hz": raw.get("pitch", 0.0),
        "pitch_variability": raw.get("pitch_variability", 0.0),
        "speaking_rate_wps": speaking_rate,
        "pause_ratio": raw.get("pause_ratio", 0.0),
        "energy_rms": raw.get("rms_energy", 0.0),
    }


def analyze_voice(audio_path, language=None, multilingual=False):
    """
    Runs both analyses and returns one dict:
      {
        # Existing fields (unchanged shape):
        "language_detected": "ur",
        "languages_detected": ["ur"],
        "raw_transcript": "...",              # native-script, untranslated
        "word_by_word_gloss": ["...", ...],
        "contextual_translation": "...",      # English, with Indian names protected
        "acoustic_features": {...},
        "words": [{"word": "...", "start": 0.0, "end": 0.4}, ...],
        "duration_sec": 3.2,

        # Added fields:
        "normalized_transcription": "...",    # raw_transcript with Indian names corrected
        "english_contextual_text": "...",     # same value as contextual_translation
        "language": {
          "primary": "ur", "secondary": [], "code_switched": False, "confidence": 0.93,
        },
        "entities": [
          {"original": "...", "entity_type": "PERSON", "normalized": "...",
           "confidence": 0.94, "correction_applied": True},
          ...
        ],
        "confidence": {"language": 0.93, "transcription": 0.87, "translation": 0.85},
      }

    See docs/voice_pipeline_notes.md for the reasoning behind these
    additions (Indian-name protection, Tamil-accuracy fix) and their known
    limitations.

    `language`: ISO code (e.g. "ta") to force, skipping Whisper's auto-detect.
    Use this when the WHOLE clip is one known language — e.g. every clip from
    the Tamil data-collection sessions should pass "ta", since auto-detect can
    misidentify close language pairs (Urdu was misread as Hindi in testing).

    `multilingual`: set this instead of `language` when a single clip may mix
    languages mid-sentence (e.g. Tamil with English words dropped in,
    "Tanglish"-style code-switching). `languages_detected` will then list every
    language actually found, and word_by_word_gloss leaves already-English
    words untouched instead of mistranslating them.
    """
    native = _transcribe_native(audio_path, language=language, multilingual=multilingual)

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr if sr else 0.0
    acoustic = _acoustic_features(audio_path, len(native["words"]), duration)

    if native["languages_detected"] == ["en"]:
        # Nothing but English -- nothing to translate. Forcing English text
        # through a translation model produces garbage (tested: it
        # hallucinated content that was never said), so just pass the native
        # words straight through.
        contextual = native["raw_transcript"]
        translation_confidence = native["transcription_confidence"]
        gloss = [w["word"] for w in native["words"]]
    else:
        contextual, translation_confidence = _translate_contextual(
            audio_path, language=language, multilingual=multilingual
        )
        gloss = _word_by_word_gloss(native["words"])

    # Indian-name protection -- a dedicated stage (ai/voice/name_protection.py),
    # deliberately run on BOTH the native transcript and the English text
    # independently, since Whisper's translate task decodes straight from
    # audio in its own pass rather than consuming our corrected transcript;
    # protecting only one side would leave the other's name errors as-is.
    # Only Latin-script name mentions are checked against the Indian-name
    # dictionary (native-script name correction is out of scope here -- see
    # docs/voice_pipeline_notes.md for why).
    native_protection = protect_names(native["raw_transcript"])
    contextual_protection = protect_names(contextual)
    normalized_transcription = native_protection["corrected_text"]
    english_contextual_text = contextual_protection["corrected_text"]

    entities = native_protection["entities"] + [
        e for e in contextual_protection["entities"]
        if e["original"] not in {ne["original"] for ne in native_protection["entities"]}
    ]

    language_profile = _build_language_profile(
        native["language"], native.get("language_probability"), native["languages_detected"]
    )

    return {
        # --- Existing fields, unchanged shape (backward compatible with
        # ai/video/existing_voice_adapter.py and existing tests) ---
        "language_detected": native["language"],
        "languages_detected": native["languages_detected"],
        "raw_transcript": native["raw_transcript"],
        "word_by_word_gloss": gloss,
        "contextual_translation": english_contextual_text,
        "acoustic_features": acoustic,
        "words": native["words"],
        "duration_sec": round(duration, 3),

        # --- New fields: the three preserved versions, structured language
        # detection, and Indian-name protection ---
        "normalized_transcription": normalized_transcription,
        "english_contextual_text": english_contextual_text,
        "language": language_profile,
        "entities": entities,
        "confidence": {
            "language": language_profile["confidence"],
            # exp(mean segment avg_logprob) from the model's own decoding
            # -- a real proxy signal, since faster-whisper doesn't expose
            # a first-class confidence score directly.
            "transcription": native["transcription_confidence"],
            "translation": translation_confidence,
        },
    }


if __name__ == "__main__":
    # python -m ai.voice.voice_analysis <path_to_audio.wav> [language_code | multi]
    # "multi" as the 2nd arg turns on multilingual/code-switching mode instead
    # of forcing a single language.
    audio_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "recordings", "recording.wav"
    )
    lang_arg = sys.argv[2] if len(sys.argv) > 2 else None
    use_multilingual = lang_arg == "multi"
    forced_language = None if use_multilingual else lang_arg

    if not os.path.exists(audio_path):
        print(f"Audio file not found at {audio_path}")
        sys.exit(1)

    result = analyze_voice(audio_path, language=forced_language, multilingual=use_multilingual)
    print(json.dumps(result, indent=2, ensure_ascii=False))
