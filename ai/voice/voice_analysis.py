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
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from ai.voice.voice_features import extract_voice_features

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
    """Faster-Whisper ("small", CPU, int8) — loaded once, cached here."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("small", device="cpu", compute_type="int8")
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
    )

    words = []
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
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
        "languages_detected": languages_detected,
        "raw_transcript": " ".join(p for p in text_parts if p).strip(),
        "words": words,
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
        audio_path, beam_size=5, task="translate", language=language, multilingual=multilingual
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


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
        "language_detected": "ur",
        "languages_detected": ["ur"],
        "raw_transcript": "...",
        "word_by_word_gloss": ["...", ...],
        "contextual_translation": "...",
        "acoustic_features": {...},
        "words": [{"word": "...", "start": 0.0, "end": 0.4}, ...],
        "duration_sec": 3.2
      }

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
        gloss = [w["word"] for w in native["words"]]
    else:
        contextual = _translate_contextual(audio_path, language=language, multilingual=multilingual)
        gloss = _word_by_word_gloss(native["words"])

    return {
        "language_detected": native["language"],
        "languages_detected": native["languages_detected"],
        "raw_transcript": native["raw_transcript"],
        "word_by_word_gloss": gloss,
        "contextual_translation": contextual,
        "acoustic_features": acoustic,
        # Per-word (word, start_sec, end_sec) timing -- already computed
        # internally by _transcribe_native, just not previously surfaced.
        # Exposed here (purely additive, no existing field changed) so
        # callers needing real transcript timing (e.g. video/voice
        # timestamp synchronization) don't have to re-run Whisper.
        "words": native["words"],
        "duration_sec": round(duration, 3),
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
