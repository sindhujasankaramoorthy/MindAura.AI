"""
Builds the pipeline's stable JSON output (and error) shape. Centralized here
so pipeline.py's orchestration logic doesn't also own presentation/schema
concerns.
"""
from typing import Any, Dict, List

# TokenLanguage string labels -> the display language name required by the
# output schema. "Tanglish" deliberately maps to "Tamil" -- Tanglish is a
# fallback mechanism, not a language (see ai/text/tanglish/rules.py).
_TOKEN_LABEL_TO_LANGUAGE = {
    "English": "English",
    "Tanglish": "Tamil",
    "TAMIL": "Tamil",
    "HINDI": "Hindi",
    "TELUGU": "Telugu",
    "MALAYALAM": "Malayalam",
    "KANNADA": "Kannada",
    "BENGALI": "Bengali",
    "MARATHI": "Marathi",
    "GUJARATI": "Gujarati",
    "PUNJABI": "Punjabi",
    "URDU": "Urdu",
}

# langdetect ISO code -> display name, for "other supported" world languages
# that the Indic-focused token classifier doesn't recognize (Spanish,
# French, etc.) and that aren't Tamil-via-Tanglish.
_ISO_TO_LANGUAGE_NAME = {
    "en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "fr": "French",
    "es": "Spanish", "de": "German", "ar": "Arabic", "zh-cn": "Chinese",
    "zh-tw": "Chinese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ru": "Russian", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
    "pl": "Polish", "uk": "Ukrainian",
}


def token_label_to_language(label: str) -> str:
    return _TOKEN_LABEL_TO_LANGUAGE.get(label)


def iso_to_language_name(iso_code: str) -> str:
    return _ISO_TO_LANGUAGE_NAME.get(iso_code, iso_code)


def build_language_info(
    token_classifications,
    script: str,
    tanglish_fallback_used: bool,
    fallback_language_name: str = None,
) -> Dict[str, Any]:
    """
    Derives the `language` block from per-token classification (the
    pipeline's primary, most granular signal -- handles English/Tamil/
    native-Indic-script identification and mixed-language detection) with
    `fallback_language_name` (a whole-text langdetect result) used only when
    the token classifier found no recognizable language at all -- e.g. a
    genuinely "other" Latin-script language like Spanish or French that
    isn't Tamil-via-Tanglish and isn't in the token classifier's Indic set.
    """
    ordered_languages: List[str] = []
    counts: Dict[str, int] = {}

    for _token, label in token_classifications:
        language = token_label_to_language(label)
        if not language:
            continue
        if language not in counts:
            ordered_languages.append(language)
        counts[language] = counts.get(language, 0) + 1

    if not ordered_languages and fallback_language_name:
        ordered_languages = [fallback_language_name]
        counts = {fallback_language_name: 1}

    if not ordered_languages:
        primary_language = "unknown"
    else:
        primary_language = max(ordered_languages, key=lambda lang: counts[lang])

    return {
        "primary_language": primary_language,
        "languages_detected": ordered_languages,
        "mixed_language": len(ordered_languages) > 1,
        "script": script,
        "tanglish_fallback_used": tanglish_fallback_used,
    }


def build_success_output(
    raw_input: str,
    contextual_final_sentence: str,
    expressive_texting: Dict[str, Any],
    language: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "raw_input": raw_input,
        "contextual_final_sentence": contextual_final_sentence,
        "expressive_texting": expressive_texting,
        "language": language,
        "status": "success",
    }


def build_error_output(raw_input: str, error_type: str, message: str) -> Dict[str, Any]:
    return {
        "raw_input": raw_input,
        "contextual_final_sentence": None,
        "expressive_texting": {"detected": False, "extensions": []},
        "language": {
            "primary_language": "unknown",
            "languages_detected": [],
            "mixed_language": False,
            "script": "unknown",
            "tanglish_fallback_used": False,
        },
        "status": "error",
        "error": {"type": error_type, "message": message},
    }
