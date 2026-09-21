"""
Centralized translation routing -- the single place that decides which
model handles a given language, instead of that decision being duplicated
across modules.

Routing:
    English            -> passthrough (no translation)
    IndicTrans2-supported language -> IndicTrans2, falling back to NLLB-200
                                       if IndicTrans2 can't load (e.g. it's
                                       a gated HF model and no access/token
                                       is configured in this environment)
    everything else    -> NLLB-200
"""
import logging

from ai.text.translation import indictrans2, nllb

logger = logging.getLogger(__name__)


def translate_to_english(text: str, iso_lang_code: str) -> str:
    """
    Route `text` to the appropriate translation backend based on
    `iso_lang_code` (an ISO 639-1 code, e.g. from langdetect) and return
    English text. English input is passed through unchanged.
    """
    if not text or not text.strip():
        return text

    if iso_lang_code == "en":
        return text

    if indictrans2.is_supported(iso_lang_code):
        translated = indictrans2.translate_to_english(text, iso_lang_code)
        if translated != text:
            return translated
        # IndicTrans2 returns the original text unchanged on any internal
        # failure (see indictrans2.py) -- most commonly because
        # ai4bharat/indictrans2-indic-en-1B is a gated model and this
        # environment has no HF access to it. Fall back to NLLB rather
        # than silently shipping untranslated native-script text.
        logger.warning(
            f"IndicTrans2 did not translate '{iso_lang_code}' text "
            "(likely no access to the gated model) -- falling back to NLLB."
        )
        return nllb.translate_to_english(text, iso_lang_code)

    return nllb.translate_to_english(text, iso_lang_code)


def is_indic(iso_lang_code: str) -> bool:
    return indictrans2.is_supported(iso_lang_code)
