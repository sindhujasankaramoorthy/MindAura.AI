"""
NLLB-200 translation -- the fallback path for any language not covered by
IndicTrans2 (see router.py for the routing decision), and ALSO the fallback
for Indic languages when IndicTrans2 itself can't load (e.g. it's a gated
HF model and no access/token is configured -- see router.py).
"""
import logging

from ai.model_registry import get_nllb_600m

logger = logging.getLogger(__name__)

# ISO 639-1 -> FLORES-200 code, for languages this pipeline expects to route
# through NLLB. NLLB itself supports ~200 languages; this map only needs
# entries for languages we actually detect upstream (langdetect's output
# codes) or that IndicTrans2 also covers (as its fallback), not the model's
# full inventory.
ISO_TO_FLORES = {
    "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "ar": "arb_Arab", "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl", "pt": "por_Latn",
    "it": "ita_Latn", "nl": "nld_Latn", "tr": "tur_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "id": "ind_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl",
    "sw": "swh_Latn", "fa": "pes_Arab", "he": "heb_Hebr", "el": "ell_Grek",
    # Indic languages -- primarily routed to IndicTrans2, but NLLB covers
    # them too so translation still works if IndicTrans2 is unavailable.
    "ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu", "ml": "mal_Mlym",
    "kn": "kan_Knda", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "ur": "urd_Arab",
}

_tokenizer = None
_model = None
_device = None


def _load():
    global _tokenizer, _model, _device
    if _model is None:
        _tokenizer, _model, _device = get_nllb_600m()
    return _tokenizer, _model, _device


def translate_to_english(text: str, iso_lang_code: str = None) -> str:
    """Translate `text` to English via NLLB-200. Falls back to the original
    text (rather than raising) if translation fails for any reason."""
    if not text or not text.strip():
        return text

    try:
        tokenizer, model, device = _load()

        flores_code = ISO_TO_FLORES.get(iso_lang_code)
        if flores_code:
            tokenizer.src_lang = flores_code

        inputs = tokenizer(text, return_tensors="pt")
        if device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=512,
        )
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"NLLB translation failed: {e}")
        return text
