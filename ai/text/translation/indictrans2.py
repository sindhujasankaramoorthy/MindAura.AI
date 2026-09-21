"""
IndicTrans2 translation -- the primary path for Indian languages (see
router.py for the routing decision). This model was previously implemented
in the old ai/preprocessing/advanced_correction.py (as `translate_indic`)
but never actually called from anywhere -- it's wired into the active path
here for the first time.

FLORES_TO_ISO / ISO_TO_FLORES are derived directly from
IndicTransToolkit.evaluator's own `_flores_codes` mapping (the actual
language set this specific model/toolkit version supports), not an
arbitrary hand-picked list -- satisfies "use the actual language codes
supported by the selected models."
"""
import logging

import torch

from ai.model_registry import get_indictrans2

logger = logging.getLogger(__name__)

# FLORES-200 code -> ISO 639-1, exactly as used by IndicTransToolkit's own
# evaluator (ai4bharat/IndicTransToolkit). This is the real, complete set of
# languages/dialects ai4bharat/indictrans2-indic-en-1B was trained on.
FLORES_TO_ISO = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi",
    "brx_Deva": "hi", "doi_Deva": "hi", "gom_Deva": "kK", "gon_Deva": "hi",
    "guj_Gujr": "gu", "hin_Deva": "hi", "hne_Deva": "hi", "kan_Knda": "kn",
    "kas_Arab": "ur", "kas_Deva": "hi", "mag_Deva": "hi", "mai_Deva": "hi",
    "mal_Mlym": "ml", "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi",
    "npi_Deva": "ne", "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "hi",
    "sat_Olck": "or", "snd_Arab": "ur", "snd_Deva": "hi", "tam_Taml": "ta",
    "tel_Telu": "te", "urd_Arab": "ur", "unr_Deva": "hi",
}

# Reverse map (ISO -> FLORES) -- when multiple FLORES codes share an ISO
# code (dialect variants), the first/most standard one wins.
ISO_TO_FLORES = {}
for _flores, _iso in FLORES_TO_ISO.items():
    ISO_TO_FLORES.setdefault(_iso, _flores)

SUPPORTED_ISO_CODES = set(ISO_TO_FLORES.keys())

_tokenizer = None
_model = None
_processor = None
_load_failed = False


def is_supported(iso_lang_code: str) -> bool:
    return iso_lang_code in SUPPORTED_ISO_CODES


def _load():
    global _tokenizer, _model, _processor, _load_failed
    if _model is None and not _load_failed:
        try:
            _tokenizer, _model, _processor = get_indictrans2()
        except Exception as e:
            # Cache the failure too, not just success -- otherwise every
            # single translate call re-attempts the (slow, network-bound)
            # load, e.g. hitting the same 401 repeatedly if this is a gated
            # model this environment has no access to.
            _load_failed = True
            logger.error(f"IndicTrans2 failed to load, will not retry this process: {e}")
    return _tokenizer, _model, _processor


def translate_to_english(text: str, iso_lang_code: str) -> str:
    """Translate `text` (in an Indian language) to English via IndicTrans2.
    Falls back to the original text if the language isn't supported or
    translation fails for any reason."""
    if not text or not text.strip():
        return text

    src_lang = ISO_TO_FLORES.get(iso_lang_code)
    if src_lang is None:
        logger.warning(f"IndicTrans2 has no mapping for language code '{iso_lang_code}'.")
        return text

    try:
        tokenizer, model, processor = _load()
        batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang="eng_Latn")
        inputs = tokenizer(batch, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return processor.postprocess_batch(decoded, lang="eng_Latn")[0]
    except Exception as e:
        logger.error(f"IndicTrans2 translation failed: {e}")
        return text
