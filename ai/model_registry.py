"""
Process-wide cache for expensive model/resource loads (transformers models,
tokenizers, SymSpell dictionaries, etc.), keyed by a short string name.

Every heavy load in ai/ should go through ModelRegistry.get() instead of
calling from_pretrained()/etc. directly, so that:
  - a given checkpoint is only ever loaded once per process, no matter how
    many EmotionAnalyzer/TextNormalizer/EmotionPreservingCorrector/... are
    instantiated.
  - tests can monkeypatch ModelRegistry.get in one place to stub out every
    model at once (see tests/conftest.py).

Loader functions here own the checkpoint name and from_pretrained() call for
models that don't depend on other ai.preprocessing/* constants. Loaders that
need those constants (e.g. SymSpell, which is boosted with protected-word
lists defined in advanced_correction.py) stay defined next to their
dependencies and just route through ModelRegistry.get() for caching.
"""
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    _cache: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key: str, loader: Callable[[], Any]) -> Any:
        if key in cls._cache:
            return cls._cache[key]
        with cls._lock:
            if key not in cls._cache:
                logger.info(f"[ModelRegistry] Loading '{key}'...")
                cls._cache[key] = loader()
                logger.info(f"[ModelRegistry] '{key}' loaded.")
        return cls._cache[key]

    @classmethod
    def is_loaded(cls, key: str) -> bool:
        return key in cls._cache

    @classmethod
    def clear(cls, key: Optional[str] = None) -> None:
        with cls._lock:
            if key is None:
                cls._cache.clear()
            else:
                cls._cache.pop(key, None)


# ── Named loaders ────────────────────────────────────────────────────────

def get_roberta_go_emotions():
    """Returns (tokenizer, model, device) for SamLowe/roberta-base-go_emotions."""

    def _load():
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "SamLowe/roberta-base-go_emotions"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        return tokenizer, model, device

    return ModelRegistry.get("roberta_go_emotions", _load)


def get_nllb_600m():
    """Returns (tokenizer, model, device) for facebook/nllb-200-distilled-600M."""

    def _load():
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            model = model.to('cuda')
        return tokenizer, model, device

    return ModelRegistry.get("nllb_600m", _load)


def get_gpt2_perplexity_model():
    """Returns (tokenizer, model) for gpt2, used only as a perplexity scorer."""

    def _load():
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model

    return ModelRegistry.get("gpt2_perplexity", _load)


def get_indictrans2():
    """Returns (tokenizer, model, processor) for ai4bharat/indictrans2-indic-en-1B."""

    def _load():
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from IndicTransToolkit import IndicProcessor

        model_name = "ai4bharat/indictrans2-indic-en-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        processor = IndicProcessor(inference=True)
        return tokenizer, model, processor

    return ModelRegistry.get("indictrans2", _load)


def get_gliner_ner():
    """Returns a GLiNER zero-shot NER model (NeuML/gliner-bert-tiny)."""

    def _load():
        from gliner import GLiNER

        return GLiNER.from_pretrained("NeuML/gliner-bert-tiny")

    return ModelRegistry.get("gliner_ner", _load)


def get_bert_ner_fallback():
    """Returns a transformers NER pipeline (dslim/bert-base-NER-uncased)."""

    def _load():
        from transformers import pipeline

        return pipeline("ner", model="dslim/bert-base-NER-uncased")

    return ModelRegistry.get("bert_ner_fallback", _load)


def get_wav2vec2_ser():
    """Returns (feature_extractor, model) for superb/wav2vec2-base-superb-er."""

    def _load():
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_name = "superb/wav2vec2-base-superb-er"
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        model.eval()
        return processor, model

    return ModelRegistry.get("wav2vec2_ser", _load)


def get_faster_whisper():
    """Returns a faster_whisper WhisperModel ("base", CPU, int8)."""

    def _load():
        from faster_whisper import WhisperModel

        return WhisperModel("base", device="cpu", compute_type="int8")

    return ModelRegistry.get("faster_whisper", _load)
