import os
import sys
import logging
from transformers import pipeline

try:
    from ai.model_registry import get_roberta_go_emotions
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from ai.model_registry import get_roberta_go_emotions

from ai.text.pipeline import TextNormalizer

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    """
    Analyzes the transcribed text to detect emotions.
    Uses RoBERTa go_emotions model.

    Runs transcripts through the same TextNormalizer pipeline (Tanglish
    correction, NER protection, negation recovery, etc.) that text journal
    entries get in ai/inference/emotion_predict.py, so a spoken Tanglish
    journal gets the same preprocessing quality as a typed one instead of
    going straight from Whisper into RoBERTa unprocessed.
    """

    _instance = None
    _classifier = None
    _normalizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextEmotionAnalyzer, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._classifier is None:
            logger.info("Loading Text Emotion Recognition Model...")
            # Shares the same underlying RoBERTa GoEmotions model/tokenizer
            # as the text-journal pipeline (ai/inference/emotion_predict.py)
            # via the registry, so it's only ever loaded once per process.
            tokenizer, model, device = get_roberta_go_emotions()
            self._classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=3,
                device=device
            )
            logger.info("Text Emotion Model Loaded.")

    def _load_normalizer(self):
        if self._normalizer is None:
            self._normalizer = TextNormalizer()

    def predict(self, text: str):
        """
        Predicts top emotions from text.
        Returns:
            list of dicts containing 'label' and 'score'
        """
        if not text or not text.strip():
            return []

        self._load_normalizer()
        # translator_fn is left unset: voice transcripts are normalized/
        # corrected the same as text, but non-English/non-Tanglish machine
        # translation via NLLB is not wired in here (would pull in another
        # large model from the voice path for a case that hasn't come up
        # in practice yet — journal entries are English/Tanglish).
        normalized = self._normalizer.normalize(text)
        inference_text = normalized["corrected_sentence"]

        self._load_model()
        results = self._classifier(inference_text)

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            return results[0]
        return results

if __name__ == "__main__":
    analyzer = TextEmotionAnalyzer()
    print(analyzer.predict("I feel so sad and lonely falling behind."))
