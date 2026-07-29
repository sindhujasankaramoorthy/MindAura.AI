import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    """
    Analyzes the transcribed text to detect emotions.
    Uses RoBERTa go_emotions model.
    """
    
    _instance = None
    _classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextEmotionAnalyzer, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._classifier is None:
            logger.info("Loading Text Emotion Recognition Model...")
            # Using the fast RoBERTa GoEmotions model
            self._classifier = pipeline(
                "text-classification", 
                model="SamLowe/roberta-base-go_emotions", 
                top_k=3
            )
            logger.info("Text Emotion Model Loaded.")

    def predict(self, text: str):
        """
        Predicts top emotions from text.
        Returns:
            list of dicts containing 'label' and 'score'
        """
        if not text or not text.strip():
            return []
            
        self._load_model()
        results = self._classifier(text)
        
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            return results[0]
        return results

if __name__ == "__main__":
    analyzer = TextEmotionAnalyzer()
    print(analyzer.predict("I feel so sad and lonely falling behind."))
