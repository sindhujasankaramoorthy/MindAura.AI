import logging
import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from transformers import pipeline
import langdetect
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from ai.preprocessing.text_normalizer import TextNormalizer
from ai.inference.psychological_signals import PsychologicalSignalExtractor
from ai.model_registry import get_roberta_go_emotions, get_nllb_600m


class EmotionAnalyzer:
    """
    EmotionAnalyzer for MindAura.
    Performs deep emotion analysis on journal/blog entries using RoBERTa.
    """
    
    def __init__(self):
        """
        Initialize the EmotionAnalyzer. Model loading is deferred until first
        use (see the `classifier`/`trans_tokenizer`/`trans_model` properties
        below), all backed by the shared ai.model_registry.ModelRegistry so
        the RoBERTa/NLLB checkpoints are loaded at most once per process.
        """
        self.model_name = "SamLowe/roberta-base-go_emotions"
        self.normalizer = TextNormalizer()
        self.signal_extractor = PsychologicalSignalExtractor()

        self.NLLB_LANG_CODES = {
            'en': 'eng_Latn', 'ta': 'tam_Taml', 'hi': 'hin_Deva', 'te': 'tel_Telu',
            'ml': 'mal_Mlym', 'kn': 'kan_Knda', 'bn': 'ben_Beng', 'mr': 'mar_Deva',
            'fr': 'fra_Latn', 'es': 'spa_Latn', 'de': 'deu_Latn', 'ar': 'arb_Arab',
            'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant', 'zh': 'zho_Hans'
        }

        # Define all 28 GoEmotions labels
        self.ALL_GOEMOTIONS = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]

        self.translation_model_name = "facebook/nllb-200-distilled-600M"

        self._classifier = None
        self._device = None
        self._trans_tokenizer = None
        self._trans_model = None

    @property
    def device(self) -> int:
        if self._device is None:
            _, _, self._device = get_roberta_go_emotions()
        return self._device

    @property
    def classifier(self):
        """Lazily-built text-classification pipeline over the shared RoBERTa GoEmotions model."""
        if self._classifier is None:
            tokenizer, model, device = get_roberta_go_emotions()
            self._device = device
            # top_k=None ensures we get confidence scores for all supported GoEmotions labels
            self._classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                device=device
            )
        return self._classifier

    @property
    def trans_tokenizer(self):
        if self._trans_tokenizer is None:
            self._trans_tokenizer, self._trans_model, self._device = get_nllb_600m()
        return self._trans_tokenizer

    @property
    def trans_model(self):
        if self._trans_model is None:
            self._trans_tokenizer, self._trans_model, self._device = get_nllb_600m()
        return self._trans_model


    def _classify_emotions(self, inference_text: str) -> Dict[str, Any]:
        """
        Run RoBERTa GoEmotions inference and return sorted emotion scores.

        Neutral-suppression rule:
          If 'neutral' is the highest-scoring emotion but its lead over the
          second-highest is less than 0.05, the second-highest emotion is
          promoted as the dominant emotion for downstream reasoning.
          A 'dominant_narrative' field is set to 'Neutral' in that case to
          signal the overall flat/neutral profile to callers.
        """
        predictions = self.classifier(inference_text)
        scores_list = predictions[0]

        emotion_scores = {label: 0.0 for label in self.ALL_GOEMOTIONS}
        for item in scores_list:
            emotion_scores[item['label']] = float(item['score'])

        emotion_scores = dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))

        sorted_emotions = list(emotion_scores.items())
        highest_name, highest_score = sorted_emotions[0] if sorted_emotions else ("", 0.0)
        second_name, second_score = sorted_emotions[1] if len(sorted_emotions) > 1 else ("", 0.0)

        dominant_narrative = highest_name

        if highest_name == "neutral" and (highest_score - second_score) < 0.05:
            # Neutral leads but by too small a margin — promote second-highest
            dominant_emotion = second_name
            dominant_narrative = "Neutral"
        else:
            dominant_emotion = highest_name

        logger.debug(
            f"Dominant emotion: '{dominant_emotion}' "
            f"(highest={highest_name}:{highest_score:.4f}, "
            f"second={second_name}:{second_score:.4f}, "
            f"narrative='{dominant_narrative}')"
        )

        return {
            "dominant_emotion": dominant_emotion,
            "dominant_narrative": dominant_narrative,
            "emotion_scores": emotion_scores,
        }

    def translate_to_english(self, text: str, src_lang_code: str = None) -> str:
        """
        Translate the input text to English using NLLB-200.
        """
        try:
            logger.info(f"Translating text from {src_lang_code} to English...")
            
            # Set source language if we know it
            if src_lang_code and src_lang_code in self.NLLB_LANG_CODES:
                self.trans_tokenizer.src_lang = self.NLLB_LANG_CODES[src_lang_code]
            
            inputs = self.trans_tokenizer(text, return_tensors="pt")
            if self.device == 0:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
            translated_tokens = self.trans_model.generate(
                **inputs, 
                forced_bos_token_id=self.trans_tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=512
            )
            
            translated_text = self.trans_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            logger.info(f"Translation complete: '{translated_text}'")
            return translated_text
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            # Fallback to returning the original text if translation completely fails
            return text

    def analyze_journal(self, text: str) -> Dict[str, Any]:
        """
        Run emotion classification on the input text and return all sorted scores.
        Applies text preprocessing before inference.

        Returns:
            {
                "original_text": str,
                "processed_text": str,
                "emotion_scores": Dict[str, float]
            }
        """
        # Validate empty input
        if not text or not text.strip():
            logger.warning("Empty input received.")
            raise ValueError("Input text cannot be empty.")
            
        try:
            # Preprocess, detect language, translate using TextNormalizer
            norm_result = self.normalizer.normalize(text, translator_fn=self.translate_to_english)
            
            # The inference uses the corrected sentence
            inference_text = norm_result["corrected_sentence"]
            classification = self._classify_emotions(inference_text)
            
            return {
                "original_language": norm_result["original_language"],
                "original_text": norm_result["original_text"],
                "processed_text": norm_result["processed_text"],
                "translated_text": norm_result["translated_text"],
                "corrected_sentence": norm_result["corrected_sentence"],
                "preprocessing_metadata": norm_result["metadata"],
                "dominant_emotion": classification["dominant_emotion"],
                "dominant_narrative": classification["dominant_narrative"],
                "emotion_scores": classification["emotion_scores"],
            }
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

    def get_top_emotions(self, text: str, top_n: int = 5) -> Dict[str, float]:
        """
        Return the highest scoring emotions from the journal text.
        """
        result = self.analyze_journal(text)
        emotion_scores = result["emotion_scores"]
        return dict(list(emotion_scores.items())[:top_n])

    def _calculate_intensity(self, top_scores: List[float]) -> int:
        """
        Calculate Emotional Intensity Score (0-100) based on the strength of top emotions.
        """
        if not top_scores:
            return 0
        
        # Using the average of the top 3 highest scoring emotions to determine intensity
        avg_top = sum(top_scores[:3]) / min(len(top_scores), 3)
        
        # Scale to 0-100
        return min(100, max(0, int(avg_top * 100)))

    def _calculate_diversity(self, scores: Dict[str, float]) -> int:
        """
        Calculate Emotional Diversity Score (0-100) via normalized Shannon entropy
        of the emotion probability distribution.
        """
        probs = [p for p in scores.values() if p > 0]
        if len(scores) <= 1 or not probs:
            return 0

        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(len(scores))

        diversity = (entropy / max_entropy) * 100
        return min(100, max(0, int(diversity)))

    def _calculate_complexity(self, top_scores: List[float]) -> int:
        """
        Calculate Emotional Complexity Score (0-100) based on how mixed the emotional profile is.
        """
        if len(top_scores) < 2 or top_scores[0] == 0:
            return 0
            
        # Complexity is higher if the second top emotion is also very strong (competing emotions)
        complexity = (top_scores[1] / top_scores[0]) * 100
        return min(100, max(0, int(complexity)))

    def generate_visualization_data(self, top_emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate chart-ready JSON data.
        Returns data structures for bar and radar charts for React Native consumption.
        """
        labels = list(top_emotions.keys())
        scores = [round(score, 4) for score in top_emotions.values()]
        
        return {
            "bar_chart": {
                "labels": labels,
                "scores": scores
            },
            "radar_chart": {
                "labels": labels,
                "scores": scores
            }
        }

    def process(self, text: str) -> Dict[str, Any]:
        """
        End-to-end processing of a journal entry, returning structured analytical output.
        """
        logger.info("Processing journal entry for deep emotion analysis...")

        if not text or not text.strip():
            logger.warning("Empty input received.")
            raise ValueError("Input text cannot be empty.")

        norm_result = self.normalizer.normalize(text, translator_fn=self.translate_to_english)
        inference_text = norm_result["corrected_sentence"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            emotion_future = executor.submit(self._classify_emotions, inference_text)
            signal_future = executor.submit(
                self.signal_extractor.extract_text_scores,
                norm_result["original_text"],
                norm_result["processed_text"],
                norm_result["translated_text"],
            )

            classification = emotion_future.result()
            text_signal_scores = signal_future.result()

        psychological_signals = self.signal_extractor.merge_with_emotions(
            text_signal_scores,
            classification["emotion_scores"],
        )

        analysis_result = {
            "original_language": norm_result["original_language"],
            "original_text": norm_result["original_text"],
            "processed_text": norm_result["processed_text"],
            "translated_text": norm_result["translated_text"],
            "corrected_sentence": norm_result["corrected_sentence"],
            "preprocessing_metadata": norm_result["metadata"],
            "dominant_emotion": classification["dominant_emotion"],
            "dominant_narrative": classification["dominant_narrative"],
            "emotion_scores": classification["emotion_scores"],
            "psychological_signals": psychological_signals,
        }
        
        emotion_scores = analysis_result["emotion_scores"]
        preprocess_original = analysis_result["original_text"]
        preprocess_processed = analysis_result["processed_text"]
        translated_text = analysis_result["translated_text"]
        lang_name = analysis_result["original_language"]
        
        # Get top 5 emotions
        top_emotions_full = list(emotion_scores.items())[:5]
        top_emotions = {k: v for k, v in top_emotions_full}
        top_scores_list = [v for k, v in top_emotions_full]
        
        # 1. Dominant Emotion
        dominant_emotion = analysis_result["dominant_emotion"]
        
        # 2, 3, 4. Deep Analysis Scores
        intensity = self._calculate_intensity(top_scores_list)
        diversity = self._calculate_diversity(emotion_scores)
        complexity = self._calculate_complexity(top_scores_list)
        
        # 5. Visualization Data
        vis_data = self.generate_visualization_data(top_emotions)
        
        # Final Structured Output
        result = {
            "original_language": lang_name,
            "original_text": preprocess_original,
            "processed_text": preprocess_processed,
            "translated_text": translated_text,
            "corrected_sentence": analysis_result["corrected_sentence"],
            "preprocessing_metadata": analysis_result["preprocessing_metadata"],
            "dominant_emotion": dominant_emotion,
            "dominant_narrative": analysis_result["dominant_narrative"],
            "emotion_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
            "top_emotions": {k: round(v, 4) for k, v in top_emotions.items()},
            "emotional_intensity": intensity,
            "emotional_diversity": diversity,
            "emotional_complexity": complexity,
            "psychological_signals": psychological_signals,
            "visualization_data": vis_data
        }
        
        logger.info("Deep emotion analysis completed.")
        return result


if __name__ == "__main__":
    print("Initializing EmotionAnalyzer...")
    analyzer = EmotionAnalyzer()
    
    print("\n" + "=" * 55)
    print("   MindAura - Deep Emotion Analysis")
    print("   Write your journal entry below.")
    print("   Type 'exit' or 'quit' to stop.")
    print("=" * 55)

    while True:
        print()
        user_input = input("📝 Enter your journal text: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye! Take care of yourself. 💙")
            break

        if not user_input:
            print("⚠️  Empty input. Please write something.")
            continue

        try:
            results = analyzer.process(user_input)

            print(f"\n{'─' * 55}")
            print(f"🌐 Detected Language   : {results['original_language']}")
            print(f"📝 Original Text       : {results['original_text']}")
            print(f"🔧 Processed Text      : {results['processed_text']}")
            print(f"🔄 Translated Text     : {results['translated_text']}")
            print(f"🎯 Dominant Emotion    : {results['dominant_emotion']}")
            print(f"⚡ Emotional Intensity : {results['emotional_intensity']}/100")
            print(f"🌈 Emotional Diversity : {results['emotional_diversity']}/100")
            print(f"🧩 Emotional Complexity: {results['emotional_complexity']}/100")

            print("\n🏆 Top Emotions:")
            for em, score in results['top_emotions'].items():
                bar = "█" * int(score * 40)
                print(f"   {em:20s} {score:.4f} {bar}")

            print(f"\n📊 Full Analysis JSON:")
            print(json.dumps(results, indent=4, ensure_ascii=False))
            print("─" * 55)

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
