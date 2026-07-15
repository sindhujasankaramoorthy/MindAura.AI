import logging
import re
from typing import Dict, Any, Callable

import langdetect

from langdetect.lang_detect_exception import LangDetectException

from .tanglish_patterns import WORD_REPLACEMENTS, normalize_tanglish_semantics
from .advanced_correction import EmotionPreservingCorrector

logger = logging.getLogger(__name__)

# Constants and Mappings
LANGUAGE_NAMES = {
    'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi', 'te': 'Telugu',
    'ml': 'Malayalam', 'kn': 'Kannada', 'bn': 'Bengali', 'mr': 'Marathi',
    'fr': 'French', 'es': 'Spanish', 'de': 'German', 'ar': 'Arabic',
    'zh-cn': 'Chinese', 'zh-tw': 'Chinese', 'zh': 'Chinese'
}

CHAT_ABBREVIATION_MAP = {
    "tbh": "to be honest", "imo": "in my opinion",
    "idk": "I don't know", "smh": "shaking my head", "ngl": "not going to lie",
    "irl": "in real life", "btw": "by the way", "omg": "oh my god",
    "brb": "be right back", "ty": "thank you", "thx": "thanks",
    "pls": "please", "plz": "please", "rn": "right now", "nvm": "never mind",
    "cuz": "because", "u": "you",
}

TANGLISH_DICT = WORD_REPLACEMENTS

class TextNormalizer:
    """
    Robust text normalizer handling English, Tamil, Tanglish, informal language,
    and typos before emotion inference.
    """

    def __init__(self):
        self.advanced_corrector = EmotionPreservingCorrector()

    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect language using langdetect. Fallback to English on error or low confidence.
        """
        if len(text.split()) < 3:
            return {"language_code": "en", "language_name": "English"}
            
        try:
            lang_code = langdetect.detect(text)
            lang_name = LANGUAGE_NAMES.get(lang_code, "Unknown")
            return {
                "language_code": lang_code,
                "language_name": lang_name
            }
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
            return {"language_code": "en", "language_name": "English"}
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {str(e)}")
            return {"language_code": "en", "language_name": "English"}

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace and tame excessive repeated characters.
        """
        processed = re.sub(r'\s+', ' ', text).strip()
        processed = re.sub(r'(.)\1{2,}', r'\1\1', processed)
        processed = re.sub(r'(?<![\w])i(?![\w])', 'I', processed)
        return processed

    def expand_chat_abbreviations(self, text: str) -> str:
        """
        Expand chat abbreviations (tbh, idk, pls, etc.) to full forms.
        """
        processed = text
        processed = re.sub(r'\bw/o\b', 'without', processed, flags=re.IGNORECASE)
        processed = re.sub(r'\bw/', 'with ', processed)
        
        for pattern, replacement in CHAT_ABBREVIATION_MAP.items():
            processed = re.sub(
                r'\b' + re.escape(pattern) + r'\b',
                replacement, processed, flags=re.IGNORECASE
            )
        return processed

    def expand_tanglish(self, text: str) -> str:
        """
        Map common Tanglish journaling expressions to semantic English.
        """
        return normalize_tanglish_semantics(text)

    def normalize(self, text: str, translator_fn: Callable[[str, str], str] = None) -> Dict[str, str]:
        """
        Orchestrates the full preprocessing pipeline.
        """
        # 1. Raw Input
        logger.info(f"Raw Input: '{text}'")
        original_text = text

        # Document-level routing for Indian languages
        initial_lang_info = self.detect_language(text)
        initial_lang_code = initial_lang_info['language_code']
        
        supported_indic_langs = ['ta', 'hi', 'te', 'ml', 'kn', 'bn', 'mr', 'gu', 'pa', 'ur']
        if initial_lang_code in supported_indic_langs:
            text = self.advanced_corrector.translate_indic(text, initial_lang_code)

        # 2. Text Normalization (clean + chat abbreviations)
        cleaned = self.clean_text(text)
        cleaned = self.expand_chat_abbreviations(cleaned)
        logger.debug(f"After Text Normalization: '{cleaned}'")

        # 3. Detect Named Entities (without masking yet)
        all_entities = []
        if self.advanced_corrector.ner_protection:
            all_entities = self.advanced_corrector.ner_protection.detect_entities(cleaned)

        # 4. Language Detection Layer (Token-level)
        from .language_detector import TokenLanguage
        token_classifications = self.advanced_corrector.language_detector.detect(cleaned, all_entities)
        token_langs_str = ", ".join([f"'{token}': {lang}" for token, lang in token_classifications if token.strip()])
        logger.info(f"Language Detection: [{token_langs_str}]")

        # 5. NER Protection Layer (run only on Unknown tokens)
        ner_entities = []
        protected_text, placeholder_map = cleaned, {}
        
        if self.advanced_corrector.ner_protection:
            
            # Find spans of all tokens classified as Unknown
            unknown_spans = []
            current_idx = 0
            for token, lang in token_classifications:
                start = cleaned.find(token, current_idx)
                if start != -1:
                    end = start + len(token)
                    current_idx = end
                    if lang == TokenLanguage.UNKNOWN:
                        unknown_spans.append((start, end))
                else:
                    # Fallback
                    current_idx += len(token)

            # Filter entities: keep RELATION or entities overlapping with Unknown tokens
            for start, end, ent_type, word in all_entities:
                if ent_type == 'RELATION':
                    ner_entities.append((start, end, ent_type, word))
                else:
                    # Check overlap with any Unknown token span
                    overlaps = False
                    for u_start, u_end in unknown_spans:
                        if not (end <= u_start or start >= u_end):
                            overlaps = True
                            break
                    if overlaps:
                        ner_entities.append((start, end, ent_type, word))
                        
            # Log NER Entities
            if ner_entities:
                ner_log = "\n".join([f"- {ent[2]}: '{ent[3]}' (at index {ent[0]}:{ent[1]})" for ent in ner_entities])
                logger.info(f"NER Entities:\n{ner_log}")
            else:
                logger.info("NER Entities:\n[]")
                
            # Apply protection masking
            protected_text, placeholder_map = self.advanced_corrector.ner_protection.protect(cleaned, ner_entities)
        else:
            logger.info("NER Entities:\n[]")

        # 5. Language-Specific Auto Correction (English + Tanglish paths)
        corrected = self.advanced_corrector.correct(protected_text)
        logger.info(f"After Correction: '{corrected}'")

        # 6. Context Correction (slang replacement)
        context_corrected = self.advanced_corrector.context_correct(corrected)
        logger.debug(f"After Context Correction: '{context_corrected}'")

        # 7. Tanglish Semantic Normalization
        semantic_normalized = normalize_tanglish_semantics(context_corrected)
        logger.info(f"After Semantic Normalization: '{semantic_normalized}'")

        # 8. Negation Recovery
        neg_recovered = self.advanced_corrector.recover_negations(semantic_normalized)
        logger.debug(f"After Negation Recovery: '{neg_recovered}'")

        # 9. Phrase Standardization
        standardized = self.advanced_corrector.standardize_phrases(neg_recovered)
        logger.debug(f"After Phrase Standardization: '{standardized}'")

        # 10. Sentence Reconstruction
        reconstructed = self.advanced_corrector.reconstruct_sentence(standardized)
        logger.debug(f"After Sentence Reconstruction: '{reconstructed}'")

        # 11. NER Restoration
        if self.advanced_corrector.ner_protection:
            final_text = self.advanced_corrector.ner_protection.restore(reconstructed, placeholder_map)
        else:
            final_text = reconstructed

        # Final Emotion Input
        logger.info(f"Final Emotion Input: '{final_text}'")

        # Final Language Detection
        lang_info = self.detect_language(final_text)
        lang_code = lang_info['language_code']
        
        # Translation
        translated_text = final_text
        if lang_code != 'en' and translator_fn is not None:
            try:
                translated_text = translator_fn(final_text, lang_code)
            except Exception as e:
                logger.error(f"Translation failed, falling back to processed text: {e}")
                translated_text = final_text

        return {
            "original_language": initial_lang_info["language_name"],
            "original_text": original_text,
            "processed_text": final_text,
            "translated_text": translated_text
        }
