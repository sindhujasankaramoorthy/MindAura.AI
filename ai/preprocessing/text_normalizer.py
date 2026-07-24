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

        initial_lang_info = self.detect_language(text)
        initial_lang_code = initial_lang_info['language_code']

        # 2. Text Normalization (clean + chat abbreviations)
        cleaned = self.clean_text(text)
        cleaned = self.expand_chat_abbreviations(cleaned)
        logger.debug(f"After Text Normalization: '{cleaned}'")

        # 3. Detect Named Entities & Protect (Masking)
        all_entities = []
        protected_text, placeholder_map = cleaned, {}
        
        if self.advanced_corrector.ner_protection:
            all_entities = self.advanced_corrector.ner_protection.detect_entities(cleaned)
            
            # Log NER Entities
            if all_entities:
                ner_log = "\n".join([f"- {ent[2]}: '{ent[3]}' (at index {ent[0]}:{ent[1]})" for ent in all_entities])
                logger.info(f"NER Entities:\n{ner_log}")
            else:
                logger.info("NER Entities:\n[]")
                
            # Apply protection masking FIRST
            protected_text, placeholder_map = self.advanced_corrector.ner_protection.protect(cleaned, all_entities)
        else:
            logger.info("NER Entities:\n[]")

        # 4. Token-Level Language Detection (runs on protected text)
        from .language_detector import TokenLanguage
        token_classifications = self.advanced_corrector.language_detector.detect(protected_text, [])
        token_langs_str = ", ".join([f"'{token}': {lang}" for token, lang in token_classifications if token.strip()])
        logger.info(f"Language Detection: [{token_langs_str}]")

        # 5. Pipeline execution (English Spell Correction, Tanglish Autocorrect -> IndicXlit -> AI4Bharat)
        corrected = self.advanced_corrector.correct(protected_text)
        logger.info(f"After Correction: '{corrected}'")

        # 8. Context Correction (slang replacement)
        context_corrected = self.advanced_corrector.context_correct(corrected)
        logger.debug(f"After Context Correction: '{context_corrected}'")

        # 9. Tanglish Semantic Normalization (applies English mappings)
        semantic_normalized = normalize_tanglish_semantics(context_corrected)
        logger.info(f"After Semantic Normalization: '{semantic_normalized}'")

        # 10. Negation Recovery
        neg_recovered = self.advanced_corrector.recover_negations(semantic_normalized)
        logger.debug(f"After Negation Recovery: '{neg_recovered}'")

        # 11. Phrase Standardization
        standardized = self.advanced_corrector.standardize_phrases(neg_recovered)
        logger.debug(f"After Phrase Standardization: '{standardized}'")

        # 12. Sentence Reconstruction
        reconstructed = self.advanced_corrector.reconstruct_sentence(standardized)
        logger.debug(f"After Sentence Reconstruction: '{reconstructed}'")

        # 13. Restore Protected NER Entities
        if self.advanced_corrector.ner_protection:
            final_text = self.advanced_corrector.ner_protection.restore(reconstructed, placeholder_map)
        else:
            final_text = reconstructed

        # Final Emotion Input
        logger.info(f"Final Emotion Input: '{final_text}'")

        # Final Language Detection
        lang_info = self.detect_language(final_text)
        lang_code = lang_info['language_code']
        
        # Translation Fallback (NLLB)
        translated_text = final_text
        normalization_type = None
        if lang_code != 'en' and translator_fn is not None:
            try:
                translated_text = translator_fn(final_text, lang_code)
                normalization_type = "indic_translation"
            except Exception as e:
                logger.error(f"Translation failed, falling back to processed text: {e}")
                translated_text = final_text

        # Build Metadata
        metadata = []
        tanglish_meta_idx = 0
        for token, lang in token_classifications:
            if not token.strip():
                continue
                
            is_punct = not token.replace("'", "").isalnum()
            ner_type = None
            ner_original_word = None
            is_ner_placeholder = False
            
            # Identify NER matches
            if token in placeholder_map:
                is_ner_placeholder = True
                ner_original_word = placeholder_map[token]
                m = re.match(r"<([A-Z_]+)_\d+>", token)
                if m:
                    ner_type = m.group(1)
                    
            # Check for elongation in original text (rudimentary check on token)
            has_elongation = bool(re.search(r'(.)\1{2,}', token))
            
            # Get corrected token simulation to see if it was modified
            corrected_token = token
            was_corrected = False
            
            # Use TokenLanguage string representations
            lang_str = str(lang)
            tanglish_info = {}
            translated_token = token
            confidence = 100.0
            
            if lang_str == 'TokenLanguage.ENGLISH' or lang_str == 'English':
                cands = self.advanced_corrector.correct_english_token(token)
                if cands and cands[0] != token:
                    corrected_token = cands[0]
                    was_corrected = True
            elif lang_str == 'TokenLanguage.TANGLISH' or lang_str == 'Tanglish':
                cand = self.advanced_corrector.correct_tanglish_token(token)
                if cand != token:
                    corrected_token = cand
                    was_corrected = True
                        
                # Compute translated token for Tanglish
                translated_token = normalize_tanglish_semantics(corrected_token)
                confidence = tanglish_info.get("confidence", 100.0)
                    
            meta_obj = {
                "original_token": token,
                "detected_language": lang_str,
                "corrected_token": corrected_token,
                "translated_token": translated_token,
                "confidence": confidence,
                "normalized_token": token,
                "final_token": corrected_token, # Approximation of final token
                "language": lang_str,
                "is_ner_placeholder": is_ner_placeholder,
                "ner_type": ner_type,
                "ner_original_word": ner_original_word,
                "normalization_type": normalization_type,
                "has_elongation": has_elongation,
                "has_punctuation": is_punct,
                "was_corrected": was_corrected,
            }
            # Merge extra tanglish info if any
            for k, v in tanglish_info.items():
                if k not in meta_obj:
                    meta_obj[f"tanglish_{k}"] = v
                    
            metadata.append(meta_obj)

        return {
            "corrected_sentence": translated_text,
            "metadata": metadata,
            # Backward compatibility fields
            "original_language": initial_lang_info["language_name"],
            "original_text": original_text,
            "processed_text": final_text,
            "translated_text": translated_text
        }
