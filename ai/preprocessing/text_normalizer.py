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
        Normalize whitespace, tame excessive repeated characters, and fix specific english misspellings.
        """
        processed = re.sub(r'\s+', ' ', text).strip()
        processed = re.sub(r'(.)\1{2,}', r'\1\1', processed)
        processed = re.sub(r'(?<![\w])i(?![\w])', 'I', processed)
        
        from .language_boundary import ENGLISH_MISSPELLINGS
        for k, v in ENGLISH_MISSPELLINGS.items():
            processed = re.sub(r'\b' + re.escape(k) + r'\b', v, processed, flags=re.IGNORECASE)
            
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

        logger.debug("Raw language detection input:")
        
        from .language_boundary import classify_sentence_language
        routing_info = classify_sentence_language(text)
        logger.debug(f"English confidence: {routing_info['confidence']}")
        
        has_native = bool(re.search(r'[\u0900-\u0DFF]', text))
        
        pipeline = "UNKNOWN"
        if routing_info['pipeline'] == 'ENGLISH':
            pipeline = "ENGLISH"
        elif has_native:
            pipeline = "NATIVE"
        else:
            pipeline = "TANGLISH"
            
        logger.debug(f"Selected language pipeline: {pipeline}")

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
        token_classifications = self.advanced_corrector.language_detector.detect(protected_text, [], pipeline=pipeline)
        
        # Calculate Sentence Analysis stats
        eng_count = sum(1 for pred in token_classifications if pred[1] == 'English' and pred[0].strip())
        tan_count = sum(1 for pred in token_classifications if pred[1] == 'Tanglish' and pred[0].strip())
        total_valid = eng_count + tan_count or 1
        
        sentence_analysis = (
            "Sentence Analysis\n\n"
            "Sentence Language Confidence:\n"
            f"English: {(eng_count/total_valid)*100:.1f}%\n"
            f"Tanglish: {(tan_count/total_valid)*100:.1f}%\n"
            f"Mixed: {(100 if eng_count > 0 and tan_count > 0 else 0):.1f}%\n\n"
            "Token Analysis\n"
        )
        
        token_analysis_lines = []
        for pred in token_classifications:
            if not pred[0].strip():
                continue
            token_analysis_lines.append(
                f"Token: {pred[0]}\n"
                f"Language: {pred[1]}\n"
                f"Confidence: {getattr(pred, 'confidence', 1.0)}\n"
                f"Reason: {getattr(pred, 'reason', 'Unknown')}\n"
            )
            
        logger.info(sentence_analysis + "\n".join(token_analysis_lines))

        initial_lang_info = self.detect_language(text)
        
        # 5. Pipeline execution (English Spell Correction, Tanglish Autocorrect -> IndicXlit -> AI4Bharat)
        corrected, stages_dict, transliteration_map = self.advanced_corrector.correct(protected_text, pipeline=pipeline)
        
        logger.info("After Autocorrect:\n" + stages_dict.get("autocorrect", corrected))
        logger.info("After Phrase Normalization:\n" + stages_dict.get("phrase", corrected))
        logger.info("After Canonical Normalization:\n" + stages_dict.get("canonical", corrected))
        logger.info("After IndicXlit:\n" + stages_dict.get("indic", corrected))
        logger.info(f"After Correction (Final string from chunk processing): '{corrected}'")

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
                
        logger.info(f"After Tamil→English Translation:\n{translated_text}")

        # Build Metadata
        metadata = []
        tanglish_meta_idx = 0
        for token_pred in token_classifications:
            token, lang = token_pred
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
                    
            # Determine Expressive Texting (Elongation) metadata
            m = re.search(r'(.)\1{2,}', token)
            is_english_elongation = bool(m)
            elongation_character = m.group(1) if m else None
            elongation_count = 0
            if m:
                m2 = re.search(f'({re.escape(elongation_character)}{{3,}})', token)
                if m2:
                    elongation_count = len(m2.group(1))
            
            # Get corrected token simulation to see if it was modified
            corrected_token = token
            was_corrected = False
            
            # Use TokenLanguage string representations
            lang_str = str(lang)
            translated_token = token
            canonical_token = token
            transliterated_token = token
            
            if lang_str == 'TokenLanguage.ENGLISH' or lang_str == 'English':
                cands = self.advanced_corrector.correct_english_token(token)
                if cands and cands[0] != token:
                    corrected_token = cands[0]
                    was_corrected = True
                canonical_token = corrected_token
                transliterated_token = corrected_token
            elif lang_str == 'TokenLanguage.TANGLISH' or lang_str == 'Tanglish':
                cand = self.advanced_corrector.correct_tanglish_token(token)
                if cand != token:
                    corrected_token = cand
                    was_corrected = True
                        
                from ai.preprocessing.tanglish_canonical import normalize_canonical_tanglish
                canonical_token = normalize_canonical_tanglish(corrected_token)
                transliterated_token = transliteration_map.get(canonical_token, canonical_token)
                translated_token = normalize_tanglish_semantics(corrected_token)
            
            elongation_base_word = corrected_token if is_english_elongation else None
                    
            meta_obj = {
                "original_token": token,
                "corrected_token": corrected_token,
                "normalized_token": token,
                "canonical_token": canonical_token,
                "transliterated_token": transliterated_token,
                "translated_token": translated_token,
                "final_token": corrected_token, # Approximation of final token
                "detected_language": lang_str,
                "was_corrected": was_corrected,
                "is_english_elongation": is_english_elongation,
                "elongation_character": elongation_character,
                "elongation_count": elongation_count,
                "elongation_base_word": elongation_base_word,
                # Additional fields to preserve compatibility
                "is_ner_placeholder": is_ner_placeholder,
                "ner_type": ner_type,
                "ner_original_word": ner_original_word,
                # New fields for Context-Aware Detector
                "language_confidence": getattr(token_pred, 'confidence', 1.0),
                "language_reason": getattr(token_pred, 'reason', "Unknown")
            }
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
