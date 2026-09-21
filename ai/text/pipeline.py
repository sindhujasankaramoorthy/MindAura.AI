"""
Standalone multilingual text-processing pipeline.

Public entry point: process_text(text) -> dict (see bottom of this file).
Everything above it (TextNormalizer and friends) is the existing
preprocessing engine (language routing, NER protection, spell correction,
Tanglish handling, sentence reconstruction) that process_text() builds on.

This pipeline does NOT import or initialize RoBERTa, Qwen, Ollama, or any
emotion/psychiatrist-reasoning code -- it stops at JSON. Those components
live under temporary/models/ until a later task reconnects them.
"""
import logging
import re
from typing import Dict, Any, Callable

import langdetect

from langdetect.lang_detect_exception import LangDetectException

from ai.text.tanglish.rules import WORD_REPLACEMENTS, normalize_tanglish_semantics
from ai.text.tanglish.detector import is_tanglish_fallback
from ai.text.normalization.correction import EmotionPreservingCorrector
from ai.text.language.script_detector import detect_script
from ai.text.translation.router import translate_to_english as route_translate
from ai.text.expressive.detector import detect_expressive_texting
from ai.text.schemas.output import (
    build_language_info,
    build_success_output,
    build_error_output,
    iso_to_language_name,
)

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
        
        from ai.text.language.boundary import ENGLISH_MISSPELLINGS
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
        
        from ai.text.language.boundary import classify_sentence_language
        routing_info = classify_sentence_language(text)
        logger.debug(f"English confidence: {routing_info['confidence']}")
        
        has_native = bool(re.search(r'[\u0900-\u0DFF]', text))

        pipeline = "UNKNOWN"
        if routing_info['pipeline'] == 'ENGLISH':
            pipeline = "ENGLISH"
        elif has_native:
            pipeline = "NATIVE"
        else:
            # Without a confident English or native-Indic-script signal,
            # the old assumption was always Tanglish (Tamil in Latin
            # script). That breaks for genuinely different Latin-script
            # languages (Spanish, French, ...): with no fastText available
            # to disambiguate, a single coincidental hit against the
            # ~242k-word Tanglish vocabulary isn't enough evidence (short
            # common words collide across languages by chance) -- require a
            # clear majority of words to match before trusting "Tanglish".
            # Otherwise, if langdetect confidently names a specific other
            # language, route as OTHER so it skips Tanglish-specific
            # correction/transliteration and goes straight to the
            # translation router.
            tanglish_vocab = self.advanced_corrector.language_detector.word_classifier.tanglish_words
            words = re.findall(r"[a-zA-Z']+", text.lower())
            tanglish_hit_ratio = (
                sum(1 for w in words if w in tanglish_vocab) / len(words) if words else 0.0
            )

            pipeline = "TANGLISH"
            # langdetect is unreliable on short input (verified: single
            # nonsense words like "anxios"/"minf" get confidently but
            # wrongly guessed as Portuguese/Estonian/etc) -- require at
            # least 3 words before trusting its verdict for this routing
            # decision, matching the same threshold already used by
            # detect_language() elsewhere in this file. Below that, stay on
            # the TANGLISH path so single ambiguous words still reach
            # correct()'s custom-overrides / SymSpell correction instead of
            # skipping it.
            if tanglish_hit_ratio < 0.5 and words and len(words) >= 3:
                try:
                    if langdetect.detect(text) != "en":
                        pipeline = "OTHER"
                except LangDetectException:
                    pass
            
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
        # "OTHER" (a Latin-script language that's neither English nor
        # Tanglish, e.g. Spanish/French) skips this entirely -- running
        # English SymSpell or Tanglish fuzzy-matching over it would mangle
        # words in a language neither of those was ever meant to handle.
        # The translation router (langdetect + NLLB) handles it untouched.
        if pipeline == "OTHER":
            corrected, stages_dict, transliteration_map = protected_text, {}, {}
        else:
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
                        
                from ai.text.tanglish.canonical import normalize_canonical_tanglish
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
            "translated_text": translated_text,
            # Raw per-token (token, language_label) pairs, e.g. ("enaku",
            # "Tanglish") -- used by process_text() to build the language/
            # mixed-language/tanglish_fallback_used summary without
            # re-parsing the stringified `metadata` list above.
            "token_classifications": [(t[0], t[1]) for t in token_classifications],
            "language_pipeline": pipeline,
        }


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────

_normalizer_instance = None

# Generous cap for a journal/blog entry -- guards against pathologically
# large input rather than limiting normal use.
MAX_INPUT_LENGTH = 20000

_KNOWN_TOKEN_LABELS = {
    "English", "Tanglish", "TAMIL", "HINDI", "TELUGU", "MALAYALAM",
    "KANNADA", "BENGALI", "MARATHI", "GUJARATI", "PUNJABI", "URDU",
}


def _get_normalizer() -> "TextNormalizer":
    """Lazy singleton -- TextNormalizer's own models (SymSpell, NER) load
    once per process, never per-call."""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer()
    return _normalizer_instance


def process_text(text: str) -> Dict[str, Any]:
    """
    The Text Module's single public entry point.

        from ai.text.pipeline import process_text
        result = process_text(user_input)

    Handles language detection (including Tamil-via-Latin/"Tanglish"
    fallback), routes translation through IndicTrans2 (Indian languages) or
    NLLB-200 (everything else), reconstructs a contextual English sentence,
    detects expressive alphabet extension, and returns one JSON-serializable
    dict. Never raises -- always returns a dict with a "status" field.

    Does NOT touch RoBERTa/Qwen/Ollama/emotion analysis/psychiatrist
    reasoning; the pipeline stops at JSON.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return build_error_output(
            text if isinstance(text, str) else "",
            "empty_input",
            "Input text is empty or invalid.",
        )

    if len(text) > MAX_INPUT_LENGTH:
        return build_error_output(
            text,
            "input_too_long",
            f"Input exceeds the maximum supported length of {MAX_INPUT_LENGTH} characters.",
        )

    try:
        normalizer = _get_normalizer()
        norm_result = normalizer.normalize(text, translator_fn=route_translate)
    except Exception as e:
        logger.exception("Text pipeline processing failed.")
        return build_error_output(text, "processing_failed", str(e))

    try:
        script = detect_script(text)
        token_classifications = norm_result["token_classifications"]

        if norm_result.get("language_pipeline") == "OTHER":
            # The pipeline already determined this is a genuinely different
            # Latin-script language (not English, not Tanglish -- e.g.
            # Spanish/French). Per-word classification isn't reliable here
            # (no fastText available to tell "unknown Spanish word" apart
            # from "unknown Tanglish word"), so trust the whole-text
            # langdetect verdict instead of the token classifier.
            fallback_name = None
            try:
                fallback_name = iso_to_language_name(langdetect.detect(text))
            except LangDetectException:
                pass
            language_info = build_language_info([], script, False, fallback_name)
        else:
            tanglish_used = is_tanglish_fallback(token_classifications)

            # If the token classifier didn't recognize any language at all,
            # fall back to a whole-text language guess so `language` isn't
            # just empty.
            fallback_language_name = None
            if not any(label in _KNOWN_TOKEN_LABELS for _token, label in token_classifications):
                try:
                    if len(text.split()) >= 2:
                        fallback_language_name = iso_to_language_name(langdetect.detect(text))
                except LangDetectException:
                    pass

            language_info = build_language_info(
                token_classifications, script, tanglish_used, fallback_language_name
            )
        expressive = detect_expressive_texting(text)

        contextual_final_sentence = norm_result["translated_text"]
        if language_info["mixed_language"] or language_info["tanglish_fallback_used"]:
            # Chunk-substitution reconstruction can leave code-switched
            # sentences grammatically awkward even when the meaning is
            # right (e.g. "Feel Ram told me I very difficult."). Run an
            # NLLB English->English fluency pass to smooth it -- only for
            # this case, not clean single-language input, since translating
            # already-fluent English through NLLB has been observed to
            # hallucinate content that was never said. Falls back to the
            # pre-fluency-pass sentence if this fails or returns nothing.
            try:
                from ai.text.translation.nllb import translate_to_english as _fluency_pass
                fluent = _fluency_pass(contextual_final_sentence, "en")
                if fluent and fluent.strip():
                    contextual_final_sentence = fluent
            except Exception as e:
                logger.warning(f"Fluency pass failed, keeping pre-pass sentence: {e}")

        return build_success_output(
            raw_input=text,
            contextual_final_sentence=contextual_final_sentence,
            expressive_texting=expressive,
            language=language_info,
        )
    except Exception as e:
        logger.exception("Failed to build text pipeline output.")
        return build_error_output(text, "output_construction_failed", str(e))
