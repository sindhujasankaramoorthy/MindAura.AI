"""
Phase 3 — Advanced Emotion-Preserving Text Correction

Multi-stage preprocessing layer that sits between Tanglish normalization
and RoBERTa GoEmotions inference.

Pipeline order:
    1. Language Detection
    2. NER Protection
    3. Context-Aware Auto Correction
    4. Tanglish Semantic Normalization
    5. Negation Recovery
    6. Sentence Reconstruction
"""
import re
import importlib.resources
import logging
import itertools
from symspellpy import SymSpell, Verbosity
from typing import Set, Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .tanglish_patterns import (
    normalize_tanglish_semantics, 
    WORD_REPLACEMENTS
)
from .language_detector import LanguageDetector, TokenLanguage

logger = logging.getLogger(__name__)

# Levenshtein distance helper
def LevenshteinDistance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return LevenshteinDistance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

# Generalized phonetic tanglish mapping and distance helpers
def tanglish_key(word: str) -> str:
    w = word.lower()
    # Collapse consecutive duplicate letters
    w = re.sub(r'(.)\1+', r'\1', w)
    # Common Romanized Tamil spelling variants mapped phonetically
    w = w.replace('dh', 'd').replace('th', 'd').replace('t', 'd')
    w = w.replace('zh', 'l').replace('r', 'l')
    w = w.replace('w', 'v')
    w = w.replace('sh', 's').replace('z', 's')
    w = w.replace('g', 'k').replace('h', '')
    # Vowels mappings
    w = w.replace('oo', 'u').replace('ee', 'i').replace('aa', 'a').replace('ae', 'e').replace('ai', 'e')
    # Reduce vowels to simplify mappings (e.g. u -> i, o -> u, e -> i)
    w = w.replace('o', 'u').replace('u', 'i').replace('e', 'i')
    return w

def tanglish_distance(w1: str, w2: str) -> int:
    return LevenshteinDistance(tanglish_key(w1), tanglish_key(w2))

# ──────────────────────────────────────────────────────────────────────
# A. NEGATION RECOVERY MAP — Configurable & Extensible
# ──────────────────────────────────────────────────────────────────────
NEGATION_RECOVERY_MAP: Dict[str, str] = {
    "cnt": "can't",
    "cant": "can't",
    "dont": "don't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "aint": "ain't",
    "mustnt": "mustn't",
    "neednt": "needn't",
    "mightnt": "mightn't",
    "idnt": "I don't",
    "dnt": "don't",
    "dhnt": "don't",
    "cnat": "can't",
    "cannt": "can't",
    "donot": "do not",
    "cannot": "cannot",
    "wnt": "won't",
    "shldnt": "shouldn't",
    "cldnt": "couldn't",
    "wldnt": "wouldn't",
    "dsnt": "doesn't",
    "nvr": "never",
}

_NEGATION_PAIRS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
    for k, v in sorted(NEGATION_RECOVERY_MAP.items(), key=lambda x: len(x[0]), reverse=True)
]

# ──────────────────────────────────────────────────────────────────────
# C. EMOTIONAL VOCABULARY PROTECTION
# ──────────────────────────────────────────────────────────────────────
PROTECTED_EMOTIONS: Set[str] = {
    "sad", "sadness", "lonely", "alone", "blank", "empty",
    "fear", "afraid", "scared", "anxious", "panic", "stress", "stressed",
    "overthink", "overthinking", "hurt", "broken", "cry", "crying",
    "angry", "frustrated", "disturbed", "upset", "hopeless", "helpless",
    "worthless", "tired", "exhausted", "burnout", "burnt", "burned",
    "numb", "restless", "confused", "lost", "depressed", "worried",
    "grief", "grieving", "shame", "guilty", "guilt", "regret",
    "miserable", "suffering", "pain", "painful", "trauma", "traumatic",
    "insecure", "insecurity", "nervous", "dread", "agony", "anguish",
    "bitter", "resentful", "resentment", "jealous", "jealousy", "envy",
    "desperate", "despair", "gloomy", "melancholy", "sorrow", "sorrowful",
    "vulnerable", "weak", "fatigue", "fatigued", "overwhelmed", "overwhelm",
}

PROTECTED_NEGATIONS: Set[str] = {
    "not", "don't", "doesn't", "can't", "cannot", "won't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "didn't", "couldn't", "shouldn't", "wouldn't", "ain't", "never",
    "no", "none", "nor", "neither", "nobody", "nothing", "nowhere",
    "mustn't", "needn't", "mightn't",
}

# Protected words — SymSpell must never touch these
PROTECTED_WORDS: Set[str] = PROTECTED_NEGATIONS | PROTECTED_EMOTIONS | {'person', 'gpe', 'org', 'relation'} | set(NEGATION_RECOVERY_MAP.keys())

# Slang adjustments (minimal fallback rules)
CONTEXT_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bwanna\b", re.IGNORECASE), "want to"),
    (re.compile(r"\bgonna\b", re.IGNORECASE), "going to"),
    (re.compile(r"\bgotta\b", re.IGNORECASE), "got to"),
    (re.compile(r"\bkinda\b", re.IGNORECASE), "kind of"),
    (re.compile(r"\bsorta\b", re.IGNORECASE), "sort of"),
    (re.compile(r"\blemme\b", re.IGNORECASE), "let me"),
    (re.compile(r"\bgimme\b", re.IGNORECASE), "give me"),
    (re.compile(r"\bdunno\b", re.IGNORECASE), "don't know"),
]

PSYCHOLOGICAL_STANDARDIZATION: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bmind\s+disturbed\b", re.IGNORECASE), "mentally disturbed"),
    (re.compile(r"\bmentally\s+exhausted\b", re.IGNORECASE), "mental fatigue"),
]


CUSTOM_OVERRIDES: Dict[str, str] = {
    "thnkng": "thinking",
    "thnking": "thinking",
    "thinkng": "thinking",
    "thnk": "think",
    "feling": "feeling",
    "feelin": "feeling",
    "feelng": "feeling",
    "undrstnd": "understand",
    "undrstd": "understood",
    "smthing": "something",
    "nthing": "nothing",
    "evrthing": "everything",
    "wrkng": "working",
    "hlping": "helping",
    "tlking": "talking",
    "sleping": "sleeping",
    "brking": "breaking",
    # Common emotional misspellings (short words misclassified by Tanglish heuristic)
    "minf": "mind",
    "mnd": "mind",
    "lonley": "lonely",
    "lonley": "lonely",
    "lonly": "lonely",
    "scard": "scared",
    "scrd": "scared",
    "anxios": "anxious",
    "anxous": "anxious",
    "ovrthinking": "overthinking",
    "overthinkng": "overthinking",
    "ovrthnking": "overthinking",
    "depresed": "depressed",
    "depressd": "depressed",
    "frustrated": "frustrated",
    "wurried": "worried",
    "wrried": "worried",
    "lonly": "lonely",
    "painfull": "painful",
    "emty": "empty",
    "emtpy": "empty",
}


class EmotionPreservingCorrector:
    """
    Advanced text correction layer preserving psychological and emotional context.
    """
    EDIT_DISTANCE_RATIO_THRESHOLD = 0.6

    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Load the default English frequency dictionary
        try:
            ref = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            with importlib.resources.as_file(ref) as path:
                self.sym_spell.load_dictionary(str(path), term_index=0, count_index=1)
        except AttributeError:
            import pkg_resources
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        # Boost protected words to max frequency
        for word in PROTECTED_WORDS:
            self.sym_spell.create_dictionary_entry(word, 999_999_999)

        # Boost negation recovery map keys so they are recognized as exact matches
        # and won't be corrected by SymSpell (they will be expanded by recover_negations)
        for key in NEGATION_RECOVERY_MAP:
            self.sym_spell.create_dictionary_entry(key, 999_999_999)

        # Boost common emotion/mental-health words that may be absent from standard dict
        _extra_emotional = [
            "overthinking", "overthink", "burnout", "helpless", "hopeless",
            "worthless", "numb", "restless", "depressed", "anxious",
            "fatigued", "overwhelmed", "insecure", "traumatic", "thinking",
            "lonely", "scared", "worried", "afraid", "miserable",
            "melancholy", "anguish", "mindfulness", "mind", "mental",
            "overthink", "overwhelm", "panic", "grief", "sorrow",
        ]
        for w in _extra_emotional:
            self.sym_spell.create_dictionary_entry(w, 500_000_000)

        # Register canonical Tanglish roots in SymSpell so they are
        # "known" and never substituted as English typo corrections.
        # The LanguageDetector will still correctly route them as Tanglish.
        from .tanglish_patterns import WORD_REPLACEMENTS
        for root in WORD_REPLACEMENTS:
            self.sym_spell.create_dictionary_entry(root, 1)  # low freq = known but not English

        # Pre-SymSpell overrides
        self._custom_overrides = CUSTOM_OVERRIDES

        # Context Model (MLM)
        model_name = "gpt2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.mlm_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.mlm_model.eval()
            self._has_context_model = True
        except Exception as e:
            logger.error(f"Failed to load MLM for context evaluation: {e}")
            self._has_context_model = False

        # IndicTrans2 Model
        try:
            from transformers import AutoModelForSeq2SeqLM
            from IndicTransToolkit import IndicProcessor
            indic_model_name = "ai4bharat/indictrans2-indic-en-1B"
            self.indic_tokenizer = AutoTokenizer.from_pretrained(indic_model_name, trust_remote_code=True)
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained(indic_model_name, trust_remote_code=True)
            self.indic_model.eval()
            self.indic_processor = IndicProcessor(inference=True)
            self._has_indic_model = True
        except Exception as e:
            logger.error(f"Failed to load IndicTrans2 model: {e}")
            self._has_indic_model = False

        # NER Protection Layer
        try:
            from ai.preprocessing.ner_protection import NERProtection
            self.ner_protection = NERProtection()
        except Exception as e:
            logger.error(f"Failed to initialize NER Protection: {e}")
            self.ner_protection = None

        # Language Detector
        self.language_detector = LanguageDetector(
            self.sym_spell, 
            PROTECTED_WORDS, 
            set(NEGATION_RECOVERY_MAP.keys())
        )

    def _score_sentence(self, sentence: str) -> float:
        if not self._has_context_model:
            return float('inf')
        
        inputs = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.mlm_model(inputs['input_ids'], labels=inputs['input_ids'])
            return outputs.loss.item() * inputs['input_ids'].size(1)

    def translate_indic(self, text: str, src_lang: str) -> str:
        if not self._has_indic_model:
            return text
        
        lang_map = {
            'ta': 'tam_Taml', 'hi': 'hin_Deva', 'te': 'tel_Telu', 
            'ml': 'mal_Mlym', 'kn': 'kan_Knda', 'bn': 'ben_Beng', 
            'mr': 'mar_Deva', 'gu': 'guj_Gujr', 'pa': 'pan_Guru', 
            'ur': 'urd_Arab'
        }
        reverse_map = {
            'TAMIL': 'tam_Taml', 'HINDI': 'hin_Deva', 'TELUGU': 'tel_Telu',
            'MALAYALAM': 'mal_Mlym', 'KANNADA': 'kan_Knda', 'BENGALI': 'ben_Beng',
            'MARATHI': 'mar_Deva', 'GUJARATI': 'guj_Gujr', 'PUNJABI': 'pan_Guru',
            'URDU': 'urd_Arab'
        }
        mapped_src = lang_map.get(src_lang) or reverse_map.get(src_lang, 'hin_Deva')

        tgt_lang = "eng_Latn"
        try:
            batch = self.indic_processor.preprocess_batch([text], src_lang=mapped_src, tgt_lang=tgt_lang)
            inputs = self.indic_tokenizer(batch, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.indic_model.generate(**inputs, max_new_tokens=512)
            decoded = self.indic_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return self.indic_processor.postprocess_batch(decoded, lang=tgt_lang)[0]
        except Exception as e:
            logger.error(f"Indic translation error: {e}")
            return text

    def recover_negations(self, text: str) -> str:
        processed = text
        for pattern, replacement in _NEGATION_PAIRS:
            processed = pattern.sub(replacement, processed)
        return processed

    def correct_english_token(self, token: str) -> List[str]:
        """
        Applies English spelling correction and returns candidates.
        """
        word = token
        word_lower = word.lower()

        if (len(word) <= 1 or 
            word_lower in ("i", "a", "im", "ok", "no") or 
            "'" in word or 
            word_lower in PROTECTED_WORDS):
            return [word]

        if word_lower in self._custom_overrides:
            corrected = self._custom_overrides[word_lower]
            if word.istitle(): corrected = corrected.title()
            elif word.isupper(): corrected = corrected.upper()
            return [corrected]

        # Check if word is already correct
        suggestions = self.sym_spell.lookup(
            word_lower, Verbosity.TOP, max_edit_distance=2, include_unknown=True
        )
        if not suggestions or suggestions[0].term == word_lower:
            return [word]

        # Get correction candidates
        best_suggestions = self.sym_spell.lookup(
            word_lower, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False
        )
        
        valid_candidates = []
        for s in best_suggestions[:3]:
            ratio = s.distance / max(len(word_lower), 1)
            if ratio <= self.EDIT_DISTANCE_RATIO_THRESHOLD:
                cand = s.term
                if word.istitle(): cand = cand.title()
                elif word.isupper(): cand = cand.upper()
                valid_candidates.append(cand)

        if valid_candidates:
            if word not in valid_candidates:
                valid_candidates.append(word)
            return valid_candidates
        
        return [word]

    def correct_tanglish_token(self, token: str, candidate_pool: Set[str] = None) -> str:
        """
        Applies generalized Tanglish spelling correction using phonetic key distance.
        """
        word = token
        word_lower = word.lower()

        if candidate_pool is None:
            from .tanglish_patterns import WORD_REPLACEMENTS
            candidate_pool = set(WORD_REPLACEMENTS.keys())

        # Find best canonical root
        best_root = None
        min_dist = 99
        for root in candidate_pool:
            dist = tanglish_distance(word_lower, root)
            if dist < min_dist:
                min_dist = dist
                best_root = root

        # Apply length-dependent edit distance constraints to avoid matching short words
        max_allowed_dist = 2
        if len(word_lower) <= 2:
            max_allowed_dist = 0
        elif len(word_lower) <= 3:
            max_allowed_dist = 1

        if min_dist <= max_allowed_dist:
            corrected = best_root
            if word.istitle(): corrected = corrected.title()
            elif word.isupper(): corrected = corrected.upper()
            logger.debug(f"[Tanglish Correction Path] Corrected '{word}' -> '{corrected}' (distance: {min_dist})")
            return corrected

        return word

    def correct(self, text: str) -> str:
        """
        Applies Context-Aware Auto Correction with separate English and Tanglish paths.
        Negation recovery runs first to protect negation contractions from SymSpell corruption.
        """
        # Step 0: Recover negations FIRST (before spell correction corrupts them)
        text = self.recover_negations(text)

        classifications = self.language_detector.detect(text, [])
        candidate_options = []
        has_misspelling = False

        for token, lang in classifications:
            if not token:
                candidate_options.append([token])
                continue

            word = token
            word_lower = word.lower()

            # 1. Unknown tokens or Delimiters/Placeholders (preserved as-is)
            if lang == TokenLanguage.UNKNOWN:
                candidate_options.append([word])
                continue

            # 1b. Custom overrides applied universally (before language-path branching)
            if word_lower in self._custom_overrides:
                corrected = self._custom_overrides[word_lower]
                if word.istitle(): corrected = corrected.title()
                elif word.isupper(): corrected = corrected.upper()
                candidate_options.append([corrected])
                continue

            # 2. English Correction Path
            if lang == TokenLanguage.ENGLISH:
                cands = self.correct_english_token(word)
                if len(cands) > 1 or cands[0] != word:
                    has_misspelling = True
                candidate_options.append(cands)

            # 3. Tanglish Correction Path
            elif lang == TokenLanguage.TANGLISH:
                corrected = self.correct_tanglish_token(word)
                candidate_options.append([corrected])

            # 4. Indian Language — defer translation: store a sentinel tuple so
            # consecutive Indic tokens can be grouped and translated together
            # as a single string (preserving sentence context for IndicTrans2).
            elif lang in ['TAMIL', 'HINDI', 'TELUGU', 'MALAYALAM', 'KANNADA',
                          'BENGALI', 'MARATHI', 'GUJARATI', 'PUNJABI', 'URDU']:
                candidate_options.append([(word, lang, True)])

        # ── Indic Chunk Translation ─────────────────────────────────────────
        # Group consecutive Indic-sentinel entries (including any UNKNOWN
        # delimiter entries between them) into chunks and translate each chunk
        # as a single string so IndicTrans2 receives full sentence context
        # instead of isolated words.
        merged_options: list = []
        i = 0
        while i < len(candidate_options):
            slot = candidate_options[i]

            # Detect an Indic sentinel slot
            if (slot and isinstance(slot[0], tuple) and
                    len(slot[0]) == 3 and slot[0][2] is True):
                chunk_lang = slot[0][1]
                chunk_parts: list = [slot[0][0]]  # first Indic word
                j = i + 1

                # Absorb subsequent slots that are either:
                #   a) another Indic token of the SAME language, or
                #   b) a pure delimiter (UNKNOWN) that sits between Indic tokens
                #      of the same language (preserves spaces/punctuation).
                while j < len(candidate_options):
                    next_slot = candidate_options[j]

                    # Next slot is another Indic sentinel of the same language
                    if (next_slot and isinstance(next_slot[0], tuple) and
                            len(next_slot[0]) == 3 and next_slot[0][2] is True and
                            next_slot[0][1] == chunk_lang):
                        chunk_parts.append(next_slot[0][0])
                        j += 1
                        continue

                    # Next slot is a delimiter — include it only when the slot
                    # after it is still the same Indic language.
                    if (next_slot and len(next_slot) == 1 and
                            isinstance(next_slot[0], str)):
                        if j + 1 < len(candidate_options):
                            peek = candidate_options[j + 1]
                            if (peek and isinstance(peek[0], tuple) and
                                    len(peek[0]) == 3 and peek[0][2] is True and
                                    peek[0][1] == chunk_lang):
                                chunk_parts.append(next_slot[0])
                                j += 1
                                continue
                    break

                # Translate the entire chunk as one string so the model receives
                # full sentence/paragraph context rather than isolated words.
                full_chunk = "".join(chunk_parts)
                translated_chunk = self.translate_indic(full_chunk, chunk_lang)
                merged_options.append([translated_chunk])
                i = j
            else:
                merged_options.append(slot)
                i += 1

        if not has_misspelling or not self._has_context_model:
            return "".join([opts[0] for opts in merged_options])

        # MLM scoring for English candidates
        combinations = list(itertools.islice(itertools.product(*merged_options), 100))
        best_sentence = text
        best_score = float('inf')

        for combo in combinations:
            candidate_sentence = "".join(combo)
            score = self._score_sentence(candidate_sentence)
            if score < best_score:
                best_score = score
                best_sentence = candidate_sentence

        return best_sentence

    def context_correct(self, text: str) -> str:
        processed = text
        for pattern, replacement in CONTEXT_REPLACEMENTS:
            processed = pattern.sub(replacement, processed)
        return processed

    def standardize_phrases(self, text: str) -> str:
        processed = text
        for pattern, replacement in PSYCHOLOGICAL_STANDARDIZATION:
            processed = pattern.sub(replacement, processed)
        return processed

    def reconstruct_sentence(self, text: str) -> str:
        processed = text.strip()
        processed = re.sub(r'\s+', ' ', processed)

        _sentence_starters = [
            r'I cannot', r'I can\'t', r'I don\'t', r'I feel', r'I am',
            r'I have', r'I need', r'I want', r'I will', r'I won\'t',
            r'My mind', r'My head',
            r'need\b', r'want\b', r'everything\b',
        ]
        for starter in _sentence_starters:
            pattern = re.compile(
                r'(?<=[a-z])\s+(' + starter + r'\b)',
                re.IGNORECASE
            )
            processed = pattern.sub(r'. \1', processed)

        processed = re.sub(r'\s*,\s*', ', ', processed)
        processed = re.sub(r'\s*\.\s*', '. ', processed)
        processed = re.sub(r'\.{2,}', '.', processed)
        processed = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'I', processed)

        if processed:
            processed = processed[0].upper() + processed[1:]

        def _cap_after_punct(m: re.Match) -> str:
            return m.group(1) + ' ' + m.group(2).upper()

        processed = re.sub(r'([.!?])\s+([a-z])', _cap_after_punct, processed)
        processed = processed.rstrip()
        if processed and processed[-1] not in '.!?':
            processed += '.'

        return re.sub(r'\s+', ' ', processed).strip()

    def process(self, text: str) -> str:
        """
        Coordinates the complete preprocessing pipeline matching the target architecture.
        """
        logger.debug(f"[AdvancedCorrection] Raw Input: '{text}'")

        # 1. NER Protection (detect and mask entities)
        if self.ner_protection:
            entities = self.ner_protection.detect_entities(text)
            protected_text, placeholder_map = self.ner_protection.protect(text)
        else:
            protected_text, placeholder_map = text, {}

        # 2. Context-Aware Auto Correction (English + Tanglish paths)
        step1 = self.correct(protected_text)
        logger.debug(f"[AdvancedCorrection] After Correction: '{step1}'")

        # 3. Context Correction (slang replacement)
        step2 = self.context_correct(step1)

        # 4. Tanglish Semantic Normalization
        step3 = normalize_tanglish_semantics(step2)
        logger.debug(f"[AdvancedCorrection] After Semantic Normalization: '{step3}'")

        # 5. Negation Recovery
        step4 = self.recover_negations(step3)

        # 6. Phrase Standardization
        step5 = self.standardize_phrases(step4)

        # 7. Sentence Reconstruction
        step6 = self.reconstruct_sentence(step5)

        # 8. NER Restoration
        if self.ner_protection:
            final_text = self.ner_protection.restore(step6, placeholder_map)
        else:
            final_text = step6

        logger.debug(f"[AdvancedCorrection] Final Emotion Input: '{final_text}'")
        return final_text
