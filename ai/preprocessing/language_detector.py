

import re
import logging
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)

class TokenLanguage:
    ENGLISH = "English"
    TANGLISH = "Tanglish"
    UNKNOWN = "Unknown"

class LanguageDetector:
    """
    Token-level language detector for code-mixed English/Tanglish text.
    Classifies each token as English, Tanglish, or Unknown using dynamic
    morphological heuristics — no hardcoded root dictionaries.
    """
    def __init__(self, sym_spell, protected_words: Set[str], negation_words: Set[str]):
        self.sym_spell = sym_spell
        self.protected_words = {w.lower() for w in protected_words}
        self.negation_words = {w.lower() for w in negation_words}

    def _is_tanglish(self, word: str) -> bool:
        w = word.lower()
        
        # 1. English dict exact match check
        from symspellpy import Verbosity
        exact = self.sym_spell.lookup(w, Verbosity.TOP, max_edit_distance=0, include_unknown=False)
        if exact and exact[0].count > 1:
            return False
            
        # 2. English typo check with tight ratio (avoid classifying typos as Tanglish unless very far)
        close = self.sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False)
        if close:
            ratio = close[0].distance / max(len(w), 1)
            if ratio <= 0.45:
                return False
                
        # 3. Known common short particles/pronouns (fallback list)
        if w in ("oru", "ah", "dha", "sol", "yen", "nee", "en", "un", "da", "di", "la", "ve"):
            return True
            
        # 4. English suffix protection (English words like 'feeling', 'sadness' aren't Tanglish)
        ENGLISH_SUFFIXES = ("ing", "tion", "ness", "ment", "edly", "ness", "ed", "er", "est", "ly", "ful")
        if any(w.endswith(sfx) for sfx in ENGLISH_SUFFIXES) and len(w) > 5:
            return False
            
        # 5. Romanized Tamil morphological endings (vowels + common consonants)
        # Tamil words highly tend to end in vowels (aeiou) or m, n, l, r, y, h
        TAMIL_ENDINGS = set('aeiou') | {'m', 'n', 'l', 'r', 'y', 'h'}
        return w[-1] in TAMIL_ENDINGS

    def detect(self, text: str, ner_entities: List[Tuple[int, int, str, str]] = None) -> List[Tuple[str, str]]:
        """
        Tokenize the text and classify each token's language dynamically.
        Returns a list of (token, language_tag) tuples.
        """
        # Split by placeholders OR words, keeping delimiters
        token_pattern = r"(\*\*[A-Z_]+_\d+\*\*|[a-zA-Z]+(?:'[a-zA-Z]+)?)"
        tokens_with_delimiters = re.split(token_pattern, text)
        
        token_classifications = []
        current_char_idx = 0
        
        for i, token in enumerate(tokens_with_delimiters):
            if not token:
                continue
                
            start_idx = current_char_idx
            end_idx = start_idx + len(token)
            current_char_idx = end_idx
            
            # 1a. Check if delimiter
            if i % 2 == 0:
                token_classifications.append((token, TokenLanguage.UNKNOWN))
                continue
                
            word = token
            word_lower = word.lower()
            
            # 1b. Check if NER placeholder (e.g. **PERSON_1**)
            if word.startswith("**") and word.endswith("**"):
                token_classifications.append((word, TokenLanguage.UNKNOWN))
                continue
                
            # 1c. Check if non-alphabetic
            if not word.replace("'", "").isalpha():
                token_classifications.append((word, TokenLanguage.UNKNOWN))
                continue

            # 1d. Check if overlapping with a PERSON, ORG, GPE, LOC entity in ner_entities
            is_ner_entity = False
            if ner_entities:
                for s, e, ent_type, _ in ner_entities:
                    if ent_type in ('PERSON', 'ORG', 'GPE', 'LOC'):
                        if not (end_idx <= s or start_idx >= e):
                            is_ner_entity = True
                            break
            if is_ner_entity:
                token_classifications.append((word, TokenLanguage.UNKNOWN))
                continue

            # 2. Check if negation or protected word
            if word_lower in self.negation_words or word_lower in self.protected_words:
                token_classifications.append((word, TokenLanguage.ENGLISH))
                continue
                
            # 3. Check if in English dictionary (SymSpell exact frequency lookup)
            from symspellpy import Verbosity
            suggestions = self.sym_spell.lookup(
                word_lower, Verbosity.TOP, max_edit_distance=0, include_unknown=False
            )
            # Filter out custom Tanglish entries with count=1 from being treated as English
            if suggestions and suggestions[0].count > 1:
                token_classifications.append((word, TokenLanguage.ENGLISH))
                continue

            # 4. Check if Morphologically Tanglish
            if self._is_tanglish(word_lower):
                token_classifications.append((word, TokenLanguage.TANGLISH))
                continue

            # 5. Check if it's an English typo (edit distance <= 2 to a high-frequency word, length >= 4)
            is_english_typo = False
            if len(word_lower) >= 4:
                suggestions = self.sym_spell.lookup(
                    word_lower, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False
                )
                if suggestions:
                    for s in suggestions[:2]:
                        ratio = s.distance / max(len(word_lower), 1)
                        if ratio <= 0.45:
                            is_english_typo = True
                            break
            if is_english_typo:
                token_classifications.append((word, TokenLanguage.ENGLISH))
                continue
                    
            # 6. Default fallback for other unmatched words
            token_classifications.append((word, TokenLanguage.UNKNOWN))
            
        return token_classifications
