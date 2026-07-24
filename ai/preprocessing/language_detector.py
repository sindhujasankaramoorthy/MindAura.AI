import re
import logging
from typing import List, Tuple, Set

from .word_classifier import WordClassifier

logger = logging.getLogger(__name__)

class TokenLanguage:
    ENGLISH = "English"
    TANGLISH = "Tanglish"
    UNKNOWN = "Unknown"
    TAMIL = "TAMIL"
    HINDI = "HINDI"
    TELUGU = "TELUGU"
    MALAYALAM = "MALAYALAM"
    KANNADA = "KANNADA"
    BENGALI = "BENGALI"
    MARATHI = "MARATHI"
    GUJARATI = "GUJARATI"
    PUNJABI = "PUNJABI"
    URDU = "URDU"

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
        self.word_classifier = WordClassifier()

    def detect(self, text: str, ner_entities: List[Tuple[int, int, str, str]] = None) -> List[Tuple[str, str]]:
        """
        Tokenize the text and classify each token's language dynamically.
        Returns a list of (token, language_tag) tuples.
        """
        # Split by placeholders OR words, keeping delimiters.
        # The character class [\w\u0080-\uFFFF]+ captures complete Indic script
        # words (Tamil, Hindi, Telugu, Malayalam, etc.) as single tokens instead
        # of shattering them into individual Unicode code points.
        token_pattern = r"(\*\*[A-Z_]+_\d+\*\*|[\w\u0080-\uFFFF]+(?:'[\w\u0080-\uFFFF]+)?)"
        tokens_with_delimiters = re.split(token_pattern, text, flags=re.UNICODE)
        
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
                
            # 3. Classify with WordClassifier
            lang_str = self.word_classifier.classify(word)
            if lang_str == "ENGLISH":
                token_classifications.append((word, TokenLanguage.ENGLISH))
            elif lang_str == "TANGLISH":
                token_classifications.append((word, TokenLanguage.TANGLISH))
            elif lang_str == "UNKNOWN":
                token_classifications.append((word, TokenLanguage.UNKNOWN))
            else:
                # It's an Indian language (e.g. "TAMIL")
                token_classifications.append((word, getattr(TokenLanguage, lang_str, lang_str)))
                
        return token_classifications
