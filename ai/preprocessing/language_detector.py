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

class LanguagePrediction(tuple):
    def __new__(cls, token, language, confidence, reason):
        return super(LanguagePrediction, cls).__new__(cls, (token, language))
        
    def __init__(self, token, language, confidence, reason):
        self.confidence = round(confidence, 4)
        self.reason = reason

class LanguageDetector:
    """
    Context-aware language detector for code-mixed English/Tanglish text.
    Uses a Viterbi-style sequence labeling algorithm to maximize chunk coherence.
    """
    def __init__(self, sym_spell, protected_words: Set[str], negation_words: Set[str]):
        self.sym_spell = sym_spell
        self.protected_words = {w.lower() for w in protected_words}
        self.negation_words = {w.lower() for w in negation_words}
        self.word_classifier = WordClassifier()
        
    def get_transition_prob(self, prev_state: str, curr_state: str) -> float:
        # High cohesion: encourage staying in the same language
        if prev_state == curr_state:
            return 0.8
        # Code-switching between English and Tanglish
        if {prev_state, curr_state} == {TokenLanguage.ENGLISH, TokenLanguage.TANGLISH}:
            return 0.15
        # Everything else (e.g., native scripts, unknown transitions)
        return 0.05

    def get_emission_scores(self, word: str, is_ner: bool, boundary_lang: str, pipeline: str) -> dict:
        scores = {}
        word_lower = word.lower()
        
        # 1. NER Override
        if is_ner:
            return {TokenLanguage.UNKNOWN: 1.0}
            
        # 2. Negation and Protected words
        if word_lower in self.negation_words or word_lower in self.protected_words:
            return {TokenLanguage.ENGLISH: 1.0}
            
        # 3. Base scores from WordClassifier
        lang_str = self.word_classifier.classify(word)
        if lang_str == "ENGLISH":
            scores[TokenLanguage.ENGLISH] = 0.7
            scores[TokenLanguage.TANGLISH] = 0.2
        elif lang_str == "TANGLISH":
            scores[TokenLanguage.TANGLISH] = 0.8
            scores[TokenLanguage.ENGLISH] = 0.1
        elif lang_str == "UNKNOWN":
            scores[TokenLanguage.UNKNOWN] = 0.5
            scores[TokenLanguage.ENGLISH] = 0.25
            scores[TokenLanguage.TANGLISH] = 0.25
        else:
            native_lang = getattr(TokenLanguage, lang_str, lang_str)
            scores[native_lang] = 0.9
            
        # 4. English Boundary Bias
        if boundary_lang == "ENGLISH":
            scores[TokenLanguage.ENGLISH] = scores.get(TokenLanguage.ENGLISH, 0) + 0.4
            
        # 5. Pipeline Bias
        if pipeline == "ENGLISH":
            scores[TokenLanguage.ENGLISH] = scores.get(TokenLanguage.ENGLISH, 0) + 0.2
            
        # Normalize
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def detect(self, text: str, ner_entities: List[Tuple[int, int, str, str]] = None, pipeline: str = "UNKNOWN") -> List[LanguagePrediction]:
        token_pattern = r"(<[A-Z_]+_\d+>|[\w\u0080-\uFFFF]+(?:'[\w\u0080-\uFFFF]+)?)"
        tokens_with_delimiters = re.split(token_pattern, text, flags=re.UNICODE)
        
        # Extract word tokens that need classification
        words_to_classify = []
        current_char_idx = 0
        
        from .language_boundary import classify_token_language
        
        for i, token in enumerate(tokens_with_delimiters):
            if not token:
                continue
            start_idx = current_char_idx
            end_idx = start_idx + len(token)
            current_char_idx = end_idx
            
            # Identify delimiters or non-alphabetic tokens
            is_word = False
            if i % 2 != 0:
                is_word = token.replace("'", "").isalpha()
                # Check for NER placeholders
                if token.startswith("<") and token.endswith(">") and "_" in token:
                    is_word = False
                    
            if not is_word:
                words_to_classify.append({'token': token, 'skip': True, 'lang': TokenLanguage.UNKNOWN, 'reason': 'Delimiter/Punctuation'})
                continue
                
            # Check NER overlap
            is_ner = False
            if ner_entities:
                for s, e, ent_type, _ in ner_entities:
                    if ent_type in ('PERSON', 'ORG', 'GPE', 'LOC'):
                        if not (end_idx <= s or start_idx >= e):
                            is_ner = True
                            break
                            
            boundary_lang = classify_token_language(token)
            words_to_classify.append({
                'token': token, 
                'skip': False, 
                'is_ner': is_ner, 
                'boundary_lang': boundary_lang
            })
            
        # Viterbi decoding over actual words
        word_indices = [idx for idx, item in enumerate(words_to_classify) if not item['skip']]
        
        if not word_indices:
            return [LanguagePrediction(item['token'], item['lang'], 1.0, item['reason']) for item in words_to_classify]
            
        # Initialize Viterbi for the first word
        V = [{}] # V[t][state] = (prob, prev_state, emission_prob)
        first_idx = word_indices[0]
        emissions = self.get_emission_scores(words_to_classify[first_idx]['token'], words_to_classify[first_idx]['is_ner'], words_to_classify[first_idx]['boundary_lang'], pipeline)
        
        for state, prob in emissions.items():
            V[0][state] = (prob, None, prob)
            
        # Forward pass
        for t in range(1, len(word_indices)):
            V.append({})
            idx = word_indices[t]
            curr_emissions = self.get_emission_scores(words_to_classify[idx]['token'], words_to_classify[idx]['is_ner'], words_to_classify[idx]['boundary_lang'], pipeline)
            
            for curr_state, e_prob in curr_emissions.items():
                max_p = -1
                best_prev = None
                for prev_state, (prev_p, _, _) in V[t-1].items():
                    p = prev_p * self.get_transition_prob(prev_state, curr_state)
                    if p > max_p:
                        max_p = p
                        best_prev = prev_state
                V[t][curr_state] = (max_p * e_prob, best_prev, e_prob)
                
            # Normalize to prevent underflow
            total_p = sum(val[0] for val in V[t].values())
            if total_p > 0:
                for state in V[t]:
                    p, prev, e_prob = V[t][state]
                    V[t][state] = (p / total_p, prev, e_prob)
                    
        # Backtrack
        best_final_state = max(V[-1].keys(), key=lambda k: V[-1][k][0])
        curr = best_final_state
        
        for t in range(len(word_indices)-1, -1, -1):
            idx = word_indices[t]
            prob, prev, e_prob = V[t][curr]
            
            words_to_classify[idx]['lang'] = curr
            words_to_classify[idx]['confidence'] = prob
            
            # Determine reason
            if words_to_classify[idx]['is_ner']:
                reason = "NER entity protection"
            elif e_prob > 0.9:
                reason = "Strong intrinsic emission probability"
            elif prev and prev == curr:
                reason = "Contextual cohesion with previous token"
            else:
                reason = "Viterbi optimal sequence path"
                
            words_to_classify[idx]['reason'] = reason
            curr = prev
            
        # Construct final output
        final_classifications = []
        for item in words_to_classify:
            conf = item.get('confidence', 1.0)
            reason = item.get('reason', 'Unknown')
            final_classifications.append(LanguagePrediction(item['token'], item['lang'], conf, reason))
            
        return final_classifications
