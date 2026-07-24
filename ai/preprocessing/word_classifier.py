"""
Word Classifier
Classifies each token into one of:
- ENGLISH
- TANGLISH
- ENTITY
- UNKNOWN
- Native Indian languages (TAMIL, HINDI, etc.)
only labels words
"""
import csv
import os
import logging
from wordfreq import zipf_frequency as z
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class WordClassifier:

    def __init__(self):
        self.tanglish_words = set()
        
        try:
            from ai.tanglish_model.src.vocabulary import load_vocabulary, create_word_set
            df = load_vocabulary()
            self.tanglish_words = create_word_set(df)
        except Exception as e:
            logger.warning(f"Failed to load official Tanglish vocabulary: {e}")
            
        # Add protected Tanglish particles so they are never treated as English
        particles = {"nu", "la", "da", "di", "ah", "nga", "uh", "ehh"}
        self.tanglish_words.update(particles)
            
        try:
            import fasttext
            # Try to load from current directory or relative to this file
            model_path = "lid.176.bin"
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), "lid.176.bin")
            self.ft_model = fasttext.load_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load fasttext model: {e}")
            self.ft_model = None

        self.supported_langs = {
            'ta': 'TAMIL', 'hi': 'HINDI', 'te': 'TELUGU', 'ml': 'MALAYALAM',
            'kn': 'KANNADA', 'bn': 'BENGALI', 'mr': 'MARATHI', 'gu': 'GUJARATI',
            'pa': 'PUNJABI', 'ur': 'URDU'
        }

    def is_eng(self,word):
        return z(word.lower(),"en")>2.5

    def classify(self,word):
        w_lower = word.lower()
        if w_lower in self.tanglish_words:
            return "TANGLISH"

        if self.is_eng(word):#to check if english 
            return "ENGLISH"
            
        if self.ft_model is not None:
            # FastText expects single line string without newlines
            safe_word = word.replace('\n', ' ').strip()
            if safe_word:
                preds = self.ft_model.predict(safe_word)
                lang_code = preds[0][0].replace('__label__', '')
                if lang_code in self.supported_langs:
                    return self.supported_langs[lang_code]
        
        # If it falls through to UNKNOWN, perform a fuzzy check against Tanglish vocabulary
        if self.tanglish_words:
            match = process.extractOne(w_lower, self.tanglish_words, scorer=fuzz.ratio)
            if match and match[1] >= 85.0:
                return "TANGLISH"

        return "UNKNOWN"

if __name__ == "__main__":

    clf= WordClassifier()

    print(clf.classify("happy"))
    print(clf.classify("romba"))
    print(clf.classify("Sindhuja"))