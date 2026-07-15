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

logger = logging.getLogger(__name__)

class WordClassifier:

    def __init__(self):
        self.tanglish_words = set()
        tanglish_csv_path = os.path.join(os.path.dirname(__file__), "data", "third_enhanced_transliterated_words.csv")
        if os.path.exists(tanglish_csv_path):
            with open(tanglish_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row.get('word', '').strip().lower()
                    if word:
                        self.tanglish_words.add(word)
        else:
            logger.warning(f"Tanglish CSV not found at {tanglish_csv_path}")
            
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
        
        return "UNKNOWN"

if __name__ == "__main__":

    clf= WordClassifier()

    print(clf.classify("happy"))
    print(clf.classify("romba"))
    print(clf.classify("Sindhuja"))