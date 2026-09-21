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

from .lexical_constants import ZIPF_ENGLISH_THRESHOLD_LOOSE, fuzzy_match_tanglish
from .script_detector import detect_script

logger = logging.getLogger(__name__)

class WordClassifier:

    def __init__(self):
        self.tanglish_words = set()
        
        try:
            from ai.text.tanglish.model.src.vocabulary import load_vocabulary, create_word_set
            df = load_vocabulary()
            self.tanglish_words = create_word_set(df)
        except Exception as e:
            logger.warning(f"Failed to load official Tanglish vocabulary: {e}")
            
        # Add protected Tanglish particles so they are never treated as English
        particles = {"nu", "la", "da", "di", "ah", "nga", "uh", "ehh"}
        self.tanglish_words.update(particles)
            
        try:
            import fasttext

            # Fixed, project-relative location (ai/text/language/models/) so
            # this works regardless of the caller's current working
            # directory -- not a personal/absolute path. This mirrors the
            # existing models/ convention used elsewhere in the repo (e.g.
            # ai/training/models/), and is gitignored the same way (*.bin,
            # models/ in .gitignore) since it's a ~130MB binary that
            # shouldn't be committed. See README.md "Installation" for the
            # download command.
            model_path = os.path.join(os.path.dirname(__file__), "models", "lid.176.bin")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"fastText language-id model not found at {model_path}. "
                    "Download it with:\n"
                    "  mkdir -p ai/text/language/models && "
                    "curl -L -o ai/text/language/models/lid.176.bin "
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin\n"
                    "Falling back to zipf-frequency + Tanglish-vocabulary "
                    "classification only (no fastText language ID)."
                )
            self.ft_model = fasttext.load_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load fasttext model, falling back to non-fastText classification: {e}")
            self.ft_model = None

        self.supported_langs = {
            'ta': 'TAMIL', 'hi': 'HINDI', 'te': 'TELUGU', 'ml': 'MALAYALAM',
            'kn': 'KANNADA', 'bn': 'BENGALI', 'mr': 'MARATHI', 'gu': 'GUJARATI',
            'pa': 'PUNJABI', 'ur': 'URDU'
        }

        # Unicode-script -> language, as a deterministic, always-available
        # fallback for native-script Indic words. Doesn't depend on
        # fastText (which requires a ~130MB model file this environment
        # doesn't reliably have) -- script identity for e.g. Devanagari or
        # Telugu characters is unambiguous regardless of model availability.
        self._script_to_lang = {
            "Tamil": "TAMIL", "Devanagari": "HINDI", "Telugu": "TELUGU",
            "Malayalam": "MALAYALAM", "Kannada": "KANNADA", "Bengali": "BENGALI",
            "Gujarati": "GUJARATI", "Gurmukhi": "PUNJABI", "Arabic": "URDU",
        }

    def is_eng(self,word):
        return z(word.lower(),"en") > ZIPF_ENGLISH_THRESHOLD_LOOSE

    def classify(self,word):
        w_lower = word.lower()
        if w_lower in self.tanglish_words:
            return "TANGLISH"

        if self.is_eng(word):#to check if english
            return "ENGLISH"

        script = detect_script(word)
        if script in self._script_to_lang:
            return self._script_to_lang[script]

        if self.ft_model is not None:
            # FastText expects single line string without newlines
            safe_word = word.replace('\n', ' ').strip()
            if safe_word:
                try:
                    preds = self.ft_model.predict(safe_word)
                    lang_code = preds[0][0].replace('__label__', '')
                    if lang_code in self.supported_langs:
                        return self.supported_langs[lang_code]
                except Exception as e:
                    # A load-time success doesn't guarantee predict() won't
                    # fail at use-time (e.g. a fasttext/numpy version
                    # mismatch) -- don't let that crash classification, just
                    # fall through to the fuzzy-vocabulary check below.
                    logger.warning(f"fastText predict() failed, falling back: {e}")
        
        # If it falls through to UNKNOWN, perform a fuzzy check against Tanglish vocabulary
        if self.tanglish_words:
            if fuzzy_match_tanglish(w_lower, self.tanglish_words) is not None:
                return "TANGLISH"

        return "UNKNOWN"

if __name__ == "__main__":

    clf= WordClassifier()

    print(clf.classify("happy"))
    print(clf.classify("romba"))
    print(clf.classify("Sindhuja"))