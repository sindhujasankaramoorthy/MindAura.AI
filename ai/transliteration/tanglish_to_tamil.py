import logging

logger = logging.getLogger(__name__)

class TanglishToTamil:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            from ai4bharat.transliteration import XlitEngine
            self.engine = XlitEngine("ta", beam_width=4, rescore=False)
            self._available = True
            logger.info("Successfully loaded IndicXlit engine for Tamil.")
        except ImportError:
            logger.error("ai4bharat-transliteration is not installed. Run 'pip install ai4bharat-transliteration'")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize IndicXlit: {e}")
            self._available = False
            
        self.custom_overrides = {
            "nu": "என்று",
            "la": "இல்லையா",
            "da": "டா",
            "di": "டி",
            "ah": "ஆ",
            "nga": "ங்க",
            "uh": "உ",
            "ehh": "ஏ",
            "gwen": "gwen", # Prevent weird transliteration of English names if they slip through
        }

    def transliterate_word(self, word: str) -> str:
        if not self._available or not word.strip():
            return word
            
        word_lower = word.lower()
        if word_lower in self.custom_overrides:
            return self.custom_overrides[word_lower]
            
        try:
            # The API behavior can vary, handle list, dict, or string
            res = self.engine.translit_word(word_lower, topk=1)
            
            if isinstance(res, dict):
                # E.g. {"ta": ["எனக்கு"]}
                for lang_code, options in res.items():
                    if options and isinstance(options, list):
                        return options[0]
            elif isinstance(res, list) and len(res) > 0:
                # E.g. ["எனக்கு"]
                return res[0]
            elif isinstance(res, str):
                return res
                
        except Exception as e:
            logger.warning(f"Transliteration failed for '{word}': {e}")
            
        return word

    def transliterate_sentence(self, sentence: str) -> str:
        """
        Transliterates a full sentence while preserving spaces and delimiters.
        """
        if not self._available or not sentence.strip():
            return sentence
            
        import re
        
        # Split by non-word characters to preserve them
        tokens = re.split(r'(\W+)', sentence)
        out_tokens = []
        for t in tokens:
            if not t.strip():
                out_tokens.append(t)
            elif re.match(r'\W+', t):
                out_tokens.append(t)
            else:
                out_tokens.append(self.transliterate_word(t))
                
        return "".join(out_tokens)

