# Implementation Plan

## Goal
Add multilingual language support to the existing MindAura preprocessing pipeline without changing the current architecture, workflow, or outputs. Support English, Tanglish, Tamil, Hindi, Telugu, Malayalam, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu.

## Proposed Changes

### 1. `ai/preprocessing/word_classifier.py`
- Add `fasttext` model loading (`lid.176.bin`).
- Add `tanglish_words.csv` loading to a set for fast lookup.
- Update `classify(self, word)`:
  - Check if word is in Tanglish dataset -> return "TANGLISH"
  - Check if word is English (`wordfreq`) -> return "ENGLISH"
  - Check with FastText -> return matched Indian language (e.g., "TAMIL", "HINDI") or "UNKNOWN".
- Map FastText language codes (`ta`, `hi`, `te`, etc.) to language names.
- Ensure models and resources are loaded only once in `__init__`.

### 2. `ai/preprocessing/language_detector.py`
- Import `WordClassifier`.
- Add Indian languages to `TokenLanguage`.
- Update `token_pattern` to `r"(\*\*[A-Z_]+_\d+\*\*|[^\W\d_]+(?:'[^\W\d_]+)?)"` to ensure native Indian script tokens are not skipped.
- Replace the existing hardcoded English/Tanglish fallback logic in `detect()` with a call to `self.word_classifier.classify(word)`.

### 3. `ai/preprocessing/text_normalizer.py`
- In `normalize(self, text)`, add document-level routing at the very beginning:
  - Use `self.detect_language(text)` to get the overall document language.
  - If the language is one of the supported Indian languages, translate the entire document to English using `IndicTrans2` (via `self.advanced_corrector.translate_indic`) *before* continuing the pipeline.
  - If English or Tanglish, proceed as normal.

### 4. `ai/preprocessing/advanced_correction.py`
- Add `IndicTrans2` model loading in `EmotionPreservingCorrector.__init__`.
- Add a new helper method `translate_indic(self, text, src_lang)` which translates text to English using `IndicTrans2`.
- Update `correct(self, text)`:
  - Add language-aware routing for Indian language tokens.
  - If a token is an Indian language, translate it to English using `translate_indic(word, lang)`.
  - Continue the pipeline.

## Open Questions
- Is `ai4bharat/indictrans2-indic-en-1B` the correct model name to use for `IndicTrans2` via `transformers`? We will assume yes and use it with `trust_remote_code=True` and `IndicTransToolkit`.
