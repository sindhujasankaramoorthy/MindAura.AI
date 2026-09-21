"""
Ported from the old root-level test_tanglish_debug.py manual script.
Only needs pandas/rapidfuzz (via ai.tanglish_model.src), no torch — belongs
in the fast tier.
"""
from ai.text.tanglish.model.src.vocabulary import load_vocabulary, create_word_set
from ai.text.tanglish.model.src.autocorrect import correct_word

KNOWN_TANGLISH_WORDS = [
    "enaku", "romba", "kastama", "iruku", "pudikula", "mudiyala", "theriyum",
]


def test_known_tanglish_words_are_in_vocabulary():
    df = load_vocabulary()
    word_set = create_word_set(df)
    for word in KNOWN_TANGLISH_WORDS:
        assert word.lower() in word_set, f"expected '{word}' in Tanglish vocabulary"


def test_correct_word_returns_a_correction_for_known_words():
    for word in KNOWN_TANGLISH_WORDS:
        result = correct_word(word)
        assert result["corrected"], f"expected a non-empty correction for '{word}'"
