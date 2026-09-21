from ai.text.language.lexical_constants import (
    FUZZY_MATCH_THRESHOLD,
    ZIPF_ENGLISH_THRESHOLD_LOOSE,
    ZIPF_ENGLISH_THRESHOLD_STANDARD,
    ZIPF_ENGLISH_THRESHOLD_STRICT,
    fuzzy_match_tanglish,
)


def test_thresholds_are_ordered_loose_to_strict():
    assert ZIPF_ENGLISH_THRESHOLD_LOOSE < ZIPF_ENGLISH_THRESHOLD_STANDARD < ZIPF_ENGLISH_THRESHOLD_STRICT


def test_fuzzy_match_finds_close_word():
    vocabulary = {"romba", "kastama", "manasu"}
    assert fuzzy_match_tanglish("rombaa", vocabulary) == "romba"


def test_fuzzy_match_returns_none_below_threshold():
    vocabulary = {"romba", "kastama", "manasu"}
    assert fuzzy_match_tanglish("zzz", vocabulary) is None


def test_fuzzy_match_respects_custom_threshold():
    vocabulary = {"romba"}
    # "roomba" vs "romba" scores well but not perfectly; a very high bar rejects it.
    assert fuzzy_match_tanglish("roomba", vocabulary, threshold=99.9) is None
    assert fuzzy_match_tanglish("roomba", vocabulary, threshold=FUZZY_MATCH_THRESHOLD) == "romba"
