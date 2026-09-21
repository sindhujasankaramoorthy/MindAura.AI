from ai.text.language.boundary import classify_sentence_language, classify_token_language


def test_classify_sentence_language_english():
    result = classify_sentence_language("I am feeling very happy today")
    assert result["pipeline"] == "ENGLISH"


def test_classify_sentence_language_tanglish_particles_not_english():
    result = classify_sentence_language("enaku romba bayama iruku")
    assert result["pipeline"] == "UNKNOWN"


def test_classify_token_language_strong_english_word():
    assert classify_token_language("the") == "ENGLISH"


def test_classify_token_language_tanglish_particle_is_unknown():
    assert classify_token_language("nu") == "UNKNOWN"
