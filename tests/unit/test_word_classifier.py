from ai.preprocessing.word_classifier import WordClassifier


def test_english_word_classified_as_english():
    clf = WordClassifier()
    assert clf.classify("happy") == "ENGLISH"


def test_tanglish_particle_classified_as_tanglish():
    clf = WordClassifier()
    assert clf.classify("romba") == "TANGLISH"


def test_is_eng_uses_loose_zipf_threshold():
    clf = WordClassifier()
    assert clf.is_eng("the") is True
