from ai.text.tanglish.rules import normalize_tanglish_semantics, reorder_sov_to_svo


def test_normalize_tanglish_semantics_translates_core_words():
    result = normalize_tanglish_semantics("enaku romba bayama iruku")
    assert "afraid" in result
    assert "very" in result


def test_reorder_sov_to_svo_moves_subject_to_front():
    result = reorder_sov_to_svo("very afraid feel I")
    assert result.strip().lower().startswith("i")
