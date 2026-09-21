"""
Real-model integration coverage for the full TextNormalizer pipeline
(NER protection, SymSpell, Viterbi language detection, Tanglish correction,
negation recovery). Ported from the old root-level manual scripts
(test_pipeline.py, test_advanced_correction.py, test_tanglish_pipeline.py),
which printed ✅/❌ by eye with no real assertions.

Assertions are loose (substring containment) rather than exact-string-match,
since exact wording can drift slightly as the underlying correction/NER
models change — what matters is that the pipeline still recovers the right
words, not the exact punctuation/casing of the final sentence.

Needs torch/transformers/symspellpy; marked slow since it loads real models
(SymSpell + NER layer at minimum). Run with `pytest -m slow`.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("symspellpy")

from ai.text.pipeline import TextNormalizer

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def normalizer():
    return TextNormalizer()


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("i cnt do it ,idnt feel very good", "can't"),
        ("my minf aint silent", "mind"),
        ("enaku oru madhiri mind disturbed ah iruku", "mentally disturbed"),
    ],
)
def test_success_criteria_sentences(normalizer, text, expected_fragment):
    result = normalizer.normalize(text)
    assert expected_fragment.lower() in result["processed_text"].lower()


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("cnt", "can't"),
        ("dont", "don't"),
        ("didnt", "didn't"),
        ("idnt", "I don't"),
        ("isnt", "isn't"),
        ("arent", "aren't"),
        ("aint", "ain't"),
    ],
)
def test_negation_recovery(normalizer, text, expected_fragment):
    result = normalizer.normalize(text)
    assert expected_fragment.lower() in result["processed_text"].lower()


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("minf", "mind"),
        ("lonley", "lonely"),
        ("scard", "scared"),
        ("anxios", "anxious"),
        ("thnkng", "thinking"),
        ("ovrthinking", "overthinking"),
    ],
)
def test_spell_correction_does_not_mangle_emotional_words(normalizer, text, expected_fragment):
    result = normalizer.normalize(text)
    assert expected_fragment.lower() in result["processed_text"].lower()


def test_critical_negation_bug_fix(normalizer):
    """
    Regression check for a previously-fixed bug where combined negation
    contractions ("cnt", "idnt") were lost during spell correction.
    """
    result = normalizer.normalize("i cnt do it ,idnt feel very good")
    actual = result["processed_text"]
    assert "can't" in actual
    assert "don't" in actual


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("i had a friend vishal he is the reason", "Vishal"),
        ("I live in chennai and work for google", "Chennai"),
        ("my sister and mother are with me", "sister"),
        ("Ram went to New York", "Ram"),
    ],
)
def test_ner_protection_preserves_entities(normalizer, text, expected_fragment):
    result = normalizer.normalize(text)
    assert expected_fragment.lower() in result["processed_text"].lower()


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("pudikula enaku", "do not like"),
        ("romba kavalaya iruku", "worried"),
        ("enaku romba kastama iruku", "difficult"),
    ],
)
def test_tanglish_generalization(normalizer, text, expected_fragment):
    result = normalizer.normalize(text)
    assert expected_fragment.lower() in result["processed_text"].lower()
