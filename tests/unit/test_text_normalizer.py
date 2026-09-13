"""
TextNormalizer orchestration wiring, using the mocked ModelRegistry (see
conftest.py) so this exercises the 13-stage pipeline's control flow without
downloading/running any real model.

Needs torch/transformers/symspellpy importable (advanced_correction.py and
text_normalizer.py import them at module scope even though the actual model
*loading* is lazy) — skips cleanly wherever those aren't installed, e.g. on
a machine that only edits code and never runs the pipeline locally.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("symspellpy")

from ai.preprocessing.text_normalizer import TextNormalizer


def test_normalize_returns_expected_keys():
    normalizer = TextNormalizer()
    result = normalizer.normalize("I feel very happy today")

    assert set(result.keys()) >= {
        "corrected_sentence",
        "metadata",
        "original_language",
        "original_text",
        "processed_text",
        "translated_text",
    }


def test_normalize_does_not_load_gpt2_or_indictrans2_for_clean_english():
    """
    A plain, correctly-spelled English sentence should never need the GPT2
    perplexity scorer or IndicTrans2 (Phase 2's lazy-loading win) — but
    since ModelRegistry.get is mocked here, this only verifies the *call*
    pattern, not real memory behavior. See tests/integration for a real
    verification of this on a machine with the models installed.
    """
    normalizer = TextNormalizer()
    normalizer.normalize("I feel very happy today")

    corrector = normalizer.advanced_corrector
    assert corrector._mlm_load_attempted is False
    assert corrector._indic_load_attempted is False
