"""
Unit tests for ai/voice/name_protection.py -- the dedicated Indian-name
protection stage for the voice pipeline. Pure text-in/text-out logic, no
audio or ASR involved, so these run fast and deterministically.
"""
import pytest

from ai.voice.name_protection import protect_names


def test_known_name_is_left_unchanged():
    result = protect_names("Ramesh went to Chennai for work.")
    entities = [e for e in result["entities"] if e["entity_type"] == "PERSON"]
    assert any(e["original"] == "Ramesh" for e in entities)
    ramesh = next(e for e in entities if e["original"] == "Ramesh")
    assert ramesh["correction_applied"] is False
    assert ramesh["normalized"] == "Ramesh"
    assert result["corrected_text"] == "Ramesh went to Chennai for work."


def test_never_translates_or_mangles_a_name_present_in_text():
    text = "Suresh Kumar went to the doctor yesterday."
    result = protect_names(text)
    # The name must still be present in the corrected text, unmodified.
    assert "Suresh" in result["corrected_text"]
    assert "Kumar" in result["corrected_text"]


def test_common_surname_is_not_mis_corrected_to_an_unrelated_first_name():
    # Regression test: "Kumar" (a common standalone surname) was, before
    # the dictionary/threshold fix, being fuzzy-matched to "Kumaran" (an
    # unrelated first name) -- exactly the kind of blind autocorrection
    # this stage must never do.
    result = protect_names("Sooresh Kumar went to the doctor yesterday.")
    kumar = next((e for e in result["entities"] if e["original"] == "Kumar"), None)
    assert kumar is not None
    assert kumar["correction_applied"] is False
    assert kumar["normalized"] == "Kumar"


def test_initial_is_never_corrected():
    result = protect_names("I spoke with R. Suresh about the appointment.")
    initials = [e for e in result["entities"] if e["original"] in ("R", "R.")]
    for e in initials:
        assert e["correction_applied"] is False


def test_no_correction_below_threshold_keeps_original():
    # A single unfamiliar syllable should not be force-matched to some
    # unrelated dictionary name just because it scores highest among a
    # bad set of options.
    result = protect_names("Xyz called me today.")
    for e in result["entities"]:
        if e["confidence"] is not None and e["confidence"] < 0.85:
            assert e["correction_applied"] is False


def test_no_entities_for_text_without_names():
    result = protect_names("I feel very anxious and tired today.")
    assert result["entities"] == []
    assert result["corrected_text"] == "I feel very anxious and tired today."


def test_output_schema_matches_spec():
    result = protect_names("Priya said she feels okay.")
    assert "entities" in result
    assert "corrected_text" in result
    for e in result["entities"]:
        assert set(e.keys()) == {"original", "entity_type", "normalized", "confidence", "correction_applied"}
        assert e["entity_type"] == "PERSON"
        assert isinstance(e["confidence"], float)
        assert isinstance(e["correction_applied"], bool)
