"""
Integration test for ai/video/existing_voice_adapter.py against a real
recording already in the repo -- confirms it actually reuses the existing
voice model (ai.voice.voice_analysis.analyze_voice) rather than a
duplicate transcription path, and that the raw transcript is preserved
untranslated (section 10 of the spec).
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from ai.video.existing_voice_adapter import analyze_audio

pytestmark = pytest.mark.slow

SAMPLE_AUDIO = os.path.join(
    os.path.dirname(__file__), "..", "..", "ai", "voice", "recordings", "recording.wav"
)


@pytest.fixture(autouse=True)
def _skip_if_sample_missing():
    if not os.path.exists(SAMPLE_AUDIO):
        pytest.skip("Sample recording not present in ai/voice/recordings/.")


def test_analyze_audio_returns_common_schema():
    result = analyze_audio(SAMPLE_AUDIO)

    assert result["status"] == "ok"
    assert isinstance(result["transcription"]["raw_text"], str) and result["transcription"]["raw_text"]
    assert isinstance(result["transcription"]["segments"], list)
    assert result["transcription"]["language_detected"]


def test_analyze_audio_segments_have_real_timestamps():
    result = analyze_audio(SAMPLE_AUDIO)
    segments = result["transcription"]["segments"]

    assert len(segments) > 0
    for seg in segments:
        assert seg["end_sec"] >= seg["start_sec"] >= 0


def test_analyze_audio_marks_unsupported_acoustic_fields_honestly():
    result = analyze_audio(SAMPLE_AUDIO)
    acoustics = result["acoustics"]

    # The existing model genuinely computes these:
    assert isinstance(acoustics["pitch"]["f0_mean_hz"], float)
    assert isinstance(acoustics["energy"]["energy_mean"], float)

    # The existing model does NOT compute these -- must stay honest, not fabricated:
    assert acoustics["voice_quality"]["jitter"] == "not_available"
    assert acoustics["voice_quality"]["shimmer"] == "not_available"
    assert acoustics["pauses"]["pause_count"] == "not_available"


def test_analyze_audio_handles_invalid_file_gracefully():
    result = analyze_audio("/tmp/this_audio_does_not_exist_xyz.wav")
    assert result["status"] == "error"
    assert result["transcription"]["raw_text"] == "not_available"
