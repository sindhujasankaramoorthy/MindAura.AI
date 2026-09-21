"""
Tests for ai/video/video_io.py using small synthetic videos (generated via
ffmpeg's testsrc/sine filters -- no real patient footage needed for these
structural checks). Only validate/extract_metadata/extract_audio are
covered here (ffmpeg-only); extract_frames (needs OpenCV) is covered in the
integration test once mediapipe/opencv are installed.
"""
import os

import pytest

from ai.video import video_io

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
VIDEO_WITH_AUDIO = os.path.join(FIXTURES_DIR, "synthetic_with_audio.mp4")
VIDEO_NO_AUDIO = os.path.join(FIXTURES_DIR, "synthetic_no_audio.mp4")


@pytest.fixture(autouse=True)
def _skip_if_fixtures_missing():
    if not os.path.exists(VIDEO_WITH_AUDIO):
        pytest.skip("Synthetic test fixtures not generated (see tests/fixtures/).")


def test_validate_video_accepts_valid_file():
    video_io.validate_video(VIDEO_WITH_AUDIO)  # should not raise


def test_validate_video_rejects_missing_file():
    with pytest.raises(video_io.VideoValidationError):
        video_io.validate_video("/tmp/this_file_does_not_exist_12345.mp4")


def test_validate_video_rejects_non_video_file(tmp_path):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_text("this is just text, not a video")
    with pytest.raises(video_io.VideoValidationError):
        video_io.validate_video(str(fake))


def test_extract_metadata_reports_resolution_and_fps():
    metadata = video_io.extract_metadata(VIDEO_WITH_AUDIO)
    assert metadata["resolution"]["width"] == 320
    assert metadata["resolution"]["height"] == 240
    assert metadata["fps"] == 10.0
    assert metadata["has_audio"] is True
    assert metadata["duration_sec"] != "not_available"


def test_extract_metadata_detects_no_audio():
    metadata = video_io.extract_metadata(VIDEO_NO_AUDIO)
    assert metadata["has_audio"] is False
    assert metadata["audio_duration_sec"] == "not_available"


def test_extract_audio_produces_wav_file(tmp_path):
    out_path = str(tmp_path / "extracted.wav")
    result = video_io.extract_audio(VIDEO_WITH_AUDIO, out_path)
    assert result == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_extract_audio_returns_none_for_video_without_audio(tmp_path):
    out_path = str(tmp_path / "extracted.wav")
    result = video_io.extract_audio(VIDEO_NO_AUDIO, out_path)
    assert result is None
    assert not os.path.exists(out_path)
