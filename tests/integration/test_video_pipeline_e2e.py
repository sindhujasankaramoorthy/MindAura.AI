"""
End-to-end test of ai.video.pipeline.analyze_video() on real (synthetic)
video files -- exercises the full path: validate -> metadata -> audio
extraction -> existing voice model -> frame extraction -> MediaPipe face
pipeline -> sync -> final JSON. Needs mediapipe + opencv, so this is a slow
integration test, not a unit test.
"""
import json
import os

import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")
pytest.importorskip("torch")

from ai.video.pipeline import analyze_video

pytestmark = pytest.mark.slow

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
VIDEO_WITH_AUDIO = os.path.join(FIXTURES_DIR, "synthetic_with_audio.mp4")
VIDEO_NO_AUDIO = os.path.join(FIXTURES_DIR, "synthetic_no_audio.mp4")


@pytest.fixture(autouse=True)
def _skip_if_fixtures_missing():
    if not os.path.exists(VIDEO_WITH_AUDIO):
        pytest.skip("Synthetic test fixtures not generated.")


def test_analyze_video_produces_valid_json_end_to_end():
    result = analyze_video(VIDEO_WITH_AUDIO, video_id="e2e-test-1")

    # Must be JSON-serializable -- the whole point of this pipeline's output.
    serialized = json.dumps(result)
    assert json.loads(serialized)["video_id"] == "e2e-test-1"

    for key in ("analysis_id", "video_id", "analysis_version", "video", "face", "voice",
                "synchronized_events", "analysis_metadata"):
        assert key in result


def test_analyze_video_no_face_in_synthetic_pattern():
    # testsrc is a color-bar pattern, not a real face -- confirms the
    # "no detectable face" path works end-to-end without crashing.
    result = analyze_video(VIDEO_WITH_AUDIO)
    assert result["face"]["detected"] is False
    assert result["face"]["face_presence_ratio"] == 0.0


def test_analyze_video_without_audio_track():
    result = analyze_video(VIDEO_NO_AUDIO)
    assert result["video"]["has_audio"] is False
    assert result["voice"]["status"] == "no_audio"


def test_analyze_video_never_calls_forbidden_components():
    result = analyze_video(VIDEO_WITH_AUDIO)
    metadata = result["analysis_metadata"]
    assert metadata["emotion_classification"] is False
    assert metadata["clinical_interpretation"] is False
    assert metadata["diagnosis"] is False
    assert metadata["qwen_used"] is False


def test_analyze_video_invalid_file_returns_valid_error_json():
    result = analyze_video("/tmp/no_such_video_xyz.mp4")
    assert result["face"]["detected"] is False
    assert "reason" in result["face"]
    # Still a complete, valid JSON shape even on failure.
    json.dumps(result)
