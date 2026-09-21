"""
Tests for ai/face/pipeline.aggregate_face_observations() -- the pure
aggregation logic, tested with synthetic per-frame feature sequences so it
doesn't need MediaPipe/a real video. Covers the "no face detected" and
"temporary face loss" scenarios from the spec's test list (section 14)
directly, without requiring real patient footage.
"""
from ai.face.pipeline import aggregate_face_observations


def _full_frame(t, face_detected=True):
    if not face_detected:
        return {"timestamp_sec": t, "face_detected": False}
    return {
        "timestamp_sec": t,
        "face_detected": True,
        "left_eye_openness": 0.9, "right_eye_openness": 0.9,
        "left_eye_squint": 0.0, "right_eye_squint": 0.0,
        "gaze_direction": "center",
        "left_eyebrow_raise": 0.1, "right_eyebrow_raise": 0.1,
        "left_eyebrow_lower": 0.0, "right_eyebrow_lower": 0.0,
        "inner_eyebrow_raise": 0.0, "eyebrow_asymmetry": 0.0,
        "mouth_opening": 0.2, "mouth_width": 0.5, "jaw_opening": 0.0,
        "lip_pucker": 0.0, "lip_stretch_left": 0.0, "lip_stretch_right": 0.0,
        "smile_left": 0.1, "smile_right": 0.1,
        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    }


def test_empty_video_returns_not_detected():
    result = aggregate_face_observations([])
    assert result["detected"] is False
    assert result["face_presence_ratio"] == 0.0


def test_no_face_in_any_frame():
    frames = [_full_frame(i * 0.1, face_detected=False) for i in range(10)]
    result = aggregate_face_observations(frames)
    assert result["detected"] is False
    assert result["face_presence_ratio"] == 0.0
    assert result["reason"] == "No face detected in any frame."


def test_temporary_face_loss_computes_partial_presence_ratio():
    frames = (
        [_full_frame(i * 0.1) for i in range(5)]
        + [_full_frame((5 + i) * 0.1, face_detected=False) for i in range(3)]  # temporary loss
        + [_full_frame((8 + i) * 0.1) for i in range(2)]
    )
    result = aggregate_face_observations(frames)

    assert result["detected"] is True
    assert result["face_presence_ratio"] == 0.7  # 7 of 10 frames had a face
    assert "eyes" in result and result["eyes"]["left"]["average_openness"] > 0


def test_full_detection_produces_complete_schema():
    frames = [_full_frame(i * 0.1) for i in range(20)]
    result = aggregate_face_observations(frames)

    assert result["detected"] is True
    assert result["face_presence_ratio"] == 1.0
    for key in ("head_pose", "eyes", "eyebrows", "mouth", "smile_observation", "tear_observation", "temporal_events"):
        assert key in result

    assert result["tear_observation"]["status"] == "not_trained"
