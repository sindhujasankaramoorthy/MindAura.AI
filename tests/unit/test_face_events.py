"""
Unit tests for ai/face/events.py's threshold-based temporal event detector,
using synthetic per-frame feature sequences (no MediaPipe/OpenCV needed --
this tests the aggregation/segmentation logic in isolation).
"""
from ai.face import events


def _frame(t, **kwargs):
    d = {"timestamp_sec": t}
    d.update(kwargs)
    return d


def test_blink_detection_counts_one_blink():
    # Eye openness dips below threshold for 3 frames then recovers = 1 blink.
    frames = [
        _frame(0.0, left_eye_openness=0.9, right_eye_openness=0.9),
        _frame(0.1, left_eye_openness=0.9, right_eye_openness=0.9),
        _frame(0.2, left_eye_openness=0.2, right_eye_openness=0.2),
        _frame(0.3, left_eye_openness=0.1, right_eye_openness=0.1),
        _frame(0.4, left_eye_openness=0.2, right_eye_openness=0.2),
        _frame(0.5, left_eye_openness=0.9, right_eye_openness=0.9),
        _frame(0.6, left_eye_openness=0.9, right_eye_openness=0.9),
    ]
    result = events.detect_blink_events(frames)
    assert result["left"]["blink_count"] == 1
    assert result["right"]["blink_count"] == 1
    assert result["left"]["events"][0]["start_sec"] == 0.2
    assert result["left"]["events"][0]["end_sec"] == 0.4


def test_blink_detection_no_blink_when_eyes_stay_open():
    frames = [_frame(i * 0.1, left_eye_openness=0.9, right_eye_openness=0.9) for i in range(10)]
    result = events.detect_blink_events(frames)
    assert result["left"]["blink_count"] == 0
    assert result["left"]["events"] == []


def test_smile_detection_counts_event_and_symmetry():
    frames = [
        _frame(0.0, smile_left=0.0, smile_right=0.0),
        _frame(0.5, smile_left=0.8, smile_right=0.8),
        _frame(1.0, smile_left=0.9, smile_right=0.7),
        _frame(1.5, smile_left=0.0, smile_right=0.0),
    ]
    result = events.detect_smile_events(frames)
    assert result["detected"] is True
    assert result["event_count"] == 1
    assert 0.0 <= result["symmetry"] <= 1.0


def test_smile_not_detected_below_threshold():
    frames = [_frame(i * 0.1, smile_left=0.1, smile_right=0.1) for i in range(5)]
    result = events.detect_smile_events(frames)
    assert result["detected"] is False
    assert result["event_count"] == 0


def test_eyebrow_asymmetry_averaged():
    frames = [
        _frame(0.0, left_eyebrow_raise=0.5, right_eyebrow_raise=0.5, left_eyebrow_lower=0.0,
               right_eyebrow_lower=0.0, eyebrow_asymmetry=0.1),
        _frame(0.1, left_eyebrow_raise=0.6, right_eyebrow_raise=0.4, left_eyebrow_lower=0.0,
               right_eyebrow_lower=0.0, eyebrow_asymmetry=0.3),
    ]
    result = events.detect_eyebrow_events(frames)
    assert result["asymmetry"] == 0.2


def test_gaze_summary_dominant_direction_and_changes():
    frames = [
        _frame(0.0, gaze_direction="center"),
        _frame(0.1, gaze_direction="center"),
        _frame(0.2, gaze_direction="left"),
        _frame(0.3, gaze_direction="center"),
    ]
    result = events.summarize_gaze(frames)
    assert result["dominant_direction"] == "center"
    assert result["direction_changes"] == 2


def test_mouth_summary_reports_average_and_max():
    frames = [
        _frame(0.0, mouth_opening=0.1, mouth_width=0.5, jaw_opening=0.0),
        _frame(0.1, mouth_opening=0.4, mouth_width=0.5, jaw_opening=0.3),
    ]
    result = events.summarize_mouth(frames)
    assert result["average_opening"] == 0.25
    assert result["maximum_opening"] == 0.4


def test_head_pose_summary_reports_movement_range():
    frames = [
        _frame(0.0, head_pose={"pitch": 0.0, "yaw": -5.0, "roll": 1.0}),
        _frame(0.1, head_pose={"pitch": 4.0, "yaw": 5.0, "roll": 1.0}),
    ]
    result = events.summarize_head_pose(frames)
    assert result["movement_range"]["pitch"] == 4.0
    assert result["movement_range"]["yaw"] == 10.0


def test_empty_input_returns_not_available_not_crash():
    result = events.detect_blink_events([])
    assert result["left"]["average_openness"] == "not_available"
    assert result["left"]["blink_count"] == 0

    result = events.summarize_gaze([])
    assert result["dominant_direction"] == "not_available"
