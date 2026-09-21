"""
Face pipeline orchestration: runs the FaceLandmarker across every extracted
video frame, aggregates the per-frame features into the observations
required by the video-analysis spec, and returns the "face" section of the
final JSON. Never classifies emotion -- only measurable behavior.
"""
import logging
from typing import Any, Dict, Iterator, Tuple

from ai.face.landmarker import FaceLandmarker
from ai.face import events as face_events
from ai.face import features as face_features
from ai.face.tear_observation import analyze_tear_presence

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "not_available"


def run_face_pipeline(frame_iter: Iterator[Tuple[float, Any]]) -> Dict[str, Any]:
    """
    `frame_iter` yields (timestamp_sec, frame) pairs, e.g. from
    ai.video.video_io.extract_frames(). Runs face detection once per frame
    (VIDEO mode, tracked across the whole sequence) and returns the
    complete "face" JSON section. Requires MediaPipe -- see
    aggregate_face_observations() for the mediapipe-free aggregation logic
    on its own (unit-tested directly with synthetic per-frame features,
    e.g. to cover "temporary face loss" without needing a real video).
    """
    per_frame_features = []

    with FaceLandmarker(num_faces=1) as landmarker:
        for timestamp_sec, frame in frame_iter:
            timestamp_ms = int(timestamp_sec * 1000)
            detection = landmarker.detect(frame, timestamp_ms)
            frame_features = face_features.compute_frame_features(detection)
            frame_features["timestamp_sec"] = timestamp_sec
            per_frame_features.append(frame_features)

    return aggregate_face_observations(per_frame_features)


def aggregate_face_observations(per_frame_features: list) -> Dict[str, Any]:
    """
    Pure aggregation logic: takes a list of per-frame feature dicts (as
    produced by face_features.compute_frame_features(), each with a
    "face_detected" bool and a "timestamp_sec") and returns the complete
    "face" JSON section. No MediaPipe dependency -- directly testable with
    synthetic frame sequences, including ones that simulate temporary face
    loss (some frames face_detected=False in the middle of the sequence).
    """
    total_count = len(per_frame_features)
    detected_count = sum(1 for f in per_frame_features if f.get("face_detected"))

    if total_count == 0:
        return build_empty_face_result("No frames were extracted from the video.")

    face_presence_ratio = round(detected_count / total_count, 4)
    detected_frames = [f for f in per_frame_features if f.get("face_detected")]

    if not detected_frames:
        result = build_empty_face_result("No face detected in any frame.")
        result["face_presence_ratio"] = face_presence_ratio
        return result

    eyes = face_events.detect_blink_events(detected_frames)
    gaze = face_events.summarize_gaze(detected_frames)
    eyebrows = face_events.detect_eyebrow_events(detected_frames)
    mouth = face_events.summarize_mouth(detected_frames)
    smile = face_events.detect_smile_events(detected_frames)
    head_pose = face_events.summarize_head_pose(detected_frames)
    tear = analyze_tear_presence(detected_frames)

    temporal_events = _collect_temporal_events(eyes, smile, eyebrows)

    return {
        "detected": True,
        "detection_confidence": NOT_AVAILABLE,  # FaceLandmarker doesn't expose a per-frame detection score
        "tracking_confidence": NOT_AVAILABLE,    # not exposed by this MediaPipe task
        "face_presence_ratio": face_presence_ratio,
        "head_pose": {"pitch": head_pose["pitch"], "yaw": head_pose["yaw"], "roll": head_pose["roll"]},
        "head_movement_range": head_pose["movement_range"],
        "eyes": {
            "left": {
                "average_openness": eyes["left"]["average_openness"],
                "blink_count": eyes["left"]["blink_count"],
                "average_blink_duration_sec": eyes["left"]["average_blink_duration_sec"],
            },
            "right": {
                "average_openness": eyes["right"]["average_openness"],
                "blink_count": eyes["right"]["blink_count"],
                "average_blink_duration_sec": eyes["right"]["average_blink_duration_sec"],
            },
            "gaze": gaze,
        },
        "eyebrows": {
            "left": {"average_raise": eyebrows["left"]["average_raise"], "average_lower": eyebrows["left"]["average_lower"]},
            "right": {"average_raise": eyebrows["right"]["average_raise"], "average_lower": eyebrows["right"]["average_lower"]},
            "asymmetry": eyebrows["asymmetry"],
        },
        "mouth": mouth,
        "smile_observation": {
            "detected": smile["detected"],
            "event_count": smile["event_count"],
            "average_intensity": smile["average_intensity"],
            "symmetry": smile["symmetry"],
        },
        "tear_observation": tear,
        "temporal_events": temporal_events,
    }


def _collect_temporal_events(eyes, smile, eyebrows):
    events = []
    for side in ("left", "right"):
        for e in eyes[side]["events"]:
            e = dict(e)
            e["event_type"] = f"blink_{side}"
            events.append(e)
    for e in smile["events"]:
        events.append(dict(e))
    for side in ("left", "right"):
        for e in eyebrows[side]["raise_events"]:
            e = dict(e)
            e["event_type"] = f"eyebrow_raise_{side}"
            events.append(e)
    return sorted(events, key=lambda e: e["start_sec"])


def build_empty_face_result(reason: str) -> Dict[str, Any]:
    logger.warning("Face pipeline: %s", reason)
    return {
        "detected": False,
        "reason": reason,
        "detection_confidence": NOT_AVAILABLE,
        "tracking_confidence": NOT_AVAILABLE,
        "face_presence_ratio": 0.0,
        "head_pose": {"pitch": NOT_AVAILABLE, "yaw": NOT_AVAILABLE, "roll": NOT_AVAILABLE},
        "head_movement_range": NOT_AVAILABLE,
        "eyes": {"left": {}, "right": {}, "gaze": {"dominant_direction": NOT_AVAILABLE, "direction_changes": NOT_AVAILABLE}},
        "eyebrows": {"left": {}, "right": {}, "asymmetry": NOT_AVAILABLE},
        "mouth": {"average_opening": NOT_AVAILABLE, "maximum_opening": NOT_AVAILABLE, "lip_width": NOT_AVAILABLE, "jaw_movement": NOT_AVAILABLE},
        "smile_observation": {"detected": False, "event_count": 0, "average_intensity": NOT_AVAILABLE, "symmetry": NOT_AVAILABLE},
        "tear_observation": analyze_tear_presence([]),
        "temporal_events": [],
    }
