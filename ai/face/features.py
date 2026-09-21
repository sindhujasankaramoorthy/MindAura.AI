"""
Per-frame geometric/expression feature computation from MediaPipe Face
Landmarker output (blendshapes + landmarks + transformation matrix).

Blendshapes are MediaPipe's standard 52-category ARKit-style set (a real,
documented capability of the model -- not something invented here); a few
mouth/jaw measurements are computed directly from landmark distances since
blendshapes alone give expression *intensity*, not raw geometric extent.

This module only computes numbers for ONE frame. Aggregating across frames
into blink counts, smile events, etc. is events.py's job.
"""
import math
from typing import Any, Dict, List, Optional

# Well-known MediaPipe FaceMesh landmark indices (468/478-point topology).
_LM_LEFT_EYE_UPPER = 159
_LM_LEFT_EYE_LOWER = 145
_LM_RIGHT_EYE_UPPER = 386
_LM_RIGHT_EYE_LOWER = 374
_LM_UPPER_LIP = 13
_LM_LOWER_LIP = 14
_LM_MOUTH_LEFT = 61
_LM_MOUTH_RIGHT = 291
_LM_JAW_CHIN = 152
_LM_FOREHEAD = 10
_LM_LEFT_EYE_OUTER = 33
_LM_RIGHT_EYE_OUTER = 263


def _dist(landmarks: List[List[float]], i: int, j: int) -> float:
    ax, ay, _ = landmarks[i]
    bx, by, _ = landmarks[j]
    return math.hypot(ax - bx, ay - by)


def compute_frame_features(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    `detection` is one FaceLandmarker.detect() result. Returns a flat dict
    of per-frame measurements, or {"face_detected": False} if no face was
    found in this frame.
    """
    if not detection.get("face_detected"):
        return {"face_detected": False}

    landmarks = detection["landmarks"]
    blendshapes = detection.get("blendshapes") or {}
    matrix = detection.get("transformation_matrix")

    # Face-size reference (inter-ocular-ish distance) to normalize the
    # landmark-distance measurements below, so "mouth opening" means
    # roughly the same thing regardless of how close the face is to camera.
    face_scale = _dist(landmarks, _LM_LEFT_EYE_OUTER, _LM_RIGHT_EYE_OUTER) or 1.0

    b = lambda name: blendshapes.get(name, 0.0)

    features: Dict[str, Any] = {
        "face_detected": True,

        # --- Eyes ---
        "left_eye_openness": round(1.0 - b("eyeBlinkLeft"), 4),
        "right_eye_openness": round(1.0 - b("eyeBlinkRight"), 4),
        "left_eye_squint": round(b("eyeSquintLeft"), 4),
        "right_eye_squint": round(b("eyeSquintRight"), 4),
        "gaze_direction": _gaze_direction(b),

        # --- Eyebrows ---
        "left_eyebrow_raise": round(b("browOuterUpLeft"), 4),
        "right_eyebrow_raise": round(b("browOuterUpRight"), 4),
        "left_eyebrow_lower": round(b("browDownLeft"), 4),
        "right_eyebrow_lower": round(b("browDownRight"), 4),
        "inner_eyebrow_raise": round(b("browInnerUp"), 4),
        "eyebrow_asymmetry": round(abs(b("browOuterUpLeft") - b("browOuterUpRight")), 4),

        # --- Mouth / lips / jaw (landmark-distance geometry, normalized) ---
        "mouth_opening": round(_dist(landmarks, _LM_UPPER_LIP, _LM_LOWER_LIP) / face_scale, 4),
        "mouth_width": round(_dist(landmarks, _LM_MOUTH_LEFT, _LM_MOUTH_RIGHT) / face_scale, 4),
        "jaw_opening": round(b("jawOpen"), 4),
        "lip_pucker": round(b("mouthPucker"), 4),
        "lip_stretch_left": round(b("mouthStretchLeft"), 4),
        "lip_stretch_right": round(b("mouthStretchRight"), 4),
        "upper_lip_raise": round((b("mouthUpperUpLeft") + b("mouthUpperUpRight")) / 2, 4),
        "lower_lip_movement": round((b("mouthLowerDownLeft") + b("mouthLowerDownRight")) / 2, 4),

        # --- Smile (mouthSmileLeft/Right are MediaPipe's standard smile blendshapes) ---
        "smile_left": round(b("mouthSmileLeft"), 4),
        "smile_right": round(b("mouthSmileRight"), 4),

        # --- Head pose (from the transformation matrix, if available) ---
        "head_pose": _head_pose_from_matrix(matrix),
    }
    return features


def _gaze_direction(b) -> str:
    """
    Coarse gaze-direction label from the eyeLookX blendshapes (both eyes
    averaged). "center" if no direction clearly dominates.
    """
    up = (b("eyeLookUpLeft") + b("eyeLookUpRight")) / 2
    down = (b("eyeLookDownLeft") + b("eyeLookDownRight")) / 2
    left = (b("eyeLookOutLeft") + b("eyeLookInRight")) / 2
    right = (b("eyeLookOutRight") + b("eyeLookInLeft")) / 2

    directions = {"up": up, "down": down, "left": left, "right": right}
    dominant, score = max(directions.items(), key=lambda kv: kv[1])
    return dominant if score > 0.3 else "center"


def _head_pose_from_matrix(matrix: Optional[List[List[float]]]) -> Optional[Dict[str, float]]:
    """
    Decomposes the 4x4 facial transformation matrix's rotation component
    into pitch/yaw/roll (degrees), using the standard rotation-matrix ->
    Euler-angle formula. Returns None if no matrix was produced for this
    frame (e.g. tracking briefly lost).
    """
    if not matrix:
        return None

    r = matrix
    sy = math.sqrt(r[0][0] ** 2 + r[1][0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(-r[2][0], sy)
        roll = math.atan2(r[1][0], r[0][0])
    else:
        pitch = math.atan2(-r[1][2], r[1][1])
        yaw = math.atan2(-r[2][0], sy)
        roll = 0.0

    return {
        "pitch": round(math.degrees(pitch), 2),
        "yaw": round(math.degrees(yaw), 2),
        "roll": round(math.degrees(roll), 2),
    }
