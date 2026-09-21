"""
Temporal facial event detection from a per-frame feature sequence.

This is a threshold/hysteresis-based detector, NOT a trained model. See
train_temporal_model.py for the trained-model path this is a working
stand-in for: the datasets specified for that (CK+, DFEW, FERV39k) all
require manual, license-gated registration this environment cannot
complete (see the final report), so this rule-based detector is what
actually runs today. It only measures movement (onset/apex/offset/
intensity) -- it never labels an event as an emotion.
"""
from typing import Any, Dict, List, Optional

# Thresholds are on MediaPipe's own 0-1 blendshape scale.
BLINK_CLOSE_THRESHOLD = 0.4    # eye counted "closed" below this openness
SMILE_ON_THRESHOLD = 0.3       # mouthSmile counted "active" above this
EYEBROW_RAISE_THRESHOLD = 0.3


def _threshold_events(
    timestamps: List[float],
    values: List[float],
    on_threshold: float,
    event_type: str,
    above: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generic onset/apex/offset event detector: an event runs from the frame
    the signal crosses `on_threshold` (in the `above`/`below` direction)
    until it crosses back. Confidence isn't fabricated -- it's left out for
    this detector, which is deterministic threshold-crossing, not a model
    with a real confidence score (see train_temporal_model.py for where a
    trained model's confidence would eventually go).
    """
    events = []
    in_event = False
    start_idx = None
    peak_val = None

    def is_active(v):
        return v >= on_threshold if above else v <= on_threshold

    for i, v in enumerate(values):
        if not in_event and is_active(v):
            in_event = True
            start_idx = i
            peak_val = v
        elif in_event:
            peak_val = max(peak_val, v) if above else min(peak_val, v)
            if not is_active(v):
                events.append(_make_event(timestamps, start_idx, i - 1, event_type, peak_val))
                in_event = False

    if in_event:
        events.append(_make_event(timestamps, start_idx, len(values) - 1, event_type, peak_val))

    return events


def _make_event(timestamps, start_idx, end_idx, event_type, intensity) -> Dict[str, Any]:
    start_sec = timestamps[start_idx]
    end_sec = timestamps[end_idx]
    return {
        "event_type": event_type,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "duration_sec": round(max(end_sec - start_sec, 0.0), 3),
        "intensity": round(float(intensity), 4),
    }


def detect_blink_events(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns per-eye blink events + summary stats, from left/right eye openness."""
    result = {}
    for side in ("left", "right"):
        key = f"{side}_eye_openness"
        timestamps = [f["timestamp_sec"] for f in frames if key in f]
        values = [f[key] for f in frames if key in f]

        if not values:
            result[side] = {"average_openness": "not_available", "blink_count": 0, "average_blink_duration_sec": "not_available", "events": []}
            continue

        events = _threshold_events(timestamps, values, BLINK_CLOSE_THRESHOLD, "blink", above=False)
        avg_duration = round(sum(e["duration_sec"] for e in events) / len(events), 3) if events else "not_available"

        result[side] = {
            "average_openness": round(sum(values) / len(values), 4),
            "blink_count": len(events),
            "average_blink_duration_sec": avg_duration,
            "events": events,
        }
    return result


def detect_smile_events(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Smile events from the average of left/right mouthSmile blendshapes."""
    timestamps, left_vals, right_vals = [], [], []
    for f in frames:
        if "smile_left" in f and "smile_right" in f:
            timestamps.append(f["timestamp_sec"])
            left_vals.append(f["smile_left"])
            right_vals.append(f["smile_right"])

    if not timestamps:
        return {"detected": False, "event_count": 0, "average_intensity": "not_available", "symmetry": "not_available", "events": []}

    avg_vals = [(l + r) / 2 for l, r in zip(left_vals, right_vals)]
    events = _threshold_events(timestamps, avg_vals, SMILE_ON_THRESHOLD, "smile", above=True)

    # Symmetry at peak of each event: 1.0 = perfectly symmetric.
    for e in events:
        # find peak frame within the event window for symmetry
        window = [i for i, t in enumerate(timestamps) if e["start_sec"] <= t <= e["end_sec"]]
        if window:
            peak_i = max(window, key=lambda i: avg_vals[i])
            l, r = left_vals[peak_i], right_vals[peak_i]
            e["symmetry"] = round(1.0 - abs(l - r) / max(l, r, 1e-6), 4)
        else:
            e["symmetry"] = "not_available"

    return {
        "detected": len(events) > 0,
        "event_count": len(events),
        "average_intensity": round(sum(e["intensity"] for e in events) / len(events), 4) if events else "not_available",
        "symmetry": round(sum(e["symmetry"] for e in events if isinstance(e["symmetry"], float)) / len(events), 4) if events else "not_available",
        "events": events,
    }


def detect_eyebrow_events(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for side in ("left", "right"):
        raise_key = f"{side}_eyebrow_raise"
        lower_key = f"{side}_eyebrow_lower"
        timestamps = [f["timestamp_sec"] for f in frames if raise_key in f]
        raise_vals = [f[raise_key] for f in frames if raise_key in f]
        lower_vals = [f[lower_key] for f in frames if lower_key in f]

        if not raise_vals:
            result[side] = {"average_raise": "not_available", "average_lower": "not_available", "raise_events": []}
            continue

        raise_events = _threshold_events(timestamps, raise_vals, EYEBROW_RAISE_THRESHOLD, "eyebrow_raise", above=True)
        result[side] = {
            "average_raise": round(sum(raise_vals) / len(raise_vals), 4),
            "average_lower": round(sum(lower_vals) / len(lower_vals), 4) if lower_vals else "not_available",
            "raise_events": raise_events,
        }

    asymmetry_vals = [f["eyebrow_asymmetry"] for f in frames if "eyebrow_asymmetry" in f]
    result["asymmetry"] = round(sum(asymmetry_vals) / len(asymmetry_vals), 4) if asymmetry_vals else "not_available"
    return result


def summarize_gaze(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    directions = [f["gaze_direction"] for f in frames if "gaze_direction" in f]
    if not directions:
        return {"dominant_direction": "not_available", "direction_changes": "not_available"}

    counts: Dict[str, int] = {}
    changes = 0
    for i, d in enumerate(directions):
        counts[d] = counts.get(d, 0) + 1
        if i > 0 and d != directions[i - 1]:
            changes += 1

    dominant = max(counts, key=counts.get)
    return {"dominant_direction": dominant, "direction_changes": changes}


def summarize_mouth(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    opening_vals = [f["mouth_opening"] for f in frames if "mouth_opening" in f]
    width_vals = [f["mouth_width"] for f in frames if "mouth_width" in f]
    jaw_vals = [f["jaw_opening"] for f in frames if "jaw_opening" in f]

    if not opening_vals:
        return {"average_opening": "not_available", "maximum_opening": "not_available", "lip_width": "not_available", "jaw_movement": "not_available"}

    return {
        "average_opening": round(sum(opening_vals) / len(opening_vals), 4),
        "maximum_opening": round(max(opening_vals), 4),
        "lip_width": round(sum(width_vals) / len(width_vals), 4) if width_vals else "not_available",
        "jaw_movement": round(max(jaw_vals) - min(jaw_vals), 4) if jaw_vals else "not_available",
    }


def summarize_head_pose(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    poses = [f["head_pose"] for f in frames if f.get("head_pose")]
    if not poses:
        return {"pitch": "not_available", "yaw": "not_available", "roll": "not_available", "movement_range": "not_available"}

    def avg(key):
        vals = [p[key] for p in poses]
        return round(sum(vals) / len(vals), 2)

    def rng(key):
        vals = [p[key] for p in poses]
        return round(max(vals) - min(vals), 2)

    return {
        "pitch": avg("pitch"),
        "yaw": avg("yaw"),
        "roll": avg("roll"),
        "movement_range": {"pitch": rng("pitch"), "yaw": rng("yaw"), "roll": rng("roll")},
    }
