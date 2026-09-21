"""
Timestamp synchronization: finds temporal overlap between face events and
voice events on the shared video timeline. Reports ONLY that events
overlap -- never infers any psychological meaning from the overlap.
"""
from typing import Any, Dict, List


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start <= b_end and b_start <= a_end


def _overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_voice_events(voice_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turns the voice adapter's word-level segments into simple speech-span
    events (one event per contiguous run of words with no large gap) --
    the only voice-side "event" the existing voice model's output actually
    supports without fabricating pause-by-pause detail it doesn't provide
    (pause_count/exact pause boundaries are "not_available", see
    existing_voice_adapter.py).
    """
    if not voice_segments:
        return []

    events = []
    current_start = voice_segments[0]["start_sec"]
    current_end = voice_segments[0]["end_sec"]
    GAP_THRESHOLD_SEC = 0.6

    for seg in voice_segments[1:]:
        if seg["start_sec"] - current_end > GAP_THRESHOLD_SEC:
            events.append({"event_type": "speech", "start_sec": current_start, "end_sec": current_end})
            current_start = seg["start_sec"]
        current_end = seg["end_sec"]

    events.append({"event_type": "speech", "start_sec": current_start, "end_sec": current_end})
    return events


def synchronize_events(face_events: List[Dict[str, Any]], voice_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns every (face_event, voice_event) pair whose [start_sec, end_sec]
    windows overlap on the shared video timeline. Purely a temporal-overlap
    report -- no interpretation of what the overlap "means".
    """
    synchronized = []
    for f in face_events:
        for v in voice_events:
            if _overlaps(f["start_sec"], f["end_sec"], v["start_sec"], v["end_sec"]):
                synchronized.append({
                    "face_event": {"event_type": f["event_type"], "start_sec": f["start_sec"], "end_sec": f["end_sec"]},
                    "voice_event": {"event_type": v["event_type"], "start_sec": v["start_sec"], "end_sec": v["end_sec"]},
                    "overlap_sec": round(_overlap_duration(f["start_sec"], f["end_sec"], v["start_sec"], v["end_sec"]), 3),
                })
    return synchronized
