"""Unit tests for ai/video/sync.py -- timestamp overlap logic only."""
from ai.video.sync import build_voice_events, synchronize_events


def test_build_voice_events_groups_contiguous_words():
    segments = [
        {"word": "hello", "start_sec": 0.0, "end_sec": 0.3},
        {"word": "there", "start_sec": 0.35, "end_sec": 0.6},
        # gap > 0.6s threshold -> new speech event
        {"word": "world", "start_sec": 2.0, "end_sec": 2.4},
    ]
    events = build_voice_events(segments)
    assert len(events) == 2
    assert events[0]["start_sec"] == 0.0
    assert events[0]["end_sec"] == 0.6
    assert events[1]["start_sec"] == 2.0


def test_build_voice_events_empty_input():
    assert build_voice_events([]) == []


def test_synchronize_events_finds_overlap():
    face_events = [{"event_type": "smile", "start_sec": 5.2, "end_sec": 6.6}]
    voice_events = [{"event_type": "speech", "start_sec": 5.8, "end_sec": 6.4}]

    result = synchronize_events(face_events, voice_events)
    assert len(result) == 1
    assert result[0]["face_event"]["event_type"] == "smile"
    assert result[0]["voice_event"]["event_type"] == "speech"
    assert result[0]["overlap_sec"] == 0.6


def test_synchronize_events_no_overlap_returns_empty():
    face_events = [{"event_type": "blink_left", "start_sec": 1.0, "end_sec": 1.2}]
    voice_events = [{"event_type": "speech", "start_sec": 5.0, "end_sec": 6.0}]

    result = synchronize_events(face_events, voice_events)
    assert result == []


def test_synchronize_events_does_not_add_interpretation_fields():
    face_events = [{"event_type": "smile", "start_sec": 0.0, "end_sec": 1.0}]
    voice_events = [{"event_type": "speech", "start_sec": 0.5, "end_sec": 1.5}]

    result = synchronize_events(face_events, voice_events)
    keys = set(result[0].keys())
    assert keys == {"face_event", "voice_event", "overlap_sec"}
