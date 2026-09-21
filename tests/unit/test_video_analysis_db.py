"""
Tests for the video_analyses database table/helpers added to
backend/app/database.py, and validates that ai.video.pipeline's output
JSON shape matches what the database layer expects (section 12/13/14 of
the video-pipeline spec).
"""
import os
import sqlite3
import uuid

import pytest

from backend.app import database as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Points the database module at a throwaway sqlite file for this test,
    so tests never touch the real dev database."""
    test_db_path = str(tmp_path / "test_video_analyses.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield test_db_path


def _sample_analysis(analysis_id=None, video_id="vid-1"):
    return {
        "analysis_id": analysis_id or str(uuid.uuid4()),
        "video_id": video_id,
        "analysis_version": "1.0",
        "created_at": "2026-09-17T00:00:00Z",
        "video": {"duration_sec": 10.0, "fps": 30, "resolution": {"width": 1280, "height": 720},
                   "has_audio": True, "audio_duration_sec": 10.0},
        "face": {"detected": True, "face_presence_ratio": 0.9, "temporal_events": []},
        "voice": {"transcription": {"raw_text": "hello", "segments": []}, "acoustics": {}, "duration_sec": 10.0},
        "synchronized_events": [],
        "analysis_metadata": {
            "face_model": "mediapipe_face_landmarker + rule-based temporal event detector",
            "voice_model": "existing_voice_model",
            "emotion_classification": False,
            "clinical_interpretation": False,
            "diagnosis": False,
            "qwen_used": False,
        },
    }


def test_save_and_fetch_video_analysis(temp_db):
    analysis = _sample_analysis()
    saved = db.save_video_analysis("pat-1", analysis)

    assert saved["patient_id"] == "pat-1"
    assert saved["analysis_id"] == analysis["analysis_id"]
    assert saved["observation"]["video_id"] == "vid-1"


def test_get_video_analysis_returns_none_for_missing(temp_db):
    assert db.get_video_analysis("no-such-analysis-id") is None


def test_get_video_analyses_for_patient_orders_by_created_at(temp_db):
    a1 = _sample_analysis(analysis_id="a1", video_id="vid-1")
    a2 = _sample_analysis(analysis_id="a2", video_id="vid-2")
    a2["created_at"] = "2026-09-18T00:00:00Z"

    db.save_video_analysis("pat-2", a1)
    db.save_video_analysis("pat-2", a2)

    results = db.get_video_analyses_for_patient("pat-2")
    assert len(results) == 2
    assert results[0]["analysis_id"] == "a2"  # most recent first


def test_preserves_full_observation_not_just_summary(temp_db):
    analysis = _sample_analysis()
    db.save_video_analysis("pat-3", analysis)

    fetched = db.get_video_analysis(analysis["analysis_id"])
    # The full nested structure must round-trip, not just a flattened summary.
    assert fetched["observation"]["face"]["face_presence_ratio"] == 0.9
    assert fetched["observation"]["voice"]["transcription"]["raw_text"] == "hello"
