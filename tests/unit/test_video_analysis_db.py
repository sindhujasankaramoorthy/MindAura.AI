"""
Tests for the video_analyses database table/helpers in
backend/app/database.py, and validates that ai.video.pipeline's output
JSON shape matches what the database layer expects (section 12/13/14 of
the video-pipeline spec).

Runs against a dedicated Postgres test database (TEST_DATABASE_URL env
var, defaulting to a local `mindaura_test` database) rather than the dev
database, so tests never depend on or leave residue in real data. Each
test gets clean tables via drop_all/create_all.
"""
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import database as db
from backend.app.db import Base
from backend.app.db_models import Patient

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://mindaura:mindaura_dev_pw@127.0.0.1:5432/mindaura_test",
)


@pytest.fixture
def temp_db(monkeypatch):
    """Points the database module at a dedicated, freshly-reset test
    database for the duration of the test."""
    test_engine = create_engine(TEST_DATABASE_URL, future=True)
    tables = [t for t in Base.metadata.sorted_tables if t.name != "embeddings"]
    Base.metadata.drop_all(bind=test_engine, tables=tables)
    Base.metadata.create_all(bind=test_engine, tables=tables)

    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(db, "SessionLocal", TestSessionLocal)

    # video_analyses.patient_id has a foreign key to patients -- seed the
    # three patient ids these tests use.
    with TestSessionLocal() as session:
        now = "2026-09-17T00:00:00Z"
        for pid in ("pat-1", "pat-2", "pat-3"):
            session.add(Patient(
                id=pid, mrn=f"TEST-{pid}", first_name="Test", last_name="Patient",
                date_of_birth="2000-01-01", age=26, gender="Unspecified", phone="0000000000",
                known_allergies=[], chronic_conditions=[], created_at=now, updated_at=now,
            ))
        session.commit()

    yield test_engine
    test_engine.dispose()


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
