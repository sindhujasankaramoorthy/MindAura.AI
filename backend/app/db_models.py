"""
MedTrust AI - SQLAlchemy ORM models (PostgreSQL).

Mirrors the previous SQLite schema 1:1 -- same table names, same primary
key values/format (app-generated strings like "pat-1", "tj-...", not
UUIDs), so existing data and existing ID references throughout the
frontend/tests keep working unchanged. The only structural upgrade is
JSON columns: previously TEXT + json.dumps/json.loads, now native JSONB
(indexable, queryable, and doesn't need manual (de)serialization).
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import ForeignKey, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db import Base

try:
    from pgvector.sqlalchemy import Vector
    _HAS_PGVECTOR = True
except ImportError:  # pgvector python package not installed
    _HAS_PGVECTOR = False

# Embedding dimensionality is a placeholder for whichever embedding model
# is eventually chosen (e.g. 384 for MiniLM, 768 for many BERT-family
# models). Not used by any pipeline yet -- see Embedding model below.
EMBEDDING_DIM = 384


class User(Base):
    """Doctor/student/patient account identity. password_hash is NULL for
    the seeded doctor/student demo accounts (they use the role-switcher,
    not real login); set for self-registered patient accounts."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    avatar: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    registration_number: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    specialization: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    hospital_affiliation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    designation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class Patient(Base):
    """Clinical profile. Shares its id with the matching `users` row for a
    self-registered patient (see database.register_patient)."""
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    mrn: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[str] = mapped_column(String, nullable=False)
    date_of_birth: Mapped[str] = mapped_column(String, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    blood_group: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    phone: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emergency_contact_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    emergency_contact_phone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    known_allergies: Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    chronic_conditions: Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    doctor_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    student_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="scheduled")
    scheduled_time: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ended_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    google_meet: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    case_sheet_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_approved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    consultation_id: Mapped[str] = mapped_column(String, ForeignKey("consultations.id"), nullable=False)
    speaker: Mapped[str] = mapped_column(String, nullable=False)
    speaker_name: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.98)
    turn_order: Mapped[int] = mapped_column(Integer, nullable=False)


class CaseSheet(Base):
    __tablename__ = "casesheets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    consultation_id: Mapped[str] = mapped_column(String, ForeignKey("consultations.id"), unique=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    doctor_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    student_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    extraction_source: Mapped[str] = mapped_column(String, nullable=False, default="gemini")
    sections: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    multilingual_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    approval: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)


class VideoAnalysis(Base):
    """Multimodal (face + voice) observation records -- full measurable
    observation JSON, not just a summary (see ai/video/pipeline.py)."""
    __tablename__ = "video_analyses"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    video_id: Mapped[str] = mapped_column(String, nullable=False)
    analysis_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    analysis_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    face_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    voice_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    observation: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)


class TextJournal(Base):
    """Structured output of ai/text/pipeline.py for a patient-submitted
    journal entry. processing_status lets a failed AI run be recorded
    (with the original text preserved) rather than dropped."""
    __tablename__ = "text_journals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    analysis: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processing_status: Mapped[str] = mapped_column(String, nullable=False, default="completed")


class VoiceRecord(Base):
    """Structured output of ai/video/existing_voice_adapter.py for a
    patient-submitted voice check-in (audio only, no video). The audio
    file itself is never stored here -- see storage_uri."""
    __tablename__ = "voice_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    analysis: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processing_status: Mapped[str] = mapped_column(String, nullable=False, default="completed")
    # Placeholder for the object-storage URI (S3/MinIO/etc.) of the source
    # audio clip. Nothing currently writes this column -- the backend
    # only ever processes audio via a deleted temp file today (see
    # backend/app/api/checkins.py) -- but the column exists so wiring in
    # real object storage later doesn't require another migration.
    storage_uri: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class Session(Base):
    """Patient login session tokens."""
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class CheckInSession(Base):
    """One row per completed check-in flow on /check-in (which modes were
    used, and the local per-mode metadata the frontend already tracked in
    localStorage -- see src/lib/store.ts's addCheckIn()). The AI-analyzed
    content itself lives in text_journals/voice_records/video_analyses;
    this is the session-level summary those individual submissions belong
    to, e.g. for the /check-ins timeline view."""
    __tablename__ = "check_in_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    types: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    voice_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class PracticeCompletion(Base):
    """One row per guided-practice session finished on /practices. `note`
    holds the free-text written during a writing-type practice (Daily
    Reflection/Gratitude Journal) -- previously held only in React state
    and discarded when the user left the page."""
    __tablename__ = "practice_completions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    practice_id: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed_at: Mapped[str] = mapped_column(String, nullable=False)


class PatientPreferences(Base):
    """Notification preference toggles from /profile -- previously local
    React state only, reset on every page reload."""
    __tablename__ = "patient_preferences"

    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), primary_key=True)
    daily_checkin_reminder: Mapped[bool] = mapped_column(nullable=False, default=True)
    practice_reminders: Mapped[bool] = mapped_column(nullable=False, default=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class Embedding(Base):
    """
    Forward-looking table for vector similarity search (pgvector), not
    populated or queried by any pipeline yet. Generic on purpose --
    `source_type`/`source_id` point at any existing record (e.g.
    source_type="text_journal", source_id=<text_journals.id>) rather than
    one embeddings table per modality, so future embedding-backed features
    (semantic search over journals, similar-case retrieval, etc.) don't
    need their own schema each time.

    Requires `CREATE EXTENSION vector;` on the target Postgres server and
    the `pgvector` Python package (both already reflected in
    requirements.txt). If the pgvector extension isn't installed on a
    given deployment, `init_db()` skips creating this table rather than
    failing the whole startup -- see backend/app/database.py.
    """
    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    if _HAS_PGVECTOR:
        embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
