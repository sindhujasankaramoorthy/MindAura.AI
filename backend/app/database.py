"""
MedTrust AI - Database layer (PostgreSQL via SQLAlchemy).

Public function names/signatures/return shapes are unchanged from the
previous SQLite implementation on purpose -- every caller across
backend/app/api/*.py keeps working against this same interface. What
changed underneath: SQLAlchemy ORM sessions instead of raw sqlite3
cursors, and native JSONB columns instead of TEXT + manual json.dumps/
json.loads.

get_db_connection() is kept as a name (now returning a SQLAlchemy
Connection) for the handful of route handlers that run their own
hand-written multi-table JOIN SQL (consultations.py, casesheets.py) --
those are clearer as SQL than as ORM query-builder chains, and SQLAlchemy
Core's text()/`.mappings()` gives the same "dict(row)" ergonomics the old
sqlite3.Row-based code used.
"""
import hashlib
import hmac
import secrets
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from sqlalchemy import select, func, text
from sqlalchemy.exc import DBAPIError

from backend.app.db import Base, engine, SessionLocal, get_connection
from backend.app.db_models import (
    User,
    Patient,
    Consultation,
    Transcript,
    CaseSheet,
    VideoAnalysis,
    TextJournal,
    VoiceRecord,
    Session as SessionModel,
    Embedding,
    CheckInSession,
    PracticeCompletion,
    PatientPreferences,
)

# Re-exported so `from backend.app.database import get_db_connection` keeps
# working unchanged in backend/app/api/{auth,patients,consultations,casesheets}.py.
get_db_connection = get_connection


def calculate_age(dob_str: str) -> int:
    """Calculates exact age in years from YYYY-MM-DD string."""
    try:
        born = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


def init_db():
    """Creates all tables (idempotent) and seeds demo data. The
    `embeddings` table (pgvector) is created separately and best-effort --
    it's skipped, not fatal, on a Postgres server without the `vector`
    extension installed."""
    tables = [
        User.__table__, Patient.__table__, Consultation.__table__,
        Transcript.__table__, CaseSheet.__table__, VideoAnalysis.__table__,
        TextJournal.__table__, VoiceRecord.__table__, SessionModel.__table__,
        CheckInSession.__table__, PracticeCompletion.__table__, PatientPreferences.__table__,
    ]
    Base.metadata.create_all(bind=engine, tables=tables)

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine, tables=[Embedding.__table__])
    except DBAPIError:
        # pgvector extension not installed on this Postgres server -- the
        # rest of the app works fine without it; only future embedding-
        # based features would need it. See db_models.py.
        pass

    seed_initial_data()


def seed_initial_data():
    with SessionLocal() as session:
        if session.scalar(select(func.count()).select_from(User)) > 0:
            return

        now = datetime.now().isoformat()

        session.add_all([
            User(
                id="doc-1", name="Dr. Rajesh Sharma, MD", role="doctor",
                email="dr.sharma@medtrust.hospital.org",
                avatar="https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150",
                registration_number="TNMC-84920",
                specialization="Internal Medicine & Cardiology",
                hospital_affiliation="Apollo - MedTrust University Teaching Hospital",
                designation="Senior Consultant & Clinical Professor",
            ),
            User(
                id="stu-1", name="Sneha Patel", role="student",
                email="sneha.patel@student.medtrust.edu",
                avatar="https://images.unsplash.com/photo-1594824813512-1f31f9076fdf?w=150",
                registration_number="MED-2022-092",
                specialization="Medical Student (Final Year MBBS)",
                hospital_affiliation="MedTrust University Medical College",
                designation="Clinical Rotation Intern",
            ),
            User(
                id="pat-1", name="K. Sundaram", role="patient",
                email="sundaram.k@gmail.com",
                avatar="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
                hospital_affiliation="Apollo - MedTrust University Teaching Hospital",
                designation="Outpatient",
            ),
        ])

        session.add_all([
            Patient(
                id="pat-1", mrn="MT-2026-0841", first_name="Sundaram", last_name="Krishnamoorthy",
                date_of_birth="1968-05-14", age=calculate_age("1968-05-14"), gender="Male",
                blood_group="B+", phone="+91 98401 23456", email="sundaram.k@gmail.com",
                address="42 Temple View Road, Mylapore, Chennai, Tamil Nadu 600004",
                emergency_contact_name="Radha Sundaram (Spouse)", emergency_contact_phone="+91 98401 23457",
                known_allergies=["Penicillin", "Sulfa Drugs"],
                chronic_conditions=["Essential Hypertension", "Mild Dyslipidemia"],
                created_at=now, updated_at=now,
            ),
            Patient(
                id="pat-2", mrn="MT-2026-0912", first_name="Lakshmi", last_name="Narayanan",
                date_of_birth="1974-08-22", age=calculate_age("1974-08-22"), gender="Female",
                blood_group="O+", phone="+91 98412 34567", email="lakshmi.n@gmail.com",
                address="15 Anna Salai, Guindy, Chennai, Tamil Nadu 600032",
                emergency_contact_name="Narayanan S (Husband)", emergency_contact_phone="+91 98412 34568",
                known_allergies=["Aspirin (causes stomach irritation)"],
                chronic_conditions=["Type 2 Diabetes Mellitus (8 years)", "Diabetic Peripheral Neuropathy"],
                created_at=now, updated_at=now,
            ),
            Patient(
                id="pat-3", mrn="MT-2026-1004", first_name="Aarav", last_name="Mehra",
                date_of_birth="2020-03-10", age=calculate_age("2020-03-10"), gender="Male",
                blood_group="A+", phone="+91 98200 98765", email="parents.aarav@gmail.com",
                address="7B Regency Heights, Bandra West, Mumbai, Maharashtra 400050",
                emergency_contact_name="Rohan Mehra (Father)", emergency_contact_phone=None,
                known_allergies=[], chronic_conditions=[],
                created_at=now, updated_at=now,
            ),
        ])
        session.commit()


# --- Database Helper Query Functions ---

def get_patients(search_term: Optional[str] = None) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(Patient)
        if search_term:
            term = f"%{search_term.lower()}%"
            stmt = stmt.where(
                func.lower(Patient.first_name).like(term)
                | func.lower(Patient.last_name).like(term)
                | func.lower(Patient.mrn).like(term)
                | Patient.phone.like(f"%{search_term}%")
            )
        stmt = stmt.order_by(Patient.created_at.desc())
        return [_patient_to_dict(p) for p in session.scalars(stmt)]


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        patient = session.get(Patient, patient_id)
        return _patient_to_dict(patient) if patient else None


def create_patient(data: Dict[str, Any]) -> Dict[str, Any]:
    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(Patient)) + 1
        # Timestamp-derived, not a plain incrementing counter: a counter
        # scheme collides with the hardcoded seed MRNs (MT-2026-0841/0912/
        # 1004) once enough real patients exist to reach those numbers --
        # verified against a real Postgres UniqueViolation during testing.
        unique_suffix = int(datetime.now().timestamp()) % 100000
        new_id = f"pat-{count}_{int(datetime.now().timestamp())}"
        mrn = f"MT-2026-{unique_suffix:05d}"
        age = calculate_age(data["date_of_birth"])
        now = datetime.now().isoformat()

        patient = Patient(
            id=new_id, mrn=mrn, first_name=data["first_name"], last_name=data["last_name"],
            date_of_birth=data["date_of_birth"], age=age, gender=data["gender"],
            blood_group=data.get("blood_group", "Unknown"), phone=data["phone"],
            email=data.get("email"), address=data.get("address"),
            emergency_contact_name=data.get("emergency_contact_name"),
            emergency_contact_phone=data.get("emergency_contact_phone"),
            known_allergies=data.get("known_allergies", []),
            chronic_conditions=data.get("chronic_conditions", []),
            created_at=now, updated_at=now,
        )
        session.add(patient)
        session.commit()
        return _patient_to_dict(patient)


def _patient_to_dict(p: Patient) -> Dict[str, Any]:
    return {
        "id": p.id, "mrn": p.mrn, "first_name": p.first_name, "last_name": p.last_name,
        "date_of_birth": p.date_of_birth, "age": p.age, "gender": p.gender,
        "blood_group": p.blood_group, "phone": p.phone, "email": p.email,
        "address": p.address, "emergency_contact_name": p.emergency_contact_name,
        "emergency_contact_phone": p.emergency_contact_phone,
        "known_allergies": p.known_allergies or [], "chronic_conditions": p.chronic_conditions or [],
        "created_at": p.created_at, "updated_at": p.updated_at,
    }


# --- Video Analysis Helper Functions ---

def save_video_analysis(patient_id: str, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists one video-analysis observation record (see
    ai/video/pipeline.analyze_video()'s output) for a patient. Stores the
    full observation JSON, not just a summary -- future models need the
    raw measurable observations, not only a final interpretation.
    """
    with SessionLocal() as session:
        metadata = analysis_json.get("analysis_metadata", {})
        record = VideoAnalysis(
            id=f"va-{analysis_json['analysis_id']}",
            patient_id=patient_id,
            video_id=analysis_json["video_id"],
            analysis_id=analysis_json["analysis_id"],
            analysis_version=analysis_json["analysis_version"],
            created_at=analysis_json["created_at"],
            face_model=metadata.get("face_model"),
            voice_model=metadata.get("voice_model"),
            observation=analysis_json,
        )
        session.add(record)
        session.commit()
        return _video_analysis_to_dict(record)


def get_video_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        record = session.scalar(select(VideoAnalysis).where(VideoAnalysis.analysis_id == analysis_id))
        return _video_analysis_to_dict(record) if record else None


def get_video_analyses_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(VideoAnalysis)
            .where(VideoAnalysis.patient_id == patient_id)
            .order_by(VideoAnalysis.created_at.desc())
        )
        return [_video_analysis_to_dict(r) for r in session.scalars(stmt)]


def _video_analysis_to_dict(r: VideoAnalysis) -> Dict[str, Any]:
    return {
        "id": r.id, "patient_id": r.patient_id, "video_id": r.video_id,
        "analysis_id": r.analysis_id, "analysis_version": r.analysis_version,
        "created_at": r.created_at, "face_model": r.face_model, "voice_model": r.voice_model,
        "observation": r.observation,
    }


# --- Text Journal Helper Functions ---

def save_text_journal(
    patient_id: str,
    raw_input: str,
    analysis_json: Dict[str, Any],
    processing_status: str = "completed",
) -> Dict[str, Any]:
    """Persists one text check-in's structured output from
    ai/text/pipeline.process_text(). processing_status is 'completed' for
    a normal run or 'failed' when the AI pipeline raised -- the original
    raw_input is preserved either way instead of being dropped."""
    with SessionLocal() as session:
        record = TextJournal(
            id=f"tj-{int(datetime.now().timestamp() * 1000)}",
            patient_id=patient_id,
            created_at=datetime.now().isoformat(),
            raw_input=raw_input,
            analysis=analysis_json,
            processing_status=processing_status,
        )
        session.add(record)
        session.commit()
        return _text_journal_to_dict(record)


def get_text_journals_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(TextJournal)
            .where(TextJournal.patient_id == patient_id)
            .order_by(TextJournal.created_at.desc())
        )
        return [_text_journal_to_dict(r) for r in session.scalars(stmt)]


def _text_journal_to_dict(r: TextJournal) -> Dict[str, Any]:
    return {
        "id": r.id, "patient_id": r.patient_id, "created_at": r.created_at,
        "raw_input": r.raw_input, "analysis": r.analysis, "processing_status": r.processing_status,
    }


# --- Voice Record Helper Functions ---

def save_voice_record(
    patient_id: str,
    analysis_json: Dict[str, Any],
    processing_status: str = "completed",
) -> Dict[str, Any]:
    """Persists one voice check-in's structured output from
    ai/video/existing_voice_adapter.analyze_audio(). processing_status is
    'completed' for a normal run or 'failed' when the AI pipeline raised."""
    with SessionLocal() as session:
        record = VoiceRecord(
            id=f"vr-{int(datetime.now().timestamp() * 1000)}",
            patient_id=patient_id,
            created_at=datetime.now().isoformat(),
            analysis=analysis_json,
            processing_status=processing_status,
        )
        session.add(record)
        session.commit()
        return _voice_record_to_dict(record)


def get_voice_records_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(VoiceRecord)
            .where(VoiceRecord.patient_id == patient_id)
            .order_by(VoiceRecord.created_at.desc())
        )
        return [_voice_record_to_dict(r) for r in session.scalars(stmt)]


def _voice_record_to_dict(r: VoiceRecord) -> Dict[str, Any]:
    return {
        "id": r.id, "patient_id": r.patient_id, "created_at": r.created_at,
        "analysis": r.analysis, "processing_status": r.processing_status,
    }


# --- Patient Authentication ---
#
# The `users` table (role='patient') holds login identity (email +
# password_hash); the `patients` table (same id) holds the clinical
# profile that text_journals/voice_records/video_analyses already key on.
# Self-registration only collects name/email/password, so the clinical
# fields the `patients` table requires (dob, gender, phone) are seeded
# with explicit "not yet provided" placeholders for clinic staff to fill
# in later -- never fabricated as if they were real intake data.

_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        user = session.scalar(select(User).where(func.lower(User.email) == email.lower()))
        return _user_to_dict(user) if user else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        user = session.get(User, user_id)
        return _user_to_dict(user) if user else None


def get_all_users() -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        return [_user_to_dict(u) for u in session.scalars(select(User))]


def _user_to_dict(u: User) -> Dict[str, Any]:
    return {
        "id": u.id, "name": u.name, "role": u.role, "email": u.email, "avatar": u.avatar,
        "registration_number": u.registration_number, "specialization": u.specialization,
        "hospital_affiliation": u.hospital_affiliation, "designation": u.designation,
        "password_hash": u.password_hash,
    }


def register_patient(name: str, email: str, password: str) -> Dict[str, Any]:
    """Creates both the login identity (users row) and the clinical
    profile (patients row, via the existing create_patient()) for a
    self-registered patient, sharing one generated id. Raises ValueError
    if the email is already registered."""
    if get_user_by_email(email):
        raise ValueError("An account with this email already exists.")

    name_parts = name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    patient = create_patient({
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": "",
        "gender": "Not provided",
        "phone": "Not provided",
        "email": email,
    })

    with SessionLocal() as session:
        session.add(User(
            id=patient["id"], name=name.strip(), role="patient", email=email,
            password_hash=hash_password(password),
        ))
        session.commit()
    return patient


def create_session(patient_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with SessionLocal() as session:
        session.add(SessionModel(token=token, patient_id=patient_id, created_at=datetime.now().isoformat()))
        session.commit()
    return token


def get_patient_id_for_token(token: str) -> Optional[str]:
    with SessionLocal() as session:
        record = session.get(SessionModel, token)
        return record.patient_id if record else None


def delete_session(token: str) -> None:
    with SessionLocal() as session:
        record = session.get(SessionModel, token)
        if record:
            session.delete(record)
            session.commit()


# --- Check-in Session Helper Functions ---
#
# The session-level summary a /check-in run produces (which modes were
# used, plus their local metadata) -- previously held only in the
# browser's localStorage (src/lib/store.ts). The AI-analyzed content of
# each individual mode still lives in text_journals/voice_records/
# video_analyses; this is what ties them together as one check-in.

def save_check_in_session(
    patient_id: str,
    types: List[str],
    text_value: Optional[str] = None,
    voice_seconds: Optional[int] = None,
    video_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    with SessionLocal() as session:
        record = CheckInSession(
            id=f"cis-{int(datetime.now().timestamp() * 1000)}",
            patient_id=patient_id,
            types=types,
            text=text_value,
            voice_seconds=voice_seconds,
            video_seconds=video_seconds,
            created_at=datetime.now().isoformat(),
        )
        session.add(record)
        session.commit()
        return _check_in_session_to_dict(record)


def get_check_in_sessions_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(CheckInSession)
            .where(CheckInSession.patient_id == patient_id)
            .order_by(CheckInSession.created_at.desc())
        )
        return [_check_in_session_to_dict(r) for r in session.scalars(stmt)]


def _check_in_session_to_dict(r: CheckInSession) -> Dict[str, Any]:
    return {
        "id": r.id, "patient_id": r.patient_id, "types": r.types, "text": r.text,
        "voice_seconds": r.voice_seconds, "video_seconds": r.video_seconds, "created_at": r.created_at,
    }


# --- Practice Completion Helper Functions ---

def save_practice_completion(patient_id: str, practice_id: str, note: Optional[str] = None) -> Dict[str, Any]:
    with SessionLocal() as session:
        record = PracticeCompletion(
            id=f"pc-{int(datetime.now().timestamp() * 1000)}",
            patient_id=patient_id,
            practice_id=practice_id,
            note=note,
            completed_at=datetime.now().isoformat(),
        )
        session.add(record)
        session.commit()
        return _practice_completion_to_dict(record)


def get_practice_completions_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(PracticeCompletion)
            .where(PracticeCompletion.patient_id == patient_id)
            .order_by(PracticeCompletion.completed_at.desc())
        )
        return [_practice_completion_to_dict(r) for r in session.scalars(stmt)]


def _practice_completion_to_dict(r: PracticeCompletion) -> Dict[str, Any]:
    return {
        "id": r.id, "patient_id": r.patient_id, "practice_id": r.practice_id,
        "note": r.note, "completed_at": r.completed_at,
    }


# --- Patient Preferences Helper Functions ---

def get_patient_preferences(patient_id: str) -> Dict[str, Any]:
    """Returns the patient's saved preferences, creating a default row the
    first time they're requested (mirrors the frontend's previous
    defaults: reminders on, practice nudges off)."""
    with SessionLocal() as session:
        record = session.get(PatientPreferences, patient_id)
        if not record:
            record = PatientPreferences(
                patient_id=patient_id,
                daily_checkin_reminder=True,
                practice_reminders=False,
                updated_at=datetime.now().isoformat(),
            )
            session.add(record)
            session.commit()
        return _preferences_to_dict(record)


def update_patient_preferences(
    patient_id: str,
    daily_checkin_reminder: Optional[bool] = None,
    practice_reminders: Optional[bool] = None,
) -> Dict[str, Any]:
    with SessionLocal() as session:
        record = session.get(PatientPreferences, patient_id)
        if not record:
            record = PatientPreferences(
                patient_id=patient_id, daily_checkin_reminder=True, practice_reminders=False,
                updated_at=datetime.now().isoformat(),
            )
            session.add(record)
        if daily_checkin_reminder is not None:
            record.daily_checkin_reminder = daily_checkin_reminder
        if practice_reminders is not None:
            record.practice_reminders = practice_reminders
        record.updated_at = datetime.now().isoformat()
        session.commit()
        return _preferences_to_dict(record)


def _preferences_to_dict(r: PatientPreferences) -> Dict[str, Any]:
    return {
        "patient_id": r.patient_id,
        "daily_checkin_reminder": r.daily_checkin_reminder,
        "practice_reminders": r.practice_reminders,
        "updated_at": r.updated_at,
    }
