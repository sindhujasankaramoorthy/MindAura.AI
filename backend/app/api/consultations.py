"""
MedTrust AI - Consultation & Google Meet Management API
Manages video meetings, Google Meet Spaces API links, live transcript turns,
and rich clinical scenario demonstration loaders.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, Body
from sqlalchemy import text
from backend.app.database import get_db_connection, get_patient_by_id
from backend.app.services.meet_service import (
    create_google_meet_session,
    list_scenarios,
    get_scenario
)
from backend.app.models import ConsultationCreate, TranscriptTurn

router = APIRouter(prefix="/api/consultations", tags=["consultations"])


@router.get("/")
def list_consultations(
    status: Optional[str] = Query(None, description="Filter by status (scheduled, in_progress, completed)"),
    patient_id: Optional[str] = Query(None)
):
    """Lists all consultations with joined patient and doctor names."""
    conn = get_db_connection()

    query = """
    SELECT c.*,
           p.first_name || ' ' || p.last_name as patient_name,
           p.mrn as patient_mrn,
           p.age as patient_age,
           p.gender as patient_gender,
           u.name as doctor_name,
           u.specialization as doctor_specialization,
           s.name as student_name,
           (SELECT COUNT(*) FROM transcripts t WHERE t.consultation_id = c.id) as transcript_turn_count
    FROM consultations c
    JOIN patients p ON c.patient_id = p.id
    JOIN users u ON c.doctor_id = u.id
    LEFT JOIN users s ON c.student_id = s.id
    WHERE 1=1
    """
    params: Dict[str, Any] = {}

    if status:
        query += " AND c.status = :status"
        params["status"] = status
    if patient_id:
        query += " AND c.patient_id = :patient_id"
        params["patient_id"] = patient_id

    query += " ORDER BY c.scheduled_time DESC"

    try:
        rows = conn.execute(text(query), params).mappings().all()
    finally:
        conn.close()

    return [dict(r) for r in rows]


@router.get("/scenarios/list")
def get_sample_scenarios():
    """Returns the library of 5 rich realistic clinical scenarios for live demonstration."""
    return list_scenarios()


@router.get("/{consultation_id}")
def get_consultation_details(consultation_id: str):
    """Retrieves single consultation details including full transcript history."""
    conn = get_db_connection()
    try:
        row = conn.execute(text("""
        SELECT c.*,
               p.first_name || ' ' || p.last_name as patient_name,
               p.mrn as patient_mrn,
               p.age as patient_age,
               p.gender as patient_gender,
               p.known_allergies,
               p.chronic_conditions,
               u.name as doctor_name,
               u.specialization as doctor_specialization,
               u.registration_number as doctor_registration,
               s.name as student_name
        FROM consultations c
        JOIN patients p ON c.patient_id = p.id
        JOIN users u ON c.doctor_id = u.id
        LEFT JOIN users s ON c.student_id = s.id
        WHERE c.id = :consultation_id
        """), {"consultation_id": consultation_id}).mappings().first()

        if not row:
            raise HTTPException(status_code=404, detail="Consultation not found")

        consultation = dict(row)
        consultation["patient_allergies"] = consultation.pop("known_allergies") or []
        consultation["patient_chronic_conditions"] = consultation.pop("chronic_conditions") or []

        turns = conn.execute(text("""
        SELECT * FROM transcripts
        WHERE consultation_id = :consultation_id
        ORDER BY turn_order ASC
        """), {"consultation_id": consultation_id}).mappings().all()
    finally:
        conn.close()

    consultation["transcripts"] = [dict(t) for t in turns]
    return consultation


@router.post("/", status_code=201)
def create_consultation(data: ConsultationCreate):
    """
    Creates a new consultation session and automatically provisions
    a Google Meet Space URI and Google Calendar invitation link.
    """
    patient = get_patient_by_id(data.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    now = datetime.now().isoformat()
    scheduled_time = data.scheduled_time or now
    consult_id = f"cons-{int(datetime.now().timestamp())}"

    # Provision Google Meet Space
    patient_full_name = f"{patient['first_name']} {patient['last_name']}"
    meet_info = create_google_meet_session(consult_id, f"MedTrust Consultation: {patient_full_name}")

    conn = get_db_connection()
    try:
        conn.execute(text("""
        INSERT INTO consultations (
            id, patient_id, doctor_id, student_id, status, scheduled_time,
            started_at, duration_seconds, google_meet, case_sheet_id, is_approved
        ) VALUES (:id, :patient_id, :doctor_id, :student_id, 'scheduled', :scheduled_time,
                  :started_at, 0, :google_meet, NULL, 0)
        """), {
            "id": consult_id,
            "patient_id": data.patient_id,
            "doctor_id": data.doctor_id,
            "student_id": data.student_id or "stu-1",
            "scheduled_time": scheduled_time,
            "started_at": now,
            "google_meet": json.dumps(meet_info),
        })
        conn.commit()
    finally:
        conn.close()

    return get_consultation_details(consult_id)


@router.post("/{consultation_id}/status")
def update_consultation_status(consultation_id: str, status: str = Body(..., embed=True)):
    """Updates the consultation status (e.g. in_progress, completed, paused)."""
    conn = get_db_connection()
    now = datetime.now().isoformat()
    try:
        if status == "in_progress":
            conn.execute(
                text("UPDATE consultations SET status = :status, started_at = COALESCE(started_at, :now) WHERE id = :id"),
                {"status": status, "now": now, "id": consultation_id},
            )
        elif status == "completed":
            conn.execute(
                text("UPDATE consultations SET status = :status, ended_at = :now WHERE id = :id"),
                {"status": status, "now": now, "id": consultation_id},
            )
        else:
            conn.execute(
                text("UPDATE consultations SET status = :status WHERE id = :id"),
                {"status": status, "id": consultation_id},
            )
        conn.commit()
    finally:
        conn.close()

    return {"message": f"Status updated to {status}", "consultation_id": consultation_id}


@router.post("/{consultation_id}/transcripts")
def add_transcript_turn(consultation_id: str, turn: Dict[str, Any]):
    """
    Appends a live speaker turn (from Web Speech API or audio stream)
    with normalized speaker diarization (doctor, student, patient).
    """
    conn = get_db_connection()
    try:
        max_order = conn.execute(
            text("SELECT COALESCE(MAX(turn_order), 0) FROM transcripts WHERE consultation_id = :id"),
            {"id": consultation_id},
        ).scalar()
        next_order = max_order + 1
        turn_id = f"turn-{consultation_id}-{next_order}"
        timestamp = turn.get("timestamp") or datetime.now().strftime("%H:%M:%S")

        conn.execute(text("""
        INSERT INTO transcripts (
            id, consultation_id, speaker, speaker_name, timestamp, text, confidence, turn_order
        ) VALUES (:id, :consultation_id, :speaker, :speaker_name, :timestamp, :text, :confidence, :turn_order)
        """), {
            "id": turn_id,
            "consultation_id": consultation_id,
            "speaker": turn.get("speaker", "doctor"),
            "speaker_name": turn.get("speaker_name", "Doctor"),
            "timestamp": timestamp,
            "text": turn.get("text", "").strip(),
            "confidence": float(turn.get("confidence", 0.98)),
            "turn_order": next_order,
        })

        # Also transition consultation to in_progress if currently scheduled
        conn.execute(
            text("UPDATE consultations SET status = 'in_progress' WHERE id = :id AND status = 'scheduled'"),
            {"id": consultation_id},
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": turn_id,
        "turn_order": next_order,
        "consultation_id": consultation_id,
        "text": turn.get("text")
    }


@router.post("/{consultation_id}/load-scenario/{scenario_key}")
def load_scenario_into_consultation(consultation_id: str, scenario_key: str):
    """
    Demonstration Mode: Loads the complete multi-turn dialogue of one of
    the 5 clinical scenarios into the consultation's live transcript record.
    """
    scenario = get_scenario(scenario_key)
    conn = get_db_connection()
    try:
        # Clear existing transcripts for a clean demo
        conn.execute(text("DELETE FROM transcripts WHERE consultation_id = :id"), {"id": consultation_id})

        for idx, turn in enumerate(scenario["turns"], 1):
            conn.execute(text("""
            INSERT INTO transcripts (
                id, consultation_id, speaker, speaker_name, timestamp, text, confidence, turn_order
            ) VALUES (:id, :consultation_id, :speaker, :speaker_name, :timestamp, :text, :confidence, :turn_order)
            """), {
                "id": f"turn-{consultation_id}-{idx}",
                "consultation_id": consultation_id,
                "speaker": turn["speaker"],
                "speaker_name": turn["speaker_name"],
                "timestamp": turn["timestamp"],
                "text": turn["text"],
                "confidence": 0.99,
                "turn_order": idx,
            })

        conn.execute(
            text("UPDATE consultations SET status = 'in_progress', duration_seconds = 145, is_approved = 0 WHERE id = :id"),
            {"id": consultation_id},
        )
        # Reset any existing case sheet for this consultation back to draft so user can test generation
        conn.execute(
            text("UPDATE casesheets SET status = 'draft', approval = :approval WHERE consultation_id = :id"),
            {"approval": json.dumps({"is_approved": False}), "id": consultation_id},
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "message": f"Successfully loaded scenario '{scenario['title']}'",
        "turn_count": len(scenario["turns"]),
        "scenario": scenario
    }


@router.get("/google-meet/transcripts/{space_id}")
def get_google_meet_transcripts_endpoint(space_id: str):
    """
    Simulates retrieval via Google Meet REST API endpoint:
    GET spaces/{space}/transcripts
    https://developers.google.com/meet/api/reference/rest/v2/spaces.transcripts
    """
    conn = get_db_connection()
    try:
        row = conn.execute(text("""
        SELECT c.id, c.google_meet
        FROM consultations c
        WHERE c.google_meet::text LIKE :pattern
        """), {"pattern": f"%{space_id}%"}).mappings().first()

        if not row:
            return {
                "name": f"spaces/{space_id}/transcripts",
                "transcripts": [],
                "state": "ACTIVE"
            }

        consult_id = row["id"]
        turns = conn.execute(
            text("SELECT * FROM transcripts WHERE consultation_id = :id ORDER BY turn_order ASC"),
            {"id": consult_id},
        ).mappings().all()
    finally:
        conn.close()

    formatted_entries = [
        {
            "name": f"spaces/{space_id}/transcripts/{t['id']}",
            "participant": t["speaker_name"],
            "role": t["speaker"],
            "text": t["text"],
            "createTime": t["timestamp"]
        }
        for t in turns
    ]

    return {
        "name": f"spaces/{space_id}/transcripts",
        "entries": formatted_entries,
        "state": "ENDED" if len(turns) > 5 else "ACTIVE"
    }
