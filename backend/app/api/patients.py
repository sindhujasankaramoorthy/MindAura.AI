"""
MedTrust AI - Patient Management API
Handles patient directory, searching, validated registration with
automatic date-of-birth to age calculation, and consultation history.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from backend.app.database import (
    get_patients,
    get_patient_by_id,
    create_patient,
    get_db_connection,
    calculate_age
)
from backend.app.models import PatientCreate

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get("/")
def list_patients(search: Optional[str] = Query(None, description="Search by name, MRN, or phone")):
    """Returns all registered patients, optionally filtered by search string."""
    return get_patients(search)


@router.get("/{patient_id}")
def get_patient(patient_id: str):
    """Retrieves full details for a single patient."""
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.post("/", status_code=201)
def register_new_patient(data: PatientCreate):
    """
    Registers a new patient with validated demographics and
    automatically calculates the precise age in years from date of birth.
    """
    if not data.first_name.strip() or not data.last_name.strip():
        raise HTTPException(status_code=400, detail="First name and last name are required.")
    
    # Auto-calculate age to verify valid date format
    try:
        age = calculate_age(data.date_of_birth)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date of birth format. Must be YYYY-MM-DD.")
    
    dumped_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    created = create_patient(dumped_data)
    return created


@router.get("/{patient_id}/consultations")
def get_patient_consultations(patient_id: str):
    """Retrieves full consultation history for this patient."""
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT c.*, u.name as doctor_name, u.specialization as doctor_specialization,
           s.name as student_name
    FROM consultations c
    JOIN users u ON c.doctor_id = u.id
    LEFT JOIN users s ON c.student_id = s.id
    WHERE c.patient_id = ?
    ORDER BY c.scheduled_time DESC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["google_meet"] = json.loads(d.get("google_meet_json") or "{}")
        results.append(d)
    return results


@router.get("/{patient_id}/casesheets")
def get_patient_casesheets(patient_id: str):
    """Retrieves all generated clinical case sheets for this patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT cs.*, c.scheduled_time, u.name as doctor_name
    FROM casesheets cs
    JOIN consultations c ON cs.consultation_id = c.id
    JOIN users u ON cs.doctor_id = u.id
    WHERE cs.patient_id = ?
    ORDER BY cs.created_at DESC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["sections"] = json.loads(d.get("sections_json") or "{}")
        d["multilingual_summary"] = json.loads(d.get("multilingual_summary_json") or "{}")
        d["approval"] = json.loads(d.get("approval_json") or "{}")
        results.append(d)
    return results
