"""
MedTrust AI - Comprehensive Clinical Test Suite
Verifies 17-section case sheet extraction, Google Meet integration,
age calculation, doctor electronic sign-off locking, and multilingual summaries.
"""

import pytest
from datetime import date
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import calculate_age, init_db, get_db_connection
from backend.app.services.meet_service import create_google_meet_session, get_scenario, list_scenarios
from backend.app.services.ai_casesheet_service import (
    extract_case_sheet_offline_nlp,
    generate_multilingual_summaries,
    compute_approval_hash
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_calculate_age_accuracy():
    # Test age calculation from DOB
    today = date.today()
    dob_20_years_ago = f"{today.year - 20}-{today.month:02d}-{today.day:02d}"
    assert calculate_age(dob_20_years_ago) == 20

    # Specific historical dates
    assert calculate_age("1968-05-14") >= 56
    assert calculate_age("2020-03-10") >= 4


def test_google_meet_space_generation():
    session = create_google_meet_session("test-consult-1", "Test Consultation")
    assert session["space_name"].startswith("spaces/mt-")
    assert "https://meet.google.com/" in session["meeting_uri"]
    assert len(session["meeting_code"].split("-")) == 3  # Format: xxx-yyyy-zzz
    assert "calendar.google.com" in session["calendar_html_link"]


def test_offline_clinical_nlp_all_17_sections():
    scenario = get_scenario("cardiology")
    patient = {
        "first_name": "Sundaram",
        "last_name": "Krishnamoorthy",
        "mrn": "MT-2026-0841",
        "age": 58,
        "gender": "Male",
        "phone": "+91 98401 23456",
        "known_allergies": ["Penicillin"],
        "chronic_conditions": ["Hypertension"]
    }
    doctor = {
        "id": "doc-1",
        "name": "Dr. Rajesh Sharma, MD"
    }

    result = extract_case_sheet_offline_nlp(scenario["turns"], patient, doctor)

    # Verify all 17 distinct sections are present
    assert "patient_info" in result
    assert "chief_complaint" in result and len(result["chief_complaint"]) > 5
    assert "history_of_present_illness" in result and len(result["history_of_present_illness"]) > 10
    assert "symptoms" in result and len(result["symptoms"]) > 0
    assert "duration_onset" in result and len(result["duration_onset"]) > 0
    assert "past_medical_history" in result
    assert "medications" in result and len(result["medications"]) > 0
    assert "allergies" in result
    assert "family_history" in result
    assert "social_history" in result
    assert "doctor_observations" in result
    assert "vitals" in result and "blood_pressure" in result["vitals"]
    assert "investigations" in result and len(result["investigations"]) > 0
    assert "assessment_diagnosis" in result and len(result["assessment_diagnosis"]) > 5
    assert "treatment_plan" in result and len(result["treatment_plan"]) > 10
    assert "follow_up_instructions" in result
    assert "red_flag_warnings" in result and len(result["red_flag_warnings"]) > 0
    assert "missing_information" in result
    assert "uncertain_information" in result


def test_multilingual_summaries_6_languages():
    summaries = generate_multilingual_summaries(
        assessment="Exertional Angina and Hypertension",
        plan="Stop Enalapril, start Telmisartan 40mg, take Aspirin 75mg",
        patient_name="K. Sundaram"
    )

    # Must contain English, Tamil, Hindi, Telugu, Malayalam, Kannada
    assert "en" in summaries and len(summaries["en"]) > 20
    assert "ta" in summaries and len(summaries["ta"]) > 20
    assert "hi" in summaries and len(summaries["hi"]) > 20
    assert "te" in summaries and len(summaries["te"]) > 20
    assert "ml" in summaries and len(summaries["ml"]) > 20
    assert "kn" in summaries and len(summaries["kn"]) > 20


def test_patient_registration_api():
    new_patient_payload = {
        "first_name": "Devi",
        "last_name": "Sankaran",
        "date_of_birth": "1985-04-12",
        "gender": "Female",
        "blood_group": "A+",
        "phone": "+91 94444 88888",
        "email": "devi.s@gmail.com",
        "known_allergies": ["Sulfa"],
        "chronic_conditions": ["Hypothyroidism"]
    }

    resp = client.post("/api/patients/", json=new_patient_payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["first_name"] == "Devi"
    assert data["mrn"].startswith("MT-2026-")
    assert data["age"] >= 39


def test_case_sheet_approval_lock_api():
    # Fetch existing demo consultation
    resp = client.get("/api/casesheets/cons-demo-1")
    assert resp.status_code == 200
    cs = resp.json()
    cs_id = cs["id"]

    # Attempting to edit an approved_locked record must return 403 Forbidden
    edit_resp = client.put(f"/api/casesheets/{cs_id}", json={
        "sections": cs["sections"]
    })
    assert edit_resp.status_code == 403
    assert "locked" in edit_resp.json()["detail"].lower()


def test_print_html_endpoint():
    resp = client.get("/api/casesheets/cs-demo-1/print")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Apollo - MedTrust University Teaching Hospital" in resp.text
    assert "Clinical Case Sheet" in resp.text
