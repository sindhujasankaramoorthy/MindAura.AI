"""
Patient authentication dependency -- resolves the authenticated patient_id
from a bearer token (backend/app/database.sessions), never from anything
the frontend claims directly. Used by protected endpoints such as
backend/app/api/checkins.py.
"""
from fastapi import Header, HTTPException

from backend.app.database import get_patient_id_for_token


def get_current_patient_id(authorization: str = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")

    token = authorization.removeprefix("Bearer ").strip()
    patient_id = get_patient_id_for_token(token)
    if not patient_id:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")

    return patient_id
