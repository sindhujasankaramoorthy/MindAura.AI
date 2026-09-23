"""
MedTrust AI - Authentication & Role-Based Access API
Provides role management for Doctor, Medical Student, and Patient (the
1-click role-switcher demo below), plus real patient registration/login
(register_patient/create_session in backend/app/database.py) backing the
patient-facing /login and /register frontend pages.
"""

import re

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from backend.app.database import (
    get_user_by_email,
    get_user_by_id,
    get_all_users as db_get_all_users,
    register_patient,
    verify_password,
    create_session,
    delete_session,
    get_patient_by_id,
)
from backend.app.models import UserProfile
from backend.app.security import get_current_patient_id

router = APIRouter(prefix="/api/auth", tags=["auth"])

CURRENT_ACTIVE_USER_ID = "doc-1"  # Default to Senior Doctor

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.get("/users", response_model=List[UserProfile])
def get_all_users():
    """Returns all available clinical profiles for rapid role-switching."""
    return db_get_all_users()


@router.get("/me")
def get_current_user():
    """Returns the currently active user profile and active permissions."""
    user = get_user_by_id(CURRENT_ACTIVE_USER_ID)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.pop("password_hash", None)
    user["permissions"] = {
        "can_conduct_consultations": user["role"] in ["doctor", "student"],
        "can_sign_case_sheets": user["role"] == "doctor",
        "can_edit_case_sheets": user["role"] in ["doctor", "student"],
        "can_manage_patients": user["role"] in ["doctor", "student"],
        "can_access_patient_portal": True
    }
    return user


@router.post("/switch-role/{user_id}")
def switch_active_role(user_id: str):
    """1-click role switcher allowing instant testing between Doctor, Student, and Patient."""
    global CURRENT_ACTIVE_USER_ID
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"No user with ID {user_id}")

    CURRENT_ACTIVE_USER_ID = user_id
    user.pop("password_hash", None)
    return {
        "message": f"Switched active role to {user['name']} ({user['role'].upper()})",
        "user": user
    }


# --- Patient Registration & Login ---
#
# Separate from the role-switcher above: this is real credential-based
# auth for self-registered patients. Every protected check-in endpoint
# (backend/app/api/checkins.py) resolves patient_id from the session
# token created here -- never from a value the frontend sends directly.

@router.post("/register")
def register(data: RegisterRequest):
    if not data.name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    if not _EMAIL_RE.match(data.email.strip()):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    try:
        patient = register_patient(data.name, data.email.strip(), data.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token = create_session(patient["id"])
    return {
        "token": token,
        "patient": {"id": patient["id"], "name": data.name.strip(), "email": data.email.strip()},
    }


@router.post("/login")
def login(data: LoginRequest):
    user = get_user_by_email(data.email.strip())
    if not user or user["role"] != "patient" or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(user["id"])
    return {
        "token": token,
        "patient": {"id": user["id"], "name": user["name"], "email": user["email"]},
    }


@router.post("/logout")
def logout(authorization: str = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        delete_session(authorization.removeprefix("Bearer ").strip())
    return {"status": "logged_out"}


@router.get("/patient/me")
def get_authenticated_patient(patient_id: str = Depends(get_current_patient_id)):
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return {"id": patient["id"], "name": f"{patient['first_name']} {patient['last_name']}".strip(), "email": patient.get("email")}
