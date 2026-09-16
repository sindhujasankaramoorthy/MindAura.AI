"""
MedTrust AI - Authentication & Role-Based Access API
Provides role management for Doctor, Medical Student, and Patient,
with instant 1-click role switching and session profiles.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from backend.app.database import get_db_connection
from backend.app.models import UserProfile

router = APIRouter(prefix="/api/auth", tags=["auth"])

CURRENT_ACTIVE_USER_ID = "doc-1"  # Default to Senior Doctor


@router.get("/users", response_model=List[UserProfile])
def get_all_users():
    """Returns all available clinical profiles for rapid role-switching."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/me")
def get_current_user():
    """Returns the currently active user profile and active permissions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (CURRENT_ACTIVE_USER_ID,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = dict(row)
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"No user with ID {user_id}")
    
    CURRENT_ACTIVE_USER_ID = user_id
    user = dict(row)
    return {
        "message": f"Switched active role to {user['name']} ({user['role'].upper()})",
        "user": user
    }
