"""
Patient notification preferences -- connects the MindAura patient
frontend (src/routes/profile.tsx) to PostgreSQL. Previously the
"Daily check-in reminder"/"Practice reminders" toggles were local React
state only, reset on every page reload.
"""
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.database import get_patient_preferences, update_patient_preferences
from backend.app.security import get_current_patient_id

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


class PreferencesUpdateRequest(BaseModel):
    daily_checkin_reminder: Optional[bool] = None
    practice_reminders: Optional[bool] = None


@router.get("/")
def read_preferences(patient_id: str = Depends(get_current_patient_id)):
    return get_patient_preferences(patient_id)


@router.put("/")
def write_preferences(data: PreferencesUpdateRequest, patient_id: str = Depends(get_current_patient_id)):
    return update_patient_preferences(patient_id, data.daily_checkin_reminder, data.practice_reminders)
