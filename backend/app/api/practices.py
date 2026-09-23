"""
Practice-completion logging -- connects the MindAura patient frontend
(src/routes/practices.tsx) to PostgreSQL. Previously practicesDone lived
only in the browser's localStorage (src/lib/store.ts), and a writing-type
practice's note was never persisted anywhere at all.
"""
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.database import save_practice_completion, get_practice_completions_for_patient
from backend.app.security import get_current_patient_id

router = APIRouter(prefix="/api/practices", tags=["practices"])


class PracticeCompletionRequest(BaseModel):
    practice_id: str
    note: Optional[str] = None


@router.post("/complete")
def complete_practice(data: PracticeCompletionRequest, patient_id: str = Depends(get_current_patient_id)):
    saved = save_practice_completion(patient_id, data.practice_id, data.note)
    return {"status": "completed", "id": saved["id"]}


@router.get("/completions")
def list_completions(patient_id: str = Depends(get_current_patient_id)):
    return get_practice_completions_for_patient(patient_id)
