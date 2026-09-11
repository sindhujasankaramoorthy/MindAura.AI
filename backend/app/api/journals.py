from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.emotion_service import analyze

router = APIRouter(prefix="/journals", tags=["journals"])


class JournalAnalyzeRequest(BaseModel):
    text: str
    include_interpretation: bool = False


@router.post("/analyze")
def analyze_journal(request: JournalAnalyzeRequest):
    """
    Run the emotion-analysis pipeline on a journal entry.

    Deliberately text-only for this first slice — voice pipeline and
    fusion-engine risk scoring are not wired in here yet.
    """
    return analyze(request.text, include_interpretation=request.include_interpretation)
