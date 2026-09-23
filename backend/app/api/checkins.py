"""
Check-in ingestion endpoints -- connects the MindAura patient frontend
(src/routes/check-in.tsx) to the AI pipelines already built in ai/text/ and
ai/video/. Each endpoint validates the input, runs the appropriate
pipeline, saves the structured result to PostgreSQL (with a
processing_status so a failed AI run is recorded rather than silently
dropped), and prints the result to the server console.

The submitting patient is always resolved from the caller's session token
(backend/app/security.get_current_patient_id) -- never from a patient_id
the frontend might send directly -- so one patient can never write to or
read another patient's records.
"""
import json
import logging
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.database import (
    save_text_journal,
    save_voice_record,
    save_video_analysis,
    save_check_in_session,
)
from backend.app.security import get_current_patient_id

logger = logging.getLogger("medtrust.checkins")
router = APIRouter(prefix="/checkins", tags=["checkins"])


def _print_result(title: str, result: dict) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70 + "\n")


@router.post("/text")
def checkin_text(text: str = Form(...), patient_id: str = Depends(get_current_patient_id)):
    """Runs the text pipeline (ai.text.pipeline.process_text), saves the
    structured result to SQLite under the authenticated patient, and
    prints it to console."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Journal text cannot be empty.")

    try:
        from ai.text.pipeline import process_text

        result = process_text(text)
        _print_result("TEXT CHECK-IN", result)
    except Exception:
        logger.exception("Text AI pipeline failed")
        try:
            save_text_journal(patient_id, text, {"error": "processing_failed"}, "failed")
        except Exception:
            logger.exception("Failed to persist failed text journal")
        return JSONResponse(
            status_code=200,
            content={
                "status": "failed",
                "module": "text",
                "error": "We couldn't analyze your journal. Please try again.",
            },
        )

    try:
        saved = save_text_journal(patient_id, text, result, "completed")
    except Exception:
        logger.exception("Failed to save text journal to database")
        raise HTTPException(status_code=500, detail="We couldn't save your journal. Please try again.")

    return {"status": "completed", "module": "text", "id": saved["id"]}


@router.post("/voice")
async def checkin_voice(file: UploadFile = File(...), patient_id: str = Depends(get_current_patient_id)):
    """Runs the existing voice model (via ai.video.existing_voice_adapter)
    on an uploaded audio recording, saves the result to SQLite under the
    authenticated patient, and prints it to console."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio was received.")

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        try:
            from ai.video.existing_voice_adapter import analyze_audio

            result = analyze_audio(tmp_path)
            _print_result("VOICE CHECK-IN", result)
        except Exception:
            logger.exception("Voice AI pipeline failed")
            try:
                save_voice_record(patient_id, {"error": "processing_failed"}, "failed")
            except Exception:
                logger.exception("Failed to persist failed voice record")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "failed",
                    "module": "voice",
                    "error": "We couldn't process your recording. Please try again.",
                },
            )

        try:
            saved = save_voice_record(patient_id, result, "completed")
        except Exception:
            logger.exception("Failed to save voice record to database")
            raise HTTPException(status_code=500, detail="We couldn't save your recording. Please try again.")

        return {"status": "completed", "module": "voice", "id": saved["id"]}
    finally:
        os.unlink(tmp_path)


@router.post("/video")
async def checkin_video(file: UploadFile = File(...), patient_id: str = Depends(get_current_patient_id)):
    """Runs the full video pipeline (face + existing voice model + sync)
    on an uploaded video recording, saves the result under the
    authenticated patient, and prints it to console."""
    from ai.video.pipeline import analyze_video

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = analyze_video(tmp_path)
        _print_result("VIDEO CHECK-IN", result)
        saved = save_video_analysis(patient_id, result)
        return {"status": "completed", "module": "video", "id": saved["id"]}
    finally:
        os.unlink(tmp_path)


class CheckInSessionRequest(BaseModel):
    types: List[str]
    text: Optional[str] = None
    voice_seconds: Optional[int] = None
    video_seconds: Optional[int] = None


@router.post("/session")
def checkin_session(data: CheckInSessionRequest, patient_id: str = Depends(get_current_patient_id)):
    """
    Logs the session-level summary of a completed /check-in flow -- which
    modes were used, and their local metadata (previously held only in
    the browser's localStorage, see src/lib/store.ts's addCheckIn()). The
    AI-analyzed content of each mode is saved separately by /text,
    /voice, /video above; this ties them together as one check-in.
    """
    if not data.types:
        raise HTTPException(status_code=400, detail="At least one check-in type is required.")

    saved = save_check_in_session(
        patient_id, data.types, data.text, data.voice_seconds, data.video_seconds
    )
    return {"status": "completed", "id": saved["id"]}
