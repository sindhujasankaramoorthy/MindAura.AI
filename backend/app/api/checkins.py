"""
Check-in ingestion endpoints -- connects the MindAura patient frontend
(src/routes/check-in.tsx) to the AI pipelines already built in ai/text/ and
ai/video/. Each endpoint runs the appropriate pipeline, saves the structured
result to the patient's record, and prints it to the server console for
visibility during development.

There is no patient login yet (see backend/app/api/auth.py's single
CURRENT_ACTIVE_USER_ID pattern for the doctor side), so check-ins default to
the seeded demo patient "pat-1" unless a patient_id is supplied.
"""
import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.database import save_text_journal, save_voice_record, save_video_analysis

logger = logging.getLogger("medtrust.checkins")
router = APIRouter(prefix="/checkins", tags=["checkins"])

DEFAULT_PATIENT_ID = "pat-1"


def _print_result(title: str, result: dict) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70 + "\n")


@router.post("/text")
def checkin_text(text: str = Form(...), patient_id: str = Form(DEFAULT_PATIENT_ID)):
    """Runs the text pipeline (ai.text.pipeline.process_text), saves the
    structured result to the patient's record, and prints it to console."""
    from ai.text.pipeline import process_text

    result = process_text(text)
    _print_result("TEXT CHECK-IN", result)
    saved = save_text_journal(patient_id, text, result)
    return {"status": "received", "module": "text", "id": saved["id"]}


@router.post("/voice")
async def checkin_voice(file: UploadFile = File(...), patient_id: str = Form(DEFAULT_PATIENT_ID)):
    """Runs the existing voice model (via ai.video.existing_voice_adapter)
    on an uploaded audio recording, saves the result, and prints it."""
    from ai.video.existing_voice_adapter import analyze_audio

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = analyze_audio(tmp_path)
        _print_result("VOICE CHECK-IN", result)
        saved = save_voice_record(patient_id, result)
        return {"status": "received", "module": "voice", "id": saved["id"]}
    finally:
        os.unlink(tmp_path)


@router.post("/video")
async def checkin_video(file: UploadFile = File(...), patient_id: str = Form(DEFAULT_PATIENT_ID)):
    """Runs the full video pipeline (face + existing voice model + sync)
    on an uploaded video recording, saves the result, and prints it."""
    from ai.video.pipeline import analyze_video

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = analyze_video(tmp_path)
        _print_result("VIDEO CHECK-IN", result)
        saved = save_video_analysis(patient_id, result)
        return {"status": "received", "module": "video", "id": saved["id"]}
    finally:
        os.unlink(tmp_path)
