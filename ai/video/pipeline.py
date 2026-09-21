"""
Top-level video analysis pipeline -- the single public entry point:

    from ai.video.pipeline import analyze_video
    result = analyze_video("patient_submission.mp4")

    video
      |-- video frames -> face pipeline (ai.face.pipeline)
      \\-- audio track  -> existing voice model (ai.video.existing_voice_adapter)
    face + voice -> timestamp synchronization (ai.video.sync)
    -> one JSON (ai.video.schemas)

Does not call Qwen, does not classify emotion, does not diagnose. See
ai/video/schemas.py's analysis_metadata block for the explicit confirmation
of what this pipeline does NOT do.
"""
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, Optional

from ai.video import video_io
from ai.video.existing_voice_adapter import analyze_audio
from ai.video.sync import build_voice_events, synchronize_events
from ai.video.schemas import build_analysis_json
from ai.face.pipeline import run_face_pipeline, build_empty_face_result

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FPS = 10  # subsample video for face processing -- see video_io.extract_frames


def analyze_video(
    video_path: str,
    video_id: Optional[str] = None,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> Dict[str, Any]:
    """
    Runs the full pipeline on one video file and returns the canonical
    observation JSON (see ai/video/schemas.py). Never raises for expected
    failure conditions (invalid file, no face, no audio) -- always returns
    a valid JSON dict, with "not_available" for anything that couldn't be
    computed.
    """
    video_id = video_id or str(uuid.uuid4())

    try:
        video_io.validate_video(video_path)
    except video_io.VideoValidationError as e:
        logger.error("Video validation failed: %s", e)
        return build_analysis_json(
            video_id=video_id,
            video_metadata={"duration_sec": "not_available", "fps": "not_available",
                             "resolution": {"width": "not_available", "height": "not_available"},
                             "has_audio": False, "audio_duration_sec": "not_available"},
            face_result=build_empty_face_result(f"Video validation failed: {e}"),
            voice_result={"status": "error", "transcription": {}, "acoustics": {}, "duration_sec": "not_available"},
            synchronized_events=[],
        )

    metadata = video_io.extract_metadata(video_path)

    # --- Voice branch ---
    voice_result: Dict[str, Any]
    if metadata["has_audio"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "extracted_audio.wav")
            try:
                video_io.extract_audio(video_path, audio_path)
                voice_result = analyze_audio(audio_path)
            except Exception as e:
                logger.exception("Audio extraction/analysis failed")
                voice_result = {"status": "error", "error": str(e), "transcription": {}, "acoustics": {}, "duration_sec": "not_available"}
    else:
        logger.info("Video has no audio track -- skipping voice analysis.")
        voice_result = {
            "status": "no_audio",
            "transcription": {"raw_text": "not_available", "segments": [], "language_detected": "not_available", "languages_detected": []},
            "acoustics": {},
            "duration_sec": "not_available",
        }

    # --- Face branch ---
    try:
        frame_iter = video_io.extract_frames(video_path, target_fps=target_fps)
        face_result = run_face_pipeline(frame_iter)
    except Exception as e:
        logger.exception("Face pipeline failed")
        face_result = build_empty_face_result(f"Face pipeline error: {e}")

    # --- Synchronization ---
    face_events = face_result.get("temporal_events", [])
    voice_events = build_voice_events(voice_result.get("transcription", {}).get("segments", []))
    synchronized_events = synchronize_events(face_events, voice_events)

    return build_analysis_json(
        video_id=video_id,
        video_metadata=metadata,
        face_result=face_result,
        voice_result=voice_result,
        synchronized_events=synchronized_events,
    )


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) < 2:
        print("Usage: python -m ai.video.pipeline <path_to_video.mp4>")
        sys.exit(1)

    result = analyze_video(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
