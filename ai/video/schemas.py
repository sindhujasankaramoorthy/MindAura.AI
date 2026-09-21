"""
Builds the final, canonical MindAura video-analysis observation JSON. This
is the ONLY place the top-level schema shape is assembled, so face/voice/
sync results all land in one predictable structure.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ANALYSIS_VERSION = "1.0"


def build_analysis_json(
    video_id: str,
    video_metadata: Dict[str, Any],
    face_result: Dict[str, Any],
    voice_result: Dict[str, Any],
    synchronized_events: List[Dict[str, Any]],
    analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "analysis_id": analysis_id or str(uuid.uuid4()),
        "video_id": video_id,
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),

        "video": {
            "duration_sec": video_metadata.get("duration_sec"),
            "fps": video_metadata.get("fps"),
            "resolution": video_metadata.get("resolution"),
            "has_audio": video_metadata.get("has_audio"),
            "audio_duration_sec": video_metadata.get("audio_duration_sec"),
        },

        "face": face_result,

        "voice": {
            "transcription": voice_result.get("transcription", {}),
            "acoustics": voice_result.get("acoustics", {}),
            "duration_sec": voice_result.get("duration_sec"),
            "status": voice_result.get("status"),
        },

        "synchronized_events": synchronized_events,

        "analysis_metadata": {
            "face_model": "mediapipe_face_landmarker + rule-based temporal event detector",
            "voice_model": "existing_voice_model (ai.voice.voice_analysis.analyze_voice)",
            "emotion_classification": False,
            "clinical_interpretation": False,
            "diagnosis": False,
            "qwen_used": False,
        },
    }
