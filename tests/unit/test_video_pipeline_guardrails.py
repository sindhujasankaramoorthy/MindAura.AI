"""
Guardrail tests for the video-analysis pipeline: confirms none of the
active ai/video/ or ai/face/ source files import Qwen, the RoBERTa/emotion
stack, or Ollama, and that the output schema never claims emotion
classification, clinical interpretation, or diagnosis.
"""
import ast
import os

FORBIDDEN_IMPORT_SUBSTRINGS = (
    "qwen_reasoning",
    "emotion_predict",
    "ollama",
    "psychological_signals",
)

PIPELINE_DIRS = ["ai/video", "ai/face"]


def _all_python_files():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = []
    for d in PIPELINE_DIRS:
        full_dir = os.path.join(repo_root, d)
        for root, _dirs, filenames in os.walk(full_dir):
            if "__pycache__" in root:
                continue
            for fname in filenames:
                if fname.endswith(".py"):
                    files.append(os.path.join(root, fname))
    return files


def test_no_forbidden_imports_in_video_or_face_pipeline():
    offenders = []
    for path in _all_python_files():
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(forbidden in name for forbidden in FORBIDDEN_IMPORT_SUBSTRINGS):
                    offenders.append((path, name))

    assert offenders == [], f"Forbidden imports found: {offenders}"


def test_analysis_metadata_declares_no_emotion_or_diagnosis():
    from ai.video.schemas import build_analysis_json

    result = build_analysis_json(
        video_id="v1",
        video_metadata={"duration_sec": 1.0, "fps": 10, "resolution": {"width": 1, "height": 1},
                         "has_audio": False, "audio_duration_sec": "not_available"},
        face_result={"detected": False},
        voice_result={"status": "no_audio", "transcription": {}, "acoustics": {}, "duration_sec": "not_available"},
        synchronized_events=[],
    )
    metadata = result["analysis_metadata"]
    assert metadata["emotion_classification"] is False
    assert metadata["clinical_interpretation"] is False
    assert metadata["diagnosis"] is False
    assert metadata["qwen_used"] is False


def test_tear_observation_never_claims_crying():
    from ai.face.tear_observation import analyze_tear_presence

    result = analyze_tear_presence([])
    assert result["status"] == "not_trained"
    assert "person_is_crying" not in result
    assert all(v in ("not_available", "not_trained") or isinstance(v, str) for v in result.values())
