# Video analysis pipeline (ai/video/)

Top-level orchestration: one patient video in, one observation JSON out.
Combines the new face pipeline (`ai/face/`) with the *existing* voice model
(`ai/voice/voice_analysis.py`) -- no duplicate transcription/acoustic
model is created here.

## Files

| File | Responsibility |
|---|---|
| `pipeline.py` | Public entry point: `analyze_video(path)` |
| `video_io.py` | Validate, extract metadata (ffprobe), extract audio (ffmpeg), extract frames (OpenCV) |
| `existing_voice_adapter.py` | Thin adapter around `ai.voice.voice_analysis.analyze_voice()` -- reshapes its real output into the common schema, marks unsupported fields `not_available` |
| `sync.py` | Timestamp overlap between face and voice events -- reports overlap only, no interpretation |
| `schemas.py` | Assembles the final canonical JSON |

## Usage

```bash
python -m ai.video.pipeline path/to/video.mp4
```

```python
from ai.video.pipeline import analyze_video
result = analyze_video("path/to/video.mp4", video_id="optional-id")
```

## Explicitly does NOT do

No Qwen, no emotion classification, no diagnosis, no clinical
interpretation -- see `result["analysis_metadata"]`, which states this
explicitly in every output.
