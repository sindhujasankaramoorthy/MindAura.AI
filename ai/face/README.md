# Face analysis (ai/face/)

Measurable facial-behavior observations from video -- NOT an emotion
classifier, NOT a diagnostic system.

## Files

| File | Responsibility |
|---|---|
| `landmarker.py` | MediaPipe Face Landmarker wrapper (VIDEO mode, landmarks + blendshapes + transformation matrix) |
| `features.py` | Per-frame geometric/expression measurements from one detection result |
| `events.py` | Rule-based (threshold/hysteresis) temporal event detection -- blink, smile, eyebrow raise -- and aggregate summaries (gaze, mouth, head pose) |
| `tear_observation.py` | Optional tear/eye-wetness interface; always returns `not_trained` (no verified dataset exists) |
| `pipeline.py` | Orchestrates the above into the "face" JSON section |
| `temporal_model.py` | Architecture for a trained movement-segmentation model (GRU-based) -- defined, not trained |
| `train_temporal_model.py` | Training script for the above -- not run; see `data/README.md` for why |
| `models/` | Downloaded MediaPipe `.task` model file (gitignored) |
| `data/` | Dataset raw/processed storage (gitignored) -- see `data/README.md` |
| `checkpoints/` | Trained model checkpoints, if/when training runs (gitignored) |

## What this measures

Eyes (openness, blink count/duration, squint, gaze direction), eyebrows
(raise/lower, asymmetry), mouth/lips/jaw (opening, width, pucker, stretch),
smile (detected, intensity, symmetry, event count), head pose
(pitch/yaw/roll, movement range), and temporal events with
start/end/duration/intensity for each of the above.

Every field that MediaPipe doesn't actually produce (e.g. a per-frame
detection confidence score) is `"not_available"`, never a fabricated
number.

## Usage

```python
from ai.face.pipeline import run_face_pipeline
from ai.video.video_io import extract_frames

result = run_face_pipeline(extract_frames("video.mp4", target_fps=10))
```

Normally called via `ai.video.pipeline.analyze_video()`, not directly.
