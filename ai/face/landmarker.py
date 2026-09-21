"""
MediaPipe Face Landmarker wrapper -- the primary facial geometry/tracking
component (per the video-pipeline spec). Runs in VIDEO mode (one timestamp
per call, tracking across frames), with landmarks, blendshapes, and the
facial transformation matrix all enabled.

This is a pretrained, off-the-shelf Google model (not something we train) --
it just needs its .task model file downloaded once. See MODEL_URL below.
"""
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model_downloaded() -> str:
    """
    Downloads MediaPipe's official face_landmarker.task file if it isn't
    already present locally. Not committed to git (it's a ~4-9MB binary,
    same convention as ai/text/language/models/lid.176.bin elsewhere in
    this repo) -- see .gitignore.
    """
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Downloading MediaPipe face_landmarker.task to %s ...", MODEL_PATH)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceLandmarker:
    """
    Thin wrapper around mediapipe.tasks.vision.FaceLandmarker in VIDEO
    running mode. One instance should be reused across all frames of a
    single video (tracking state is per-instance), then closed.
    """

    def __init__(self, num_faces: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_model_downloaded()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms: int) -> Dict[str, Any]:
        """
        Runs detection on one BGR frame (as returned by OpenCV) at
        `timestamp_ms` (must be monotonically increasing across calls on
        this instance, per MediaPipe's VIDEO-mode contract).

        Returns:
          {
            "face_detected": bool,
            "landmarks": [[x, y, z], ...] | None,       # normalized coords
            "blendshapes": {category_name: score, ...} | None,
            "transformation_matrix": [[...4x4...]] | None,
          }
        """
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return {"face_detected": False, "landmarks": None, "blendshapes": None, "transformation_matrix": None}

        landmarks = [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]

        blendshapes: Optional[Dict[str, float]] = None
        if result.face_blendshapes:
            blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}

        transformation_matrix: Optional[List[List[float]]] = None
        if result.facial_transformation_matrixes:
            transformation_matrix = result.facial_transformation_matrixes[0].tolist()

        return {
            "face_detected": True,
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "transformation_matrix": transformation_matrix,
        }

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
