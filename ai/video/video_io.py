"""
Video input pipeline: validate -> extract metadata -> extract audio ->
extract frames. Uses the system ffmpeg/ffprobe binaries (already present on
this machine, already the tool the rest of the repo relies on for audio
work) for metadata/audio extraction, and OpenCV for frame extraction --
no new heavy dependencies beyond what the face pipeline already needs.
"""
import json
import logging
import os
import subprocess
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "not_available"


class VideoValidationError(Exception):
    """Raised when the input file isn't a readable video (see validate_video)."""


def validate_video(video_path: str) -> None:
    """
    Raises VideoValidationError if `video_path` doesn't exist or ffprobe
    can't find a decodable video stream in it. Does not raise for a
    missing/absent AUDIO stream -- that's a valid, separately-handled case
    (see extract_audio), not an invalid video.
    """
    if not video_path or not os.path.exists(video_path):
        raise VideoValidationError(f"Video file not found: {video_path}")

    try:
        streams = _probe_streams(video_path)
    except Exception as e:
        raise VideoValidationError(f"ffprobe could not read '{video_path}': {e}")

    if not any(s.get("codec_type") == "video" for s in streams):
        raise VideoValidationError(f"No decodable video stream found in '{video_path}'")


def _probe_streams(video_path: str) -> list:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream", "-of", "json", video_path],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return json.loads(proc.stdout).get("streams", [])


def extract_metadata(video_path: str) -> Dict[str, Any]:
    """
    Returns duration/fps/resolution/audio-presence, using only what ffprobe
    actually reports -- "not_available" for anything it can't determine
    (e.g. a video with a video stream that has no readable frame rate).
    """
    streams = _probe_streams(video_path)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    metadata: Dict[str, Any] = {
        "duration_sec": NOT_AVAILABLE,
        "fps": NOT_AVAILABLE,
        "resolution": {"width": NOT_AVAILABLE, "height": NOT_AVAILABLE},
        "has_audio": audio_stream is not None,
        "audio_duration_sec": NOT_AVAILABLE,
    }

    if video_stream:
        if "width" in video_stream and "height" in video_stream:
            metadata["resolution"] = {"width": video_stream["width"], "height": video_stream["height"]}
        if video_stream.get("duration"):
            metadata["duration_sec"] = round(float(video_stream["duration"]), 3)
        fps = _parse_frame_rate(video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate"))
        if fps is not None:
            metadata["fps"] = fps

    if audio_stream and audio_stream.get("duration"):
        metadata["audio_duration_sec"] = round(float(audio_stream["duration"]), 3)

    return metadata


def _parse_frame_rate(rate_str: Optional[str]) -> Optional[float]:
    if not rate_str or rate_str in ("0/0", "N/A"):
        return None
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            den = float(den)
            return round(float(num) / den, 3) if den else None
        return round(float(rate_str), 3)
    except (ValueError, ZeroDivisionError):
        return None


def extract_audio(video_path: str, output_wav_path: str) -> Optional[str]:
    """
    Extracts the audio track as 16kHz mono WAV (the format the existing
    voice model's underlying Whisper/librosa calls expect) using ffmpeg.
    Returns None (not an exception) if the video has no audio track --
    that's a valid, expected case (section 14: "audio-only/video-without-
    audio failure" test), not a crash condition.
    """
    metadata = extract_metadata(video_path)
    if not metadata["has_audio"]:
        logger.warning("No audio stream in %s -- skipping audio extraction.", video_path)
        return None

    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", output_wav_path],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()[-500:]}")
    return output_wav_path


def extract_frames(video_path: str, target_fps: Optional[float] = None) -> Iterator[Tuple[float, "Any"]]:
    """
    Yields (timestamp_sec, frame) for the video, one frame at a time
    (generator, not a loaded-into-memory list, since patient videos can be
    long). `target_fps`, if given, subsamples the video to roughly that
    rate instead of processing every native frame -- MediaPipe's face
    landmarker doesn't need 30fps to detect a blink onset reliably, and
    processing every frame of a multi-minute video is wasteful.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoValidationError(f"OpenCV could not open '{video_path}'")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    step = 1
    if target_fps and native_fps > target_fps:
        step = max(1, round(native_fps / target_fps))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                timestamp_sec = frame_idx / native_fps if native_fps else frame_idx * (1 / 30)
                yield timestamp_sec, frame
            frame_idx += 1
    finally:
        cap.release()
