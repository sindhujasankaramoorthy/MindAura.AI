import os
import sys

try:
    from ai.model_registry import get_faster_whisper
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from ai.model_registry import get_faster_whisper

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_model = None

def _load_model():
    global _model
    if _model is None:
        print("Loading Faster-Whisper model...")
        _model = get_faster_whisper()
        print("[OK] Faster-Whisper Model loaded successfully.\n")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV audio file into text.

    Args:
        audio_path (str): Path to WAV file.

    Returns:
        str: Transcribed text.
    """
    _load_model()

    segments, info = _model.transcribe(
        audio_path,
        beam_size=5
    )

    transcript = ""
    for segment in segments:
        transcript += segment.text + " "

    return transcript.strip()

if __name__ == "__main__":
    audio_path = os.path.join(
        os.path.dirname(__file__),
        "recordings",
        "recording.wav"
    )

    if not os.path.exists(audio_path):
        print(f"Test audio file not found at {audio_path}")
        sys.exit(1)

    print("Transcribing...")
    print()

    text = transcribe_audio(audio_path)

    print("=" * 50)
    print("TRANSCRIPT")
    print("=" * 50)
    print(text)