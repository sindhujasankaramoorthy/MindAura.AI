import os
import sys
from faster_whisper import WhisperModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -----------------------------
# Load Whisper model
# -----------------------------
print("Loading Faster-Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("[OK] Model loaded successfully.\n")


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV audio file into text.

    Args:
        audio_path (str): Path to WAV file.

    Returns:
        str: Transcribed text.
    """

    segments, info = model.transcribe(
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

    print("Transcribing...")
    print()

    text = transcribe_audio(audio_path)

    print("=" * 50)
    print("TRANSCRIPT")
    print("=" * 50)
    print(text)