import sounddevice as sd
import soundfile as sf
import numpy as np

print("=" * 50)
print("AVAILABLE DEVICES")
print("=" * 50)

print(sd.query_devices())

print("\nDefault Device:", sd.default.device)

# CHANGE THIS IF NEEDED
MIC_DEVICE = 14      # Noise Buds N1 (Windows WASAPI)

duration = 5
sample_rate = 16000

print("\nRecording...")
print("Speak into your Noise Buds microphone...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    device=MIC_DEVICE
)

sd.wait()

sf.write("test_recording.wav", audio, sample_rate)

print("\nRecording Finished.")

print("\n========== AUDIO ANALYSIS ==========")

print("Minimum :", np.min(audio))
print("Maximum :", np.max(audio))
print("Mean    :", np.mean(np.abs(audio)))

if np.max(np.abs(audio)) < 0.001:
    print("\n❌ No microphone signal detected.")
else:
    print("\n✅ Microphone is working!")

print("\nSaved as test_recording.wav")