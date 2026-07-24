import os
import sys
import time
import queue
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from ai.voice.audio_preprocessing import preprocess_audio
except ImportError:
    from audio_preprocessing import preprocess_audio


class SmartRecorder:
    """
    High-fidelity continuous stream Voice Activity Detector & Recorder.

    Key Features:
    1. Continuous Audio Streaming (`sd.InputStream`):
       Eliminates all frame drops, gaps, and voice breakages.

    2. Pre-roll Buffer (1.5 seconds):
       Keeps a rolling buffer of recent audio while waiting, ensuring the
       very first word/syllable of speech is captured and never cut off.

    3. Gentle VAD & Dual Gate:
       Uses speech-band energy + ZCR gate with lowered start threshold (1.8x)
       so soft speech triggers cleanly.

    4. Adaptive Silence Threshold:
       Adapts dynamically to the user's vocal peak so natural pauses don't cut off speech.
    """

    def __init__(self):

        # ── Audio parameters ───────────────────────────────────────────────
        self.SAMPLE_RATE     = 16000
        self.CHANNELS        = 1
        self.CHUNK_SAMPLES   = 1600      # 100ms per processing frame
        self.CHUNK_DURATION  = self.CHUNK_SAMPLES / self.SAMPLE_RATE  # 0.1s

        # ── Timing ────────────────────────────────────────────────────────
        self.CALIBRATION_TIME = 2.0      # seconds of silence for baseline
        self.MAX_WAIT_TIME    = 20.0     # max wait before giving up
        self.MAX_RECORD_TIME  = 180.0    # hard limit (3 minutes)
        self.MIN_RECORD_SECS  = 1.5      # minimum record duration

        # ── Pre-roll buffer ───────────────────────────────────────────────
        # Holds 1.5s of audio so the start of speech is NEVER clipped
        self.PREROLL_CHUNKS   = 15       # 15 * 0.1s = 1.5 seconds

        # ── Silence-to-stop ───────────────────────────────────────────────
        # 4.5 seconds of silence to stop recording (45 * 0.1s)
        self.SILENCE_CHUNKS   = 45

        # ── Start Hysteresis ──────────────────────────────────────────────
        # 2 consecutive frames (~0.2s) of voice to start
        self.CONFIRM_CHUNKS   = 2

        # ── Threshold Multipliers ─────────────────────────────────────────
        self.START_MULT       = 1.8      # lower threshold so start of speech is easily caught
        self.SILENCE_MULT     = 1.3      # silence threshold

        # ── ZCR Gate ──────────────────────────────────────────────────────
        self.ZCR_MAX          = 0.50     # reject pure noise bursts

        # ── Adaptive Threshold ───────────────────────────────────────────
        self.ADAPTIVE_FACTOR  = 0.12     # 12% of peak voice level

        # ── Audio Device ─────────────────────────────────────────────────
        self.DEVICE = None               # default system microphone

        # ── Output Path ──────────────────────────────────────────────────
        self.recordings_dir = os.path.join(
            os.path.dirname(__file__), "recordings"
        )
        os.makedirs(self.recordings_dir, exist_ok=True)

        # State variables
        self.background_noise  = None
        self.start_threshold   = None
        self.silence_threshold = None
        self.noise_clip        = None
        self.audio_queue       = queue.Queue()

    # ──────────────────────────────────────────────────────────────────────
    # Audio Stream Callback
    # ──────────────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """Continuous callback called by sounddevice in a background thread."""
        if status:
            print(f"[STREAM STATUS] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy().flatten())

    # ──────────────────────────────────────────────────────────────────────
    # VAD Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _speech_band_energy(self, audio: np.ndarray) -> float:
        """RMS energy in human speech band (200 - 3800 Hz)."""
        if len(audio) == 0:
            return 1e-9
        fft_vals = np.fft.rfft(audio)
        freqs    = np.fft.rfftfreq(len(audio), d=1.0 / self.SAMPLE_RATE)
        mask     = (freqs >= 200) & (freqs <= 3800)
        filtered = np.fft.irfft(fft_vals * mask, n=len(audio))
        return float(max(np.sqrt(np.mean(filtered ** 2)), 1e-9))

    def _zcr(self, audio: np.ndarray) -> float:
        """Zero Crossing Rate."""
        if len(audio) <= 1:
            return 0.0
        crossings = np.sum(np.diff(np.sign(audio)) != 0)
        return crossings / (len(audio) - 1)

    def _is_voice(self, audio: np.ndarray, threshold: float) -> bool:
        energy = self._speech_band_energy(audio)
        zcr    = self._zcr(audio)
        return energy > threshold and zcr < self.ZCR_MAX

    # ──────────────────────────────────────────────────────────────────────

    def calibrate_noise(self, stream: sd.InputStream):
        """Calibrates background noise from the live stream."""
        print("=" * 60)
        print("[~] Calibrating background noise...")
        print("    Please stay silent for 2 seconds...")
        print("=" * 60)

        # Clear queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        noise_chunks = []
        noise_levels = []
        needed_chunks = int(self.CALIBRATION_TIME / self.CHUNK_DURATION)

        while len(noise_chunks) < needed_chunks:
            try:
                chunk = self.audio_queue.get(timeout=2.0)
                noise_chunks.append(chunk)
                noise_levels.append(self._speech_band_energy(chunk))
            except queue.Empty:
                break

        if noise_chunks:
            self.noise_clip = np.concatenate(noise_chunks, axis=0)
            self.background_noise = max(float(np.mean(noise_levels)), 1e-6)
        else:
            self.background_noise = 1e-3
            self.noise_clip = None

        self.start_threshold   = self.background_noise * self.START_MULT
        self.silence_threshold = self.background_noise * self.SILENCE_MULT

        print(f"\n  Background noise floor   : {self.background_noise:.6f}")
        print(f"  Speech-start threshold   : {self.start_threshold:.6f}")
        print(f"  Silence threshold        : {self.silence_threshold:.6f}")
        print(f"  Pre-roll buffer          : {self.PREROLL_CHUNKS * self.CHUNK_DURATION:.1f} s")
        print(f"  Silence window           : {self.SILENCE_CHUNKS * self.CHUNK_DURATION:.1f} s\n")

    # ──────────────────────────────────────────────────────────────────────

    def record(self, filename: str = "recording.wav") -> str | None:
        """
        Records voice with zero breakages and full pre-roll capture.
        """
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK_SAMPLES,
            device=self.DEVICE,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._audio_callback
        ):
            self.calibrate_noise(None)

            print(f"[MIC] Listening for speech (up to {self.MAX_WAIT_TIME}s)...\n")

            preroll_buffer = deque(maxlen=self.PREROLL_CHUNKS)
            recorded_chunks: list[np.ndarray] = []

            started        = False
            silence_count  = 0
            confirm_count  = 0
            record_time    = 0.0
            peak_voice     = 0.0
            wait_start     = time.time()

            while True:
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    print("[WARN] Audio queue empty")
                    continue

                energy = self._speech_band_energy(chunk)
                zcr    = self._zcr(chunk)
                voice  = self._is_voice(chunk, self.start_threshold)

                # ── Phase 1: Waiting for Speech ─────────────────────────────
                if not started:
                    elapsed = time.time() - wait_start
                    preroll_buffer.append(chunk)

                    confirm_count = (confirm_count + 1) if voice else 0

                    if confirm_count >= self.CONFIRM_CHUNKS:
                        print("\n[ON] Speech detected -- recording started!\n")
                        started = True
                        silence_count = 0
                        peak_voice    = energy

                        # Prepend the ENTIRE pre-roll buffer so start of speech is NEVER missed!
                        recorded_chunks.extend(list(preroll_buffer))
                        record_time += len(preroll_buffer) * self.CHUNK_DURATION
                        continue

                    if elapsed >= self.MAX_WAIT_TIME:
                        print("\n[X] No speech detected within the wait window.")
                        return None

                    continue

                # ── Phase 2: Recording Speech ───────────────────────────────
                recorded_chunks.append(chunk)
                record_time += self.CHUNK_DURATION

                if voice:
                    peak_voice = max(peak_voice, energy)

                adaptive_threshold = max(
                    self.silence_threshold,
                    peak_voice * self.ADAPTIVE_FACTOR
                )

                is_silent = (energy < adaptive_threshold) and not voice
                silence_count = (silence_count + 1) if is_silent else 0
                can_stop = record_time >= self.MIN_RECORD_SECS

                # Log progress every ~0.5s
                if len(recorded_chunks) % 5 == 0:
                    print(
                        f"  [recording] energy={energy:.5f}  zcr={zcr:.3f}  "
                        f"silent={silence_count:>2}/{self.SILENCE_CHUNKS}  "
                        f"dur={record_time:.1f}s"
                    )

                if can_stop and silence_count >= self.SILENCE_CHUNKS:
                    print(
                        f"\n[STOP] {self.SILENCE_CHUNKS * self.CHUNK_DURATION:.1f}s of silence "
                        f"-- recording ended."
                    )
                    break

                if record_time >= self.MAX_RECORD_TIME:
                    print(f"\n[LIMIT] Max recording time reached.")
                    break

        # ── Phase 3: Postprocessing & Saving ───────────────────────────────
        if not recorded_chunks:
            print("[WARN] Nothing was recorded.")
            return None

        audio = np.concatenate(recorded_chunks, axis=0)

        print("[CLEAN] Cleaning audio with smooth noise reduction...")
        audio = preprocess_audio(
            audio,
            sample_rate=self.SAMPLE_RATE,
            noise_clip=self.noise_clip
        )

        output_path = os.path.join(self.recordings_dir, filename)
        sf.write(output_path, audio, self.SAMPLE_RATE)

        print("\n" + "=" * 60)
        print("[OK] Recording complete!")
        print(f"     File   : {output_path}")
        print(f"     Length : {record_time:.1f}s")
        print(f"     Peak   : {peak_voice:.5f}")
        print("=" * 60 + "\n")

        return output_path


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--device", type=int, default=None, help="Device index")
    args = parser.parse_args()

    if args.list_devices:
        print("=" * 60)
        print("Available Audio Input Devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[0] else ""
                print(f"  [{i:2d}] {dev['name']}{marker}")
        sys.exit(0)

    print("Starting Smart Recorder (Continuous Stream Mode)...\n")
    recorder = SmartRecorder()

    if args.device is not None:
        recorder.DEVICE = args.device
        print(f"[INFO] Using device index {args.device}\n")

    recorder.record()