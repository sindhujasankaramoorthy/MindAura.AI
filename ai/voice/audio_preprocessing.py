import numpy as np
import scipy.signal as signal
import noisereduce as nr


def apply_bandpass_filter(
    audio: np.ndarray,
    sample_rate: int = 16000,
    lowcut: float = 80.0,
    highcut: float = 7500.0,
    order: int = 2
) -> np.ndarray:
    """
    Applies a smooth 2nd-order Butterworth bandpass filter.
    Preserves warm vocal harmonics (80 Hz - 7500 Hz) while removing low rumble and high hiss.
    """
    if len(audio) < 128:
        return audio
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    b, a = signal.butter(order, [low, high], btype="band")
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio.astype(np.float32)


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    noise_clip: np.ndarray | None = None,
    prop_decrease: float = 0.60
) -> np.ndarray:
    """
    Gentle spectral gating noise reduction to remove background hum/fans
    without causing robotic speech distortion or breakages.
    """
    if len(audio) == 0:
        return audio
    try:
        if noise_clip is not None and len(noise_clip) >= sample_rate * 0.5:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=True,
                n_fft=512,
                win_length=512
            )
        else:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=prop_decrease,
                stationary=True,
                n_fft=512,
                win_length=512
            )
        return cleaned.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Noise reduction fallback: {e}")
        return audio


def normalize_audio(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """
    Normalizes peak amplitude so speech is clear, full, and easy to hear.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1e-5:
        normalized = audio * (target_peak / max_val)
        return normalized.astype(np.float32)
    return audio


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    noise_clip: np.ndarray | None = None
) -> np.ndarray:
    """
    Clean audio processing pipeline:
    1. Smooth vocal bandpass (80 - 7500 Hz)
    2. Gentle noise reduction (prop_decrease=0.60 to avoid robotic voice breakages)
    3. Peak amplitude normalization
    """
    # 1. Smooth bandpass
    audio_filtered = apply_bandpass_filter(audio, sample_rate=sample_rate)

    # 2. Gentle noise reduction
    if noise_clip is not None and len(noise_clip) > 0:
        noise_clip_filtered = apply_bandpass_filter(noise_clip, sample_rate=sample_rate)
    else:
        noise_clip_filtered = None

    audio_denoised = reduce_noise(
        audio_filtered,
        sample_rate=sample_rate,
        noise_clip=noise_clip_filtered,
        prop_decrease=0.60
    )

    # 3. Peak Normalization
    audio_normalized = normalize_audio(audio_denoised)

    return audio_normalized
