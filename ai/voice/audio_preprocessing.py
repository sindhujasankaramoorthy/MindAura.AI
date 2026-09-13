import os
import logging
from typing import Tuple, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import noisereduce as nr

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def load_audio(path: str, target_sr: Optional[int] = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono if it is stereo, 
    and optionally resamples it to the target sample rate.

    Args:
        path (str): The file path to the audio file.
        target_sr (Optional[int]): The target sample rate. If None, original sample rate is kept.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the mono audio waveform and the sample rate.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
        ValueError: If the audio file cannot be loaded or is corrupted.
    """
    if not os.path.exists(path):
        logger.error(f"Audio file not found: {path}")
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # librosa.load natively converts to mono and resamples if sr is provided.
        # using sr=None loads with original sampling rate unless specified.
        logger.info(f"Loading audio from {path} (Target SR: {target_sr})")
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio file {path}: {e}")
        raise ValueError(f"Failed to load audio file {path}: {e}") from e


def bandpass_filter(
    audio: np.ndarray,
    sr: int = 16000,
    lowcut: float = 80.0,
    highcut: float = 7500.0,
    order: int = 2
) -> np.ndarray:
    """
    Applies a smooth 2nd-order Butterworth bandpass filter.
    Preserves warm vocal harmonics (e.g., 80 Hz - 7500 Hz) while removing low rumble and high hiss.

    Args:
        audio (np.ndarray): The audio waveform.
        sr (int): The sample rate of the audio.
        lowcut (float): The lower frequency bound for the bandpass filter.
        highcut (float): The upper frequency bound for the bandpass filter.
        order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The bandpass-filtered audio waveform.
    """
    if len(audio) < 128:
        logger.warning("Audio too short for bandpass filtering.")
        return audio

    try:
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        # Ensure highcut is strictly less than the nyquist frequency
        high = min(highcut / nyquist, 0.99)
        
        # Verify that frequencies are valid
        if low <= 0 or high >= 1 or low >= high:
            logger.warning(f"Invalid filter frequencies (low={low}, high={high}). Returning original audio.")
            return audio

        b, a = signal.butter(order, [low, high], btype="band")
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Bandpass filter failed: {e}. Returning original audio.")
        return audio


def reduce_noise(
    audio: np.ndarray,
    sr: int = 16000,
    noise_clip: Optional[np.ndarray] = None,
    prop_decrease: float = 0.60
) -> np.ndarray:
    """
    Applies gentle spectral gating noise reduction to remove background hum/fans.
    Designed to prevent robotic speech distortion or breakages.

    Args:
        audio (np.ndarray): The audio waveform.
        sr (int): The sample rate of the audio.
        noise_clip (Optional[np.ndarray]): An optional clip containing only noise to profile.
        prop_decrease (float): The proportion by which to decrease noise (0.0 to 1.0).

    Returns:
        np.ndarray: The denoised audio waveform.
    """
    if len(audio) == 0:
        return audio

    try:
        if noise_clip is not None and len(noise_clip) >= sr * 0.5:
            logger.info("Applying noise reduction using provided noise clip.")
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=True,
                n_fft=512,
                win_length=512
            )
        else:
            logger.info("Applying stationary noise reduction (no valid noise clip provided).")
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=prop_decrease,
                stationary=True,
                n_fft=512,
                win_length=512
            )
        return cleaned.astype(np.float32)
    except Exception as e:
        logger.warning(f"Noise reduction fallback due to exception: {e}")
        return audio


def normalize_audio(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """
    Normalizes peak amplitude so speech is clear, full, and easy to hear.
    Prevents clipping by ensuring target_peak is not exceeded and clipping the final output.

    Args:
        audio (np.ndarray): The audio waveform.
        target_peak (float): The target peak amplitude (should be <= 1.0).

    Returns:
        np.ndarray: The normalized and clipped audio waveform.
    """
    try:
        max_val = np.max(np.abs(audio))
        if max_val > 1e-5:
            # Normalize to target peak
            normalized = audio * (target_peak / max_val)
            # Clip to prevent any accidental overdrive over 1.0/-1.0
            clip_bound = min(target_peak, 1.0)
            normalized = np.clip(normalized, -clip_bound, clip_bound)
            return normalized.astype(np.float32)
        
        logger.debug("Audio max value is too small; skipping normalization.")
        return audio
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return audio


def save_audio(audio: np.ndarray, sr: int, path: str) -> str:
    """
    Saves the audio waveform to the specified file path.

    Args:
        audio (np.ndarray): The audio waveform.
        sr (int): The sample rate of the audio.
        path (str): The destination file path.

    Returns:
        str: The path where the audio was saved.

    Raises:
        IOError: If there's an error saving the audio.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, audio, sr)
        logger.info(f"Successfully saved audio to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save audio to {path}: {e}")
        raise IOError(f"Failed to save audio to {path}: {e}") from e


def preprocess_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    noise_clip: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, str]:
    """
    Full clean audio processing pipeline:
    1. Loads audio (safely converting to mono and standardizing sample rate).
    2. Smooth vocal bandpass filtering (80 - 7500 Hz).
    3. Gentle noise reduction (prop_decrease=0.60 to avoid robotic voice breakages).
    4. Peak amplitude normalization (prevents clipping).
    5. Saves the processed audio to output_path.

    Args:
        input_path (str): The path to the input audio file.
        output_path (str): The path to save the processed audio file.
        target_sr (int): The desired sample rate (default 16000 Hz, compatible with Faster-Whisper).
        noise_clip (Optional[np.ndarray]): An optional clip containing noise profiling.

    Returns:
        Tuple[np.ndarray, str]: The processed waveform and the output file path.
    """
    logger.info(f"Starting audio preprocessing for {input_path}")
    
    # 1. Load Audio
    audio, sr = load_audio(input_path, target_sr=target_sr)

    # 2. Smooth bandpass
    audio_filtered = bandpass_filter(audio, sr=sr)
    
    # Filter noise clip if provided
    noise_clip_filtered = None
    if noise_clip is not None and len(noise_clip) > 0:
        noise_clip_filtered = bandpass_filter(noise_clip, sr=sr)

    # 3. Gentle noise reduction
    audio_denoised = reduce_noise(
        audio_filtered,
        sr=sr,
        noise_clip=noise_clip_filtered,
        prop_decrease=0.60
    )

    # 4. Peak Normalization
    audio_normalized = normalize_audio(audio_denoised)

    # 5. Save the processed audio
    final_output_path = save_audio(audio_normalized, sr, output_path)
    
    logger.info(f"Finished audio preprocessing. Output saved to {final_output_path}")

    return audio_normalized, final_output_path

if __name__ == "__main__":
    print("Testing audio_preprocessing.py imports and functions...")
    # Generate a dummy 1-second 16kHz sine wave audio to test the functions
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    dummy_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test bandpass filter
    filtered = bandpass_filter(dummy_audio, sr=sample_rate)
    print(f"Bandpass filter returned shape {filtered.shape}")
    
    # Test noise reduction (without noise clip)
    denoised = reduce_noise(filtered, sr=sample_rate)
    print(f"Noise reduction returned shape {denoised.shape}")
    
    # Test normalization
    normalized = normalize_audio(denoised)
    print(f"Normalized max value: {np.max(np.abs(normalized)):.2f}")
    
    print("All preprocessing module functions executed successfully!")
