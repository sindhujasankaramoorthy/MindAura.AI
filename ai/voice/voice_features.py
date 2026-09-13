import os
import numpy as np
import librosa


def extract_voice_features(audio_path):
    """
    Extract important voice features from an audio file.

    Returns:
        dict: Extracted audio features.
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    features = {}

    # -------------------------
    # Duration
    # -------------------------
    features["duration"] = round(librosa.get_duration(y=y, sr=sr), 2)

    # -------------------------
    # RMS Energy (Loudness)
    # -------------------------
    rms = librosa.feature.rms(y=y)[0]
    features["rms_energy"] = round(float(np.mean(rms)), 4)

    # -------------------------
    # Zero Crossing Rate
    # -------------------------
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zero_crossing_rate"] = round(float(np.mean(zcr)), 4)

    # -------------------------
    # Spectral Centroid
    # -------------------------
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid"] = round(float(np.mean(spectral_centroid)), 2)

    # -------------------------
    # Pitch Estimation
    # -------------------------
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    voiced_pitch = f0[~np.isnan(f0)]

    if len(voiced_pitch) > 0:
        features["pitch"] = round(float(np.mean(voiced_pitch)), 2)
        features["pitch_variability"] = round(float(np.std(voiced_pitch)), 2)
        
        # Simple Voice Tremor proxy: average absolute difference between consecutive f0 points
        pitch_diffs = np.abs(np.diff(voiced_pitch))
        features["voice_tremor"] = round(float(np.mean(pitch_diffs)), 2) if len(pitch_diffs) > 0 else 0.0
    else:
        features["pitch"] = 0.0
        features["pitch_variability"] = 0.0
        features["voice_tremor"] = 0.0

    # -------------------------
    # Speech Rate & Pause Ratio
    # -------------------------
    try:
        # Split audio on silence (top_db=30 is a common threshold)
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        
        non_silent_samples = sum([end - start for start, end in non_silent_intervals])
        total_samples = len(y)
        
        pause_samples = total_samples - non_silent_samples
        pause_ratio = pause_samples / total_samples if total_samples > 0 else 0.0
        features["pause_ratio"] = round(float(pause_ratio), 4)
        
        # Speech Rate Proxy: Number of onsets (syllable proxy) per second of non-silent audio
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration_sec = features["duration"]
        
        if duration_sec > 0:
            features["speech_rate"] = round(float(len(onsets) / duration_sec), 2)
        else:
            features["speech_rate"] = 0.0
            
    except Exception as e:
        features["pause_ratio"] = 0.0
        features["speech_rate"] = 0.0

    # -------------------------
    # MFCC
    # -------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"mfcc_{i+1}"] = round(float(np.mean(mfcc[i])), 4)

    return features


if __name__ == "__main__":

    audio_path = os.path.join(
        os.path.dirname(__file__),
        "recordings",
        "recording.wav"
    )

    features = extract_voice_features(audio_path)

    print("=" * 50)
    print("VOICE FEATURES")
    print("=" * 50)

    for key, value in features.items():
        print(f"{key:<22}: {value}")