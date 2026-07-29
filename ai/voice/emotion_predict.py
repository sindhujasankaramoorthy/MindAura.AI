import os
import sys
import torch
import librosa
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

try:
    from ai.voice.voice_features import extract_voice_features
except ImportError:
    from voice_features import extract_voice_features

logger = logging.getLogger(__name__)

# ------------------------------------
# UTF-8 Console Support
# ------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_NAME = "superb/wav2vec2-base-superb-er"

_processor = None
_model = None

@dataclass
class TimelineSegment:
    start: float
    end: float
    emotion: str
    confidence: float
    raw_probs: Dict[str, float]

class EmotionMapper:
    MAPPING = {
        "ang": "Anger",
        "sad": "Sadness",
        "hap": "Happiness",
        "neu": "Neutral",
        "fea": "Fear",
        "sur": "Surprise",
        "dis": "Disgust",
        "cal": "Calm"
    }

    @classmethod
    def get_readable(cls, label: str) -> str:
        return cls.MAPPING.get(label.lower(), label.capitalize())

class SERAnalyzer:
    """Explainable Speech Emotion Analyzer."""
    
    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold
        self._load_model()

    def _load_model(self):
        global _processor, _model
        if _model is None or _processor is None:
            logger.info("Loading Speech Emotion Recognition Model...")
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            _processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
            _model.eval()

    def _predict_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        if len(audio_chunk) == 0:
            return {}
        inputs = _processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        return {_model.config.id2label[i]: score.item() for i, score in enumerate(probs)}

    def _generate_timeline(self, audio: np.ndarray, sr: int, window_sec: float = 3.0, step_sec: float = 3.0) -> List[TimelineSegment]:
        timeline = []
        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)
        total_samples = len(audio)
        
        for start_idx in range(0, total_samples, step_samples):
            end_idx = min(start_idx + window_samples, total_samples)
            chunk = audio[start_idx:end_idx]
            if len(chunk) < int(0.5 * sr):
                continue
            
            raw_probs = self._predict_chunk(chunk, sr)
            if not raw_probs:
                continue
            
            best_label = max(raw_probs.items(), key=lambda x: x[1])[0]
            confidence = raw_probs[best_label]
            
            timeline.append(TimelineSegment(
                start=round(start_idx / sr, 1),
                end=round(end_idx / sr, 1),
                emotion=EmotionMapper.get_readable(best_label),
                confidence=confidence,
                raw_probs=raw_probs
            ))
        return self._smooth_timeline(timeline)

    def _smooth_timeline(self, timeline: List[TimelineSegment]) -> List[TimelineSegment]:
        """Prevents rapid emotion changes (jitter) unless confidence is very high."""
        if len(timeline) < 3:
            return timeline
        
        smoothed = timeline.copy()
        for i in range(1, len(timeline) - 1):
            prev_emo = smoothed[i-1].emotion
            next_emo = timeline[i+1].emotion
            curr_emo = timeline[i].emotion
            curr_conf = timeline[i].confidence
            
            if prev_emo == next_emo and curr_emo != prev_emo:
                if curr_conf < 0.8:
                    smoothed[i].emotion = prev_emo
        return smoothed

    def _aggregate_weighted_voting(self, timeline: List[TimelineSegment]) -> Dict[str, float]:
        """Aggregate chunk probabilities via weighted voting (length & confidence)."""
        if not timeline:
            return {}
        
        aggregated_probs = {}
        total_weight = 0.0
        
        for seg in timeline:
            length = seg.end - seg.start
            weight = length * seg.confidence
            for label, prob in seg.raw_probs.items():
                aggregated_probs[label] = aggregated_probs.get(label, 0.0) + (prob * weight)
            total_weight += weight
            
        if total_weight > 0:
            for label in aggregated_probs:
                aggregated_probs[label] /= total_weight
        return aggregated_probs

    def _assess_voice_stress(self, features: Dict[str, Any]) -> str:
        pitch_var = features.get("pitch_variability", 0)
        energy = features.get("rms_energy", 0)
        tremor = features.get("voice_tremor", 0)
        speech_rate = features.get("speech_rate", 0)
        
        stress_score = 0
        if pitch_var > 40: stress_score += 1
        if energy > 0.05: stress_score += 1
        if tremor > 15: stress_score += 1
        if speech_rate > 3.0: stress_score += 1
        
        if stress_score >= 3: return "High Stress"
        elif stress_score >= 1: return "Moderate Stress"
        return "Low Stress"

    def _hybrid_refinement(self, raw_probs: Dict[str, float], features: Dict[str, Any], text_emotions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[str]]:
        """Acoustic feature rules and emotion similarity logic."""
        refined = {EmotionMapper.get_readable(k): v for k, v in raw_probs.items()}
        reasoning = []
        
        pitch = features.get("pitch", 0.0)
        energy = features.get("rms_energy", 0.0)
        pause_ratio = features.get("pause_ratio", 0.0)
        
        top_text = text_emotions[0]["label"].lower() if text_emotions else "neutral"
        
        # 1. Acoustic Refinements
        if pitch > 0 and pitch < 120 and energy < 0.02 and pause_ratio > 0.2:
            refined["Sadness"] = refined.get("Sadness", 0) + 0.15
            reasoning.append("Low pitch, low energy, and high pause ratio strongly suggest Sadness.")
            
        if pitch > 250 and energy > 0.05:
            refined["Anger"] = refined.get("Anger", 0) + 0.15
            reasoning.append("High pitch and high energy suggest heightened arousal (Anger/Frustration).")
            
        # 2. Similarity Logic (Frustration vs Anger)
        anger_score = refined.get("Anger", 0)
        if anger_score > 0.3 and top_text in ["disappointment", "annoyance", "frustration"]:
            # Demote anger, promote frustration
            refined["Frustration"] = anger_score + 0.2
            refined["Anger"] = anger_score - 0.15
            reasoning.append(f"Acoustic arousal is present, but text sentiment ({top_text}) shifts classification from Anger to Frustration.")

        # Re-normalize
        total = sum(refined.values())
        if total > 0:
            for k in refined: refined[k] /= total
            
        return refined, reasoning

    def _generate_explanation(self, primary_emotion: str, top_text: str, reasoning: List[str], features: Dict[str, Any]) -> str:
        explanation = f"Vocal characteristics align with {primary_emotion}."
        if primary_emotion.lower() != top_text.lower():
            explanation += f" The speaker expresses '{top_text}' in language, but vocal characteristics indicate '{primary_emotion}'."
        
        if features.get('voice_tremor', 0) > 15:
            explanation += " Elevated voice tremor suggests emotional distress or anxiety."
            
        return explanation

    def analyze(self, audio_path: str, text_emotions: List[Dict[str, Any]] = None) -> Tuple[str, float, Dict[str, Any]]:
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            features = extract_voice_features(audio_path)
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return "Neutral", 0.0, {}

        timeline = self._generate_timeline(audio, sr)
        aggregated_raw_probs = self._aggregate_weighted_voting(timeline)
        
        refined_probs, reasoning = self._hybrid_refinement(aggregated_raw_probs, features, text_emotions or [])
        
        sorted_probs = sorted(refined_probs.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [{"emotion": k, "confidence": round(v * 100, 1)} for k, v in sorted_probs[:3]]
        
        primary_emotion = sorted_probs[0][0]
        primary_conf = sorted_probs[0][1]
        secondary_emotion = sorted_probs[1][0] if len(sorted_probs) > 1 else "None"
        
        # Probability Calibration (Mixed Emotions)
        if len(sorted_probs) > 1:
            diff = sorted_probs[0][1] - sorted_probs[1][1]
            if diff < 0.15:
                primary_emotion = f"Mixed ({primary_emotion} / {secondary_emotion})"
                reasoning.append(f"Top two emotions differ by only {diff*100:.1f}%; classified as Mixed.")
                
        # Confidence Calibration
        reliability = "High"
        if primary_conf < self.confidence_threshold:
            reliability = "Low"
            primary_emotion = "Low Confidence Prediction"
            reasoning.append("Overall confidence is below threshold, indicating ambiguous or masked emotion.")
            
        stress_level = self._assess_voice_stress(features)
        
        top_text = text_emotions[0]["label"] if text_emotions else "neutral"
        explanation = self._generate_explanation(primary_emotion, top_text, reasoning, features)

        result_json = {
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "confidence": round(primary_conf, 2),
            "prediction_reliability": reliability,
            "stress_level": stress_level,
            "voice_features": {
                "average_pitch": features.get("pitch", 0),
                "pitch_variability": features.get("pitch_variability", 0),
                "energy": features.get("rms_energy", 0),
                "speech_rate": features.get("speech_rate", 0),
                "pause_ratio": features.get("pause_ratio", 0),
                "voice_tremor": features.get("voice_tremor", 0)
            },
            "top_predictions": top_predictions,
            "timeline": [{"start": t.start, "end": t.end, "emotion": t.emotion} for t in timeline],
            "explanation": explanation,
            "reasoning": reasoning
        }
        
        return primary_emotion, primary_conf, result_json

def predict_emotion(audio_path: str, text_emotions: List[Dict[str, Any]] = None) -> Tuple[str, float, Dict[str, Any]]:
    analyzer = SERAnalyzer()
    return analyzer.analyze(audio_path, text_emotions)

if __name__ == "__main__":
    import json
    audio_path = os.path.join(os.path.dirname(__file__), "recordings", "recording.wav")
    emotion, confidence, result_json = predict_emotion(audio_path, [{"label": "disappointment", "score": 0.8}])
    print(json.dumps(result_json, indent=4))
