class EmotionFusionEngine:
    """
    Fuses Voice Acoustic Emotion and Text Sentiment Emotion
    to calculate a Mental Health / Distress Score.
    """
    
    def __init__(self):
        # Base distress weights for textual GoEmotions (0.0 to 1.0)
        self.text_weights = {
            "grief": 1.0, "sadness": 0.8, "remorse": 0.7, "disappointment": 0.6,
            "fear": 0.9, "nervousness": 0.7, "confusion": 0.4,
            "anger": 0.8, "annoyance": 0.5, "disgust": 0.7, "disapproval": 0.5,
            "embarrassment": 0.6,
            "neutral": 0.1, "realization": 0.2, "surprise": 0.3,
            "joy": 0.0, "love": 0.0, "optimism": 0.0, "caring": 0.0,
            "admiration": 0.0, "amusement": 0.0, "approval": 0.0, "excitement": 0.0,
            "gratitude": 0.0, "pride": 0.0, "relief": 0.0, "desire": 0.1
        }
        
        # Base distress weights for Acoustic emotions (0.0 to 1.0)
        self.voice_weights = {
            "anger": 0.7,   "ang": 0.7,
            "frustration": 0.6,
            "sadness": 0.8, "sad": 0.8,
            "neutral": 0.2, "neu": 0.2,
            "happiness": 0.0, "hap": 0.0,
            "fear": 0.9,    "fea": 0.9,
            "surprise": 0.4,"sur": 0.4,
            "disgust": 0.7, "dis": 0.7,
            "calm": 0.1,    "cal": 0.1
        }

    def analyze(self, vocal_emotion: str, vocal_confidence: float, text_emotions: list) -> dict:
        """
        Calculates fusion metrics based on inputs.
        text_emotions is a list of dicts: [{'label': 'sadness', 'score': 0.8}, ...]
        """
        
        if not text_emotions:
            return {"error": "No text emotions provided"}
            
        top_text_emotion = text_emotions[0]['label']
        text_confidence = text_emotions[0]['score']
        
        # 1. Calculate Text Distress Score
        # Weighted sum of top 3 text emotions
        text_distress = 0.0
        total_text_weight = 0.0
        for te in text_emotions:
            label = te['label']
            score = te['score']
            weight = self.text_weights.get(label, 0.3)
            text_distress += (weight * score)
            total_text_weight += score
            
        if total_text_weight > 0:
            text_distress = text_distress / total_text_weight
        
        # 2. Calculate Vocal Distress Score
        # Handle "Mixed (Emotion1 / Emotion2)" or "Uncertain Prediction"
        vocal_lower = vocal_emotion.lower()
        if vocal_lower.startswith("mixed"):
            # Extract first emotion from the mixed string as primary for weighting
            import re
            match = re.search(r'mixed \((.*?) /', vocal_lower)
            primary_vocal = match.group(1).strip() if match else "neutral"
            voice_distress_base = self.voice_weights.get(primary_vocal, 0.4)
        elif vocal_lower in ["uncertain prediction", "low confidence prediction"]:
            voice_distress_base = 0.3
        else:
            voice_distress_base = self.voice_weights.get(vocal_lower, 0.3)
            
        # Apply confidence factor
        voice_distress = voice_distress_base * (0.5 + (0.5 * vocal_confidence))
        
        # 3. Fusion Logic (Mental Health Score)
        # Often text is more indicative of the *exact* issue (sadness vs anger)
        # Voice is indicative of the *intensity* (loud, soft, crying)
        
        # Weighting: Text 60%, Voice 40%
        overall_distress = (text_distress * 0.6) + (voice_distress * 0.4)
        
        # Normalize to 0-100 scale
        mental_health_score = max(0, min(100, int(overall_distress * 100)))
        
        # Determine risk level
        if mental_health_score >= 75:
            risk_level = "High Distress"
        elif mental_health_score >= 50:
            risk_level = "Moderate Distress"
        elif mental_health_score >= 25:
            risk_level = "Mild Distress"
        else:
            risk_level = "Low Distress / Stable"
            
        # Detect acoustic dissonance (e.g., saying happy things with a sad voice)
        dissonance = False
        if text_distress < 0.2 and voice_distress > 0.6:
            dissonance = True # Masking
        if text_distress > 0.6 and voice_distress < 0.2:
            dissonance = True # Flat affect / Detached

        return {
            "mental_health_distress_score": mental_health_score,
            "risk_level": risk_level,
            "top_text_emotion": top_text_emotion,
            "vocal_emotion": vocal_emotion,
            "acoustic_dissonance_detected": dissonance,
            "breakdown": {
                "text_distress_component": round(text_distress, 2),
                "voice_distress_component": round(voice_distress, 2)
            }
        }

if __name__ == "__main__":
    engine = EmotionFusionEngine()
    print(engine.analyze("ang", 0.68, [{'label': 'sadness', 'score': 0.8}, {'label': 'disappointment', 'score': 0.15}]))
