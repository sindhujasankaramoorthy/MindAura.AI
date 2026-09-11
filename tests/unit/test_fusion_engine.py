from ai.voice.fusion_engine import EmotionFusionEngine


def test_analyze_returns_error_for_no_text_emotions():
    engine = EmotionFusionEngine()
    result = engine.analyze("neutral", 0.5, [])
    assert result == {"error": "No text emotions provided"}


def test_analyze_high_distress_for_strong_negative_signals():
    engine = EmotionFusionEngine()
    result = engine.analyze(
        "fea", 0.9, [{"label": "grief", "score": 0.9}, {"label": "fear", "score": 0.8}]
    )
    assert result["risk_level"] in ("High Distress", "Moderate Distress")
    assert 0 <= result["mental_health_distress_score"] <= 100


def test_analyze_low_distress_for_positive_signals():
    engine = EmotionFusionEngine()
    result = engine.analyze(
        "hap", 0.9, [{"label": "joy", "score": 0.9}, {"label": "gratitude", "score": 0.7}]
    )
    assert result["risk_level"] == "Low Distress / Stable"


def test_analyze_detects_masking_dissonance():
    engine = EmotionFusionEngine()
    # Positive text, strongly distressed voice -> "masking" dissonance
    result = engine.analyze(
        "sad", 0.9, [{"label": "joy", "score": 0.95}, {"label": "gratitude", "score": 0.9}]
    )
    assert result["acoustic_dissonance_detected"] is True
