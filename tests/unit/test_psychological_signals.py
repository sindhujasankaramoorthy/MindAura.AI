from ai.inference.psychological_signals import (
    extract_text_signal_scores,
    merge_emotion_signal_scores,
    PsychologicalSignalExtractor,
)


def test_extract_text_signal_scores_detects_social_withdrawal():
    scores = extract_text_signal_scores("I feel so alone and lonely lately")
    assert scores["social_withdrawal"] > 0.0


def test_extract_text_signal_scores_zero_for_neutral_text():
    scores = extract_text_signal_scores("The weather is nice today")
    assert all(v == 0.0 for v in scores.values())


def test_merge_emotion_signal_scores_blends_text_and_emotion():
    text_scores = {"mental_fatigue": 0.28}
    emotion_scores = {"sadness": 0.5, "disappointment": 0.5}
    merged = merge_emotion_signal_scores(text_scores, emotion_scores)
    # 0.28 (text) + 0.5*0.18 (sadness) + 0.5*0.14 (disappointment) = 0.44
    assert merged["mental_fatigue"] == 0.44


def test_extractor_extract_end_to_end():
    extractor = PsychologicalSignalExtractor()
    results = {
        "original_text": "I feel helpless and cannot handle it",
        "emotion_scores": {"sadness": 0.6, "fear": 0.3},
        "processed_text": "",
        "translated_text": "",
    }
    signals = extractor.extract(results)
    assert signals["helplessness_language"] > 0.0
