from emotion_predict import EmotionAnalyzer
analyzer = EmotionAnalyzer()
results = analyzer.process("I very oru madhiri feel I want to be alone")
print(f"Processed: {results['processed_text']}")
print(f"Translated: {results['translated_text']}")
print(f"Dominant Emotion: {results['dominant_emotion']}")
