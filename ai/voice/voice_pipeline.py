import os
import sys
import json
import logging

try:
    from ai.voice.recorder import SmartRecorder
    from ai.voice.emotion_predict import predict_emotion
    from ai.voice.speech_to_text import transcribe_audio
    from ai.voice.text_emotion import TextEmotionAnalyzer
    from ai.voice.fusion_engine import EmotionFusionEngine
except ImportError:
    from recorder import SmartRecorder
    from emotion_predict import predict_emotion
    from speech_to_text import transcribe_audio
    from text_emotion import TextEmotionAnalyzer
    from fusion_engine import EmotionFusionEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

def run_voice_pipeline(audio_file_path=None):
    """
    Runs the complete end-to-end voice processing pipeline:
    1. Records user audio with VAD and pre-roll (or uses provided audio).
    2. Transcribes the audio to text using Faster-Whisper.
    3. Detects vocal emotion using Wav2Vec2.
    
    Returns:
        dict: A dictionary containing the transcript, emotion, and confidence.
    """
    print("=" * 60)
    print("MindAura.AI Voice Pipeline")
    print("=" * 60)
    
    if not audio_file_path:
        recorder = SmartRecorder()
        audio_file_path = recorder.record("pipeline_recording.wav")
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        logger.warning("No audio was recorded. Exiting pipeline.")
        return None

    print("\n" + "=" * 60)
    print("Processing Audio...")
    print("=" * 60)

    # 1. Transcribe Audio
    print("\n[1/3] Transcribing Speech...")
    transcript = transcribe_audio(audio_file_path)
    print(f"Transcript: '{transcript}'")
    
    # 2. Predict Text Emotion (Done before Voice to allow cross-validation)
    print("\n[2/3] Analyzing Text Sentiment...")
    text_analyzer = TextEmotionAnalyzer()
    text_emotions = text_analyzer.predict(transcript)
    
    # 3. Predict Vocal Emotion (Now cross-validates with text)
    print("\n[3/3] Analyzing Vocal Emotion & Fusing Results...")
    vocal_emotion, vocal_confidence, vocal_json_summary = predict_emotion(audio_file_path, text_emotions)
    print(f"Vocal Emotion: {vocal_emotion} ({vocal_confidence*100:.1f}%)")
    
    fusion_engine = EmotionFusionEngine()
    mental_health_analysis = fusion_engine.analyze(vocal_emotion, float(vocal_confidence), text_emotions)
    
    print(f"Top Text Emotion: {mental_health_analysis.get('top_text_emotion', 'None')}")
    print(f"Mental Health Distress Score: {mental_health_analysis.get('mental_health_distress_score', 0)}/100")
    print(f"Risk Level: {mental_health_analysis.get('risk_level', 'Unknown')}")
    
    result = {
        "audio_path": audio_file_path,
        "transcript": transcript,
        "raw_vocal_emotion": vocal_json_summary,
        "raw_text_emotions": text_emotions,
        "fusion_analysis": mental_health_analysis
    }
    
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_voice_pipeline(sys.argv[1])
    else:
        run_voice_pipeline()
