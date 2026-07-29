import sys
import os

# Add the project root to sys.path so we can import from ai.voice
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from ai.voice.voice_pipeline import run_voice_pipeline

def test_qwen_voice_integration(audio_path=None):
    print("=" * 60)
    print("Testing Qwen Reasoning Integration with Voice Module")
    print("=" * 60)
    
    if audio_path and os.path.exists(audio_path):
        print(f"Using provided audio file: {audio_path}")
        result = run_voice_pipeline(audio_path)
    else:
        print("No valid audio path provided. The pipeline will default to recording a new audio clip.")
        print("To test with an existing file, run: python test_qwen_voice.py <path_to_audio_file>")
        result = run_voice_pipeline()

    if result and result.get("qwen_interpretation"):
        print("\n✅ Integration Test Successful: Qwen Interpretation generated.")
    else:
        print("\n❌ Integration Test Failed: Could not generate Qwen Interpretation.")

if __name__ == "__main__":
    test_audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_qwen_voice_integration(test_audio_path)
