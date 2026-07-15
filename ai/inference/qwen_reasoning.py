import logging
import json
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI for MindAura helping psychiatrists understand journals.
You must NOT diagnose, prescribe, recommend treatment, estimate risk, or classify severity/distress level. Remain objective.

TASK:
Based on the journal, emotions, and descriptive psychological signals provided, output a concise psychological interpretation.
Use psychological signals only as internal interpretation context. Do not add a separate Psychological Signals section.

OUTPUT FORMAT:

1. Emotional Summary:
(2-3 sentences describing the core emotional state)

2. Dominant Emotion:
(Why the highest emotion is dominant based on text)

3. Key Observations:
- (Objective behavior/pattern 1)
- (Objective behavior/pattern 2)"""

class QwenReasoning:
    """
    QwenReasoning module for MindAura.
    Generates structured psychological interpretations for psychiatrists
    using Ollama running Qwen.
    """
    
    def __init__(self, model_name: str = "qwen3:14b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.endpoint = f"{self.host}/api/chat"
        logger.info(f"QwenReasoning initialized with model {self.model_name} at {self.host}")

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Generate interpretation using the Ollama API with Qwen, enhanced with deep metrics.
        """
        if not results or not isinstance(results, dict):
            raise ValueError("Results dictionary cannot be empty.")
        
        journal_text = results.get("original_text", "")
        if not journal_text or not journal_text.strip():
            raise ValueError("Journal text cannot be empty.")
            
        emotion_scores = results.get("emotion_scores", {})
        if not emotion_scores:
            raise ValueError("Emotion scores cannot be empty.")

        # Prepare enriched user input format
        user_message = f"""INPUT:

1. Patient Journal (Original):
{journal_text}

2. Patient Journal (Processed/Translated):
{results.get('translated_text', '')}

3. Dominant Emotion: {results.get('dominant_emotion', '')} (Narrative: {results.get('dominant_narrative', results.get('dominant_emotion', ''))})

4. Top Emotions:
{json.dumps(results.get('top_emotions', {}), indent=2)}

5. Deep Emotional Metrics:
- Emotional Intensity: {results.get('emotional_intensity', 0)}/100
- Emotional Diversity: {results.get('emotional_diversity', 0)}/100
- Emotional Complexity: {results.get('emotional_complexity', 0)}/100

6. Psychological Signals (descriptive, non-diagnostic JSON):
{json.dumps(results.get('psychological_signals', {}), indent=2)}"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3  # Low temperature for more analytical and grounded responses
            }
        }

        try:
            logger.info("Sending request to Ollama local instance...")
            response = requests.post(self.endpoint, json=payload, timeout=600)
            response.raise_for_status()
            
            result_json = response.json()
            interpretation = result_json.get("message", {}).get("content", "")
            if not interpretation:
                raise ValueError("Received empty response from the local reasoning model.")
                
            return interpretation
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with local Ollama service: {str(e)}")
            raise RuntimeError(f"Ollama API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in reasoning model interpretation: {str(e)}")
            raise


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Enable importing from same directory for local testing
    sys.path.append(str(Path(__file__).parent))
    try:
        from emotion_predict import EmotionAnalyzer
    except ImportError:
        print("Error: Could not import EmotionAnalyzer from emotion_predict.py.")
        sys.exit(1)

    print("Initializing EmotionAnalyzer and QwenReasoning...")
    try:
        analyzer = EmotionAnalyzer()
        reasoner = QwenReasoning()
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("   MindAura - Psychiatric Emotional Interpretation (Qwen)")
    print("   Paste or type a multi-line journal entry below.")
    print("   Press Enter twice (empty line) or type END to submit.")
    print("   Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    def make_bar(value: float, total: float = 100.0, length: int = 20) -> str:
        percentage = (value / total)
        filled = int(percentage * length)
        filled = max(0, min(length, filled))
        return "█" * filled + "░" * (length - filled)

    while True:
        print()
        print("📝 Enter patient journal (empty line or END to submit):")

        # Collect multiple lines into a single journal string.
        # Stops when the user enters an empty line (double Enter) or types END.
        journal_lines = []
        while True:
            try:
                line = input()
            except EOFError:
                # Support piped / redirected input
                break
            if line.strip().upper() == "END":
                break
            if line == "" and journal_lines:
                # Empty line after at least one line of content → end of entry
                break
            journal_lines.append(line)

        user_input = "\n".join(journal_lines).strip()

        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye! 💙")
            break

        if not user_input:
            print("⚠️  Empty input. Please write something.")
            continue

        try:
            print("\n1. Running Emotion Classification & Deep Analysis...")
            results = analyzer.process(user_input)
            emotion_scores = results["emotion_scores"]
            
            print(f"\n🌐 Detected Language   : {results['original_language']}")
            dominant = results['dominant_emotion'].upper()
            narrative = results.get('dominant_narrative', results['dominant_emotion'])
            if narrative.lower() == 'neutral' and narrative.lower() != results['dominant_emotion'].lower():
                print(f"🎯 Dominant Emotion    : {dominant}  (overall profile: Neutral)")
            else:
                print(f"🎯 Dominant Emotion    : {dominant}")
            
            print("\n📊 Deep Emotional Analysis Metrics:")
            intensity = results['emotional_intensity']
            diversity = results['emotional_diversity']
            complexity = results['emotional_complexity']
            
            print(f"   ⚡ Emotional Intensity : [{make_bar(intensity)}] {intensity}/100")
            print(f"   🌈 Emotional Diversity : [{make_bar(diversity)}] {diversity}/100")
            print(f"   🧩 Emotional Complexity: [{make_bar(complexity)}] {complexity}/100")
            
            print("\n🏆 Top Emotions Profile:")
            for em, score in list(emotion_scores.items())[:5]:
                percentage = score * 100.0
                print(f"   {em:20s} ({score:.4f}) [{make_bar(percentage)}]")
            
            print("\n📋 Complete Emotional Distribution:")
            for em, score in emotion_scores.items():
                percentage = score * 100.0
                print(f"   {em:20s} : {score:.4f}  [{make_bar(percentage)}]")
            
            print("\n2. Generating Psychiatrist Interpretation via Qwen...")
            interpretation = reasoner.interpret(results)
            
            print(f"\n{'─' * 60}")
            print(interpretation)
            print(f"{'─' * 60}")

        except Exception as e:
            print(f"❌ Operation failed: {str(e)}")
