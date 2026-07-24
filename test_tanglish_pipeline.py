import json
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.preprocessing.text_normalizer import TextNormalizer
from ai.preprocessing.word_classifier import WordClassifier

logging.basicConfig(level=logging.WARNING)

def run_test(normalizer, text, expected):
    print("\n" + "="*80)
    print(f"Input:    {text}")
    print(f"Expected: {expected}")
    
    result = normalizer.normalize(text)
    
    print(f"\nFinal Translated: {result['corrected_sentence']}")
    
    print("\nMetadata:")
    for item in result["metadata"]:
        if "TANGLISH" in str(item["detected_language"]).upper() or "UNKNOWN" in str(item["detected_language"]).upper():
            print(f"- {item['original_token']} -> {item['detected_language']} -> {item['corrected_token']} -> {item['translated_token']} (conf: {item.get('confidence', 100)})")
            
    print("="*80)

if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    test_cases = [
        (
            "enaku romba kastama iruku",
            "I feel very difficult"
        ),
        (
            "i lost my gwen adhu i cant accept enaku theriyum i should move on nu",
            "I lost my Gwen. I can't accept it. I know I should move on."
        ),
        (
            "ennala mudiyala",
            "I cannot handle it."
        ),
        (
            "Ram ku konjum stress iruku",
            "Ram preserved by NER."
        )
    ]
    
    for text, expected in test_cases:
        run_test(normalizer, text, expected)
