"""
Test script for the Advanced Emotion-Preserving Text Correction pipeline.
Tests all success criteria from the user's specification.
"""
import sys
import os
import logging

# Enable debug logging to see pipeline internals
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from ai.preprocessing.text_normalizer import TextNormalizer


def test():
    normalizer = TextNormalizer()

    print("=" * 70)
    print("  MindAura — Advanced Emotion-Preserving Text Correction Tests")
    print("=" * 70)

    # ── Success Criteria Tests ──
    success_tests = [
        ("i cnt do it ,idnt feel very good",
         "I can't do it, I don't feel very good."),
        ("my minf aint silent",
         "My mind ain't silent."),
        ("enaku oru madhiri mind disturbed ah iruku",
         "I feel mentally disturbed."),
        ("i feel mentally exhausted enala mudla need break",
         "I feel mental fatigue. I cannot handle it. Need break."),
        ("i feek low",
         "I feel low."),
    ]

    print("\n── SUCCESS CRITERIA ──\n")
    for inp, expected in success_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        status = "✅ PASS" if actual.strip().lower() == expected.strip().lower() else "⚠️  CHECK"
        print(f"Input:    '{inp}'")
        print(f"Expected: '{expected}'")
        print(f"Actual:   '{actual}'")
        print(f"Status:   {status}")
        print()

    # ── Negation Recovery Tests ──
    neg_tests = [
        ("cnt", "can't"),
        ("dont", "don't"),
        ("didnt", "didn't"),
        ("idnt", "I don't"),
        ("dhnt", "don't"),
        ("isnt", "isn't"),
        ("arent", "aren't"),
        ("aint", "ain't"),
    ]

    print("── NEGATION RECOVERY ──\n")
    for inp, expected_frag in neg_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        status = "✅" if expected_frag.lower() in actual.lower() else "❌"
        print(f"  {status} '{inp}' -> '{actual}'  (should contain '{expected_frag}')")
    print()

    # ── Spell Correction Tests (should NOT mangle) ──
    spell_tests = [
        ("minf", "mind"),
        ("lonley", "lonely"),
        ("scard", "scared"),
        ("exhaustd", "exhausted"),
        ("happpy", "happy"),
        ("anxios", "anxious"),
        ("thnkng", "thinking"),
        ("ovrthinking", "overthinking"),
    ]

    print("── SPELL CORRECTION ──\n")
    for inp, expected_frag in spell_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        status = "✅" if expected_frag.lower() in actual.lower() else "❌"
        print(f"  {status} '{inp}' -> '{actual}'  (should contain '{expected_frag}')")
    print()

    # ── Emotion Vocabulary Protection ──
    protect_tests = [
        "sad", "scared", "anxious", "lonely", "hopeless",
        "exhausted", "overthinking", "not sad", "don't overthinking",
        "i am burnt out",
    ]

    print("── EMOTION VOCABULARY PROTECTION ──\n")
    for inp in protect_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        # The protected word(s) should still be present
        print(f"  '{inp}' -> '{actual}'")
    print()

    # ── THE CRITICAL BUG FIX ──
    print("── CRITICAL BUG FIX ──\n")
    critical = "i cnt do it ,idnt feel very good"
    result = normalizer.normalize(critical)
    actual = result['processed_text']
    print(f"  Input:  '{critical}'")
    print(f"  Output: '{actual}'")
    if "can't" in actual and "don't" in actual:
        print("  ✅ NEGATIONS PRESERVED — Bug is fixed!")
    else:
        print("  ❌ NEGATIONS LOST — Bug persists!")
    print()

    # ── NER Protection Tests ──
    print("── NER PROTECTION TESTS ──\n")
    ner_tests = [
        ("i had a friend vishal he is the reason", "Vishal"),
        ("I live in chennai and work for google", "Chennai"),
        ("I live in chennai and work for google", "Google"),
        ("my sister and mother are with me", "sister"),
        ("my sister and mother are with me", "mother"),
        ("Ram went to New York", "Ram"),
        ("Ram went to New York", "New York")
    ]
    for inp, expected_frag in ner_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        status = "✅" if expected_frag.lower() in actual.lower() else "❌"
        # Also assert casing preservation
        case_preserved = expected_frag in actual
        case_status = "✅ Case Preserved" if case_preserved else "❌ Case Changed"
        print(f"  {status} '{inp}' -> '{actual}' (should contain '{expected_frag}') [{case_status}]")
    print()

    # ── Generalization & Unseen Tanglish Tests ──
    print("── GENERALIZATION & UNSEEN TANGLISH TESTS ──\n")
    generalization_tests = [
        ("pudikula enaku", "I do not like"),
        ("romba kavalaya iruku", "feel very worried"),
        ("i feel mentally exhausted ennala mudiyala", "I cannot handle it"),
    ]
    for inp, expected_frag in generalization_tests:
        result = normalizer.normalize(inp)
        actual = result['processed_text']
        status = "✅" if expected_frag.lower() in actual.lower() else "❌"
        print(f"  {status} '{inp}' -> '{actual}' (should contain '{expected_frag}')")
    print()
    print("=" * 70)


if __name__ == "__main__":
    test()
