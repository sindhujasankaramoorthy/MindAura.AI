"""
Verifies TextEmotionAnalyzer (voice transcript path) routes text through
TextNormalizer before classification, matching the text-journal pipeline.
Needs torch/transformers/symspellpy — skips cleanly where they're absent.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("symspellpy")

from unittest.mock import MagicMock

from ai.voice.text_emotion import TextEmotionAnalyzer


def test_predict_normalizes_before_classifying(monkeypatch):
    analyzer = TextEmotionAnalyzer()

    fake_normalizer = MagicMock()
    fake_normalizer.normalize.return_value = {"corrected_sentence": "I feel very afraid"}
    analyzer._normalizer = fake_normalizer

    fake_classifier = MagicMock(return_value=[{"label": "fear", "score": 0.9}])
    analyzer._classifier = fake_classifier

    analyzer.predict("enaku romba bayama iruku")

    fake_normalizer.normalize.assert_called_once_with("enaku romba bayama iruku")
    fake_classifier.assert_called_once_with("I feel very afraid")
