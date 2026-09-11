"""
Fixtures for the fast/logic-only test tier. Nothing under tests/unit/ should
need a GPU, a model download, or real inference to pass.
"""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_model_registry(monkeypatch):
    """
    Stub out every heavy model load in ai/ for the fast unit-test tier.

    Patches ModelRegistry.get so it never calls a real loader (no downloads,
    no CPU/GPU inference) — every call returns a fake handle that can be
    unpacked into as many parts as the caller expects (tokenizer/model/device
    triples, processor/model pairs, or a single object).

    Because every model load in ai/ goes through this one function
    (ai/model_registry.py), this single monkeypatch covers RoBERTa, NLLB,
    GPT2, IndicTrans2, GLiNER, BERT-NER, Wav2Vec2, and Faster-Whisper.

    ai.model_registry itself has no heavy imports at module scope (each
    loader defers its `import torch`/`from transformers import ...` until
    actually called), so this fixture works even where torch/transformers
    aren't installed. Modules that import those libraries directly at module
    scope (e.g. advanced_correction.py, emotion_predict.py) still need
    `pytest.importorskip(...)` in the test file itself.
    """
    from ai.model_registry import ModelRegistry

    class _FakeModelHandle(MagicMock):
        def __iter__(self):
            return iter([MagicMock(), MagicMock(), MagicMock()])

    monkeypatch.setattr(
        ModelRegistry, "get", staticmethod(lambda key, loader: _FakeModelHandle())
    )
