# 2026-08-19 — Cleanup, performance, and backend scaffold pass

Status: all changes below are in the working tree only. Nothing has been
staged or committed to git — see "Still open" for what's deliberately
deferred.

## Why

An initial review of the repo found the `ai/` pipeline was the only
substantive part of the project (backend/frontend/android networking/docs
were empty stubs), with real architectural issues: models loaded eagerly
and redundantly, the same fuzzy-matching logic duplicated three times with
three different thresholds, voice and text journal entries getting
inconsistent preprocessing, and leftover dead code from earlier iterations.
This pass worked through that list end to end.

## What changed

### Dependency cleanup
- Deleted `required.txt`, `compatible_requirements.txt`,
  `filtered_requirements.txt` — three divergent, ad hoc requirements files
  left over from dependency troubleshooting.
- Populated `backend/requirements.txt` (was 0 bytes): `fastapi`,
  `uvicorn[standard]`, `pydantic`.
- Added `fastapi==0.141.1`, `pytest==9.1.1`, `uvicorn[standard]==0.52.4` to
  the root `requirements.txt` so a single `pip install -r requirements.txt`
  covers the ML pipeline, backend, and test suite in one shot.

### Dead code removed
- `ai/preprocessing/tanglish_phrase_normalizer.py` — fully superseded,
  unused file, deleted.
- `ai/preprocessing/advanced_correction.py` — removed unused
  `tanglish_key`/`tanglish_distance`/`LevenshteinDistance` helpers, the dead
  `EmotionPreservingCorrector.process()` method (a superseded duplicate of
  `TextNormalizer.normalize()`), and duplicate/no-op entries in
  `CUSTOM_OVERRIDES`.
- `ai/inference/psychological_signals.py` — removed the dead
  `EmotionStatistics` class, but ported its more principled Shannon-entropy
  diversity calculation into `emotion_predict.py::_calculate_diversity`
  first (replacing a cruder heuristic that used an unexplained "10 active
  emotions = 100%" cap).
- Added `if __name__ == "__main__":` guards to
  `ai/training/build_metadata.py` and `ai/training/preprocess_audio.py`
  (previously executed on import).
- Deleted stray tracked files: `ai/voice/emotion_predict.py.save`,
  `test_recording.wav`.

### Shared model registry (`ai/model_registry.py`, new)
Thread-safe, lazy, process-wide cache for every model load in `ai/`.
Converted eager `__init__`-time loads to load-on-first-use in:
- `ai/inference/emotion_predict.py` (RoBERTa GoEmotions, NLLB)
- `ai/preprocessing/advanced_correction.py` (SymSpell — eager but cached;
  GPT2 perplexity scorer and IndicTrans2-1B — fully lazy)
- `ai/preprocessing/ner_protection.py` (GLiNER, BERT-NER fallback)
- `ai/voice/text_emotion.py`, `ai/voice/emotion_predict.py`,
  `ai/voice/speech_to_text.py` (now share the same RoBERTa checkpoint as
  the text pipeline instead of loading a second copy)

Net effect: a plain English/Tanglish sentence with no misspellings never
loads GPT2 or IndicTrans2-1B at all — previously every pipeline run loaded
6+ transformer models regardless of whether they were needed.

Also fixed `ai/preprocessing/__init__.py`, which eagerly re-exported
`TextNormalizer`/`EmotionPreservingCorrector` at package-import time —
meaning importing even a dependency-light leaf module like
`word_classifier.py` transitively pulled in torch/transformers/symspellpy.
Emptied it out (nothing else in the repo relied on the package-level
export).

### Unified fuzzy-matching and Zipf thresholds
New `ai/preprocessing/lexical_constants.py` centralizes:
- `FUZZY_MATCH_THRESHOLD` (was duplicated in `word_classifier.py` and
  `tanglish_patterns.py`)
- Three named Zipf-frequency thresholds (`_LOOSE`/`_STANDARD`/`_STRICT`,
  values 2.5/3.0/3.8) — kept distinct since they gate three different
  decisions, not merged into one number
- `fuzzy_match_tanglish()` helper, now shared by `word_classifier.py` and
  `tanglish_patterns.py`

### Voice/text preprocessing consistency (behavior change, user-approved)
`ai/voice/text_emotion.py::TextEmotionAnalyzer.predict()` previously fed
raw Whisper transcripts straight into RoBERTa with zero normalization,
while text journals got the full 13-stage `TextNormalizer` pipeline (NER
protection, Tanglish correction, negation recovery, translation) first.
Voice transcripts now go through the same `TextNormalizer` before
classification.

### Backend scaffold (first working endpoint)
- Added missing `__init__.py` files under `backend/`.
- `backend/app/services/emotion_service.py` — thin wrapper around
  `EmotionAnalyzer`, with an optional (off by default) Qwen interpretation.
- `backend/app/api/journals.py` — `POST /journals/analyze`.
- `backend/app/main.py` — FastAPI app + `GET /health` that never triggers
  model loading.
- Smoke-tested (routes, response shape, `/health` isolation from model
  loading) with the `ai.inference` modules stubbed out, since torch isn't
  installed on this machine — needs a real end-to-end run on a machine with
  the full ML stack installed.

### Test suite (new: `tests/`, `pytest.ini`)
- `tests/unit/` — 23 fast, real (not stubbed) tests covering word
  classification, Tanglish pattern normalization, language boundary
  routing, the fusion engine, psychological signals, and Tanglish
  vocabulary/autocorrect. Runs in ~10s with no ML dependencies.
- `tests/unit/conftest.py` — autouse fixture mocking
  `ModelRegistry.get`, so any test touching the registry never downloads or
  runs a real model.
- `tests/unit/test_text_normalizer.py`, `test_text_emotion.py` — guarded
  with `pytest.importorskip("torch")` etc., since `TextNormalizer` imports
  those libraries at module scope even though model *loading* is lazy;
  skip cleanly here, will run for real wherever torch/transformers are
  installed.
- `tests/integration/` — `@pytest.mark.slow`, real-model coverage ported
  from the four old root-level print-and-eyeball scripts (see below), with
  loose substring assertions instead of exact-string matches since model
  output can drift slightly across versions.
- `pytest.ini` — registers the `slow` marker, defaults to
  `-m "not slow"` so plain `pytest` stays fast everywhere.
- Deleted the four old root-level manual test scripts (`test_pipeline.py`,
  `test_advanced_correction.py`, `test_tanglish_debug.py`,
  `test_tanglish_pipeline.py`) — no real assertions, not part of any test
  runner, their intent is now in `tests/`.

### Risk-scoring disclaimer (flagged, not silently changed)
`ai/voice/fusion_engine.py` computes a hardcoded 0–100 "Distress Score" +
risk-level label via hand-tuned weights, despite
`ai/inference/qwen_reasoning.py`'s system prompt explicitly forbidding the
LLM from doing the same thing. Left the scoring logic untouched (removing
it is a product decision, not a cleanup task) but added an explicit
disclaimer to both the class docstring and the returned dict, so it reads
as an acknowledged heuristic rather than an accidental inconsistency.

## Data findings (no files moved)

- `ai/training/processed_audio/` — 8,882 preprocessed `.wav` files,
  correctly organized folder-per-emotion (angry/calm/disgust/fear/happy/
  neutral/sad/surprise) for `datasets`' `audiofolder` loader. Ready to use
  for SER fine-tuning as-is. Note: `calm` and `surprise` are
  underrepresented (192 files vs. ~1,400 for the others).
- `ai/tanglish_model/data/raw/train.tsv`/`test.tsv` — 11,335 labeled
  Tanglish sentences, correctly placed, but labeled with coarse sentiment
  (Positive/Negative/Mixed_feelings/unknown_state/not-Tamil) rather than
  the 28 GoEmotions categories the text pipeline actually uses. Would need
  a differently-labeled (or relabeled) dataset to fine-tune
  `SamLowe/roberta-base-go_emotions` directly.
- `ai/training/train_roberta.py` and `ai/training/train_ser.py` are still
  0 bytes — no training code exists yet for either model.

## Still open (deliberately not touched)

1. **Git.** Nothing staged or committed. The 762MB of tracked audio under
   `ai/training/processed_audio/`, any `.gitignore` changes, and any
   history rewrite are all deferred until explicitly requested.
2. **Fusion engine scoring.** Whether to keep the numeric distress score at
   all (vs. dropping it in favor of purely descriptive signals) is an open
   product decision — see the disclaimer added above.
3. **Text training data.** Whether to relabel/swap the Tanglish sentiment
   dataset for something GoEmotions-compatible before writing
   `train_roberta.py`.

## Suggested next steps

1. On the Alienware: `pip install -r requirements.txt`, then run
   `pytest -m slow` plus a manual `/journals/analyze` and voice-pipeline
   call to confirm the refactor holds up against real models.
2. `ai/training/train_ser.py` is the more tractable next build — the data's
   ready and the labels already match.
