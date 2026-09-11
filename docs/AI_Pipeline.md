# MindAura — AI Pipeline

Status: 2026-09 · reflects the v2 pipeline plan (`mindaura-v2-pipeline` branch).
See `docs/Architecture.md` for the system-level view and
`docs/audits/2026-08-19-system-audit.md` for the component-by-component history.

---

## Overview

Every patient submission ("blog") is one of three modalities — **text**, **voice**,
or **video** — in **any language**. Each modality has its own preprocessing
pipeline that produces a **versioned JSON document** with two channels:

| Channel | Consumer | Purpose |
|---|---|---|
| `contextual_text_*` / emotion features | emotion-vector layer | numeric emotion representation |
| `expressive_channel` / prosody / mannerisms | Ollama (Qwen) | psychologist-readable narrative summary |

The JSON is **persisted to the database first**, then read back and passed to the
analysis layer:

```
submission → modality pipeline → MODALITY JSON → DB(write #1)
          → emotion-vector layer (RoBERTa + SER + face models → fused vector)
          → Ollama narrative summary
          → AnalysisResult JSON → DB(write #2)
          → progress engine (formula + baseline + history) → ProgressPoint
          → psychologist dashboard
```

RoBERTa is **text-only**. Voice acoustic emotion and facial emotion come from
their own models; all three vectors are **late-fused** into one numeric vector
that the progress formula consumes.

---

## 1. Text pipeline

### Preprocessing (`ai/preprocessing/`)
1. **Language routing** (revised order): detect language → if **unknown**, check the
   Tanglish vocabulary/dataset → else autocorrect → use **SymSpell** to compute the
   fraction of valid-English tokens and route on that. Tanglish is a **fallback
   route**, not a first-class branch.
2. Clean whitespace / cap elongations / expand chat abbreviations.
3. **NER protection** — GLiNER (`NeuML/gliner-bert-tiny`) + `dslim/bert-base-NER`
   fallback; entities masked as `<TYPE_N>` during correction, restored after.
4. **Token language ID** — Viterbi/HMM decoder over `WordClassifier` emissions
   (242k Tanglish vocab, `wordfreq`, fastText `lid.176`).
5. **Correction** — English: SymSpell with GPT2-perplexity candidate disambiguation
   (emotion words protected). Tanglish: fuzzy autocorrect → canonical spelling →
   IndicXlit transliteration.
6. Semantic normalization (Tanglish phrase maps, SOV→SVO), negation recovery,
   phrase standardization, sentence reconstruction, NER restore.
7. **Whole-document translation** to English if still non-English — NLLB-200-600M
   (or IndicTrans2-1B for Indic languages).

Outputs a `TextModalityDoc`:
- `contextual_text_en` — normalized/translated text → RoBERTa
- `expressive_channel` — elongations, emphasis, code-switch spans, chat-abbrev
  expansions, preserved punctuation → Ollama
- `detected_languages[]`, `language_route`, `symspell_english_pct`, `ner_entities[]`,
  `translation_path[]`, `preprocessing_metadata[]`

### Emotion analysis (`ai/inference/emotion_predict.py`)
- **RoBERTa GoEmotions** (`SamLowe/roberta-base-go_emotions`, 28 labels) — pretrained,
  not fine-tuned (see plan §4). Neutral-suppression rule promotes the 2nd emotion when
  neutral leads by < 0.05.
- **Psychological signals** — 8 indicators (mental fatigue, cognitive overload,
  restlessness, emotional conflict, self-criticism, social withdrawal, helplessness
  language, motivation reduction). Currently regex; planned upgrade to a trained
  multi-label classifier.
- **Deep metrics** — emotional intensity (avg top-3), diversity (normalized Shannon
  entropy), complexity (2nd/1st ratio).

---

## 2. Voice pipeline (`ai/voice/`)

1. **Capture** — `SmartRecorder`: continuous stream VAD, 1.5 s pre-roll, adaptive
   silence threshold → 16 kHz mono WAV. (Or an uploaded file.)
2. **Cleanup** — Butterworth bandpass 80–7500 Hz → `noisereduce` → peak normalize.
3. **Transcription** — Whisper (upgrading `base` → `large-v3`/`turbo`), language ID
   per utterance.
4. **Per-word language tag → per-word gloss → reconstructed English sentence** for
   RoBERTa; word-choice notes for Ollama.
5. **Prosody block** — F0 (mean/std/range/contour), energy/RMS, jitter, shimmer, HNR,
   speaking rate, pause ratio, voiced ratio, spectral tilt, tremor.
6. **Acoustic SER** — target model: `wav2vec2-large-xlsr-53` or WavLM-large,
   fine-tuned multilingual (classification + optional valence/arousal head).
   *Current stopgap:* `superb/wav2vec2-base-superb-er` (4 classes — mismatched with
   the 8-class code around it; replace in Stage 5).
7. **Fusion** — text emotion + acoustic emotion → combined vector; acoustic-dissonance
   flag when text and voice disagree strongly.

Outputs a `VoiceModalityDoc` (`transcript_raw`, `per_word_lang[]`,
`reconstructed_sentence_en`, `word_usage_notes`, `prosody{}`, `audio_quality`).

---

## 3. Video / face pipeline (`ai/video/` — to be built, Stage 5c/6)

Three sub-models, simplest first. Baseline: OpenFace 2.0 / Py-Feat for AUs, head
pose, gaze, landmarks with zero training.

| Sub-model | Features | Datasets |
|---|---|---|
| Frame expression | 7–8 basic emotions + continuous valence/arousal | AffectNet, RAF-DB, FER+, Aff-Wild2 |
| Action units | FACS AU presence + intensity | DISFA, BP4D, EmotioNet, Aff-Wild2 |
| Micro-expression / mannerism (temporal) | onset/apex/offset, head-pose dynamics, gaze aversion, blink rate, self-touch, fidget index | CASME II, SAMM, SMIC, CAS(ME)³ |

Outputs a `FaceModalityDoc` (`frame_expressions[]`, `action_units[]`,
`micro_expressions[]`, `mannerisms{}`, `temporal_summary`).

---

## 4. Emotion-vector fusion + progress engine (`ai/progress/` — Stage 4)

- **Fusion** — confidence-weighted late fusion of text/voice/face emotion vectors →
  one numeric `fused_vector`; keep cross-modal-agreement + acoustic-dissonance
  diagnostics.
- **Progress formula v1** — pluggable module, `formula_version` stamped on every
  point. Per-patient, **relative to the psychologist-entered baseline**, over a
  rolling window: weighted combination of negative-affect proportion, valence,
  arousal dysregulation, high-distress-emotion mass, psychological-signal load,
  acoustic-dissonance frequency, negative-AU expressivity.
- Calibrated against DAIC-WOZ / E-DAIC PHQ-8 trajectories and psychologist ratings
  (Stage 7).

---

## 5. LLM interpretation (`ai/inference/qwen_reasoning.py`)

- Local **Ollama** running `qwen3:14b`, temperature 0.3.
- System prompt: strictly **non-diagnostic** — no diagnosis, no treatment, no risk
  or severity estimation. Ranks the provided emotion labels by textual evidence,
  then emits Emotional Summary / Dominant Emotion / Key Observations.
- Receives the `expressive_channel` (+ prosody for voice, + mannerisms for video)
  and the psychological signals as interpretive context.
- `interpret()` (text), `interpret_voice()`, `interpret_video()` (to add).

---

## 6. Model registry (`ai/model_registry.py`)

Thread-safe, lazy, process-wide cache. Every heavy model load routes through
`ModelRegistry.get(key, loader)`. Tests stub it in one place
(`tests/unit/conftest.py`). Current loaders: RoBERTa GoEmotions, NLLB-600M, GPT2
perplexity, IndicTrans2-1B, GLiNER, BERT-NER, wav2vec2-SER, faster-whisper.
New loaders (SER-XLSR, Whisper-large, face models) added per stage.

---

## Model inventory

| Model | Role | Trained here? |
|---|---|---|
| `SamLowe/roberta-base-go_emotions` | text emotion (28 labels) | no — pretrained |
| `facebook/nllb-200-distilled-600M` | non-English → English MT | no |
| `ai4bharat/indictrans2-indic-en-1B` | Indic → English MT | no — weights not yet downloaded |
| `gpt2` | spell-candidate perplexity scorer | no |
| GLiNER / BERT-NER | entity protection | no |
| Whisper (`base` → `large-v3`) | speech-to-text | no |
| wav2vec2 SER | acoustic emotion | **YES — Stage 5 (multilingual XLSR fine-tune)** |
| psychological-signals classifier | 8 clinical indicators | **YES — Stage 5b** |
| face expression / AU / micro-expression | facial affect | **YES — Stage 5c** |
| `qwen3:14b` (Ollama) | narrative summary | no — self-hosted |
| Tanglish rule/dictionary layer | code-mix normalization | N/A — not ML |
