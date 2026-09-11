# MindAura System Audit

Repository audit · read-only · 2026-08-19

A complete inspection of the current repository — no assumptions carried in from prior versions — followed by a direct verdict, a component-by-component keep/change decision, and a prioritized roadmap. Nothing was modified, installed, or trained to produce this.

## At a glance

**Verdict: MOSTLY ON THE RIGHT TRACK — NEEDS CHANGES**

The AI core (text + voice emotion analysis) is genuinely well-built and worth keeping. But the thing that makes MindAura a *monitoring system* rather than an emotion-analysis demo — patient-specific thresholds, alerts, psychologist notification, escalation — does not exist anywhere beyond a single disconnected scoring formula. No database, no patient identity, no client that can even reach the backend.

- **0** databases, of any kind
- **0** alerts/notifications implemented
- **1** working backend endpoint
- **2** real AI pipelines (text, voice)

---

## Phase 1 — Audit

### 1. The product, as built

The intended flow is a loop: a patient is assessed, submits blogs continuously, the AI analyses each submission, evidence accumulates over time, and a patient-specific threshold decides between quiet monitoring and an alert that reaches a psychologist and, if configured, an emergency contact.

What's actually implemented is a single link in that chain, not the loop: a patient's text can be analysed, once, statelessly, by an AI pipeline that is genuinely good at its job. Everything before it (assessment, baseline) and after it (accumulation, threshold, alert, escalation) does not exist. There is also currently no way for a patient to even reach that one working link — no client in this repo can call the backend.

> **The core requirement is 0% built as a system.** A real, working distress-scoring formula exists in `ai/voice/fusion_engine.py` — but it is not patient-specific (there is no patient identity anywhere to attach it to), not persisted, not configurable, and not connected to any alert, notification, or escalation mechanism.

### 2. Patient-specific threshold

| Question | Answer |
|---|---|
| Does a threshold exist? | Only as hardcoded constants inside `ai/voice/fusion_engine.py` (score bands at 75 / 50 / 25). |
| Where is it stored? | Nowhere. It is a Python literal, recomputed fresh on every call. |
| How / who configures it? | Not configurable by anyone — it's a source-code constant, not a setting. |
| Is it patient-specific? | No. And it structurally can't be — there is no patient-identity concept anywhere in the codebase to attach a threshold to. |
| What contributes to it? | Text emotion scores (GoEmotions) + voice emotion label (Wav2Vec2-SER), blended via a fixed 60/40 weighted sum with hand-picked per-emotion weights. |
| Static or dynamic? | Fully static — no history awareness, no adaptation, each call is independent of any prior submission. |
| How often evaluated? | Only when explicitly called from the interactive terminal script `ai/voice/voice_pipeline.py` — not on a schedule, not per-request via any API, not triggered by anything. |
| What happens when crossed? | A string label (`"High Distress"`, etc.) is included in a returned Python dict. Nothing downstream reacts to it. |
| Alerts / notifications / escalation / supervision status / close contacts / acknowledgement / duplicate-prevention? | All confirmed absent — no code, no data model, no UI logic for any of these. `backend/app/api/alerts.py` is 0 bytes. |

**Verdict on the current design:** there isn't a threshold *system* to evaluate yet — there's a scoring *formula*, sitting in isolation. The formula itself (fixed weights, three bands) is a reasonable v1 starting point for a single global heuristic, but it is fundamentally incompatible with "patient-specific" as written, because nothing in this repository currently represents a patient for a threshold to belong to.

### 3. Multimodal blogs

- **Text** — Fully implemented pipeline, reachable via one backend endpoint. Not persisted.
- **Voice** — Fully implemented pipeline, reachable only via a terminal script — no API endpoint.
- **Video** — Confirmed absent — zero code, zero stub, zero dependency declared anywhere.

Text and voice each have a real, working AI pipeline behind them. Neither is wired to a client a patient could actually use, and neither result is stored anywhere.

### 4. Text pipeline — audit

Genuinely the most mature part of the repository:

`Input (raw string)` → `TextNormalizer` (13-stage: clean → NER protect [GLiNER/BERT-NER] → Viterbi language ID → SymSpell + Tanglish autocorrect → semantic normalize → negation recovery → NLLB translation if non-English) → `RoBERTa GoEmotions` (28 labels) concurrently with a rule-based psychological-signal extractor → optional `Qwen` (Ollama) interpretation, explicitly non-diagnostic prompt.

- **Functional:** normalization, emotion classification, signal extraction, optional LLM interpretation — all confirmed working end-to-end via `EmotionAnalyzer.process()`, tested this session.
- **Incomplete:** nothing is stored; no connection to patient identity, history, or the threshold system; only one trimmed field set is exposed via the API.

### 5. Voice pipeline — audit

`Input (.wav)` → `Faster-Whisper` transcription (CPU, int8) → text emotion (RoBERTa, now normalization-aligned with the text path) + acoustic SER (Wav2Vec2, windowed + heuristic refinement) → `EmotionFusionEngine` (distress score + risk label) → printed to a terminal. No API.

Functional end-to-end, but only as a standalone script (`python -m ai.voice.voice_pipeline`). Not exposed through the backend at all — `backend/app/api/journals.py` explicitly documents voice as out of scope for its first slice.

> A real mismatch worth fixing: the SER model in use (`superb/wav2vec2-base-superb-er`) only outputs 4 classes (angry / happy / sad / neutral), but the code built around it — `EmotionMapper.MAPPING` and `fusion_engine.py`'s weight tables — expects 8–9 (also fear, surprise, disgust, calm). Those branches are currently dead code, unreachable with today's model.

### 6. Video / face

Confirmed completely absent: no directory, no import, no dependency declaration, no UI screen reference, anywhere in the repository.

Not a blocker. The architecture that already exists — a self-contained `ai/` pipeline module plus a thin backend service wrapping it — is exactly the shape a video pipeline would slot into later: a new `ai/video/` package, a new service module, a new endpoint, with zero rework required of the text or voice paths.

### 7. Tanglish

A real, working, entirely rule-and-dictionary-based approach — no custom-trained neural model exists for Tanglish specifically. A hand-tuned Viterbi (HMM) decoder does token-level English/Tanglish language routing; SymSpell plus fuzzy-matching against a 242,000-word frequency-ranked vocabulary handles spelling correction; a regex-and-dictionary layer handles semantic normalization and SOV→SVO grammatical reordering; off-the-shelf NLLB and IndicXlit models handle translation/transliteration.

This is a technically appropriate v1: it works today, without needing training data that doesn't exist. The genuine gap is that the only Tanglish text dataset in the repo (`ai/tanglish_model/data/raw/train.tsv`) is labeled with coarse sentiment, not emotion — so there's currently no path from "rule-based" to "fine-tuned neural" for text without new data.

### 8. Current AI models

| Model | Purpose | Status | Loaded via |
|---|---|---|---|
| `SamLowe/roberta-base-go_emotions` | Text emotion, 28 labels | Pretrained, unmodified | `get_roberta_go_emotions()` |
| `facebook/nllb-200-distilled-600M` | Non-English → English translation | Pretrained | `get_nllb_600m()` |
| `gpt2` | Perplexity scorer for spelling-candidate disambiguation | Pretrained, repurposed | `get_gpt2_perplexity_model()` |
| `ai4bharat/indictrans2-indic-en-1B` | Indic-language → English translation | Pretrained, **fully wired but never called** | `get_indictrans2()` |
| `NeuML/gliner-bert-tiny` | Zero-shot NER | Pretrained | `get_gliner_ner()` |
| `dslim/bert-base-NER-uncased` | NER fallback | Pretrained | `get_bert_ner_fallback()` |
| `superb/wav2vec2-base-superb-er` | Voice emotion, 4 classes | Pretrained, IEMOCAP-trained — class-count mismatch | `get_wav2vec2_ser()` |
| `faster-whisper` (base) | Speech-to-text | Pretrained, CPU int8 | `get_faster_whisper()` |
| ai4bharat XlitEngine | Tanglish → Tamil transliteration | Pretrained | `ai/transliteration/tanglish_to_tamil.py` |
| `qwen3:14b` (via Ollama) | Non-diagnostic interpretive summary | Pretrained, self-hosted | `ai/inference/qwen_reasoning.py` |
| SymSpell | English spell correction | Dictionary algorithm, not a model | `advanced_correction.py` |

All nine model checkpoints now load lazily through a single shared registry (this session's work) — a plain English/Tanglish sentence never triggers GPT2 or IndicTrans2 at all, and RoBERTa loads once, shared between the text and voice paths.

### 9–11. Datasets — text and voice

| Dataset | Location | Shape | Fit |
|---|---|---|---|
| RAVDESS + CREMA-D (preprocessed) | `ai/training/processed_audio/` | 8,882 clips, 8 emotion folders | Ready for SER fine-tuning as-is. American-accented actors — fixes class-count mismatch, not accent gap. `calm`/`surprise` underrepresented. |
| Tanglish sentiment corpus | `ai/tanglish_model/data/raw/*.tsv` | 11,335 rows, 5-class sentiment | Real, correctly placed — wrong label scheme for emotion fine-tuning. Usable as a secondary cross-check signal. |
| Tanglish vocabulary | `ai/tanglish_model/data/processed/tanglish_words.csv` | 242,338 word-frequency rows | Actively load-bearing across every fuzzy-match/autocorrect call site. |
| EmoTa *(not yet in repo)* | github.com/aaivu/EmoTa | 936 utterances, 22 native Tamil speakers, 5 emotions | Would close the accent gap RAVDESS/CREMA-D can't. |
| TamilEmo *(not yet in repo)* | github.com/Chaarangan/TamilEmo | 42,000+ Tamil YouTube comments, real emotion labels | Would unblock text emotion fine-tuning. |

**Recommendation: keep everything currently in the repo.** None of it is redundant — the sentiment corpus and a future emotion corpus measure different things and can coexist; RAVDESS/CREMA-D and EmoTa solve different halves of the same problem (volume vs. accent).

### 12. Training pipeline

`ai/training/train_roberta.py` and `ai/training/train_ser.py` are both 0 bytes. Nothing in this repository trains anything today.

- **SER (voice) — yes, worth training.** Data exists in the right shape today; the payoff (fixing the 4-class/8-class mismatch) is concrete and immediate.
- **Text emotion (RoBERTa) — not yet.** Blocked on label scheme, not on effort.
- **Qwen — no.** General-purpose LLM capability is sufficient for interpretive summarization; fine-tuning is a larger, riskier undertaking for uncertain benefit.
- **SymSpell / dictionaries — not "training."** Improve via curation, not ML training.

### 13. Actual pipeline trace

**Text:** Android `JournalScreen` "Analyze" button (`onClick = { /* TODO */ }`) ✕ no network layer exists ··· `POST /journals/analyze` works if called directly → response returned as JSON, never stored, never evaluated against anything.

**Voice:** Android mic button (`onClick = { /* Voice Journal */ }`) ✕ no network layer, no API endpoint exists at all ··· only reachable via `python -m ai.voice.voice_pipeline`, standalone.

**Video:** N/A — confirmed not implemented.

**Monitoring:** Analysis → aggregation → patient state → threshold → alert → notification → escalation. **None of these steps connect to the next one.**

Disconnected components and dead ends found this session:
- `alerts.py`, `analysis.py`, `auth.py`, `dashboard.py`, `journal_service.py`, `summary_service.py` — all 0 bytes.
- `DoctorDashboardScreen.kt`, `EmergencySupportScreen.kt`, `ContactsManagementScreen.kt` — UI shells, zero backing logic; `DoctorDashboard` isn't even reachable from any navigation path.
- `translate_indic()` / IndicTrans2 — fully implemented, never called.
- `fusion_engine.py` — real logic, but an island reachable only from a terminal script.
- Android has no networking code and no `INTERNET` permission at all.
- `frontend/` — 100% empty, disconnected from everything.
- Two Android test files reference a `MainScreen`/`MainScreenViewModel` that doesn't exist anywhere in `src/main` — broken template leftovers.

Text and voice results are shaped completely differently today — expected while standalone, but they'll need a unifying schema once persistence exists.

### 14. Database

Confirmed, exhaustively: no SQL schema, no ORM models, no migration tooling, no `.env`, no DB packages in either requirements file, no Docker Compose service. The only "persistence" anywhere in the repository is a hardcoded in-memory list in an unused Android template file.

**Every item on the checklist — patients, psychologists, observations, baselines, blogs, text/voice/video analysis, monitoring history, thresholds, risk state, alerts, notifications, close contacts, supervision status, alert history, trends — needs to be designed and built from nothing.**

### 15. Backend

FastAPI — the right framework choice. Current surface: `GET /health` (no model loading, instant) and `POST /journals/analyze` (text only, unauthenticated). Everything else — patient APIs, psychologist APIs, monitoring APIs, threshold APIs, alert APIs, notification APIs — doesn't exist. No auth, no background job processing beyond an in-request thread pool, no global error handling, no CORS configuration, no structured logging.

The pattern that exists — thin API route → thin service module → `ai/` pipeline — is sound and worth extending, not replacing.

### 16. Frontend

**Patient side (Android):** Genuinely decent Compose UI: journal entry (real local state), dashboard (hardcoded placeholder data), emergency/contacts screens (forms exist, handlers do nothing), wellness/community (static). No screen can reach the backend — no networking library declared, no `INTERNET` permission.

**Psychologist side:** One screen, `DoctorDashboardScreen.kt` — three static text labels with no data, no navigation entry point, no backing logic.

**Web (React):** Every file is 0 bytes, `package.json` included — not even a valid, runnable project.

### 17. Security and privacy

- No authentication anywhere — `auth.py` empty; Login accepts any input and navigates straight to the dashboard.
- No secrets hardcoded — checked android, frontend, config files; clean.
- No CORS policy configured in FastAPI.
- No file-upload handling exists yet — audio/video uploads will need validation, size limits, storage-access control designed in from the start.
- No consent or audit-trail mechanism — worth flagging early given clinically sensitive content and third-party (emergency contact) notification; India's DPDP Act 2023 is the relevant framework and should shape the database/API design from day one.

---

## Phase 2 — Technical verdict

### MOSTLY ON THE RIGHT TRACK — NEEDS CHANGES

Not *on the right track* unqualified — the stated core requirement (patient-specific threshold monitoring with alerting and escalation) is currently 0% implemented as a connected system.

Not *major architectural changes required* or *should be restructured* either — nothing about the existing architecture is wrong and needs tearing down. The `ai/` pipeline's design (self-contained module, models behind a shared lazy-loading registry, thin backend service wrapping it) is a genuinely good foundation. The gap isn't bad architecture — it's that the product's defining layer (patients, thresholds, alerts, a way for a client to reach any of it) hasn't been built yet.

**So: keep building on the AI core you have, and treat the monitoring loop as new construction, not a fix.**

---

## Phase 3 — What should we keep?

| Component | Recommendation | Reason |
|---|---|---|
| Text preprocessing (`TextNormalizer`) | **KEEP** | Sophisticated, functional, recently cleaned up. |
| Text emotion model (RoBERTa GoEmotions) | **KEEP** | Works well pretrained on the normalized-to-English text it receives. |
| Tanglish rule/dictionary layer | **KEEP** | Functional, deduplicated; no trained alternative ready to replace it. |
| Voice transcription (Faster-Whisper) | **KEEP** | Standard, appropriate, works. |
| Voice emotion model (Wav2Vec2-SER) | **MODIFY** | Fine-tune to 8 classes — data already exists in the repo. |
| Fusion engine / distress score | **MODIFY** | Keep the math; rebuild its context to be patient-specific and connected to a real threshold system. |
| IndicTrans2 wiring | **MODIFY** | Wire in for the intended multilingual feature, or remove the dead weight. |
| Voice + text training datasets | **KEEP** | None are redundant. |
| Backend (FastAPI pattern) | **MODIFY / EXTEND** | Right framework, right pattern; needs more endpoints, not a rebuild. |
| Database | **BUILD** | Doesn't exist. |
| Patient / psychologist data model | **BUILD** | Doesn't exist anywhere. |
| Threshold / alert / escalation system | **BUILD** | Only the scoring math exists. |
| Android — patient screens | **KEEP UI / ADD LOGIC** | Real, decent Compose UI; needs networking, auth, real data binding. |
| Android — psychologist dashboard | **REBUILD** | Three static labels, not reachable via navigation today. |
| Web frontend (React) | **REBUILD or DROP** | 100% empty; decide if needed for v1 given Android is further along. |
| Auth (backend + Android) | **BUILD** | Doesn't exist anywhere; blocks every patient/psychologist-specific feature. |
| Video / face pipeline | **DO NOT BUILD YET** | Correctly out of scope; architecture can absorb it later. |

---

## Phase 4 — What should we change?

### Critical — must change before continuing
1. No database — nothing can be "monitored over time" without persistence.
2. No patient/psychologist identity model — a threshold/alert needs something to attach to.
3. No authentication — role separation and sensitive-data protection depend on it.
4. Threshold/alert/escalation is the stated core requirement and is currently only a disconnected formula.
5. No client can reach the backend at all.

### Important — should change soon
1. Wav2Vec2-SER's 4-class/8-class mismatch.
2. Voice pipeline isn't exposed via any API.
3. README overstates progress (frontend "in progress" when empty; Android not mentioned at all).
4. No CORS policy in FastAPI.
5. `DoctorDashboardScreen` unreachable from any navigation path.

### Optional — can improve later
- Wire in or remove the unused IndicTrans2 path.
- Text-side emotion fine-tuning, once TamilEmo is integrated.
- Pull in EmoTa/TamilEmo.
- Structured logging and observability.

### Do not change — already good enough
- The `TextNormalizer` 13-stage pipeline architecture.
- The shared model-registry / lazy-loading pattern.
- The backend's thin-service-over-`ai/` pattern.
- The new test suite's structure.

---

## Phase 5 — What should we build next?

Order below is dependency order — each step unblocks the next.

### 1. Data model + database
Current: nothing exists. Missing: Patients, Psychologists, JournalEntries, AnalysisResults. Tech: Postgres/SQLite + SQLAlchemy + Alembic. Training: no. Verify: round-trip a record through create → read → migrate.

### 2. Authentication (patient + psychologist roles)
Current: `auth.py` empty, Login is a no-op. Tech: FastAPI + JWT + password hashing. Training: no. Verify: unauthenticated calls rejected; role mismatch returns 403.

### 3. Persist text analysis, attached to a patient
Current: works but forgets. Change: add `patient_id`, write a row per submission. Training: no. Verify: two submissions from the same patient both appear in their history.

### 4. Voice API endpoint, same persistence
Current: terminal-only. Change: new route wrapping `voice_pipeline.py` as a service. Training: optionally the SER fine-tune first. Verify: a real recording produces a stored, retrievable result.

### 5. Patient-specific threshold + alert creation
Current: global hardcoded formula. Change: threshold config API, evaluation step, Alert table. Training: no. Verify: a deliberately extreme submission creates exactly one alert, not duplicates.

### 6. Notification + escalation
Current: nothing. Change: notification service (email to start), escalation policy per patient. Data: close-contact info (which the app needs to actually save first). Verify: an alert reaches a psychologist and can be acknowledged; history retained.

### 7. Android networking layer
Current: zero network code, no `INTERNET` permission. Tech: Retrofit + OkHttp + coroutines. Verify: a real journal submission from the phone appears in the backend's stored data.

### 8. Psychologist dashboard, for real
Current: three static labels, unreachable. Change: decide the surface (Android vs. repurposed React), build against steps 1–6's APIs. Verify: a psychologist sees an alert generated by step 5 in near-real time.

---

## Phase 6 — Final target architecture

Not a predetermined pipeline forced onto what exists — this is what's already working, plus the minimum new construction to close the gaps found above.

```
Clients                Backend                 AI layer              Persistence            Monitoring loop
Android (extend) +  →  FastAPI — auth,      →  ai/ — unchanged    →  Postgres —          →  Evaluate → alert →
psychologist web       patient/psych APIs,      in shape, text +      patients, results,      notify → escalate
dashboard (repurpose    threshold config,        voice pipelines,      thresholds, alerts,     → acknowledge
empty React app)        alerts (all new)         lazy model registry   history (all new)       (all new)
```

Video slots in later as a new module inside the same `ai/` layer and a new endpoint — nothing above needs to change shape to accommodate it when it's time.

---

## Phase 7 — Roadmap

### Now
- Database + data model
- Authentication
- Persist text analysis per patient
- Android networking layer

### Next
- Voice API endpoint + persistence
- Patient-specific threshold + alert creation
- Notification + close-contact escalation
- Real psychologist dashboard

### Later
- SER fine-tuning (class alignment)
- TamilEmo / EmoTa integration
- Trend / longitudinal views
- Web frontend: build or drop, decisively

### Future
- Video / face pipeline
- Text emotion fine-tuning on Tanglish
- Self-supervised voice domain adaptation
- Compliance / audit tooling (DPDP Act)

### In one pass

| | |
|---|---|
| Existing code to preserve | `ai/` pipeline, model registry, backend pattern, test suite, Android patient-screen UI shells. |
| Existing data to preserve | RAVDESS/CREMA-D, Tanglish sentiment corpus, Tanglish vocabulary — all of it. |
| Data to preprocess differently | None of the existing data — new data (EmoTa/TamilEmo) needs bringing in alongside it, not replacing it. |
| Models to keep as-is | RoBERTa GoEmotions, NLLB, Whisper, GLiNER/BERT-NER, IndicXlit, Qwen. |
| Models to train | Wav2Vec2-SER (class alignment now; accent adaptation once EmoTa is in). |
| Models not to train | Qwen (general-purpose capability suffices); text RoBERTa (until label data exists). |
| New data required | Emotion-labeled Tanglish text (TamilEmo); Tamil-accented voice (EmoTa) — identified, not pulled in yet. |
| New modules required | Auth, patient/psychologist services, threshold/alert engine, notification service, Android network layer. |
| Database changes | Build from nothing. |
| Backend changes | Extend the existing FastAPI pattern with the endpoints listed in Phase 5. |
| Frontend changes | Android: add networking + wire stubs to real calls. Web: decide its purpose (likely psychologist-only) before building. |
| Threshold/alert infrastructure required | Per-patient config, evaluation trigger, Alert table, notification delivery, acknowledgment flow, alert history — all new. |

---

*Read-only audit — no files modified, no packages installed, no models trained to produce this. Awaiting review before any implementation begins.*
