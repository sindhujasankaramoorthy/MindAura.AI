# MindAura — System Architecture

Status: 2026-09 · v2 pipeline plan (`mindaura-v2-pipeline` branch).

---

## What MindAura is

A **clinician-in-the-loop monitoring tool for psychologists**. A psychologist
assesses a patient and records a **baseline**; the patient then submits journal
"blogs" (text / voice / video, any language) over time; MindAura analyses each
submission and maintains a **progress score** on the psychologist's dashboard.
It is **non-diagnostic** — it supports the clinician, it does not replace them.

---

## Layers

```
┌────────────────────────────────────────────────────────────────────────────┐
│ CLIENTS                                                                     │
│   Patient app (Android, Compose)  ── login A ──┐                            │
│   Psychologist dashboard (web, TBD)── login B ──┤                            │
└───────────────────────────────────────────────┬────────────────────────────┘
                                                │ HTTPS / JWT (role-scoped)
┌───────────────────────────────────────────────▼────────────────────────────┐
│ BACKEND  (FastAPI — backend/app/)                                           │
│   api/       auth · journals(text) · voice · video · dashboard · alerts     │
│   services/  thin wrappers over ai/                                         │
│   store/     SQLAlchemy models + Alembic migrations (SQLite)                │
│   background task queue (in-process) for slow analysis jobs                  │
└───────────────┬───────────────────────────────────────────┬────────────────┘
                │                                           │
┌───────────────▼───────────────┐          ┌────────────────▼───────────────┐
│ AI LAYER  (ai/)               │          │ PERSISTENCE  (SQLite mindaura.db)│
│  preprocessing/  text norm     │  write#1 │  patients · psychologists        │
│  voice/          speech+SER    │◄────────►│  patient_psychologist_link       │
│  video/          face (TBD)    │  write#2 │  baselines · submissions         │
│  inference/      RoBERTa+Qwen  │          │  modality_docs (JSON col)        │
│  progress/       formula (TBD) │          │  analysis_results (JSON col)     │
│  model_registry  lazy cache    │          │  progress_points · audit_log     │
│  schemas/        Pydantic      │          │  (WAL mode)                       │
└───────────────┬───────────────┘          └─────────────────────────────────┘
                │
┌───────────────▼───────────────┐
│ Ollama (local) — qwen3:14b     │
└───────────────────────────────┘
```

---

## Data flow (one submission)

1. Patient uploads a blog → `POST /journals/{text|voice|video}` (authenticated,
   role = patient).
2. Backend stores the raw asset + a `submissions` row, kicks off a background job.
3. Job runs the modality pipeline → `modality_docs` row (JSON). **[DB write #1]**
4. Job reads that doc, runs the emotion-vector layer (RoBERTa + SER + face → fused
   vector) and Ollama → `analysis_results` row (JSON). **[DB write #2]**
5. Progress engine reads `baseline` + recent `analysis_results` → `progress_points`
   row.
6. If a progress threshold is crossed → `alerts` row; psychologist notified.
7. Psychologist dashboard (`GET /dashboard/...`, role = psychologist) reads
   progress trend + latest summary per patient.

---

## Key decisions

| Area | Decision |
|---|---|
| DB engine | **SQLite** (`mindaura.db`), SQLAlchemy + Alembic, JSON columns for documents, WAL mode. `store/` abstraction allows a later Postgres move. |
| Auth | FastAPI + JWT, two roles (patient / psychologist), 403 on cross-role / cross-patient access. |
| Async | In-process background task queue (FastAPI `BackgroundTasks` / worker loop). No external broker at this scale. |
| Schemas | All request/response and pipeline documents are versioned Pydantic models in `ai/schemas/` with a `schema_version` field. |
| Progress formula | Pluggable (`ai/progress/`), research-derived, `formula_version` stamped on every point. |
| Text emotion model | Kept pretrained (translate-first); revisited after the loop + longitudinal data exist. |
| Video | Uploaded clips (not live capture, unless decided otherwise). Slots in as `ai/video/` + a new endpoint with no rework of text/voice. |
| Compliance | India DPDP Act 2023 — consent capture, audit trail, encryption at rest (Stage 8). |

---

## Build order (see the plan file for full detail)

0. Repo hygiene → 1. JSON schemas → 2. SQLite DB + auth → 3. Wire pipeline through
DB + voice API → 4. Progress engine + dashboard → 5. Model training (SER, signals,
face) → 6. Video modality → 7. Clinical calibration → 8. Compliance hardening.

---

## Current state (2026-09)

- **`ai/` text + voice pipelines** — functional; text reachable via one API endpoint,
  voice terminal-only. Fully provisioned locally (`.venv/`, 8/9 HF models, `qwen3:14b`).
- **Backend** — `GET /health` + `POST /journals/analyze` only; auth / DB / other
  endpoints are 0-byte stubs.
- **Android** — Compose UI shell, 12 screens, no networking, no audio/camera capture.
- **Web frontend** — empty.
- **Face pipeline, training code, database, progress engine** — not built.
