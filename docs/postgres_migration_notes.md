# SQLite → PostgreSQL migration

(Note: the request that triggered this said "migrate from MongoDB" --
there was no MongoDB anywhere in this project, only SQLite. This
migrates the actual database, SQLite, to PostgreSQL.)

## What changed

- **`backend/app/db.py`** (new): SQLAlchemy engine/session setup.
  `DATABASE_URL` env var, defaults to a local dev Postgres instance.
- **`backend/app/db_models.py`** (new): SQLAlchemy ORM models, one per
  existing table, same table names and same primary-key *values/format*
  (app-generated strings like `pat-1`, `tj-...`, not UUIDs) -- so every ID
  already referenced throughout the frontend/tests keeps working. JSON
  columns that used to be `TEXT` + manual `json.dumps`/`json.loads`
  (`google_meet_json`, `sections_json`, `analysis_json`, etc.) are now
  native `JSONB` (renamed without the `_json` suffix, e.g. `google_meet`,
  `sections`, `analysis`), queryable with real Postgres JSON operators.
- **`backend/app/database.py`**: same public function names/signatures/
  return shapes as before -- every caller keeps working unchanged --
  reimplemented internally with SQLAlchemy ORM sessions instead of raw
  `sqlite3` cursors.
- **`backend/app/api/{auth,patients,consultations,casesheets}.py`**: these
  four files ran their own hand-written multi-table JOIN SQL directly
  (not through `database.py`'s functions). Converted to SQLAlchemy Core's
  `text()` with named parameters and `.mappings()` (gives the same
  `dict(row)` ergonomics `sqlite3.Row` did), rather than force-fitting
  complex JOINs into ORM query-builder chains. `get_db_connection()` is
  kept as a name (now returns a SQLAlchemy `Connection`) specifically so
  these four files' calling code needed only mechanical changes
  (`?` → `:name`, `cursor.execute` → `conn.execute(text(...), {...})`),
  not a rewrite of their query logic.
- **`scripts/migrate_sqlite_to_postgres.py`** (new, one-time): copies
  every row from the old `backend/data/medtrust.db` into Postgres,
  preserving IDs exactly, via `INSERT ... ON CONFLICT DO NOTHING` (safe
  to re-run). Already run against the local dev database -- the real
  patient account and journal entries that existed in SQLite are now in
  Postgres.
- **`requirements.txt`**: added `sqlalchemy`, `psycopg2-binary`, `pgvector`.

## Frontend logs that were previously browser-only

Following the SQLite→Postgres migration, three more frontend interactions
that only ever lived in the browser's `localStorage` (or, for practice
notes, nowhere at all) were wired to Postgres too:

- **`check_in_sessions`** -- the session-level summary of a `/check-in`
  run (which modes were used + local duration metadata), via the new
  `POST /checkins/session`. The AI-analyzed content of each mode still
  lives in `text_journals`/`voice_records`/`video_analyses`; this ties
  them together.
- **`practice_completions`** -- practice completions plus the
  previously-never-persisted-anywhere writing-practice note, via
  `backend/app/api/practices.py` (`POST /api/practices/complete`,
  `GET /api/practices/completions`).
- **`patient_preferences`** -- the "Daily check-in reminder"/"Practice
  reminders" toggles on `/profile`, via `backend/app/api/preferences.py`
  (`GET`/`PUT /api/preferences/`). Previously local React state, reset on
  every page reload.

All three are protected by the same session-token auth as the check-in
endpoints (`backend/app/security.get_current_patient_id`) -- verified
with real registered-patient requests, including a 401 check for the
unauthenticated case.

## Object storage

Audio/video files were **never** stored in the database, before or after
this migration -- `backend/app/api/checkins.py` writes an uploaded clip to
a temp file, runs the AI pipeline, and deletes the temp file; only the
AI's structured JSON output is persisted. `voice_records` has a nullable
`storage_uri` column reserved for wiring in real object storage (S3/MinIO)
later, but nothing currently writes to it -- there was no existing object
storage integration to migrate.

## pgvector

`db_models.py`'s `Embedding` table is schema-ready for vector similarity
search (generic `source_type`/`source_id` pointer at any existing record,
so future embedding-backed features don't need their own table each
time), and `init_db()` attempts `CREATE EXTENSION IF NOT EXISTS vector`
before creating it.

**On this machine's Postgres 16 install, the `vector` extension itself
isn't available** (`postgresql-16-pgvector` isn't in the apt cache here,
and installing system packages needs sudo this environment doesn't have
non-interactively) -- `init_db()` catches that and skips creating the
`embeddings` table rather than failing startup. The Python `pgvector`
package is installed either way. **To actually enable it**: install the
matching `postgresql-<version>-pgvector` system package on the target
Postgres server, then re-run `init_db()` (or just restart the backend) --
the extension and table will then be created automatically.

## Verified for real, not assumed

All of this was tested against an actual local PostgreSQL 16 instance
(not just written and hoped to work):
- Fresh schema creation (`init_db()`) -- all 9 tables + JSONB columns.
- Patient registration → login → text check-in, through the real AI
  pipeline, landing in Postgres with queryable JSONB
  (`SELECT analysis->>'status' FROM text_journals` works).
- The full consultation → scenario load → AI case-sheet generation →
  doctor approval/lock → print-view flow (all four raw-SQL route files).
- Found and fixed a real pre-existing bug surfaced by clean-database
  testing: `create_patient()`'s MRN generation (`1000 + count`) collided
  with the hardcoded seed data's MRNs once patient count reached them --
  fixed to derive the MRN from a timestamp instead, consistent with how
  this codebase already generates other IDs.
- The full existing test suite (98 tests) passes against Postgres.
- Ran the real data migration against the actual dev database and
  confirmed the real registered patient account and their journal entries
  came through intact.
