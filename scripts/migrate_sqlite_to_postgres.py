"""
One-time data migration: copies existing rows from the old SQLite
database (backend/data/medtrust.db) into the new PostgreSQL database
(backend/app/database.py's SQLAlchemy models), preserving every existing
ID exactly. Safe to re-run: existing rows in Postgres are left alone
(ON CONFLICT DO NOTHING), so this only ever adds rows that aren't there
yet -- including the pre-existing seed rows the SQLite file and a freshly
init_db()'d Postgres database both already share.

Usage (from the project root):
    python scripts/migrate_sqlite_to_postgres.py [path/to/medtrust.db]

Requires the target Postgres database to already exist and have its
schema created (run backend.app.database.init_db() first, or just start
the backend once).
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db import engine
from backend.app.db_models import (
    User, Patient, Consultation, Transcript, CaseSheet,
    VideoAnalysis, TextJournal, VoiceRecord, Session as SessionModel,
)

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "backend" / "data" / "medtrust.db"


def _rows(conn: sqlite3.Connection, table: str):
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]


def _upsert_skip(pg_conn, table, rows):
    """INSERT ... ON CONFLICT DO NOTHING, so already-migrated/seeded rows
    (matched by primary key) are silently left as-is."""
    if not rows:
        return 0
    stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    result = pg_conn.execute(stmt)
    return result.rowcount


def migrate(sqlite_path: Path) -> None:
    if not sqlite_path.exists():
        print(f"No SQLite database found at {sqlite_path} -- nothing to migrate.")
        return

    sconn = sqlite3.connect(str(sqlite_path))

    with engine.begin() as pconn:
        users = [
            {**r, "password_hash": r.get("password_hash")}
            for r in _rows(sconn, "users")
        ]
        n = _upsert_skip(pconn, User.__table__, users)
        print(f"users: {n} inserted ({len(users)} in source)")

        patients = []
        for r in _rows(sconn, "patients"):
            r = dict(r)
            r["known_allergies"] = json.loads(r.pop("known_allergies_json") or "[]")
            r["chronic_conditions"] = json.loads(r.pop("chronic_conditions_json") or "[]")
            patients.append(r)
        n = _upsert_skip(pconn, Patient.__table__, patients)
        print(f"patients: {n} inserted ({len(patients)} in source)")

        consultations = []
        for r in _rows(sconn, "consultations"):
            r = dict(r)
            r["google_meet"] = json.loads(r.pop("google_meet_json") or "{}")
            consultations.append(r)
        n = _upsert_skip(pconn, Consultation.__table__, consultations)
        print(f"consultations: {n} inserted ({len(consultations)} in source)")

        transcripts = _rows(sconn, "transcripts")
        n = _upsert_skip(pconn, Transcript.__table__, transcripts)
        print(f"transcripts: {n} inserted ({len(transcripts)} in source)")

        casesheets = []
        for r in _rows(sconn, "casesheets"):
            r = dict(r)
            r["sections"] = json.loads(r.pop("sections_json") or "{}")
            r["multilingual_summary"] = json.loads(r.pop("multilingual_summary_json") or "null")
            r["approval"] = json.loads(r.pop("approval_json") or "null")
            casesheets.append(r)
        n = _upsert_skip(pconn, CaseSheet.__table__, casesheets)
        print(f"casesheets: {n} inserted ({len(casesheets)} in source)")

        video_analyses = []
        for r in _rows(sconn, "video_analyses"):
            r = dict(r)
            r["observation"] = json.loads(r.pop("observation_json") or "{}")
            video_analyses.append(r)
        n = _upsert_skip(pconn, VideoAnalysis.__table__, video_analyses)
        print(f"video_analyses: {n} inserted ({len(video_analyses)} in source)")

        text_journals = []
        for r in _rows(sconn, "text_journals"):
            r = dict(r)
            r["analysis"] = json.loads(r.pop("analysis_json") or "{}")
            text_journals.append(r)
        n = _upsert_skip(pconn, TextJournal.__table__, text_journals)
        print(f"text_journals: {n} inserted ({len(text_journals)} in source)")

        voice_records = []
        for r in _rows(sconn, "voice_records"):
            r = dict(r)
            r["analysis"] = json.loads(r.pop("analysis_json") or "{}")
            voice_records.append(r)
        n = _upsert_skip(pconn, VoiceRecord.__table__, voice_records)
        print(f"voice_records: {n} inserted ({len(voice_records)} in source)")

        sessions = _rows(sconn, "sessions")
        n = _upsert_skip(pconn, SessionModel.__table__, sessions)
        print(f"sessions: {n} inserted ({len(sessions)} in source)")

    sconn.close()
    print("Migration complete.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SQLITE_PATH
    migrate(path)
