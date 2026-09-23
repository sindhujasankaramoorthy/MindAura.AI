"""
MedTrust AI - SQLAlchemy engine/session setup for PostgreSQL.

Replaces the previous raw sqlite3 connection layer. DATABASE_URL is read
from the environment so dev/test/prod can point at different databases
without a code change; see backend/.env.example for the local-dev value.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.engine import Connection

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://mindaura:mindaura_dev_pw@127.0.0.1:5432/mindaura",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_connection() -> Connection:
    """
    A raw SQLAlchemy Connection, for the hand-written multi-table JOIN
    queries (consultations/casesheets) that are clearer as SQL than as
    ORM query-builder chains. Caller is responsible for closing it (same
    contract the old sqlite3 get_db_connection() had) -- use as:

        conn = get_connection()
        try:
            row = conn.execute(text("SELECT ..."), {"id": x}).mappings().first()
            conn.commit()
        finally:
            conn.close()
    """
    return engine.connect()
