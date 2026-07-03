"""
database.py
===========
Database engine, session factory, and initialization utilities.

Purpose
-------
Centralizes all SQLAlchemy engine/session plumbing so no other module
needs to know the connection string or engine options. Also provides
`init_db()` which creates all tables and seeds them with the sample
knowledge base (diseases, symptoms, rules, medicines) defined in
sample_data.py, and `reset_db()` for a clean rebuild.

Running this file directly (``python database.py``) creates and seeds
medical.db from scratch, which satisfies the "medical.db creation
script" requirement.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_PATH, DATABASE_URL, get_logger
from models import Base

logger = get_logger(__name__)

# `check_same_thread=False` is required for SQLite when accessed from a
# Streamlit app, which may service callbacks on different threads.
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context-managed database session.

    Purpose: Guarantee every session is closed (and rolled back on error)
             regardless of how the calling code exits the `with` block.
    Input  : none
    Output : yields a SQLAlchemy Session
    Logic  : try/yield/except-rollback/finally-close pattern.
    Time Complexity : O(1) plus whatever the wrapped queries cost.
    Space Complexity: O(1) beyond the session object itself.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session rolled back due to an error.")
        raise
    finally:
        session.close()


def _is_database_empty() -> bool:
    """Return True if the diseases table has no rows (fresh database)."""
    from models import Disease  # local import avoids a circular import at module load time

    with get_session() as session:
        count = session.query(Disease).count()
    return count == 0


def init_db(force_reseed: bool = False) -> None:
    """
    Create all tables (if missing) and seed the knowledge base.

    Purpose: Single entry point called at application startup (app.py)
             and from the CLI to guarantee medical.db exists and is
             populated before any reasoning engine runs.
    Input  : force_reseed (bool) - if True, wipes and reloads sample data
             even if the database already has rows.
    Output : None (side effect: medical.db is created/populated on disk).
    Logic  : 1) Create tables via Base.metadata.create_all.
             2) If the diseases table is empty (or force_reseed), import
                sample_data and call its seed() function.
    Time Complexity : O(D + S + M + R) where D/S/M/R are the number of
                       diseases/symptoms/medicines/rules being inserted.
    Space Complexity: O(D + S + M + R) for the ORM objects created.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured at %s", DATABASE_PATH)

    if force_reseed:
        reset_db()
        return

    if _is_database_empty():
        logger.info("Database is empty - seeding sample medical knowledge base...")
        import sample_data

        sample_data.seed(SessionLocal)
        logger.info("Seeding complete.")
    else:
        logger.info("Database already contains data - skipping seed.")


def reset_db() -> None:
    """
    Drop and recreate every table, then reseed from sample_data.

    Purpose: Give developers/recruiters a one-command way to get back to
             a known-good state (`python database.py --reset`).
    Time Complexity : O(D + S + M + R) for the reseed, plus table DDL cost.
    Space Complexity: O(D + S + M + R)
    """
    logger.warning("Resetting database: all existing data will be lost.")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    import sample_data

    sample_data.seed(SessionLocal)
    logger.info("Database reset and reseeded successfully.")


def table_names() -> list[str]:
    """Return the list of table names currently present in medical.db."""
    return inspect(engine).get_table_names()


if __name__ == "__main__":
    import sys

    if "--reset" in sys.argv:
        reset_db()
    else:
        init_db()
    print(f"Database ready at: {DATABASE_PATH}")
    print(f"Tables: {table_names()}")
