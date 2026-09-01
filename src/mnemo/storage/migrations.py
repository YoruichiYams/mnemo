"""Schema migration manager and self-healing integrity routines for Mnemo.

Tracks schema state using SQLite's built-in ``PRAGMA user_version``.
Provides automated repair for FTS5 out-of-sync states and incremental
DDL migrations without user intervention.
"""

from __future__ import annotations

import sqlite3
from typing import Any

# Current target schema version
CURRENT_SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Migration scripts
# ---------------------------------------------------------------------------


def _migration_v1(conn: sqlite3.Connection) -> None:
    """Migration v1: Base schema tables, FTS5 virtual table, triggers, and indexes."""
    from mnemo.storage.connection import _load_schema_sql

    conn.executescript(_load_schema_sql())


def _migration_v2(conn: sqlite3.Connection) -> None:
    """Migration v2: AST file scan cache table and category index."""
    sql = """
    -- Cache table for incremental AST scanning
    CREATE TABLE IF NOT EXISTS file_scan_cache (
        file_path       TEXT PRIMARY KEY,
        mtime           REAL NOT NULL,
        sha256          TEXT NOT NULL,
        entity_count    INTEGER NOT NULL DEFAULT 0,
        relation_count  INTEGER NOT NULL DEFAULT 0,
        scanned_at      REAL NOT NULL
    );

    -- Secondary lookup indexes
    CREATE INDEX IF NOT EXISTS idx_facts_category
        ON facts (category, valid_start, valid_end);

    CREATE INDEX IF NOT EXISTS idx_entities_type
        ON entities (entity_type, valid_start, valid_end);
    """
    conn.executescript(sql)


_MIGRATIONS = [
    (1, _migration_v1),
    (2, _migration_v2),
]


# ---------------------------------------------------------------------------
# Public Migration & Healing API
# ---------------------------------------------------------------------------


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the current schema version from ``PRAGMA user_version``."""
    row = conn.execute("PRAGMA user_version;").fetchone()
    if row is None:
        return 0
    return int(row[0])


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Update ``PRAGMA user_version``."""
    conn.execute(f"PRAGMA user_version = {int(version)};")


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply all pending migrations up to ``CURRENT_SCHEMA_VERSION``.

    Args:
        conn: Active SQLite connection.

    Returns:
        The new schema version after migrations.
    """
    try:
        conn.execute("BEGIN IMMEDIATE;")
    except Exception:
        # Already in a transaction or in-memory lock
        pass

    try:
        current_ver = get_schema_version(conn)

        for target_ver, migration_fn in _MIGRATIONS:
            if current_ver < target_ver:
                migration_fn(conn)
                set_schema_version(conn, target_ver)
                current_ver = target_ver

        # Ensure FTS5 synchronization after any structural updates
        verify_and_repair_fts(conn)
        try:
            conn.commit()
        except Exception:
            pass
        return current_ver
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def verify_and_repair_fts(conn: sqlite3.Connection) -> bool:
    """Verify that ``facts_fts`` is in sync with active ``facts``.

    If out of sync or if ID mismatch is detected, automatically rebuilds FTS5 index.

    Args:
        conn: Active SQLite connection.

    Returns:
        True if repair was required and executed, False if already healthy.
    """
    try:
        active_facts_row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM facts
            WHERE ingest_end IS NULL AND valid_end IS NULL;
            """
        ).fetchone()
        fts_row = conn.execute("SELECT COUNT(*) AS c FROM facts_fts;").fetchone()

        active_count = int(active_facts_row["c"]) if active_facts_row else 0
        fts_count = int(fts_row["c"]) if fts_row else 0

        # Check for missing IDs between active facts and FTS
        id_mismatch_row = conn.execute(
            """
            SELECT 1 FROM facts f
            LEFT JOIN facts_fts ft ON f.id = ft.id
            WHERE f.ingest_end IS NULL AND f.valid_end IS NULL AND ft.id IS NULL
            LIMIT 1;
            """
        ).fetchone()

        if active_count != fts_count or id_mismatch_row is not None:
            # Rebuild FTS5 index from active facts
            conn.execute("DELETE FROM facts_fts;")
            conn.execute(
                """
                INSERT INTO facts_fts (id, text, category)
                SELECT id, text, category FROM facts
                WHERE ingest_end IS NULL AND valid_end IS NULL;
                """
            )
            return True
        return False
    except Exception:
        try:
            conn.execute("DELETE FROM facts_fts;")
            conn.execute(
                """
                INSERT INTO facts_fts (id, text, category)
                SELECT id, text, category FROM facts
                WHERE ingest_end IS NULL AND valid_end IS NULL;
                """
            )
            return True
        except Exception:
            return False


def check_database_integrity(conn: sqlite3.Connection) -> dict[str, Any]:
    """Run SQLite integrity checks and collect diagnostic health statistics.

    Args:
        conn: Active SQLite connection.

    Returns:
        Dictionary containing health status, schema version, PRAGMAs, and table counts.
    """
    integrity_row = conn.execute("PRAGMA integrity_check(1);").fetchone()
    integrity_ok = integrity_row is not None and str(integrity_row[0]).lower() == "ok"

    journal_mode_row = conn.execute("PRAGMA journal_mode;").fetchone()
    foreign_keys_row = conn.execute("PRAGMA foreign_keys;").fetchone()
    user_ver = get_schema_version(conn)

    # Table counts
    tables_to_check = ["facts", "entities", "relations", "debt_ledger", "file_scan_cache"]
    counts: dict[str, int] = {}
    for tbl in tables_to_check:
        try:
            r = conn.execute(f"SELECT COUNT(*) AS c FROM {tbl};").fetchone()
            counts[tbl] = int(r["c"]) if r else 0
        except Exception:
            counts[tbl] = -1

    # FTS5 status
    fts_sync = False
    try:
        fts_row = conn.execute("SELECT COUNT(*) AS c FROM facts_fts;").fetchone()
        fts_count = int(fts_row["c"]) if fts_row else 0
        active_row = conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE ingest_end IS NULL AND valid_end IS NULL;"
        ).fetchone()
        active_count = int(active_row["c"]) if active_row else 0
        fts_sync = fts_count == active_count
    except Exception:
        pass

    return {
        "integrity_ok": integrity_ok,
        "schema_version": user_ver,
        "target_version": CURRENT_SCHEMA_VERSION,
        "journal_mode": str(journal_mode_row[0]) if journal_mode_row else "unknown",
        "foreign_keys_enabled": bool(foreign_keys_row and foreign_keys_row[0] == 1),
        "fts_in_sync": fts_sync,
        "table_counts": counts,
    }
