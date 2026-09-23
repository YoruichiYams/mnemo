"""Schema migration manager and self-healing integrity routines for Mnemo.

Tracks schema state using SQLite's built-in ``PRAGMA user_version``.
Provides automated repair for FTS5 out-of-sync states and incremental
DDL migrations without user intervention.
"""

from __future__ import annotations

import sqlite3
from typing import Any

# Current target schema version
CURRENT_SCHEMA_VERSION = 6

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


def _migration_v3(conn: sqlite3.Connection) -> None:
    """Migration v3: Fact-entity linking table and is_stale flag for AST drift."""
    sql = """
    CREATE TABLE IF NOT EXISTS fact_entity_links (
        fact_id             TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
        entity_id           TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        entity_hash_at_link TEXT NOT NULL,
        created_at          TEXT NOT NULL,
        PRIMARY KEY (fact_id, entity_id)
    );

    CREATE INDEX IF NOT EXISTS idx_fact_entity_links_entity
        ON fact_entity_links (entity_id);

    CREATE INDEX IF NOT EXISTS idx_fact_entity_links_fact
        ON fact_entity_links (fact_id);
    """
    conn.executescript(sql)

    # Idempotently add is_stale column to facts if not already present
    pragma_rows = conn.execute("PRAGMA table_info(facts);").fetchall()
    columns = {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in pragma_rows}
    if "is_stale" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;")


def _migration_v4(conn: sqlite3.Connection) -> None:
    """Migration v4: Activity ticks, project state, and bitemporal index."""
    sql = """
    CREATE TABLE IF NOT EXISTS project_state (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        activity_tick   INTEGER NOT NULL DEFAULT 0,
        updated_at      REAL    NOT NULL
    );
    INSERT OR IGNORE INTO project_state (id, activity_tick, updated_at) VALUES (1, 0, 0.0);

    CREATE INDEX IF NOT EXISTS idx_facts_bitemporal_as_of
        ON facts (valid_start, valid_end, ingest_start, ingest_end);
    """
    conn.executescript(sql)

    pragma_rows = conn.execute("PRAGMA table_info(facts);").fetchall()
    columns = {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in pragma_rows}
    if "last_accessed_tick" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN last_accessed_tick INTEGER NOT NULL DEFAULT 0;")
    if "reinforcement_count" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0;")


def _migration_v5(conn: sqlite3.Connection) -> None:
    """Migration v5: Provenance fields, audit tombstones, and embedding metadata."""
    sql = """
    CREATE TABLE IF NOT EXISTS audit_tombstones (
        id          TEXT    PRIMARY KEY,
        fact_hash   TEXT    NOT NULL,
        reason      TEXT    NOT NULL DEFAULT 'security_redaction',
        purged_at   REAL    NOT NULL
    );
    """
    conn.executescript(sql)

    pragma_rows = conn.execute("PRAGMA table_info(facts);").fetchall()
    columns = {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in pragma_rows}
    if "source_type" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN source_type TEXT NOT NULL DEFAULT 'agent';")
    if "source_ref" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN source_ref TEXT;")
    if "confidence" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0;")
    if "search_tokens" not in columns:
        conn.execute("ALTER TABLE facts ADD COLUMN search_tokens TEXT NOT NULL DEFAULT '';")

    ps_rows = conn.execute("PRAGMA table_info(project_state);").fetchall()
    ps_columns = {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in ps_rows}
    if "embedding_model" not in ps_columns:
        conn.execute(
            "ALTER TABLE project_state ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'deterministic-hash-64';"
        )
    if "embedding_dimension" not in ps_columns:
        conn.execute(
            "ALTER TABLE project_state ADD COLUMN embedding_dimension INTEGER NOT NULL DEFAULT 64;"
        )


def _migration_v6(conn: sqlite3.Connection) -> None:
    """Migration v6: Add cascading foreign keys for relations table."""
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS relations_v6 (
            id              TEXT    PRIMARY KEY,
            source_id       TEXT    NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            target_id       TEXT    NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type   TEXT    NOT NULL,
            weight          REAL    NOT NULL DEFAULT 1.0,
            valid_start     REAL    NOT NULL,
            valid_end       REAL,
            ingest_start    REAL    NOT NULL,
            ingest_end      REAL,
            FOREIGN KEY(source_id) REFERENCES entities(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES entities(id) ON DELETE CASCADE
        );

        INSERT OR IGNORE INTO relations_v6
            SELECT r.id, r.source_id, r.target_id, r.relation_type, r.weight,
                   r.valid_start, r.valid_end, r.ingest_start, r.ingest_end
            FROM relations r
            WHERE EXISTS (SELECT 1 FROM entities e1 WHERE e1.id = r.source_id)
              AND EXISTS (SELECT 1 FROM entities e2 WHERE e2.id = r.target_id);

        DROP TABLE relations;
        ALTER TABLE relations_v6 RENAME TO relations;

        CREATE INDEX IF NOT EXISTS idx_relations_source
            ON relations (source_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_relations_target
            ON relations (target_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_relations_valid_window
            ON relations (valid_start, valid_end, ingest_end);
        """
    )
    conn.execute("PRAGMA foreign_keys = ON;")


_MIGRATIONS = [
    (1, _migration_v1),
    (2, _migration_v2),
    (3, _migration_v3),
    (4, _migration_v4),
    (5, _migration_v5),
    (6, _migration_v6),
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
                SELECT id, text || ' ' || COALESCE(search_tokens, ''), category FROM facts
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
                SELECT id, text || ' ' || COALESCE(search_tokens, ''), category FROM facts
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
    tables_to_check = [
        "facts",
        "entities",
        "relations",
        "debt_ledger",
        "file_scan_cache",
        "fact_entity_links",
        "project_state",
        "audit_tombstones",
    ]
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

    # Orphan fact_entity_links check
    orphan_links = 0
    try:
        orphan_row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM fact_entity_links fel
            WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.id = fel.fact_id)
               OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = fel.entity_id);
            """
        ).fetchone()
        orphan_links = int(orphan_row["c"]) if orphan_row else 0
    except Exception:
        pass

    # Embedding metadata check
    emb_model = "unknown"
    emb_dim = 0
    try:
        ps_row = conn.execute(
            "SELECT embedding_model, embedding_dimension FROM project_state WHERE id = 1;"
        ).fetchone()
        if ps_row:
            emb_model = str(ps_row["embedding_model"])
            emb_dim = int(ps_row["embedding_dimension"])
    except Exception:
        pass

    return {
        "integrity_ok": integrity_ok,
        "schema_version": user_ver,
        "target_version": CURRENT_SCHEMA_VERSION,
        "journal_mode": str(journal_mode_row[0]) if journal_mode_row else "unknown",
        "foreign_keys_enabled": bool(foreign_keys_row and foreign_keys_row[0] == 1),
        "fts_in_sync": fts_sync,
        "orphan_links": orphan_links,
        "embedding_model": emb_model,
        "embedding_dimension": emb_dim,
        "table_counts": counts,
    }
