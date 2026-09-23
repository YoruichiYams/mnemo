"""Tests for schema migration and integrity verification."""

from __future__ import annotations

import sqlite3

from mnemo.storage.connection import Database
from mnemo.storage.migrations import (
    CURRENT_SCHEMA_VERSION,
    apply_migrations,
    check_database_integrity,
    get_schema_version,
    verify_and_repair_fts,
)


class TestMigrations:
    """Validate PRAGMA user_version migrations and self-healing."""

    def test_migrations_applied_on_init(self) -> None:
        """Database initialized in memory has current schema version."""
        db = Database(":memory:")
        with db.session() as conn:
            ver = get_schema_version(conn)
            assert ver == CURRENT_SCHEMA_VERSION

    def test_apply_migrations_idempotent(self) -> None:
        """Calling apply_migrations repeatedly produces no errors and preserves version."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        ver1 = apply_migrations(conn)
        assert ver1 == CURRENT_SCHEMA_VERSION

        ver2 = apply_migrations(conn)
        assert ver2 == CURRENT_SCHEMA_VERSION
        conn.close()

    def test_fts_repair_on_desync(self) -> None:
        """verify_and_repair_fts detects and rebuilds out-of-sync FTS index."""
        db = Database(":memory:")
        with db.session() as conn:
            # Insert a fact directly without trigger
            conn.execute(
                """
                INSERT INTO facts (
                    id, text, category, salience, access_count, tier, last_accessed_at,
                    valid_start, valid_end, ingest_start, ingest_end
                ) VALUES ('f1', 'Direct fact bypass trigger', 'general', 1.0, 0, 'working', 100.0, 100.0, NULL, 100.0, NULL);
                """
            )
            # Delete FTS entry to simulate desync
            conn.execute("DELETE FROM facts_fts WHERE id = 'f1';")

            # Check integrity shows desync
            repaired = verify_and_repair_fts(conn)
            assert repaired is True

            # Check again, now healthy
            repaired_again = verify_and_repair_fts(conn)
            assert repaired_again is False

    def test_check_database_integrity(self) -> None:
        """check_database_integrity returns healthy status for a clean database."""
        db = Database(":memory:")
        with db.session() as conn:
            health = check_database_integrity(conn)
        assert health["integrity_ok"] is True
        assert health["schema_version"] == CURRENT_SCHEMA_VERSION
        assert health["fts_in_sync"] is True
        assert "facts" in health["table_counts"]
        assert "file_scan_cache" in health["table_counts"]
        assert "fact_entity_links" in health["table_counts"]
        assert "project_state" in health["table_counts"]

