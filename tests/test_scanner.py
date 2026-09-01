"""Tests for ProjectScanner and fact auto-linking."""

from __future__ import annotations

from pathlib import Path

from mnemo.engine.scanner import ProjectScanner, link_fact_to_entities
from mnemo.storage.connection import Database


class TestProjectScanner:
    """Validate AST parsing, diff caching, and entity linking."""

    def test_scan_python_files(self, tmp_path: Path) -> None:
        """Scanner correctly extracts modules, classes, and imports from python files."""
        src_dir = tmp_path / "src" / "pkg"
        src_dir.mkdir(parents=True)

        mod_file = src_dir / "service.py"
        mod_file.write_text(
            """
import sqlite3
from pydantic import BaseModel

class AuthService(BaseModel):
    pass
""",
            encoding="utf-8",
        )

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)

        with db.session() as conn:
            res = scanner.scan(conn)
            assert res["scanned_files"] == 1
            assert res["skipped_files"] == 0
            assert res["entities_added"] >= 3

            # Verify entities in DB
            entities = conn.execute("SELECT name, entity_type FROM entities").fetchall()
            names = {e["name"] for e in entities}
            assert "pkg.service" in names
            assert "pkg.service.AuthService" in names
            assert "sqlite3" in names

            # Verify relations
            relations = conn.execute(
                """
                SELECT r.relation_type, e1.name AS src, e2.name AS tgt
                FROM relations r
                JOIN entities e1 ON e1.id = r.source_id
                JOIN entities e2 ON e2.id = r.target_id
                """
            ).fetchall()
            rel_types = {(r["src"], r["tgt"], r["relation_type"]) for r in relations}
            assert ("pkg.service", "pkg.service.AuthService", "CONTAINS") in rel_types
            assert ("pkg.service", "pydantic", "DEPENDS_ON") in rel_types

    def test_incremental_diff_caching(self, tmp_path: Path) -> None:
        """Unmodified files are skipped on subsequent scans."""
        src_dir = tmp_path / "test_lib"
        src_dir.mkdir(parents=True)
        f1 = src_dir / "mod.py"
        f1.write_text("class CoreEngine:\n    pass\n", encoding="utf-8")

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)

        with db.session() as conn:
            # First scan parses 1 file
            res1 = scanner.scan(conn)
            assert res1["scanned_files"] == 1
            assert res1["skipped_files"] == 0

            # Second scan skips unchanged file
            res2 = scanner.scan(conn)
            assert res2["scanned_files"] == 0
            assert res2["skipped_files"] == 1

            # Force scan re-parses
            res3 = scanner.scan(conn, force=True)
            assert res3["scanned_files"] == 1
            assert res3["skipped_files"] == 0

    def test_link_fact_to_entities(self) -> None:
        """Mentioning an entity in a fact text automatically links them."""
        db = Database(":memory:")
        with db.session() as conn:
            # Create an entity
            scanner = ProjectScanner()
            eid = scanner._upsert_entity(conn, "AuthManager", "class", 100.0)

            # Link a fact mentioning AuthManager
            fact_id = "fact-xyz"
            conn.execute(
                """
                INSERT INTO facts (
                    id, text, category, salience, access_count, tier, last_accessed_at,
                    valid_start, valid_end, ingest_start, ingest_end
                ) VALUES (?, 'Handles authentication via AuthManager tokens', 'security', 1.0, 0, 'working', 100.0, 100.0, NULL, 100.0, NULL);
                """,
                (fact_id,),
            )

            linked = link_fact_to_entities(
                fact_id, "Handles authentication via AuthManager tokens", conn, now=100.0
            )
            assert eid in linked

            # Check relation created
            rel = conn.execute(
                "SELECT relation_type FROM relations WHERE source_id = ? AND target_id = ?",
                (eid, fact_id),
            ).fetchone()
            assert rel is not None
            assert rel["relation_type"] == "MENTIONS"

    def test_deleted_file_reconciliation(self, tmp_path: Path) -> None:
        """Deleted files are removed from cache and their entities are soft-deleted."""
        src_dir = tmp_path / "src" / "temp"
        src_dir.mkdir(parents=True)
        f1 = src_dir / "transient.py"
        f1.write_text("class TransientWorker:\n    pass\n", encoding="utf-8")

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)

        with db.session() as conn:
            scanner.scan(conn)

            # Check entity active
            e = conn.execute(
                "SELECT name, valid_end FROM entities WHERE name = 'temp.transient.TransientWorker'"
            ).fetchone()
            assert e is not None
            assert e["valid_end"] is None

            # Delete file from disk and rescan
            f1.unlink()
            scanner.scan(conn)

            # Check cache cleaned
            cache_row = conn.execute(
                "SELECT file_path FROM file_scan_cache WHERE file_path LIKE '%transient.py%'"
            ).fetchone()
            assert cache_row is None

            # Check entity soft-deleted
            e_del = conn.execute(
                "SELECT valid_end FROM entities WHERE name = 'temp.transient.TransientWorker'"
            ).fetchone()
            assert e_del is not None
            assert e_del["valid_end"] is not None

    def test_relative_imports(self, tmp_path: Path) -> None:
        """Relative imports (from . import utils, from ..core import engine) are resolved."""
        pkg_dir = tmp_path / "src" / "app" / "api"
        pkg_dir.mkdir(parents=True)
        f1 = pkg_dir / "routes.py"
        f1.write_text(
            """
from . import handlers
from ..core import Engine
""",
            encoding="utf-8",
        )

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)

        with db.session() as conn:
            scanner.scan(conn)

            entities = conn.execute("SELECT name FROM entities").fetchall()
            names = {e["name"] for e in entities}
            assert "app.api.handlers" in names or "app.api" in names
            assert "app.core" in names
