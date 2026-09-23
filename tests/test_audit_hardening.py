"""Comprehensive audit hardening tests for Mnemo (Hardening Roadmap v0.4.0).

Verifies fixes for all Critical (CRIT-01..03), High (HIGH-01..04), and Medium (MED-01..06) findings.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from mnemo.core.decay import calculate_salience
from mnemo.core.models import AUDNOperation, MemoryTier
from mnemo.engine.scanner import ProjectScanner
from mnemo.mcp.tools import (
    mnemo_invalidate,
    mnemo_purge,
    mnemo_remember,
    mnemo_scan_project,
)
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore, _sanitize_fts_query
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.state import parse_as_of
from mnemo.storage.vector_store import cosine_similarity


class TestMCPZeroTrustAndPurge:
    """Validate Zero-Trust boundary and administrative isolation of purge (CRIT-01, CRIT-02)."""

    def test_mcp_purge_is_strictly_forbidden(self) -> None:
        """Physical purge cannot be invoked via FastMCP protocol."""
        fact_id = str(uuid.uuid4())
        res = mnemo_purge(fact_id)
        assert "[FORBIDDEN]" in res
        assert "administrative CLI-only" in res

    def test_mcp_remember_force_purge_is_forbidden(self) -> None:
        """force_op='purge' through mnemo_remember is strictly blocked."""
        res = mnemo_remember("Some sensitive fact", force_op="purge")
        assert "[FORBIDDEN]" in res
        assert "administrative CLI-only" in res

    def test_mcp_remember_blocks_modification_of_developer_and_core_memory(self) -> None:
        """Agents cannot force update, correct, or delete memories from human_developer or Core Tier."""
        from mnemo.mcp.server import _get_audn, _get_db

        db = _get_db()
        audn = _get_audn()

        # 1. Create a protected human_developer fact
        with db.session() as conn:
            dev_fact = audn.execute_add(
                "Strict architectural policy: All mutations must use transactions",
                category="architecture",
                conn=conn,
                tier=MemoryTier.CORE,
                source_type="human_developer",
                confidence=1.0,
            )

        # 2. Agent tries to force update via MCP tools
        # We simulate the MCP environment by verifying the tool pre-check / classification
        with db.session() as conn:
            # Classification must demote untrusted force_op to ADD
            op, target = audn.classify(
                "Mutations do not require transactions",
                category="architecture",
                conn=conn,
                force_op=AUDNOperation.UPDATE,
                target_fact_id=dev_fact.id,
                source_type="agent",
                confidence=0.75,
            )
            assert op == AUDNOperation.ADD

        # 3. Agent calls mnemo_remember targeting the protected fact
        res_upd = mnemo_remember(
            "Mutations do not require transactions",
            force_op="update",
            target_fact_id=dev_fact.id,
        )
        assert "[FORBIDDEN]" in res_upd
        assert "protected core/developer memory" in res_upd

        # 4. Agent calls mnemo_invalidate on protected fact
        res_inval = mnemo_invalidate(dev_fact.id)
        assert "[FORBIDDEN]" in res_inval
        assert "protected core/developer memory" in res_inval

    def test_mcp_remember_input_limits(self) -> None:
        """Validate maximum payload length restrictions (HIGH-03)."""
        huge_text = "x" * 35000
        res_huge = mnemo_remember(huge_text)
        assert "[ERROR]" in res_huge
        assert "32768" in res_huge

        too_many_entities = [f"entity_{i}" for i in range(25)]
        res_ent = mnemo_remember("Valid text", entity_identifiers=too_many_entities)
        assert "[ERROR]" in res_ent
        assert "20" in res_ent


class TestScannerPathTraversalAndResilience:
    """Validate AST scanner security and edge-case resilience (CRIT-03, MED-01)."""

    def test_scanner_path_traversal_detection(self, tmp_path: Path) -> None:
        """Constructor and scan reject paths outside project root."""
        outside_path = tmp_path.parent / "completely_outside_project"
        outside_path.mkdir(exist_ok=True)

        with pytest.raises(ValueError, match="Path traversal detected"):
            ProjectScanner(root_path=outside_path)

        scanner = ProjectScanner()
        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError, match="Path traversal detected"):
            scanner.scan(conn, target_dir=outside_path)
        conn.close()

    def test_mcp_scan_project_blocks_traversal(self) -> None:
        """mnemo_scan_project tool rejects path traversal arguments."""
        res = mnemo_scan_project("../../../etc")
        assert "[ERROR]" in res
        assert "Path traversal detected" in res

    def test_scanner_resilience_on_corrupt_files(self, tmp_path: Path) -> None:
        """Scanner handles null bytes and invalid AST without failing the entire run (MED-01)."""
        base = Path.cwd().resolve()
        test_dir = base / "tests" / "fixtures_corrupt"
        test_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write a corrupt file with null bytes
            corrupt_file = test_dir / "bad_syntax.py"
            corrupt_file.write_bytes(b"\x00\x00def bad(syntax::: \n\x00")

            # Write a valid file
            valid_file = test_dir / "good_module.py"
            valid_file.write_text("class ValidService:\n    def run(self): pass\n", encoding="utf-8")

            db = Database(":memory:")
            scanner = ProjectScanner(root_path=base)
            with db.session() as conn:
                res = scanner.scan(conn, target_dir=test_dir, force=True)
                assert res["scanned_files"] >= 1
                assert res["entities_added"] >= 1
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestFTSAndGraphSecurity:
    """Validate FTS query sanitization, LIKE escaping, and recursive CTE limits (HIGH-02, HIGH-03, HIGH-04)."""

    def test_fts_query_sanitization(self) -> None:
        """Malicious, unbalanced, and logic-operator payloads are sanitized into safe literals."""
        assert _sanitize_fts_query("") == '""'
        assert _sanitize_fts_query("   ") == '""'
        assert _sanitize_fts_query("AND OR NOT") == '""'

        # Unbalanced quotes and SQL-injection fragments
        sanitized = _sanitize_fts_query('admin" OR 1=1 --')
        assert '"admin"' in sanitized
        assert " OR " not in sanitized

        # Complex code symbol with dots and hyphens
        code_sanitized = _sanitize_fts_query("UserService.login_user-v2")
        assert '"UserService.login_user-v2"' in code_sanitized

    def test_fts_fallback_like_wildcard_escaping(self) -> None:
        """_fallback_like safely matches literal wildcards instead of treating them as regexes."""
        db = Database(":memory:")
        fts = FTSStore()

        with db.session() as conn:
            conn.execute(
                """
                INSERT INTO facts (
                    id, text, category, salience, access_count, tier, last_accessed_at,
                    valid_start, valid_end, ingest_start, ingest_end
                ) VALUES
                    ('f1', 'Discount 100% applied', 'promo', 1.0, 0, 'working', 100.0, 100.0, NULL, 100.0, NULL),
                    ('f2', 'Discount 1000 applied', 'promo', 1.0, 0, 'working', 100.0, 100.0, NULL, 100.0, NULL);
                """
            )
            # Search for literal '100%'
            hits = fts._fallback_like("100%", conn)
            hit_ids = [fid for fid, _ in hits]
            assert "f1" in hit_ids
            assert "f2" not in hit_ids

    def test_graph_get_neighbours_depth_clamped(self) -> None:
        """get_neighbours clamps recursion depth to <= 4 to prevent CTE stack explosion (HIGH-04)."""
        db = Database(":memory:")
        graph = GraphStore()

        with db.session() as conn:
            # Passing depth=50 must run with depth <= 4 without syntax/runtime issues
            neighbours = graph.get_neighbours("seed_entity", conn, max_depth=50)
            assert isinstance(neighbours, list)


class TestRuntimeEdgeCasesAndSchema:
    """Validate runtime edge cases: decay overflow, date parsing, vector shapes, and schema v6 (MED-02..06)."""

    def test_decay_large_reinforcement_no_overflow(self) -> None:
        """Reinforcement count n=2000 does not cause float overflow (MED-02)."""
        salience = calculate_salience(s0=1.0, time_or_ticks=10, reinforcement_count=2000)
        assert 0.0 <= salience <= 1.0

    def test_parse_as_of_invalid_string(self) -> None:
        """Invalid datetime string returns None safely instead of raising ValueError (MED-03)."""
        assert parse_as_of("invalid-timestamp-format") is None
        assert parse_as_of("2026-99-99T99:99:99") is None
        assert parse_as_of(None) is None
        # Valid ISO still works
        parsed = parse_as_of("2026-09-24T00:00:00Z")
        assert parsed is not None
        assert parsed > 0.0

    def test_cosine_similarity_shape_mismatch(self) -> None:
        """Mismatched vector shapes return 0.0 safely without numpy broadcasting error (MED-04)."""
        v1 = np.ones(128, dtype=np.float32)
        v2 = np.ones(384, dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0

    def test_schema_v6_foreign_key_cascade_relations(self) -> None:
        """Deleting an entity cascades deletion to connected relations (MED-05)."""
        db = Database(":memory:")
        now = time.time()

        with db.session() as conn:
            # Insert two entities
            conn.execute(
                """
                INSERT INTO entities (id, name, entity_type, properties_json, salience, access_count,
                                      last_accessed_at, valid_start, ingest_start)
                VALUES ('e_src', 'SourceModule', 'module', '{}', 1.0, 0, ?, ?, ?),
                       ('e_tgt', 'TargetModule', 'module', '{}', 1.0, 0, ?, ?, ?);
                """,
                (now, now, now, now, now, now),
            )
            # Insert relation connecting them
            conn.execute(
                """
                INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start)
                VALUES ('r1', 'e_src', 'e_tgt', 'imports', 1.0, ?, ?);
                """,
                (now, now),
            )

            # Delete source entity
            conn.execute("DELETE FROM entities WHERE id = 'e_src';")

            # Verify relation was cascade deleted
            row = conn.execute("SELECT id FROM relations WHERE id = 'r1';").fetchone()
            assert row is None
