"""Unit and integration tests for Security (Provenance & Purge), FastEmbed, and Hardening."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from pathlib import Path

from typer.testing import CliRunner

from mnemo.cli.main import app
from mnemo.core.models import AUDNOperation, MemoryTier, SourceType
from mnemo.engine.audn import AUDNClassifier
from mnemo.engine.tier_manager import TierManager
from mnemo.mcp.tools import mnemo_purge, mnemo_remember, mnemo_search
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore, split_code_tokens
from mnemo.storage.vector_store import (
    VectorStore,
    create_embedder,
    get_embedding_metadata,
    sync_embedding_metadata,
)


class TestProvenanceAndPoisoningProtection:
    """Verify source audit and memory poisoning prevention."""

    def test_fact_creation_with_provenance(self, in_memory_db: Database) -> None:
        """New facts store source_type, source_ref, and confidence."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)

        with in_memory_db.session() as conn:
            fact = audn.execute_add(
                "PostgreSQL connection timeout configured to 30s",
                "infra",
                conn,
                source_type=SourceType.GIT_COMMIT.value,
                source_ref="a1b2c3d",
                confidence=0.95,
            )

            assert fact.source_type == SourceType.GIT_COMMIT.value
            assert fact.source_ref == "a1b2c3d"
            assert fact.confidence == 0.95

            row = conn.execute(
                "SELECT source_type, source_ref, confidence FROM facts WHERE id = ?",
                (fact.id,),
            ).fetchone()
            assert row is not None
            assert row["source_type"] == SourceType.GIT_COMMIT.value
            assert row["source_ref"] == "a1b2c3d"
            assert row["confidence"] == 0.95

    def test_memory_poisoning_prevents_untrusted_overwrite(
        self, in_memory_db: Database
    ) -> None:
        """Low-confidence or tool_output facts cannot overwrite protected facts."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)

        with in_memory_db.session() as conn:
            # 1. Authoritative fact added by developer
            fact1 = audn.execute_add(
                "Use OAuth2 with PKCE flow for all mobile API authentications",
                "security",
                conn,
                source_type=SourceType.HUMAN_DEVELOPER.value,
                confidence=0.98,
            )

            # 2. Similar incoming fact from untrusted tool_output trying to update
            op, target_id = audn.classify(
                "Use OAuth2 with basic client credentials for mobile API auth",
                "security",
                conn,
                source_type=SourceType.TOOL_OUTPUT.value,
                confidence=0.5,
            )

            # Must demote to ADD and NOT overwrite fact1
            assert op == AUDNOperation.ADD
            assert target_id is None

            # 3. Existing fact remains active and valid
            row1 = conn.execute("SELECT valid_end FROM facts WHERE id = ?", (fact1.id,)).fetchone()
            assert row1 is not None
            assert row1["valid_end"] is None

    def test_core_tier_protection_clamps_untrusted_facts(
        self, in_memory_db: Database
    ) -> None:
        """Facts with confidence < 0.8 or source_type == 'tool_output' cannot enter Core Tier."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)
        tm = TierManager()

        with in_memory_db.session() as conn:
            # Try to add a tool_output fact directly to Core Tier
            fact_tool = audn.execute_add(
                "Tool output raw log analysis memory",
                "logs",
                conn,
                tier=MemoryTier.CORE,
                source_type=SourceType.TOOL_OUTPUT.value,
                confidence=0.95,
            )
            # Must be clamped to WORKING
            assert fact_tool.tier == MemoryTier.WORKING

            # Try to add low-confidence fact directly to Core Tier
            fact_low = audn.execute_add(
                "Uncertain speculative note",
                "notes",
                conn,
                tier=MemoryTier.CORE,
                source_type=SourceType.AGENT.value,
                confidence=0.6,
            )
            assert fact_low.tier == MemoryTier.WORKING

            # Attempt to decay/promote with salience 1.0
            conn.execute(
                "UPDATE facts SET salience = 0.99, tier = 'working' WHERE id IN (?, ?)",
                (fact_tool.id, fact_low.id),
            )
            tm.decay_all(conn, now=time.time())

            # Neither should have entered Core Tier
            row_tool = conn.execute("SELECT tier FROM facts WHERE id = ?", (fact_tool.id,)).fetchone()
            row_low = conn.execute("SELECT tier FROM facts WHERE id = ?", (fact_low.id,)).fetchone()
            assert row_tool["tier"] == MemoryTier.WORKING.value
            assert row_low["tier"] == MemoryTier.WORKING.value

    def test_untrusted_pinned_fact_is_not_exempt_from_decay(
        self, in_memory_db: Database
    ) -> None:
        """Pinned facts with low confidence or tool_output are not exempt from decay."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)
        tm = TierManager()

        with in_memory_db.session() as conn:
            fact = audn.execute_add(
                "Temporary pinned secret note",
                "notes",
                conn,
                metadata={"pinned": True},
                source_type=SourceType.TOOL_OUTPUT.value,
                confidence=0.5,
            )

            # Advance ticks and decay
            res = tm.decay_all(conn, current_tick=1000)
            assert res["processed"] >= 1

            row = conn.execute("SELECT salience FROM facts WHERE id = ?", (fact.id,)).fetchone()
            assert float(row["salience"]) < 1.0


class TestPhysicalPurgeRedaction:
    """Verify permanent physical destruction of facts (mnemo_purge)."""

    def test_execute_purge_destroys_all_traces(self, in_memory_db: Database) -> None:
        """Purge wipes fact from facts, links, FTS5, vectors, and creates tombstone."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)
        secret_text = "API_KEY_SECRET_sk_live_99482749281749"
        expected_hash = hashlib.sha256(secret_text.encode("utf-8")).hexdigest()

        with in_memory_db.session() as conn:
            # 1. Insert fact
            fact = audn.execute_add(secret_text, "secrets", conn)
            # Add entity link
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, properties_json, salience, access_count, last_accessed_at, valid_start, ingest_start) "
                "VALUES ('e1', 'Config', 'class', '{}', 1.0, 0, 1000.0, 1000.0, 1000.0)"
            )
            conn.execute(
                "INSERT INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
                "VALUES (?, 'e1', 'hash1', '2026-01-01')",
                (fact.id,),
            )

            # Verify fact exists in FTS and table
            assert conn.execute("SELECT COUNT(*) FROM facts WHERE id = ?", (fact.id,)).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM fact_entity_links WHERE fact_id = ?", (fact.id,)).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM facts_fts WHERE id = ?", (fact.id,)).fetchone()[0] == 1

            # 2. Execute physical purge
            purged_id = audn.execute_purge(fact.id, conn, reason="compromised_api_key")
            assert purged_id == fact.id

            # 3. Verify zero traces in facts, links, and FTS
            assert conn.execute("SELECT COUNT(*) FROM facts WHERE id = ?", (fact.id,)).fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM fact_entity_links WHERE fact_id = ?", (fact.id,)).fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM facts_fts WHERE id = ?", (fact.id,)).fetchone()[0] == 0

            # 4. Verify tombstone created with SHA256, without original text
            tomb = conn.execute(
                "SELECT * FROM audit_tombstones WHERE id = ?",
                (f"tomb_{fact.id}",),
            ).fetchone()
            assert tomb is not None
            assert tomb["fact_hash"] == expected_hash
            assert tomb["reason"] == "compromised_api_key"
            assert "API_KEY_SECRET" not in str(dict(tomb))

    def test_mcp_mnemo_purge_tool(self) -> None:
        """FastMCP mnemo_purge tool is forbidden and returns security warning."""
        unique_secret = f"secret_{uuid.uuid4().hex}"
        rem_res = mnemo_remember(f"Sensitive password {unique_secret}", category="sec", force_op="add")
        assert "[ADD]" in rem_res

        # Retrieve ID
        search_res = mnemo_search(unique_secret)
        import re
        match = re.search(r"i:([a-f0-9-]+)", search_res)
        assert match is not None
        fact_id = match.group(1)

        # Call mnemo_purge via MCP (must be rejected)
        purge_res = mnemo_purge(fact_id, reason="leak_remediation")
        assert "[FORBIDDEN]" in purge_res
        assert "administrative CLI-only" in purge_res

        # Verify fact was NOT purged
        search_after = mnemo_search(unique_secret)
        assert fact_id in search_after


class TestFTSCodeTokenization:
    """Verify subtoken extraction for CamelCase and snake_case."""

    def test_split_code_tokens(self) -> None:
        """split_code_tokens splits code identifiers into searchable sub-tokens."""
        tokens = split_code_tokens("UserService.login_user_by_id")
        assert "UserService" in tokens
        assert "User" in tokens
        assert "Service" in tokens
        assert "login_user_by_id" in tokens
        assert "login" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens

    def test_fts5_code_subtoken_search(self, in_memory_db: Database) -> None:
        """Searching sub-tokens matches facts containing compound code identifiers."""
        vs = VectorStore(create_embedder())
        audn = AUDNClassifier(vs)
        fts = FTSStore()

        with in_memory_db.session() as conn:
            fact = audn.execute_add(
                "Implemented AuthenticationController.handle_login_request method for user sessions",
                "code",
                conn,
            )

            # Search by CamelCase sub-token "Authentication"
            hits1 = fts.search("Authentication", conn)
            hit_ids1 = [h[0] for h in hits1]
            assert fact.id in hit_ids1

            # Search by snake_case sub-token "login"
            hits2 = fts.search("login", conn)
            hit_ids2 = [h[0] for h in hits2]
            assert fact.id in hit_ids2


class TestEmbeddingMetadataAndDoctor:
    """Verify model persistence, doctor diagnostics, and CLI commands."""

    def test_embedding_metadata_sync(self, in_memory_db: Database) -> None:
        """Model name and dimension are synced to project_state."""
        with in_memory_db.session() as conn:
            sync_embedding_metadata(conn, "test-model-onnx", 384)
            model, dim = get_embedding_metadata(conn)
            assert model == "test-model-onnx"
            assert dim == 384

    def test_cli_purge_and_doctor(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """CLI mnemo purge and mnemo doctor execute successfully."""
        db_file = str(tmp_path / "cli_sec_test.db")

        # 1. Init
        init_res = cli_runner.invoke(app, ["init", "--db", db_file])
        assert init_res.exit_code == 0

        # 2. Remember with provenance
        rem_res = cli_runner.invoke(
            app,
            [
                "remember",
                "Database credentials db_password_12345",
                "--source-type",
                "human_developer",
                "--confidence",
                "0.99",
                "--db",
                db_file,
            ],
        )
        assert rem_res.exit_code == 0
        assert "add" in rem_res.output

        # Extract ID
        db = Database(db_path=db_file)
        with db.session() as conn:
            row = conn.execute("SELECT id FROM facts WHERE text LIKE '%db_password_12345%'").fetchone()
            assert row is not None
            fact_id = row["id"]
        db.close()

        # 3. Purge via CLI
        purge_res = cli_runner.invoke(
            app,
            ["purge", fact_id, "--reason", "secret_leak", "--db", db_file],
        )
        assert purge_res.exit_code == 0
        assert "redacted" in purge_res.output

        # 4. Doctor reports health, tombstones, and model
        doc_res = cli_runner.invoke(app, ["doctor", "--db", db_file])
        assert doc_res.exit_code == 0
        assert "audit tombstones" in doc_res.output
        assert "orphan links" in doc_res.output
        assert "embedding model" in doc_res.output


class TestDatabaseConcurrencyRetry:
    """Verify exponential backoff retry on database locked/busy."""

    def test_execute_with_retry_on_transient_lock(self, in_memory_db: Database) -> None:
        """execute_with_retry retries on sqlite3.OperationalError: database is locked."""
        attempts = 0

        def flaky_transaction(conn: sqlite3.Connection) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = in_memory_db.execute_with_retry(flaky_transaction, max_retries=5)
        assert result == "success"
        assert attempts == 3
