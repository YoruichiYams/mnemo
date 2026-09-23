"""Unit and integration tests for Fact-Entity Linking & AST Drift detection (Phase 1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mnemo.engine.retriever import HybridRetriever
from mnemo.engine.scanner import ProjectScanner
from mnemo.engine.tier_manager import TierManager
from mnemo.mcp.tools import mnemo_inspect, mnemo_remember
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, _vec_to_blob, create_embedder


class TestFactEntityLinkingAndASTDrift:
    """Validate fact-entity links, AST drift detection, and graph-channel RRF."""

    @pytest.fixture(autouse=True)
    def _chdir_to_tmp_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

    def test_remember_with_entity_identifiers(self) -> None:
        """Calling mnemo_remember with entity_identifiers creates fact_entity_links rows."""
        text = "UserService handles user logins and JWT issuance"
        res = mnemo_remember(
            text,
            category="auth",
            force_op="add",
            entity_identifiers=["UserService.login", "AuthModule"],
        )
        assert "[ADD]" in res

        from mnemo.mcp.server import _get_db

        db = _get_db()
        with db.session() as conn:
            # Find the fact that was just created
            row = conn.execute(
                "SELECT id, is_stale FROM facts WHERE text = ? ORDER BY ingest_start DESC LIMIT 1;",
                (text,),
            ).fetchone()
            assert row is not None
            fact_id = str(row["id"])
            assert row["is_stale"] == 0

            # Inspect via MCP tool
            inspect_res = mnemo_inspect(fact_id)
            assert fact_id[:8] in inspect_res

            # Check fact_entity_links
            links = conn.execute(
                """
                SELECT fel.entity_id, fel.entity_hash_at_link, e.name
                FROM fact_entity_links fel
                JOIN entities e ON e.id = fel.entity_id
                WHERE fel.fact_id = ?;
                """,
                (fact_id,),
            ).fetchall()

            assert len(links) >= 2
            linked_names = {lnk["name"] for lnk in links}
            assert "UserService.login" in linked_names
            assert "AuthModule" in linked_names

            for lnk in links:
                assert len(lnk["entity_hash_at_link"]) == 64

    def test_ast_drift_marks_fact_stale(self, tmp_path: Path) -> None:
        """Modifying a function in code changes its hash and marks linked facts as stale."""
        src_dir = tmp_path / "src" / "finance"
        src_dir.mkdir(parents=True)
        py_file = src_dir / "tax.py"
        py_file.write_text(
            """
def calculate_tax(amount: float) -> float:
    return amount * 0.2
""",
            encoding="utf-8",
        )

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)

        now = time.time()
        with db.session() as conn:
            # 1. Initial scan
            res1 = scanner.scan(conn)
            assert res1["scanned_files"] == 1
            assert res1["entities_added"] >= 2

            # Verify entity exists
            ent = conn.execute(
                "SELECT id, name, properties_json FROM entities WHERE name = 'finance.tax.calculate_tax';"
            ).fetchone()
            assert ent is not None
            ent_id = str(ent["id"])
            props = json.loads(ent["properties_json"])
            initial_hash = props["hash"]
            assert len(initial_hash) == 64

            # 2. Create a fact linked to calculate_tax
            fact_id = "fact_tax_rule"
            conn.execute(
                """
                INSERT INTO facts (
                    id, text, category, salience, access_count, tier, last_accessed_at,
                    valid_start, valid_end, ingest_start, ingest_end, is_stale
                ) VALUES (?, 'Standard tax rate is 20 percent calculated by tax module', 'tax', 1.0, 0, 'working', ?, ?, NULL, ?, NULL, 0);
                """,
                (fact_id, now, now, now),
            )

            conn.execute(
                """
                INSERT INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (fact_id, ent_id, initial_hash, str(now)),
            )

            # Check fact is initially NOT stale
            f_row = conn.execute("SELECT is_stale FROM facts WHERE id = ?;", (fact_id,)).fetchone()
            assert f_row["is_stale"] == 0

            # 3. Modify function body in file
            py_file.write_text(
                """
def calculate_tax(amount: float) -> float:
    return amount * 0.25 + 5.0
""",
                encoding="utf-8",
            )

            # Rescan
            res2 = scanner.scan(conn, force=True)
            assert res2["scanned_files"] == 1

            # 4. Check fact is now marked stale (is_stale = 1)
            f_row_updated = conn.execute(
                "SELECT is_stale FROM facts WHERE id = ?;", (fact_id,)
            ).fetchone()
            assert f_row_updated["is_stale"] == 1

            # 5. Check debt ledger has drift entry
            debt_row = conn.execute(
                "SELECT trigger, ceiling, code_context FROM debt_ledger WHERE trigger = 'ast_drift';"
            ).fetchone()
            assert debt_row is not None
            assert debt_row["ceiling"] == "high"
            assert fact_id in debt_row["code_context"]
            assert "finance.tax.calculate_tax" in debt_row["code_context"]

            # 6. Check TierManager detect_debt includes this stale fact
            tier_mgr = TierManager()
            items = tier_mgr.detect_debt(conn)
            stale_items = [i for i in items if i.trigger == "stale_fact_code_drift"]
            assert len(stale_items) >= 1
            assert any(fact_id in i.code_context for i in stale_items)

    def test_entity_removal_marks_fact_stale(self, tmp_path: Path) -> None:
        """Removing a function from a file marks linked facts as stale and records debt."""
        src_dir = tmp_path / "src" / "api"
        src_dir.mkdir(parents=True)
        py_file = src_dir / "endpoints.py"
        py_file.write_text(
            """
def legacy_endpoint():
    pass

def active_endpoint():
    pass
""",
            encoding="utf-8",
        )

        db = Database(":memory:")
        scanner = ProjectScanner(root_path=tmp_path)
        now = time.time()

        with db.session() as conn:
            scanner.scan(conn)

            legacy_ent = conn.execute(
                "SELECT id, properties_json FROM entities WHERE name = 'api.endpoints.legacy_endpoint';"
            ).fetchone()
            assert legacy_ent is not None
            legacy_id = str(legacy_ent["id"])
            legacy_hash = json.loads(legacy_ent["properties_json"])["hash"]

            fact_id = "fact_endpoint"
            conn.execute(
                """
                INSERT INTO facts (
                    id, text, category, salience, access_count, tier, last_accessed_at,
                    valid_start, valid_end, ingest_start, ingest_end, is_stale
                ) VALUES (?, 'Legacy endpoint is exposed on /legacy', 'api', 1.0, 0, 'working', ?, ?, NULL, ?, NULL, 0);
                """,
                (fact_id, now, now, now),
            )

            conn.execute(
                """
                INSERT INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (fact_id, legacy_id, legacy_hash, str(now)),
            )

            # Update file removing legacy_endpoint
            py_file.write_text(
                """
def active_endpoint():
    pass
""",
                encoding="utf-8",
            )

            scanner.scan(conn, force=True)

            # Legacy entity soft deleted
            ent_del = conn.execute(
                "SELECT valid_end FROM entities WHERE id = ?;", (legacy_id,)
            ).fetchone()
            assert ent_del["valid_end"] is not None

            # Fact marked stale
            fact_del = conn.execute(
                "SELECT is_stale FROM facts WHERE id = ?;", (fact_id,)
            ).fetchone()
            assert fact_del["is_stale"] == 1

            # Debt ledger has critical deletion entry
            debt_del = conn.execute(
                "SELECT trigger, ceiling FROM debt_ledger WHERE trigger = 'entity_deleted';"
            ).fetchone()
            assert debt_del is not None
            assert debt_del["ceiling"] == "critical"

    def test_graph_retrieval_via_links_with_rrf(self) -> None:
        """GraphStore.find_related_facts traverses AST neighbours via fact_entity_links and penalizes stale facts."""
        db = Database(":memory:")
        now = time.time()

        with db.session() as conn:
            # 1. Entities: Module -> ServiceA -> ServiceB
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES ('e_mod', 'core', 'module', ?, ?, ?);",
                (now, now, now),
            )
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES ('e_svc_a', 'core.ServiceA', 'class', ?, ?, ?);",
                (now, now, now),
            )
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES ('e_svc_b', 'core.ServiceB', 'class', ?, ?, ?);",
                (now, now, now),
            )

            # Relations: core.ServiceA -> core.ServiceB
            conn.execute(
                "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) VALUES ('r_ab', 'e_svc_a', 'e_svc_b', 'DEPENDS_ON', 1.0, ?, ?);",
                (now, now),
            )

            # 2. Facts
            # f_a linked to ServiceA (depth 0, active)
            # f_b linked to ServiceB (depth 1, active)
            # f_stale linked to ServiceB (depth 1, stale)
            embedder = create_embedder()
            for fid, txt, stale in [
                ("f_a", "ServiceA manages pipeline routing", 0),
                ("f_b", "ServiceB manages data caching", 0),
                ("f_stale", "ServiceB uses deprecated v1 protocol", 1),
            ]:
                emb = embedder.embed([txt])[0]
                conn.execute(
                    """
                    INSERT INTO facts (
                        id, text, category, salience, access_count, tier, last_accessed_at,
                        valid_start, valid_end, ingest_start, ingest_end, embedding_blob, is_stale
                    ) VALUES (?, ?, 'general', 1.0, 0, 'working', ?, ?, NULL, ?, NULL, ?, ?);
                    """,
                    (fid, txt, now, now, now, _vec_to_blob(emb), stale),
                )

            # Link facts
            conn.execute(
                "INSERT INTO fact_entity_links VALUES ('f_a', 'e_svc_a', 'hash_a', ?);",
                (str(now),),
            )
            conn.execute(
                "INSERT INTO fact_entity_links VALUES ('f_b', 'e_svc_b', 'hash_b', ?);",
                (str(now),),
            )
            conn.execute(
                "INSERT INTO fact_entity_links VALUES ('f_stale', 'e_svc_b', 'hash_b_old', ?);",
                (str(now),),
            )

            graph_store = GraphStore()
            hits = graph_store.find_related_facts("core.ServiceA", conn, max_depth=2)

            # Verify hits:
            # f_a: depth 0 -> score 1.0
            # f_b: depth 1 -> score 0.5
            # f_stale: depth 1 -> score 0.5 * 0.5 = 0.25 (stale penalty)
            assert len(hits) == 3
            hit_map = dict(hits)
            assert hit_map["f_a"] == 1.0
            assert hit_map["f_b"] == 0.5
            assert hit_map["f_stale"] == 0.25

            # Verify full HybridRetriever with RRF
            vs = VectorStore(embedder)
            retriever = HybridRetriever(
                vector_store=vs,
                fts_store=FTSStore(),
                graph_store=graph_store,
            )

            search_results = retriever.search("ServiceA", conn, limit=5)
            assert len(search_results) >= 2

            # Check that stale fact has warning in metadata
            stale_hit = next((r for r in search_results if r.fact.id == "f_stale"), None)
            if stale_hit:
                assert stale_hit.fact.metadata.get("warning") == "stale_code_drift"
                assert stale_hit.fact.is_stale is True
