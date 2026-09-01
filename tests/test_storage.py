"""Unit tests for the storage layer: SQLite, FTS5, VectorStore, and GraphStore."""

from __future__ import annotations

import time

import numpy as np
import pytest

from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore, _sanitize_fts_query
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import (
    VectorStore,
    _blob_to_vec,
    _vec_to_blob,
    cosine_similarity,
)


class TestFTSStore:
    """Test suite for SQLite FTS5 BM25 search and synchronisation triggers."""

    def test_query_sanitizer(self) -> None:
        """Sanitizer handles edge characters safely without crashing FTS parser."""
        assert _sanitize_fts_query("") == '""'
        assert _sanitize_fts_query("hello world") == '"hello" "world"'
        assert _sanitize_fts_query("special !@#$%^&*() characters") == '"special" "characters"'
        # Quoted phrase preserved
        assert _sanitize_fts_query('"exact phrase"') == '"exact phrase"'

    def test_fts5_bm25_search_and_bitemporal_filter(
        self, populated_db: Database, fts_store: FTSStore
    ) -> None:
        """FTS5 search ranks matching facts and respects bitemporal validity."""
        now = time.time()
        with populated_db.session() as conn:
            results = fts_store.search("SQLite concurrency", conn, valid_at=now)
            assert len(results) >= 1
            assert results[0][0] == "f1"  # "Mnemo uses SQLite with WAL mode..."

            # Point in the past before fact ingestion
            past_results = fts_store.search("SQLite concurrency", conn, valid_at=now - 1000.0)
            assert len(past_results) == 0

            # Empty query search
            assert fts_store.search("", conn) == []

    def test_fts5_triggers_soft_delete_synchronization(self, in_memory_db: Database) -> None:
        """FTS5 triggers automatically purge records when valid_end or ingest_end is closed."""
        now = time.time()
        with in_memory_db.session() as conn:
            # 1. Insert active fact
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start) "
                "VALUES ('f_trigger', 'Unique searchable token xyz123', 'test', 1.0, 0, 'working', ?, ?, ?)",
                (now, now, now),
            )
            hits = conn.execute(
                "SELECT id FROM facts_fts WHERE facts_fts MATCH 'xyz123'"
            ).fetchall()
            assert len(hits) == 1
            assert hits[0]["id"] == "f_trigger"

            # 2. Soft-delete (close valid_end)
            conn.execute(
                "UPDATE facts SET valid_end = ? WHERE id = 'f_trigger'",
                (now + 1.0,),
            )
            hits_after_del = conn.execute(
                "SELECT id FROM facts_fts WHERE facts_fts MATCH 'xyz123'"
            ).fetchall()
            assert len(hits_after_del) == 0, (
                "Soft-deleted fact must be removed from facts_fts trigger!"
            )

            # 3. Reactivate (clear valid_end)
            conn.execute(
                "UPDATE facts SET valid_end = NULL WHERE id = 'f_trigger'",
            )
            hits_reactivated = conn.execute(
                "SELECT id FROM facts_fts WHERE facts_fts MATCH 'xyz123'"
            ).fetchall()
            assert len(hits_reactivated) == 1

            # 4. Hard delete
            conn.execute("DELETE FROM facts WHERE id = 'f_trigger'")
            hits_after_hard_del = conn.execute(
                "SELECT id FROM facts_fts WHERE facts_fts MATCH 'xyz123'"
            ).fetchall()
            assert len(hits_after_hard_del) == 0

    def test_fts_fallback_like(self, populated_db: Database, fts_store: FTSStore) -> None:
        """Fallback LIKE search works when direct FTS query is invoked."""
        with populated_db.session() as conn:
            hits = fts_store._fallback_like("SQLite", conn, limit=5)
            assert len(hits) >= 1
            assert hits[0][0] == "f1"


class TestVectorStore:
    """Test suite for VectorStore, embedding generator, and cosine similarity."""

    def test_cosine_similarity_properties(self) -> None:
        """Cosine similarity is 1.0 for identical vectors, 0.0 for orthogonal, -1.0 for opposite."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-5)
        assert cosine_similarity(v1, v3) == pytest.approx(0.0, abs=1e-5)
        assert cosine_similarity(v1, v4) == pytest.approx(-1.0, abs=1e-5)
        # Zero vector handling
        assert cosine_similarity(np.zeros(3, dtype=np.float32), v1) == 0.0

    def test_blob_serialization_roundtrip(self) -> None:
        """Vector blob serialization preserves exact float32 values."""
        vec = np.array([0.1234, -0.5678, 0.9999], dtype=np.float32)
        blob = _vec_to_blob(vec)
        restored = _blob_to_vec(blob)

        np.testing.assert_allclose(vec, restored, rtol=1e-5)

    def test_vector_search_and_store_embedding(
        self, in_memory_db: Database, vector_store: VectorStore
    ) -> None:
        """Vector store generates embeddings, stores blobs, and searches accurately."""
        now = time.time()
        with in_memory_db.session() as conn:
            texts = [
                ("v1", "Artificial intelligence neural networks and deep learning"),
                ("v2", "Baking delicious chocolate chip cookies and pastries"),
                ("v3", "Training machine learning transformer models for NLP"),
            ]
            for fid, txt in texts:
                conn.execute(
                    "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start) "
                    "VALUES (?, ?, 'test', 1.0, 0, 'working', ?, ?, ?)",
                    (fid, txt, now, now, now),
                )
                emb = vector_store.embed_text(txt)
                vector_store.store_embedding(conn, fid, emb)

            # Query semantically closer to ML/neural networks
            hits = vector_store.search("neural network machine learning", conn, limit=3)
            assert len(hits) >= 2
            top_id = hits[0][0]
            # Must match v1 or v3 before v2 (cooking)
            assert top_id in ("v1", "v3")

            # embed_texts batch
            batch_emb = vector_store.embed_texts(["hello", "world"])
            assert len(batch_emb) == 2


class TestGraphStore:
    """Test suite for GraphStore recursive CTE traversal."""

    def test_recursive_cte_k_hop_neighbourhood(
        self, populated_db: Database, graph_store: GraphStore
    ) -> None:
        """Recursive CTE finds multi-hop connected entities and weights."""
        now = time.time()
        with populated_db.session() as conn:
            # Seed: 'e_mnemo' -> e_sqlite (depth 1) -> e_fts5 (depth 2)
            neighbours = graph_store.get_neighbours("e_mnemo", conn, max_depth=2, valid_at=now)

            assert len(neighbours) == 2
            # 1-hop: e_sqlite
            assert neighbours[0][0] == "e_sqlite"
            assert neighbours[0][1] == 1
            assert neighbours[0][2] == "uses_storage"
            # 2-hop: e_fts5
            assert neighbours[1][0] == "e_fts5"
            assert neighbours[1][1] == 2

    def test_find_related_facts(self, populated_db: Database, graph_store: GraphStore) -> None:
        """Graph store finds facts mentioning connected entities."""
        now = time.time()
        with populated_db.session() as conn:
            # Entity name 'Mnemo' connects to 'SQLite' which is mentioned in fact f1
            hits = graph_store.find_related_facts("Mnemo", conn, max_depth=2, valid_at=now)
            assert len(hits) >= 1
            assert hits[0][0] == "f1"

            # Unknown entity returns empty list
            assert graph_store.find_related_facts("UnknownEntity", conn) == []

    def test_graph_cycle_prevention(self, in_memory_db: Database, graph_store: GraphStore) -> None:
        """Circular graph relationships do not cause infinite recursion."""
        now = time.time()
        with in_memory_db.session() as conn:
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES ('n1', 'Node1', 'concept', ?, ?, ?)",
                (now, now, now),
            )
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES ('n2', 'Node2', 'concept', ?, ?, ?)",
                (now, now, now),
            )

            # Cycle: n1 -> n2 -> n1
            conn.execute(
                "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) VALUES ('r_cyc1', 'n1', 'n2', 'links', 1.0, ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) VALUES ('r_cyc2', 'n2', 'n1', 'links', 1.0, ?, ?)",
                (now, now),
            )

            # Must terminate cleanly
            neighbours = graph_store.get_neighbours("n1", conn, max_depth=5, valid_at=now)
            assert len(neighbours) == 1
            assert neighbours[0][0] == "n2"

    def test_substring_id_cycle_prevention(
        self, in_memory_db: Database, graph_store: GraphStore
    ) -> None:
        """Entities with prefix IDs (e.g. node_1 and node_10) do not trigger false cycle prevention."""
        now = time.time()
        with in_memory_db.session() as conn:
            # node_1 -> node_10 -> node_100
            for nid in ("node_1", "node_10", "node_100"):
                conn.execute(
                    "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) VALUES (?, ?, 'concept', ?, ?, ?)",
                    (nid, f"Name_{nid}", now, now, now),
                )
            conn.execute(
                "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) VALUES ('r1', 'node_1', 'node_10', 'links', 1.0, ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) VALUES ('r2', 'node_10', 'node_100', 'links', 1.0, ?, ?)",
                (now, now),
            )

            neighbours = graph_store.get_neighbours("node_1", conn, max_depth=3, valid_at=now)
            assert len(neighbours) == 2
            eids = [n[0] for n in neighbours]
            assert eids == ["node_10", "node_100"]


class TestDatabaseConnection:
    """Test suite for Database connection manager and schema initialization."""

    def test_init_db_and_session(self, tmp_path: pytest.TempPathFactory) -> None:
        """Database creates file, runs schema, and handles transactions."""
        db_file = str(tmp_path / "conn_test.db")
        db = Database(db_path=db_file)
        assert db.db_path == db_file

        with db.session() as conn:
            conn.execute(
                "INSERT INTO facts (id, text, last_accessed_at, valid_start, ingest_start) VALUES ('t1', 'text', 0, 0, 0)"
            )

        with db.session() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = 't1'").fetchone()
            assert row is not None

        # Rollback on exception
        with pytest.raises(RuntimeError):
            with db.session() as conn:
                conn.execute(
                    "INSERT INTO facts (id, text, last_accessed_at, valid_start, ingest_start) VALUES ('t2', 'text2', 0, 0, 0)"
                )
                raise RuntimeError("Forced error")

        with db.session() as conn:
            row2 = conn.execute("SELECT * FROM facts WHERE id = 't2'").fetchone()
            assert row2 is None

        db.close()
