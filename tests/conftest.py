"""Pytest fixtures and test configuration for Mnemo."""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mnemo.engine.audn import AUDNClassifier
from mnemo.engine.retriever import HybridRetriever
from mnemo.engine.tier_manager import TierManager
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, _vec_to_blob, create_embedder


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def in_memory_db() -> Generator[Database, None, None]:
    """Fixture providing a fresh in-memory SQLite database initialized with schema."""
    db = Database(db_path=":memory:")
    yield db
    db.close()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary filesystem SQLite database path."""
    return tmp_path / "test_memory.db"


@pytest.fixture
def vector_store() -> VectorStore:
    """Fixture providing a VectorStore instance using deterministic embedder."""
    return VectorStore(create_embedder())


@pytest.fixture
def fts_store() -> FTSStore:
    """Fixture providing an FTSStore instance."""
    return FTSStore()


@pytest.fixture
def graph_store() -> GraphStore:
    """Fixture providing a GraphStore instance."""
    return GraphStore()


@pytest.fixture
def audn_classifier(vector_store: VectorStore) -> AUDNClassifier:
    """Fixture providing an AUDNClassifier instance."""
    return AUDNClassifier(vector_store)


@pytest.fixture
def tier_manager() -> TierManager:
    """Fixture providing a TierManager instance."""
    return TierManager(lambda_param=0.01, gamma=0.2)


@pytest.fixture
def hybrid_retriever(
    vector_store: VectorStore,
    fts_store: FTSStore,
    graph_store: GraphStore,
) -> HybridRetriever:
    """Fixture providing a configured HybridRetriever instance."""
    return HybridRetriever(
        vector_store=vector_store,
        fts_store=fts_store,
        graph_store=graph_store,
        rrf_k=60,
    )


@pytest.fixture
def populated_db(
    in_memory_db: Database,
    vector_store: VectorStore,
) -> Database:
    """Fixture providing a populated in-memory database with sample facts, entities, and relations."""
    now = time.time()
    sample_facts = [
        (
            "f1",
            "Mnemo uses SQLite with WAL mode for local storage concurrency",
            "database",
            "core",
            1.0,
        ),
        (
            "f2",
            "Reciprocal Rank Fusion combines vector, FTS5 and graph scores with k=60",
            "algorithms",
            "core",
            0.95,
        ),
        (
            "f3",
            "TOON format compresses JSON objects by 40-60 percent for token savings",
            "serialization",
            "working",
            0.85,
        ),
        (
            "f4",
            "Ebbinghaus forgetting curve models long-term memory salience decay",
            "psychology",
            "working",
            0.75,
        ),
        (
            "f5",
            "Archived memories have salience below 0.40 threshold",
            "general",
            "peripheral",
            0.45,
        ),
    ]

    with in_memory_db.session() as conn:
        for fid, text, cat, tier, salience in sample_facts:
            emb = vector_store.embed_text(text)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, "
                "valid_start, valid_end, ingest_start, ingest_end, embedding_blob, metadata_json) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, NULL, ?, NULL, ?, '{}')",
                (fid, text, cat, salience, tier, now, now, now, _vec_to_blob(emb)),
            )

        # Entities
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, salience, access_count, last_accessed_at, valid_start, ingest_start) "
            "VALUES (?, ?, ?, 1.0, 0, ?, ?, ?)",
            ("e_mnemo", "Mnemo", "system", now, now, now),
        )
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, salience, access_count, last_accessed_at, valid_start, ingest_start) "
            "VALUES (?, ?, ?, 1.0, 0, ?, ?, ?)",
            ("e_sqlite", "SQLite", "technology", now, now, now),
        )
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, salience, access_count, last_accessed_at, valid_start, ingest_start) "
            "VALUES (?, ?, ?, 1.0, 0, ?, ?, ?)",
            ("e_fts5", "FTS5", "technology", now, now, now),
        )

        # Relations: Mnemo -> SQLite -> FTS5
        conn.execute(
            "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) "
            "VALUES (?, ?, ?, 'uses_storage', 1.0, ?, ?)",
            ("r1", "e_mnemo", "e_sqlite", now, now),
        )
        conn.execute(
            "INSERT INTO relations (id, source_id, target_id, relation_type, weight, valid_start, ingest_start) "
            "VALUES (?, ?, ?, 'includes_module', 0.9, ?, ?)",
            ("r2", "e_sqlite", "e_fts5", now, now),
        )

    return in_memory_db
