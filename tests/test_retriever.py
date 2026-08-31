"""Unit tests for the Hybrid 3-channel retriever and ranking engine."""

from __future__ import annotations

import time

from mnemo.core.models import MemoryTier
from mnemo.engine.retriever import HybridRetriever
from mnemo.storage.connection import Database


class TestHybridRetriever:
    """Test suite for 3-channel retrieval (Vector + FTS5 + Graph) and RRF fusion."""

    def test_hybrid_search_ranks_relevant_facts(
        self,
        populated_db: Database,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        """Retriever merges results across channels and ranks best match highest."""
        with populated_db.session() as conn:
            results = hybrid_retriever.search("SQLite WAL concurrency", conn, limit=3)
            assert len(results) >= 1
            top_hit = results[0]
            # Fact f1: "Mnemo uses SQLite with WAL mode..."
            assert top_hit.fact.id == "f1"
            assert top_hit.score > 0.0
            assert top_hit.channel == "rrf"

    def test_tier_filtering(
        self,
        populated_db: Database,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        """Min_tier filter strictly excludes facts from lower tiers."""
        with populated_db.session() as conn:
            # Query matching all facts, but min_tier = CORE
            core_results = hybrid_retriever.search("Mnemo", conn, min_tier=MemoryTier.CORE)
            for r in core_results:
                assert r.fact.tier == MemoryTier.CORE

            # min_tier = WORKING should include CORE and WORKING, but exclude PERIPHERAL
            working_results = hybrid_retriever.search("Mnemo", conn, min_tier=MemoryTier.WORKING)
            for r in working_results:
                assert r.fact.tier in (MemoryTier.CORE, MemoryTier.WORKING)

    def test_salience_weighted_ranking(
        self,
        in_memory_db: Database,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        """Higher salience facts with equal RRF scores rank higher."""
        now = time.time()
        with in_memory_db.session() as conn:
            from mnemo.storage.vector_store import _vec_to_blob, create_embedder

            embedder = create_embedder()

            txt1 = "Knowledge about graph databases and triples"
            txt2 = "Knowledge about graph databases and edges"
            emb1 = embedder.embed([txt1])[0]
            emb2 = embedder.embed([txt2])[0]

            # High salience (1.0)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start, embedding_blob) "
                "VALUES ('f_high', ?, 'graph', 1.0, 0, 'core', ?, ?, ?, ?)",
                (txt1, now, now, now, _vec_to_blob(emb1)),
            )
            # Decayed salience (0.4)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start, embedding_blob) "
                "VALUES ('f_low', ?, 'graph', 0.4, 0, 'peripheral', ?, ?, ?, ?)",
                (txt2, now, now, now, _vec_to_blob(emb2)),
            )

            results = hybrid_retriever.search("graph databases", conn, limit=2)
            assert len(results) == 2
            assert results[0].fact.id == "f_high"
            assert results[0].score > results[1].score
