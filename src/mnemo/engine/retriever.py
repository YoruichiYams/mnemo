"""Hybrid 3-channel retriever with RRF fusion.

Channels:
  1. **Vector** — cosine-similarity over fact embeddings.
  2. **FTS5**   — BM25 full-text search.
  3. **Graph**  — k-hop entity-neighbourhood scoring.

Scores from each channel are normalised to [0, 1] and merged via
Reciprocal Rank Fusion (k=60).  The final list is re-weighted by
current salience.
"""

from __future__ import annotations

import sqlite3
import time

from mnemo.core.models import Fact, MemoryTier, SearchResult
from mnemo.core.rrf import reciprocal_rank_fusion
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore


def _load_fact_by_id(fact_id: str, conn: sqlite3.Connection) -> Fact | None:
    """Load a Fact from the database by primary key."""
    row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    if row is None:
        return None
    return Fact(
        id=str(row["id"]),
        text=str(row["text"]),
        category=str(row["category"]),
        salience=float(row["salience"]),
        access_count=int(row["access_count"]),
        tier=MemoryTier(str(row["tier"])),
        last_accessed_at=float(row["last_accessed_at"]),
        valid_start=float(row["valid_start"]),
        valid_end=row["valid_end"],
        ingest_start=float(row["ingest_start"]),
        ingest_end=row["ingest_end"],
    )


class HybridRetriever:
    """Orchestrates 3-channel search and RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        fts_store: FTSStore | None = None,
        graph_store: GraphStore | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._vec = vector_store
        self._fts = fts_store or FTSStore()
        self._graph = graph_store or GraphStore()
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        conn: sqlite3.Connection,
        *,
        limit: int = 20,
        min_tier: MemoryTier | None = None,
        valid_at: float | None = None,
    ) -> list[SearchResult]:
        """Run hybrid 3-channel search and return ranked results.

        Args:
            query: Natural-language query.
            conn: Active SQLite connection.
            limit: Max results to return.
            min_tier: Minimum tier filter (e.g. ``WORKING`` excludes peripheral/archived).
            valid_at: Point-in-time for bitemporal filter.

        Returns:
            Sorted list of ``SearchResult`` objects.
        """
        now = valid_at if valid_at is not None else time.time()

        # --- Run channels in parallel (sequential here, could be threaded) ---

        vec_hits = self._vec.search(query, conn, limit=limit * 3, valid_at=now)
        fts_hits = self._fts.search(query, conn, limit=limit * 3, valid_at=now)
        graph_hits = self._graph.find_related_facts(query, conn, limit=limit * 3, valid_at=now)

        # Build ranked ID lists (ordered by score desc)
        vec_ranked = [fid for fid, _ in vec_hits]
        fts_ranked = [fid for fid, _ in fts_hits]
        graph_ranked = [fid for fid, _ in graph_hits]

        # --- RRF fusion ---
        ranked_lists: list[list[str]] = []
        if vec_ranked:
            ranked_lists.append(vec_ranked)
        if fts_ranked:
            ranked_lists.append(fts_ranked)
        if graph_ranked:
            ranked_lists.append(graph_ranked)

        if not ranked_lists:
            return []

        rrf_merged = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)

        # --- Build results ---
        results: list[SearchResult] = []
        for fact_id, rrf_score in rrf_merged:
            fact = _load_fact_by_id(fact_id, conn)
            if fact is None:
                continue

            # Apply tier filter
            if min_tier is not None:
                tier_order = {
                    MemoryTier.CORE: 4,
                    MemoryTier.WORKING: 3,
                    MemoryTier.PERIPHERAL: 2,
                    MemoryTier.ARCHIVED: 1,
                }
                if tier_order.get(fact.tier, 0) < tier_order.get(min_tier, 0):
                    continue

            # Salience-weighted final score
            combined = rrf_score * fact.salience

            results.append(
                SearchResult(
                    fact=fact,
                    score=round(combined, 8),
                    channel="rrf",
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
