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

import json
import sqlite3

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

    meta = {}
    if "metadata_json" in row.keys() and row["metadata_json"]:
        try:
            meta = json.loads(row["metadata_json"])
        except Exception:
            meta = {}

    return Fact(
        id=str(row["id"]),
        text=str(row["text"]),
        category=str(row["category"]),
        salience=float(row["salience"]),
        access_count=int(row["access_count"]),
        tier=MemoryTier(str(row["tier"])),
        last_accessed_at=float(row["last_accessed_at"]),
        last_accessed_tick=int(row["last_accessed_tick"])
        if "last_accessed_tick" in row.keys() and row["last_accessed_tick"] is not None
        else 0,
        reinforcement_count=int(row["reinforcement_count"])
        if "reinforcement_count" in row.keys() and row["reinforcement_count"] is not None
        else 0,
        valid_start=float(row["valid_start"]),
        valid_end=row["valid_end"],
        ingest_start=float(row["ingest_start"]),
        ingest_end=row["ingest_end"],
        metadata=meta,
        is_stale=bool(row["is_stale"]) if "is_stale" in row.keys() and row["is_stale"] else False,
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
        as_of: str | float | None = None,
        include_archived: bool = False,
        channels: tuple[str, ...] = ("vector", "fts", "graph"),
    ) -> list[SearchResult]:
        """Run hybrid 3-channel search and return ranked results.

        Search is purely read-only and does NOT reinforce facts to prevent the
        'rich get richer' feedback loop.

        Args:
            query: Natural-language query.
            conn: Active SQLite connection.
            limit: Max results to return.
            min_tier: Minimum tier filter (e.g. ``WORKING`` excludes peripheral/archived).
            valid_at: Deprecated point-in-time for bitemporal filter.
            as_of: Point-in-time for bitemporal search (RFC3339 or epoch).
            include_archived: Whether to fallback to archived tier when higher tiers lack results.
            channels: Tuple of channels to query: any combination of ``"vector"``, ``"fts"``, ``"graph"``.

        Returns:
            Sorted list of ``SearchResult`` objects.
        """
        active_channels = set(channels)

        # --- Run requested channels ---
        vec_hits = (
            self._vec.search(query, conn, limit=limit * 3, valid_at=valid_at, as_of=as_of)
            if "vector" in active_channels
            else []
        )
        fts_hits = (
            self._fts.search(query, conn, limit=limit * 3, valid_at=valid_at, as_of=as_of)
            if "fts" in active_channels
            else []
        )
        graph_hits = (
            self._graph.find_related_facts(
                query, conn, limit=limit * 3, valid_at=valid_at, as_of=as_of
            )
            if "graph" in active_channels
            else []
        )

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

        tier_order = {
            MemoryTier.CORE: 4,
            MemoryTier.WORKING: 3,
            MemoryTier.PERIPHERAL: 2,
            MemoryTier.ARCHIVED: 1,
        }

        # --- Build primary and archived fallback candidates ---
        primary_results: list[SearchResult] = []
        archived_fallback: list[SearchResult] = []

        for fact_id, rrf_score in rrf_merged:
            fact = _load_fact_by_id(fact_id, conn)
            if fact is None:
                continue

            combined = rrf_score * fact.salience
            if fact.is_stale:
                fact.metadata["warning"] = "stale_code_drift"

            is_archived = fact.tier == MemoryTier.ARCHIVED
            if is_archived:
                fact.metadata["archived_warning"] = (
                    "Факт находится в архиве из-за низкой активности"
                )

            ch_val = "fts5" if channels[0] == "fts" else channels[0]
            search_res = SearchResult(
                fact=fact,
                score=round(combined, 8),
                channel=ch_val if len(channels) == 1 else "rrf",
            )

            if min_tier is not None and tier_order.get(fact.tier, 0) < tier_order.get(min_tier, 0):
                if is_archived and include_archived:
                    archived_fallback.append(search_res)
            else:
                primary_results.append(search_res)

        # Sort primary results descending by score
        primary_results.sort(key=lambda r: r.score, reverse=True)

        # If primary results are insufficient and archived fallback is enabled, fill up to limit
        if len(primary_results) < limit and include_archived and archived_fallback:
            archived_fallback.sort(key=lambda r: r.score, reverse=True)
            needed = limit - len(primary_results)
            for res in archived_fallback[:needed]:
                res.fact.metadata.setdefault(
                    "archived_warning", "Факт находится в архиве из-за низкой активности"
                )
                primary_results.append(res)

        return primary_results[:limit]
