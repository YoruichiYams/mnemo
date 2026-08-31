"""Knowledge-graph traversal via recursive CTEs.

Provides k-hop neighbourhood extraction around a seed entity,
with bitemporal validity filtering on both relations and entities.
"""

from __future__ import annotations

import sqlite3
import time


class GraphStore:
    """Recursive CTE graph traversal over ``entities`` and ``relations``."""

    def get_neighbours(
        self,
        entity_id: str,
        conn: sqlite3.Connection,
        *,
        max_depth: int = 2,
        limit: int = 50,
        valid_at: float | None = None,
    ) -> list[tuple[str, int, str, float]]:
        """Extract the k-hop neighbourhood of an entity.

        Uses a recursive CTE to walk outgoing edges up to ``max_depth``
        hops, filtering by bitemporal validity at ``valid_at``.

        Args:
            entity_id: Seed entity ID.
            conn: Active SQLite connection.
            max_depth: Maximum number of hops (default 2).
            limit: Maximum rows returned.
            valid_at: Point-in-time for the validity window (epoch seconds).

        Returns:
            ``[(entity_id, depth, relation_type, weight), ...]``
            ordered by depth ascending, weight descending.
        """
        now = valid_at if valid_at is not None else time.time()

        sql = """
            WITH RECURSIVE hop(eid, depth, rel_type, weight, path) AS (
                -- Base case: seed entity
                SELECT
                    ?,         -- eid
                    0,         -- depth
                    '',        -- rel_type (none for seed)
                    1.0,       -- weight
                    ?          -- path (just seed id)

                UNION ALL

                -- Recursive step: follow outgoing relations
                SELECT
                    r.target_id,
                    hop.depth + 1,
                    r.relation_type,
                    r.weight,
                    hop.path || '>' || r.target_id
                FROM hop
                JOIN relations r ON r.source_id = hop.eid
                WHERE hop.depth < ?
                  -- Relation bitemporal validity
                  AND r.valid_start  <= ?
                  AND (r.valid_end   IS NULL OR r.valid_end  > ?)
                  AND r.ingest_end   IS NULL
                  -- Cycle prevention
                  AND instr(hop.path, r.target_id) = 0
            )
            SELECT DISTINCT
                hop.eid,
                hop.depth,
                hop.rel_type,
                hop.weight
            FROM hop
            JOIN entities e ON e.id = hop.eid
            WHERE e.valid_start  <= ?
              AND (e.valid_end   IS NULL OR e.valid_end  > ?)
              AND e.ingest_end   IS NULL
              AND hop.depth > 0           -- exclude the seed itself
            ORDER BY hop.depth ASC, hop.weight DESC
            LIMIT ?;
        """

        rows = conn.execute(
            sql,
            (entity_id, entity_id, max_depth, now, now, now, now, limit),
        ).fetchall()

        return [
            (str(r["eid"]), int(r["depth"]), str(r["rel_type"]), float(r["weight"])) for r in rows
        ]

    def find_related_facts(
        self,
        entity_name: str,
        conn: sqlite3.Connection,
        *,
        max_depth: int = 2,
        limit: int = 50,
        valid_at: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find facts whose text mentions entities in the k-hop neighbourhood.

        This is used as the *graph channel* in the hybrid retriever:
        facts are scored by ``1 / depth`` of the nearest related entity.

        Args:
            entity_name: Human-readable name of the seed entity.
            conn: Active SQLite connection.
            max_depth: Maximum hops.
            limit: Maximum results.
            valid_at: Point-in-time (epoch seconds).

        Returns:
            ``[(fact_id, graph_score), ...]`` ordered by score descending.
        """
        now = valid_at if valid_at is not None else time.time()

        # Step 1: resolve name -> entity id
        row = conn.execute(
            "SELECT id FROM entities WHERE name = ? "
            "AND valid_start <= ? AND (valid_end IS NULL OR valid_end > ?) "
            "AND ingest_end IS NULL LIMIT 1",
            (entity_name, now, now),
        ).fetchone()
        if row is None:
            return []

        seed_id = str(row["id"])

        # Step 2: get k-hop neighbour entity names
        neighbours = self.get_neighbours(
            seed_id, conn, max_depth=max_depth, limit=200, valid_at=now
        )
        if not neighbours:
            return []

        # Build name -> best_score mapping  (score = 1/depth)
        eid_to_depth: dict[str, int] = {}
        for eid, depth, _rt, _w in neighbours:
            if eid not in eid_to_depth or depth < eid_to_depth[eid]:
                eid_to_depth[eid] = depth

        # Fetch entity names
        placeholders = ",".join("?" for _ in eid_to_depth)
        ent_rows = conn.execute(
            f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
            list(eid_to_depth.keys()),
        ).fetchall()

        name_scores: dict[str, float] = {}
        for er in ent_rows:
            d = eid_to_depth.get(str(er["id"]), 1)
            name_scores[str(er["name"]).lower()] = 1.0 / max(1, d)

        # Step 3: scan active facts for mentions of neighbour names
        facts_sql = """
            SELECT id, text FROM facts
            WHERE valid_start <= ? AND (valid_end IS NULL OR valid_end > ?)
              AND ingest_end IS NULL;
        """
        fact_rows = conn.execute(facts_sql, (now, now)).fetchall()

        scored: list[tuple[str, float]] = []
        for fr in fact_rows:
            text_lower = str(fr["text"]).lower()
            best = 0.0
            for name, sc in name_scores.items():
                if name in text_lower:
                    best = max(best, sc)
            if best > 0.0:
                scored.append((str(fr["id"]), round(best, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
