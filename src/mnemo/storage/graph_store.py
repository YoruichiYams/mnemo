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
        as_of: str | float | None = None,
    ) -> list[tuple[str, int, str, float]]:
        """Extract the k-hop neighbourhood of an entity.

        Uses a recursive CTE to walk outgoing edges up to ``max_depth``
        hops, filtering by bitemporal validity at ``as_of`` or ``valid_at``.

        Args:
            entity_id: Seed entity ID.
            conn: Active SQLite connection.
            max_depth: Maximum number of hops (default 2).
            limit: Maximum rows returned.
            valid_at: Deprecated point-in-time for the validity window (epoch seconds).
            as_of: Point-in-time for bitemporal search (RFC3339 or epoch).

        Returns:
            ``[(entity_id, depth, relation_type, weight), ...]``
            ordered by depth ascending, weight descending.
        """
        max_depth = min(max(1, int(max_depth)), 4)
        from mnemo.storage.state import parse_as_of

        target_time = parse_as_of(as_of) if as_of is not None else (valid_at if valid_at is not None else None)

        if target_time is not None:
            sql = """
                WITH RECURSIVE hop(eid, depth, rel_type, weight, path) AS (
                    -- Base case: seed entity
                    SELECT
                        ?,                    -- eid
                        0,                    -- depth
                        '',                   -- rel_type (none for seed)
                        1.0,                  -- weight
                        '>' || ? || '>'       -- path with bounding delimiters

                    UNION ALL

                    -- Recursive step: follow outgoing relations
                    SELECT
                        r.target_id,
                        hop.depth + 1,
                        r.relation_type,
                        r.weight,
                        hop.path || r.target_id || '>'
                    FROM hop
                    JOIN relations r ON r.source_id = hop.eid
                    WHERE hop.depth < ?
                      -- Relation bitemporal validity
                      AND r.valid_start  <= ?
                      AND (r.valid_end   IS NULL OR r.valid_end  > ?)
                      AND r.ingest_start <= ?
                      AND (r.ingest_end   IS NULL OR r.ingest_end > ?)
                      -- Cycle prevention: exact match with delimiters
                      AND instr(hop.path, '>' || r.target_id || '>') = 0
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
                  AND e.ingest_start <= ?
                  AND (e.ingest_end   IS NULL OR e.ingest_end > ?)
                  AND hop.depth > 0           -- exclude the seed itself
                ORDER BY hop.depth ASC, hop.weight DESC
                LIMIT ?;
            """
            rows = conn.execute(
                sql,
                (
                    entity_id,
                    entity_id,
                    max_depth,
                    target_time,
                    target_time,
                    target_time,
                    target_time,
                    target_time,
                    target_time,
                    target_time,
                    target_time,
                    limit,
                ),
            ).fetchall()
        else:
            now = time.time()
            sql = """
                WITH RECURSIVE hop(eid, depth, rel_type, weight, path) AS (
                    -- Base case: seed entity
                    SELECT
                        ?,                    -- eid
                        0,                    -- depth
                        '',                   -- rel_type (none for seed)
                        1.0,                  -- weight
                        '>' || ? || '>'       -- path with bounding delimiters

                    UNION ALL

                    -- Recursive step: follow outgoing relations
                    SELECT
                        r.target_id,
                        hop.depth + 1,
                        r.relation_type,
                        r.weight,
                        hop.path || r.target_id || '>'
                    FROM hop
                    JOIN relations r ON r.source_id = hop.eid
                    WHERE hop.depth < ?
                      -- Relation bitemporal validity
                      AND r.valid_start  <= ?
                      AND (r.valid_end   IS NULL OR r.valid_end  > ?)
                      AND r.ingest_end   IS NULL
                      -- Cycle prevention: exact match with delimiters
                      AND instr(hop.path, '>' || r.target_id || '>') = 0
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
        as_of: str | float | None = None,
    ) -> list[tuple[str, float]]:
        """Find facts linked to entities in the k-hop neighbourhood via fact_entity_links.

        This is used as the *graph channel* in the hybrid retriever:
        facts are scored by ``1 / (depth + 1)`` of the nearest related entity.
        Facts marked with ``is_stale = 1`` receive a 0.5x score penalty.

        Args:
            entity_name: Human-readable name or qualified path of the seed entity.
            conn: Active SQLite connection.
            max_depth: Maximum hops.
            limit: Maximum results.
            valid_at: Deprecated point-in-time (epoch seconds).
            as_of: Point-in-time for bitemporal search (RFC3339 or epoch).

        Returns:
            ``[(fact_id, graph_score), ...]`` ordered by score descending.
        """
        from mnemo.storage.state import parse_as_of

        target_time = parse_as_of(as_of) if as_of is not None else (valid_at if valid_at is not None else None)
        escaped_name = entity_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        # Step 1: resolve name -> entity id (exact match or suffix match with escaping)
        if target_time is not None:
            row = conn.execute(
                """
                SELECT id FROM entities WHERE name = ?
                  AND valid_start <= ? AND (valid_end IS NULL OR valid_end > ?)
                  AND ingest_start <= ? AND (ingest_end IS NULL OR ingest_end > ?) LIMIT 1;
                """,
                (entity_name, target_time, target_time, target_time, target_time),
            ).fetchone()

            if row is None:
                row = conn.execute(
                    r"""
                    SELECT id FROM entities
                    WHERE (name = ? OR name LIKE ? ESCAPE '\')
                      AND valid_start <= ? AND (valid_end IS NULL OR valid_end > ?)
                      AND ingest_start <= ? AND (ingest_end IS NULL OR ingest_end > ?)
                    ORDER BY length(name) DESC LIMIT 1;
                    """,
                    (entity_name, f"%.{escaped_name}", target_time, target_time, target_time, target_time),
                ).fetchone()
        else:
            now = time.time()
            row = conn.execute(
                """
                SELECT id FROM entities WHERE name = ?
                  AND valid_start <= ? AND (valid_end IS NULL OR valid_end > ?)
                  AND ingest_end IS NULL LIMIT 1;
                """,
                (entity_name, now, now),
            ).fetchone()

            if row is None:
                row = conn.execute(
                    r"""
                    SELECT id FROM entities
                    WHERE (name = ? OR name LIKE ? ESCAPE '\')
                      AND valid_start <= ? AND (valid_end IS NULL OR valid_end > ?)
                      AND ingest_end IS NULL
                    ORDER BY length(name) DESC LIMIT 1;
                    """,
                    (entity_name, f"%.{escaped_name}", now, now),
                ).fetchone()

        if row is None:
            return []

        seed_id = str(row["id"])

        # Step 2: get k-hop neighbourhood around seed entity
        neighbours = self.get_neighbours(
            seed_id, conn, max_depth=max_depth, limit=200, valid_at=valid_at, as_of=as_of
        )

        # Build entity_id -> depth map (seed entity has depth 0)
        eid_to_depth: dict[str, int] = {seed_id: 0}
        for eid, depth, _rt, _w in neighbours:
            if eid not in eid_to_depth or depth < eid_to_depth[eid]:
                eid_to_depth[eid] = depth

        # Step 3: query fact_entity_links directly for associated facts
        placeholders = ",".join("?" for _ in eid_to_depth)
        if target_time is not None:
            links_sql = f"""
                SELECT fel.fact_id, fel.entity_id, f.is_stale
                FROM fact_entity_links fel
                JOIN facts f ON f.id = fel.fact_id
                WHERE fel.entity_id IN ({placeholders})
                  AND f.valid_start <= ? AND (f.valid_end IS NULL OR f.valid_end > ?)
                  AND f.ingest_start <= ? AND (f.ingest_end IS NULL OR f.ingest_end > ?);
            """
            params = list(eid_to_depth.keys()) + [target_time, target_time, target_time, target_time]
        else:
            now = time.time()
            links_sql = f"""
                SELECT fel.fact_id, fel.entity_id, f.is_stale
                FROM fact_entity_links fel
                JOIN facts f ON f.id = fel.fact_id
                WHERE fel.entity_id IN ({placeholders})
                  AND f.valid_start <= ? AND (f.valid_end IS NULL OR f.valid_end > ?)
                  AND f.ingest_end IS NULL;
            """
            params = list(eid_to_depth.keys()) + [now, now]

        link_rows = conn.execute(links_sql, params).fetchall()

        fact_scores: dict[str, float] = {}
        for lr in link_rows:
            fid = str(lr["fact_id"])
            eid = str(lr["entity_id"])
            is_stale = int(lr["is_stale"]) if "is_stale" in lr.keys() and lr["is_stale"] else 0

            depth = eid_to_depth.get(eid, 0)
            score = 1.0 / (depth + 1)

            # Penalize stale facts by 0.5x
            if is_stale:
                score *= 0.5

            if fid not in fact_scores or score > fact_scores[fid]:
                fact_scores[fid] = score

        if fact_scores:
            scored = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
            return [(fid, round(sc, 6)) for fid, sc in scored[:limit]]

        # Step 4: Fallback for unlinked legacy facts via FTS5 with LIMIT 50
        ent_rows = conn.execute(
            f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
            list(eid_to_depth.keys()),
        ).fetchall()

        name_scores: dict[str, float] = {}
        fts_tokens: list[str] = []
        for er in ent_rows:
            d = eid_to_depth.get(str(er["id"]), 1)
            raw_name = str(er["name"]).strip()
            if raw_name:
                name_scores[raw_name.lower()] = 1.0 / (d + 1)
                safe_t = raw_name.replace('"', "").replace("'", "")
                if safe_t:
                    fts_tokens.append(f'"{safe_t}"')

        if not fts_tokens:
            return []

        fts_query = " OR ".join(fts_tokens[:10])

        if target_time is not None:
            facts_sql = """
                SELECT f.id, f.text, f.is_stale
                FROM facts_fts
                JOIN facts f ON f.id = facts_fts.id
                WHERE facts_fts MATCH ?
                  AND f.valid_start <= ? AND (f.valid_end IS NULL OR f.valid_end > ?)
                  AND f.ingest_start <= ? AND (f.ingest_end IS NULL OR f.ingest_end > ?)
                LIMIT 50;
            """
            try:
                fact_rows = conn.execute(
                    facts_sql, (fts_query, target_time, target_time, target_time, target_time)
                ).fetchall()
            except sqlite3.OperationalError:
                fact_rows = []
        else:
            now = time.time()
            facts_sql = """
                SELECT f.id, f.text, f.is_stale
                FROM facts_fts
                JOIN facts f ON f.id = facts_fts.id
                WHERE facts_fts MATCH ?
                  AND f.valid_start <= ? AND (f.valid_end IS NULL OR f.valid_end > ?)
                  AND f.ingest_end IS NULL
                LIMIT 50;
            """
            try:
                fact_rows = conn.execute(facts_sql, (fts_query, now, now)).fetchall()
            except sqlite3.OperationalError:
                fact_rows = []

        scored = []
        for fr in fact_rows:
            text_lower = str(fr["text"]).lower()
            best = 0.0
            for name, sc in name_scores.items():
                if name in text_lower:
                    best = max(best, sc)
            if best > 0.0:
                is_stale = int(fr["is_stale"]) if "is_stale" in fr.keys() and fr["is_stale"] else 0
                if is_stale:
                    best *= 0.5
                scored.append((str(fr["id"]), round(best, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
