"""FTS5 full-text search with BM25 ranking and bitemporal filtering."""

from __future__ import annotations

import re
import sqlite3
import time


def _sanitize_fts_query(query: str) -> str:
    """Sanitize user input for the FTS5 query parser.

    Strips characters that cause FTS5 syntax errors while preserving
    words, quoted phrases, and prefix wildcards.
    """
    cleaned = re.sub(r'[^\w\s"*-]', " ", query).strip()
    if not cleaned:
        return '""'
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
        return cleaned
    words = cleaned.split()
    tokens: list[str] = []
    for w in words:
        w = re.sub(r"[^\w-]", "", w)
        if w and w.upper() not in {"AND", "OR", "NOT"}:
            tokens.append(f'"{w}"')
        elif w:
            tokens.append(w)
    return " ".join(tokens) if tokens else '""'


class FTSStore:
    """BM25 full-text search over the ``facts_fts`` virtual table."""

    def search(
        self,
        query: str,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
        valid_at: float | None = None,
    ) -> list[tuple[str, float]]:
        """Run a BM25 full-text search with bitemporal validity filter.

        Args:
            query: Natural-language or keyword query.
            conn: Active SQLite connection.
            limit: Maximum results.
            valid_at: Point-in-time for the validity window (epoch seconds).
                      Defaults to *now*.

        Returns:
            ``[(fact_id, bm25_score), ...]`` ordered best-first.
        """
        now = valid_at if valid_at is not None else time.time()
        sanitized = _sanitize_fts_query(query)
        if sanitized == '""':
            return []

        sql = """
            SELECT facts_fts.id, -bm25(facts_fts) AS rank_score
            FROM facts_fts
            JOIN facts f ON f.id = facts_fts.id
            WHERE facts_fts MATCH ?
              AND f.valid_start  <= ?
              AND (f.valid_end   IS NULL OR f.valid_end  > ?)
              AND f.ingest_end   IS NULL
            ORDER BY rank_score DESC
            LIMIT ?;
        """
        try:
            rows = conn.execute(sql, (sanitized, now, now, limit)).fetchall()
            return [(str(r["id"]), float(r["rank_score"])) for r in rows]
        except sqlite3.OperationalError:
            return self._fallback_like(query, conn, limit=limit, valid_at=now)

    # ------------------------------------------------------------------
    # Fallback: plain LIKE when FTS syntax fails
    # ------------------------------------------------------------------
    def _fallback_like(
        self,
        query: str,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
        valid_at: float | None = None,
    ) -> list[tuple[str, float]]:
        now = valid_at if valid_at is not None else time.time()
        pattern = f"%{query}%"
        sql = """
            SELECT id, salience FROM facts
            WHERE (text LIKE ? OR category LIKE ?)
              AND valid_start  <= ?
              AND (valid_end   IS NULL OR valid_end  > ?)
              AND ingest_end   IS NULL
            ORDER BY salience DESC
            LIMIT ?;
        """
        rows = conn.execute(sql, (pattern, pattern, now, now, limit)).fetchall()
        return [(str(r["id"]), float(r["salience"])) for r in rows]
