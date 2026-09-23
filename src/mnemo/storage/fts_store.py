"""FTS5 full-text search with BM25 ranking and bitemporal filtering."""

from __future__ import annotations

import re
import sqlite3
import time


def split_code_tokens(text: str) -> list[str]:
    """Extract sub-tokens from code identifiers (CamelCase, snake_case, qualified paths).

    Examples:
        UserService.login_user_by_id -> ['UserService', 'User', 'Service', 'login_user_by_id', 'login', 'user', 'by', 'id']
    """
    raw_tokens = re.findall(r"[A-Za-z0-9_.]+", text)
    subtokens: set[str] = set()

    for token in raw_tokens:
        # Split by dot for qualified paths (e.g. module.Class.method)
        parts = token.split(".")
        for p in parts:
            if not p:
                continue
            subtokens.add(p)
            # Split snake_case
            if "_" in p:
                for sp in p.split("_"):
                    if sp:
                        subtokens.add(sp)
            # Split CamelCase / PascalCase: e.g. UserService -> User, Service; ASTNode -> AST, Node
            camel_parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", p)
            for cp in camel_parts:
                if cp and len(cp) > 1:
                    subtokens.add(cp)

    return sorted(subtokens)


def extract_search_tokens(text: str) -> str:
    """Generate space-separated code search subtokens for FTS indexing."""
    return " ".join(split_code_tokens(text))


def _sanitize_fts_query(query: str) -> str:
    """Sanitize user input for the FTS5 query parser.

    Extracts valid alphanumeric/code words, strips raw quotes and special
    characters to prevent syntax errors and injection, filters out logical
    operators (AND, OR, NOT, NEAR), and wraps each token in double quotes.
    Preserves whole quoted phrases when safely formatted.
    """
    if not query:
        return '""'

    trimmed = query.strip()
    if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) > 2:
        inner = trimmed[1:-1].replace('"', "").strip()
        if inner:
            return f'"{inner}"'

    # Extract valid tokens (letters, digits, underscore, dot, hyphen)
    raw_tokens = re.findall(r"[\w.-]+", query)
    tokens: list[str] = []
    for t in raw_tokens:
        clean = t.strip(".-")
        if not clean:
            continue
        if clean.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        safe_token = clean.replace('"', "")
        if safe_token:
            tokens.append(f'"{safe_token}"')

    if not tokens:
        return '""'
    return " ".join(tokens)


class FTSStore:
    """BM25 full-text search over the ``facts_fts`` virtual table."""

    def search(
        self,
        query: str,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
        valid_at: float | None = None,
        as_of: str | float | None = None,
    ) -> list[tuple[str, float]]:
        """Run a BM25 full-text search with bitemporal validity filter.

        Args:
            query: Natural-language or keyword query.
            conn: Active SQLite connection.
            limit: Maximum results.
            valid_at: Deprecated point-in-time for the validity window (epoch seconds).
            as_of: Point-in-time for bitemporal point-in-time search (RFC3339 or epoch).

        Returns:
            ``[(fact_id, bm25_score), ...]`` ordered best-first.
        """
        from mnemo.storage.state import parse_as_of

        target_time = parse_as_of(as_of) if as_of is not None else (valid_at if valid_at is not None else None)
        sanitized = _sanitize_fts_query(query)
        if sanitized == '""':
            return []

        if target_time is not None:
            sql = """
                SELECT facts_fts.id, -bm25(facts_fts) AS rank_score
                FROM facts_fts
                JOIN facts f ON f.id = facts_fts.id
                WHERE facts_fts MATCH ?
                  AND f.valid_start  <= ?
                  AND (f.valid_end   IS NULL OR f.valid_end  > ?)
                  AND f.ingest_start <= ?
                  AND (f.ingest_end   IS NULL OR f.ingest_end > ?)
                ORDER BY rank_score DESC
                LIMIT ?;
            """
            try:
                rows = conn.execute(
                    sql, (sanitized, target_time, target_time, target_time, target_time, limit)
                ).fetchall()
                if rows:
                    return [(str(r["id"]), float(r["rank_score"])) for r in rows]
                return self._fallback_like(query, conn, limit=limit, valid_at=target_time)
            except sqlite3.OperationalError:
                return self._fallback_like(query, conn, limit=limit, valid_at=target_time)
        else:
            now = time.time()
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
        as_of: str | float | None = None,
    ) -> list[tuple[str, float]]:
        from mnemo.storage.state import parse_as_of

        target_time = parse_as_of(as_of) if as_of is not None else (valid_at if valid_at is not None else None)
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        if target_time is not None:
            sql = r"""
                SELECT id, salience FROM facts
                WHERE (text LIKE ? ESCAPE '\' OR category LIKE ? ESCAPE '\')
                  AND valid_start  <= ?
                  AND (valid_end   IS NULL OR valid_end  > ?)
                  AND ingest_start <= ?
                  AND (ingest_end   IS NULL OR ingest_end > ?)
                ORDER BY salience DESC
                LIMIT ?;
            """
            rows = conn.execute(
                sql, (pattern, pattern, target_time, target_time, target_time, target_time, limit)
            ).fetchall()
        else:
            now = time.time()
            sql = r"""
                SELECT id, salience FROM facts
                WHERE (text LIKE ? ESCAPE '\' OR category LIKE ? ESCAPE '\')
                  AND valid_start  <= ?
                  AND (valid_end   IS NULL OR valid_end  > ?)
                  AND ingest_end   IS NULL
                ORDER BY salience DESC
                LIMIT ?;
            """
            rows = conn.execute(sql, (pattern, pattern, now, now, limit)).fetchall()

        return [(str(r["id"]), float(r["salience"])) for r in rows]
