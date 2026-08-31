"""FastMCP resource definitions for Mnemo.

Resources provide read-only views into the memory store that LLMs can
request as context.
"""

from __future__ import annotations

import time

from mnemo.core.models import MemoryTier
from mnemo.mcp.server import _get_db, mcp_app
from mnemo.serialization.toon import encode_fact


# -----------------------------------------------------------------------
# memory://core-identity
# -----------------------------------------------------------------------
@mcp_app.resource(
    "memory://core-identity",
    name="core-identity",
    description="All Core-tier (S >= 0.9) facts -- permanent architectural rules and persona.",
)
def core_identity_resource() -> str:
    """Return all Core-tier facts in TOON format."""
    db = _get_db()
    now = time.time()

    with db.session() as conn:
        rows = conn.execute(
            "SELECT * FROM facts "
            "WHERE tier = ? AND ingest_end IS NULL "
            "AND (valid_end IS NULL OR valid_end > ?) "
            "ORDER BY salience DESC",
            (MemoryTier.CORE.value, now),
        ).fetchall()

    if not rows:
        return "[EMPTY] no core-tier facts"

    lines: list[str] = []
    for r in rows:
        d = {
            "id": r["id"],
            "text": r["text"],
            "category": r["category"],
            "salience": r["salience"],
            "access_count": r["access_count"],
            "tier": r["tier"],
        }
        lines.append(encode_fact(d))
    return "\n".join(lines)


# -----------------------------------------------------------------------
# memory://active-context
# -----------------------------------------------------------------------
@mcp_app.resource(
    "memory://active-context",
    name="active-context",
    description="Working-tier (S >= 0.7) facts -- current task context.",
)
def active_context_resource() -> str:
    """Return all Working-tier facts in TOON format."""
    db = _get_db()
    now = time.time()

    with db.session() as conn:
        rows = conn.execute(
            "SELECT * FROM facts "
            "WHERE tier = ? AND ingest_end IS NULL "
            "AND (valid_end IS NULL OR valid_end > ?) "
            "ORDER BY salience DESC LIMIT 100",
            (MemoryTier.WORKING.value, now),
        ).fetchall()

    if not rows:
        return "[EMPTY] no working-tier facts"

    lines: list[str] = []
    for r in rows:
        d = {
            "id": r["id"],
            "text": r["text"],
            "category": r["category"],
            "salience": r["salience"],
            "access_count": r["access_count"],
            "tier": r["tier"],
        }
        lines.append(encode_fact(d))
    return "\n".join(lines)
