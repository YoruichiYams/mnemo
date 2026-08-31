"""FastMCP tool definitions for Mnemo.

All text returned to the LLM is serialised via TOON for token efficiency.
"""

from __future__ import annotations

from mnemo.core.models import AUDNOperation, MemoryTier
from mnemo.mcp.server import (
    _get_audn,
    _get_db,
    _get_retriever,
    _get_tier_manager,
    mcp_app,
)
from mnemo.serialization.toon import encode_debt, encode_fact


# -----------------------------------------------------------------------
# mnemo_remember
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_remember",
    description=(
        "Store a new memory fact. The AUDN classifier automatically "
        "detects duplicates (NOOP), updates, or fresh additions."
    ),
)
def mnemo_remember(
    text: str,
    category: str = "general",
    force_op: str | None = None,
) -> str:
    """Remember a fact through the AUDN pipeline.

    Args:
        text: The fact text to remember.
        category: Fact category (default ``general``).
        force_op: Force a specific operation: ``add``, ``update``, ``delete``, ``noop``.
    """
    db = _get_db()
    audn = _get_audn()

    forced = AUDNOperation(force_op) if force_op else None

    with db.session() as conn:
        op, existing_id = audn.classify(text, category, conn, force_op=forced)

        if op == AUDNOperation.ADD:
            fact = audn.execute_add(text, category, conn)
            return f"[ADD] {encode_fact(fact.model_dump(exclude={'embedding'}))}"

        if op == AUDNOperation.UPDATE and existing_id:
            fact = audn.execute_update(existing_id, text, category, conn)
            return f"[UPD] old={existing_id[:8]}.. {encode_fact(fact.model_dump(exclude={'embedding'}))}"

        if op == AUDNOperation.NOOP and existing_id:
            audn.execute_noop(existing_id, conn)
            return f"[NOOP] reinforced={existing_id[:8]}.."

        return f"[{op.value.upper()}] done"


# -----------------------------------------------------------------------
# mnemo_search
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_search",
    description=(
        "Hybrid 3-channel search (Vector + FTS5 + Graph) with RRF fusion. "
        "Returns TOON-serialised results for token efficiency."
    ),
)
def mnemo_search(
    query: str,
    limit: int = 10,
    min_tier: str = "archived",
) -> str:
    """Search memory with hybrid retrieval.

    Args:
        query: Natural-language search query.
        limit: Maximum number of results (default 10).
        min_tier: Minimum tier filter: ``core``, ``working``, ``peripheral``, ``archived``.
    """
    db = _get_db()
    retriever = _get_retriever()
    tier = MemoryTier(min_tier.lower())

    with db.session() as conn:
        results = retriever.search(query, conn, limit=limit, min_tier=tier)

    if not results:
        return "[EMPTY] no matching memories"

    lines: list[str] = []
    for r in results:
        fact_d = r.fact.model_dump(exclude={"embedding"})
        toon = encode_fact(fact_d)
        lines.append(f"sc:{r.score:.6f}|ch:{r.channel}|f:({toon})")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# mnemo_invalidate
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_invalidate",
    description="Soft-delete a fact by closing its valid_end timestamp.",
)
def mnemo_invalidate(fact_id: str) -> str:
    """Invalidate (soft-delete) a memory fact.

    Args:
        fact_id: The UUID of the fact to invalidate.
    """
    db = _get_db()
    audn = _get_audn()

    with db.session() as conn:
        audn.execute_delete(fact_id, conn)

    return f"[DEL] invalidated={fact_id[:8]}.."


# -----------------------------------------------------------------------
# mnemo_reinforce
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_reinforce",
    description="Reinforce a fact's salience by incrementing its access count.",
)
def mnemo_reinforce(fact_id: str, boost: float = 0.10) -> str:
    """Reinforce a memory fact.

    Args:
        fact_id: The UUID of the fact to reinforce.
        boost: Salience boost amount (default 0.10).
    """
    db = _get_db()
    audn = _get_audn()

    with db.session() as conn:
        audn.execute_noop(fact_id, conn, boost=boost)

    return f"[REINFORCED] id={fact_id[:8]}.. boost={boost}"


# -----------------------------------------------------------------------
# mnemo_inspect
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_inspect",
    description="Inspect a single fact by ID, returned in TOON format.",
)
def mnemo_inspect(fact_id: str) -> str:
    """Retrieve the full details of a fact.

    Args:
        fact_id: The UUID of the fact to inspect.
    """
    db = _get_db()

    with db.session() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()

    if row is None:
        return f"[NOT_FOUND] id={fact_id[:8]}.."

    from mnemo.core.models import Fact

    fact = Fact(
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
    return encode_fact(fact.model_dump(exclude={"embedding"}))


# -----------------------------------------------------------------------
# mnemo_get_debt_ledger
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_get_debt_ledger",
    description=(
        "Detect and return knowledge / architectural debt: "
        "decaying working-tier facts, stale core facts, orphaned entities."
    ),
)
def mnemo_get_debt_ledger() -> str:
    """Scan memory for debt items and return them in TOON format."""
    db = _get_db()
    tier_mgr = _get_tier_manager()

    with db.session() as conn:
        # First recompute tiers
        tier_mgr.decay_all(conn)
        # Then detect debt
        items = tier_mgr.detect_debt(conn)
        if items:
            tier_mgr.persist_debt(items, conn)

    if not items:
        return "[CLEAN] no debt detected"

    lines = [encode_debt(item.model_dump()) for item in items]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# mnemo_scan_project
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_scan_project",
    description=(
        "Perform an incremental AST code scan of the current repository to "
        "extract modules, classes, and dependencies into the knowledge graph."
    ),
)
def mnemo_scan_project(path: str = ".") -> str:
    """Perform AST scan of project path and populate graph entities."""
    from mnemo.engine.scanner import ProjectScanner

    db = _get_db()
    scanner = ProjectScanner(root_path=path)

    with db.session() as conn:
        res = scanner.scan(conn)

    return (
        f"[SCAN] scanned={res['scanned_files']} skipped={res['skipped_files']} "
        f"entities=+{res['entities_added']} relations=+{res['relations_added']}"
    )
