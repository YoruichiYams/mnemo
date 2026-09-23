"""FastMCP tool definitions for Mnemo.

All text returned to the LLM is serialised via TOON for token efficiency.
"""

from __future__ import annotations

import sqlite3

from mnemo.core.models import AUDNOperation, MemoryTier
from mnemo.mcp.server import (
    _get_audn,
    _get_db,
    _get_retriever,
    _get_tier_manager,
    mcp_app,
)
from mnemo.serialization.toon import encode_debt, encode_fact


def _link_explicit_entities(
    fact_id: str,
    entity_identifiers: list[str] | None,
    conn: sqlite3.Connection,
    now: float,
) -> None:
    """Link fact to explicit AST entity identifiers, storing current entity hashes."""
    if not entity_identifiers:
        return

    import hashlib
    import json
    import uuid

    for ident in entity_identifiers:
        row = conn.execute(
            """
            SELECT id, name, properties_json FROM entities
            WHERE (name = ? OR name LIKE ?)
              AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)
            ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END LIMIT 1;
            """,
            (ident, f"%.{ident}", now, ident),
        ).fetchone()

        if row:
            eid = str(row["id"])
            props = {}
            if "properties_json" in row.keys() and row["properties_json"]:
                try:
                    props = json.loads(row["properties_json"])
                except Exception:
                    pass
            ent_hash = (
                props.get("hash") or hashlib.sha256(str(row["name"]).encode("utf-8")).hexdigest()
            )
        else:
            eid = str(uuid.uuid4())
            ent_hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()
            props = {"hash": ent_hash}
            conn.execute(
                """
                INSERT INTO entities (
                    id, name, entity_type, properties_json, salience, access_count,
                    last_accessed_at, valid_start, valid_end, ingest_start, ingest_end
                ) VALUES (?, ?, 'concept', ?, 1.0, 0, ?, ?, NULL, ?, NULL);
                """,
                (eid, ident, json.dumps(props), now, now, now),
            )

        conn.execute(
            """
            INSERT INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(fact_id, entity_id) DO UPDATE SET
                entity_hash_at_link = excluded.entity_hash_at_link,
                created_at = excluded.created_at;
            """,
            (fact_id, eid, ent_hash, str(now)),
        )


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# mnemo_remember
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_remember",
    description=(
        "Store a new memory fact. The AUDN classifier automatically "
        "detects duplicates (NOOP), updates, physical corrections, or fresh additions."
    ),
)
def mnemo_remember(
    text: str,
    category: str = "general",
    force_op: str | None = None,
    target_fact_id: str | None = None,
    entity_identifiers: list[str] | None = None,
) -> str:
    """Remember a fact through the AUDN pipeline.

    Args:
        text: The fact text to remember (max 32768 chars).
        category: Fact category (default ``general``).
        force_op: Force a specific operation: ``add``, ``update``, ``delete``, ``noop``, ``correct``, ``reinforce``.
        target_fact_id: Optional ID of the fact to target (for ``correct`` or ``update``).
        entity_identifiers: Optional list of qualified entity names linked to this fact (max 20).
    """
    import time

    # Input length validation
    if len(text) > 32768:
        return "[ERROR] text exceeds maximum allowed length of 32768 characters"
    if entity_identifiers and len(entity_identifiers) > 20:
        return "[ERROR] entity_identifiers exceeds maximum allowed length of 20 elements"

    source_type = "agent"
    confidence = 0.75

    if force_op:
        op_str = force_op.lower()
        if op_str == "purge":
            return "[FORBIDDEN] Physical purge is an administrative CLI-only operation"
        if op_str == "reinforce":
            op_str = "noop"
        try:
            forced = AUDNOperation(op_str)
        except ValueError:
            valid_ops = (
                ", ".join(op.value for op in AUDNOperation if op != AUDNOperation.PURGE)
                + ", reinforce"
            )
            return f"[ERROR] invalid force_op '{force_op}'. Expected one of: {valid_ops}"
    else:
        forced = None

    db = _get_db()
    audn = _get_audn()

    now = time.time()
    with db.session() as conn:
        # Pre-check target fact protection if force_op specified with target_fact_id
        if (
            forced in (AUDNOperation.UPDATE, AUDNOperation.CORRECT, AUDNOperation.DELETE)
            and target_fact_id
        ):
            row = conn.execute(
                "SELECT source_type, tier FROM facts WHERE id = ?", (target_fact_id,)
            ).fetchone()
            if row:
                ex_src = row["source_type"] or "agent"
                ex_tier = str(row["tier"])
                if ex_src in ("human_developer", "git_commit") or ex_tier == "core":
                    return (
                        "[FORBIDDEN] Cannot force modification of protected core/developer memory"
                    )

        op, existing_id = audn.classify(
            text,
            category,
            conn,
            force_op=forced,
            target_fact_id=target_fact_id,
            source_type=source_type,
            confidence=confidence,
        )
        target_id = target_fact_id or existing_id

        if op == AUDNOperation.PURGE:
            return "[FORBIDDEN] Physical purge is an administrative CLI-only operation"

        # Check target fact protection after classification
        if (
            forced in (AUDNOperation.UPDATE, AUDNOperation.CORRECT, AUDNOperation.DELETE)
            and target_id
        ):
            row = conn.execute(
                "SELECT source_type, tier FROM facts WHERE id = ?", (target_id,)
            ).fetchone()
            if row:
                ex_src = row["source_type"] or "agent"
                ex_tier = str(row["tier"])
                if ex_src in ("human_developer", "git_commit") or ex_tier == "core":
                    return (
                        "[FORBIDDEN] Cannot force modification of protected core/developer memory"
                    )

        if op == AUDNOperation.CORRECT:
            if target_id:
                fact = audn.execute_correct(
                    target_id,
                    text,
                    category,
                    conn,
                    source_type=source_type,
                    source_ref=None,
                    confidence=confidence,
                )
                _link_explicit_entities(fact.id, entity_identifiers, conn, now)
                return f"[CORRECT] old={target_id[:8]}.. {encode_fact(fact.model_dump(exclude={'embedding'}))}"
            else:
                fact = audn.execute_add(
                    text,
                    category,
                    conn,
                    source_type=source_type,
                    source_ref=None,
                    confidence=confidence,
                )
                _link_explicit_entities(fact.id, entity_identifiers, conn, now)
                return f"[CORRECT] (new) {encode_fact(fact.model_dump(exclude={'embedding'}))}"

        if op == AUDNOperation.UPDATE:
            if target_id:
                fact = audn.execute_update(
                    target_id,
                    text,
                    category,
                    conn,
                    source_type=source_type,
                    source_ref=None,
                    confidence=confidence,
                )
                _link_explicit_entities(fact.id, entity_identifiers, conn, now)
                return f"[UPD] old={target_id[:8]}.. {encode_fact(fact.model_dump(exclude={'embedding'}))}"
            else:
                # Fallback to ADD if no existing fact was found to update
                fact = audn.execute_add(
                    text,
                    category,
                    conn,
                    source_type=source_type,
                    source_ref=None,
                    confidence=confidence,
                )
                _link_explicit_entities(fact.id, entity_identifiers, conn, now)
                return f"[ADD] (fallback) {encode_fact(fact.model_dump(exclude={'embedding'}))}"

        if op == AUDNOperation.ADD:
            fact = audn.execute_add(
                text,
                category,
                conn,
                source_type=source_type,
                source_ref=None,
                confidence=confidence,
            )
            _link_explicit_entities(fact.id, entity_identifiers, conn, now)
            return f"[ADD] {encode_fact(fact.model_dump(exclude={'embedding'}))}"

        if op == AUDNOperation.NOOP and target_id:
            audn.execute_noop(target_id, conn)
            _link_explicit_entities(target_id, entity_identifiers, conn, now)
            return f"[NOOP] reinforced={target_id[:8]}.."

        return f"[{op.value.upper()}] done"


# -----------------------------------------------------------------------
# mnemo_purge
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_purge",
    description=("Administrative physical purge (disabled via MCP for security)."),
)
def mnemo_purge(
    fact_id: str,
    reason: str = "security_redaction",
) -> str:
    """Physically and irrevocably purge a memory fact.

    Disabled via MCP — physical purge is an administrative CLI-only operation.
    """
    return "[FORBIDDEN] Physical purge is an administrative CLI-only operation"


# -----------------------------------------------------------------------
# mnemo_search
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_search",
    description=(
        "Hybrid 3-channel search (Vector + FTS5 + Graph) with RRF fusion. "
        "Returns TOON-serialised results for token efficiency. Purely read-only."
    ),
)
def mnemo_search(
    query: str,
    limit: int = 10,
    min_tier: str = "archived",
    as_of: str | None = None,
) -> str:
    """Search memory with hybrid retrieval.

    Args:
        query: Natural-language search query.
        limit: Maximum number of results (default 10).
        min_tier: Minimum tier filter: ``core``, ``working``, ``peripheral``, ``archived``.
        as_of: Optional retrospective point-in-time (RFC3339 string or epoch timestamp).
    """
    db = _get_db()
    retriever = _get_retriever()

    try:
        tier = MemoryTier(min_tier.lower())
    except ValueError:
        valid_tiers = ", ".join(t.value for t in MemoryTier)
        return f"[ERROR] invalid min_tier '{min_tier}'. Expected one of: {valid_tiers}"

    with db.session(readonly=True) as conn:
        results = retriever.search(
            query, conn, limit=limit, min_tier=tier, as_of=as_of, include_archived=True
        )

    if not results:
        return "[EMPTY] no matching memories"

    lines: list[str] = []
    for r in results:
        fact_d = r.fact.model_dump(exclude={"embedding"})
        toon = encode_fact(fact_d)
        lines.append(f"sc:{r.score:.6f}|ch:{r.channel}|f:({toon})")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# mnemo_acknowledge_usage
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_acknowledge_usage",
    description=(
        "Explicitly acknowledge that retrieved facts were actually used by the agent, "
        "triggering Spaced Repetition reinforcement without search hit bias."
    ),
)
def mnemo_acknowledge_usage(fact_ids: list[str]) -> str:
    """Reinforce facts that were actually utilized in the task.

    Args:
        fact_ids: List of fact UUIDs to reinforce.
    """
    db = _get_db()
    audn = _get_audn()

    reinforced_count = 0
    with db.session() as conn:
        for fid in fact_ids:
            row = conn.execute("SELECT id FROM facts WHERE id = ?", (fid,)).fetchone()
            if row:
                audn.execute_noop(fid, conn)
                reinforced_count += 1

    return f"[ACKNOWLEDGED] reinforced={reinforced_count} facts"


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
        row = conn.execute(
            "SELECT source_type, tier FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row:
            ex_source = row["source_type"] or "agent"
            ex_tier = str(row["tier"])
            if ex_source in ("human_developer", "git_commit") or ex_tier == "core":
                return "[FORBIDDEN] Cannot force modification of protected core/developer memory"
        audn.execute_delete(fact_id, conn)

    return f"[DEL] invalidated={fact_id[:8]}.."


# -----------------------------------------------------------------------
# mnemo_reinforce
# -----------------------------------------------------------------------
@mcp_app.tool(
    name="mnemo_reinforce",
    description="Reinforce a fact's salience by incrementing its access count and reinforcement count.",
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
    import json

    db = _get_db()

    with db.session(readonly=True) as conn:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()

    if row is None:
        return f"[NOT_FOUND] id={fact_id[:8]}.."

    from mnemo.core.models import Fact

    meta = {}
    if "metadata_json" in row.keys() and row["metadata_json"]:
        try:
            meta = json.loads(row["metadata_json"])
        except Exception:
            pass

    fact = Fact(
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
    from pathlib import Path

    from mnemo.engine.scanner import ProjectScanner
    from mnemo.storage.state import increment_activity_tick

    base = Path.cwd().resolve()
    target = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not (target == base or base in target.parents):
        return "[ERROR] Path traversal detected: scan path must be inside project root"

    db = _get_db()
    scanner = ProjectScanner(root_path=target)

    with db.session() as conn:
        res = scanner.scan(conn)
        increment_activity_tick(conn)

    return (
        f"[SCAN] scanned={res['scanned_files']} skipped={res['skipped_files']} "
        f"entities=+{res['entities_added']} relations=+{res['relations_added']}"
    )
