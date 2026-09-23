"""AUDN classifier: Add / Update / Delete / Noop.

Determines what operation to perform when new information arrives:
  ADD    — no similar fact exists; insert new.
  UPDATE — a semantically equivalent fact exists but content differs;
           close ``valid_end`` on the old record and insert a new one.
  DELETE — explicit invalidation request;
           close ``valid_end`` on the target record.
  NOOP   — a substantially identical fact is already active;
           optionally reinforce its salience.
"""

from __future__ import annotations

import json
import sqlite3
import time

from mnemo.core.models import AUDNOperation, Fact, MemoryTier
from mnemo.storage.state import get_activity_tick, increment_activity_tick
from mnemo.storage.vector_store import VectorStore, _vec_to_blob, cosine_similarity

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DUPLICATE_THRESHOLD = 0.92  # cosine >= this  ->  NOOP  (same fact)
_UPDATE_THRESHOLD = 0.70  # cosine >= this  ->  UPDATE (related but changed)


class AUDNClassifier:
    """Classify an incoming text against the existing fact store."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vs = vector_store

    def classify(
        self,
        text: str,
        category: str,
        conn: sqlite3.Connection,
        *,
        force_op: AUDNOperation | None = None,
        target_fact_id: str | None = None,
        source_type: str = "agent",
        confidence: float = 1.0,
    ) -> tuple[AUDNOperation, str | None]:
        """Classify incoming text and return ``(operation, existing_fact_id)``.

        Args:
            text: Incoming fact text.
            category: Fact category.
            conn: Active SQLite connection.
            force_op: If set, use this operation subject to provenance validation.
            target_fact_id: Optional ID of the targeted fact.
            source_type: Origin type of the incoming memory.
            confidence: Reliability score of the incoming memory.

        Returns:
            ``(AUDNOperation, existing_fact_id_or_None)``.
        """
        # If force_op is set, check provenance protection before applying it
        if force_op is not None:
            target = target_fact_id
            if target:
                row = conn.execute(
                    "SELECT source_type, confidence, tier FROM facts WHERE id = ?",
                    (target,),
                ).fetchone()
                if row:
                    ex_source = row["source_type"] or "agent"
                    ex_conf = float(row["confidence"]) if row["confidence"] is not None else 1.0
                    ex_tier = str(row["tier"])
                    is_protected = (
                        ex_source in ("human_developer", "git_commit")
                        or ex_conf >= 0.9
                        or ex_tier == "core"
                    )
                    is_untrusted = (source_type in ("agent", "tool_output")) or (confidence < ex_conf)
                    if is_protected and is_untrusted and force_op in (
                        AUDNOperation.UPDATE,
                        AUDNOperation.CORRECT,
                        AUDNOperation.DELETE,
                        AUDNOperation.PURGE,
                    ):
                        return AUDNOperation.ADD, None
            return force_op, target

        # Embed incoming text
        query_vec = self._vs.embed_text(text)

        # Find the most similar active fact
        now = time.time()
        sql = """
            SELECT id, text, embedding_blob FROM facts
            WHERE embedding_blob IS NOT NULL
              AND ingest_end IS NULL
              AND (valid_end IS NULL OR valid_end > ?)
        """
        rows = conn.execute(sql, (now,)).fetchall()

        best_id: str | None = None
        best_sim: float = 0.0

        from mnemo.storage.vector_store import _blob_to_vec

        for row in rows:
            row_vec = _blob_to_vec(row["embedding_blob"])
            sim = cosine_similarity(query_vec, row_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = str(row["id"])

        if best_sim >= _DUPLICATE_THRESHOLD:
            return AUDNOperation.NOOP, best_id
        if best_sim >= _UPDATE_THRESHOLD:
            # Memory poisoning protection: check provenance of existing fact
            if best_id:
                row = conn.execute(
                    "SELECT source_type, confidence, tier FROM facts WHERE id = ?",
                    (best_id,),
                ).fetchone()
                if row:
                    ex_source = row["source_type"] or "agent"
                    ex_conf = float(row["confidence"]) if row["confidence"] is not None else 1.0
                    ex_tier = str(row["tier"])
                    is_protected = (
                        ex_source in ("human_developer", "git_commit")
                        or ex_conf >= 0.9
                        or ex_tier == "core"
                    )
                    is_untrusted_update = (source_type in ("agent", "tool_output")) or (
                        confidence < ex_conf
                    )
                    if is_protected and is_untrusted_update:
                        return AUDNOperation.ADD, None
            return AUDNOperation.UPDATE, best_id
        return AUDNOperation.ADD, None

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def execute_add(
        self,
        text: str,
        category: str,
        conn: sqlite3.Connection,
        *,
        tier: MemoryTier = MemoryTier.WORKING,
        metadata: dict[str, object] | None = None,
        source_type: str = "agent",
        source_ref: str | None = None,
        confidence: float = 1.0,
    ) -> Fact:
        from mnemo.storage.fts_store import extract_search_tokens

        now = time.time()
        current_tick = increment_activity_tick(conn)

        # Core tier protection: untrusted facts cannot enter Core Tier
        if tier == MemoryTier.CORE and (source_type == "tool_output" or confidence < 0.8):
            tier = MemoryTier.WORKING

        search_tokens = extract_search_tokens(text)

        fact = Fact(
            text=text,
            category=category,
            tier=tier,
            metadata=metadata or {},
            last_accessed_at=now,
            last_accessed_tick=current_tick,
            reinforcement_count=0,
            valid_start=now,
            ingest_start=now,
            source_type=source_type,
            source_ref=source_ref,
            confidence=confidence,
            search_tokens=search_tokens,
        )
        embedding = self._vs.embed_text(text)

        meta_json = json.dumps(fact.metadata or metadata or {})

        conn.execute(
            "INSERT INTO facts "
            "(id, text, category, salience, access_count, tier, last_accessed_at, "
            " last_accessed_tick, reinforcement_count, "
            " valid_start, valid_end, ingest_start, ingest_end, embedding_blob, metadata_json, "
            " source_type, source_ref, confidence, search_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id,
                fact.text,
                fact.category,
                fact.salience,
                fact.access_count,
                fact.tier.value,
                now,
                current_tick,
                0,
                now,
                None,
                now,
                None,
                _vec_to_blob(embedding),
                meta_json,
                fact.source_type,
                fact.source_ref,
                fact.confidence,
                fact.search_tokens,
            ),
        )
        fact.embedding = embedding.tolist()

        # Autonomous zero-touch: automatically link fact to mentioned AST entities
        from mnemo.engine.scanner import link_fact_to_entities

        link_fact_to_entities(fact.id, fact.text, conn, now=now)
        return fact

    def execute_update(
        self,
        old_fact_id: str,
        new_text: str,
        category: str,
        conn: sqlite3.Connection,
        *,
        source_type: str = "agent",
        source_ref: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> Fact:
        """Close the old fact's ``valid_end`` and insert a replacement."""
        now = time.time()
        # Close old record in real-world time
        conn.execute(
            "UPDATE facts SET valid_end = ? WHERE id = ?",
            (now, old_fact_id),
        )
        # Insert new
        return self.execute_add(
            new_text,
            category,
            conn,
            metadata=metadata,
            source_type=source_type,
            source_ref=source_ref,
            confidence=confidence,
        )

    def execute_correct(
        self,
        old_fact_id: str,
        new_text: str,
        category: str,
        conn: sqlite3.Connection,
        *,
        source_type: str = "agent",
        source_ref: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> Fact:
        """Physical correction: close old fact's system time (ingest_end) and insert new fact.

        Preserves valid_start and valid_end from the original fact.
        """
        now = time.time()
        old_row = conn.execute(
            "SELECT * FROM facts WHERE id = ?",
            (old_fact_id,),
        ).fetchone()

        if old_row is None:
            # Fallback to normal add if old fact not found
            return self.execute_add(
                new_text,
                category,
                conn,
                source_type=source_type,
                source_ref=source_ref,
                confidence=confidence,
                metadata=metadata,
            )

        # 1. Close system time of the old fact (do NOT close valid_end!)
        conn.execute(
            "UPDATE facts SET ingest_end = ? WHERE id = ? AND ingest_end IS NULL",
            (now, old_fact_id),
        )

        # 2. Advance activity tick
        current_tick = increment_activity_tick(conn)

        valid_start = (
            float(old_row["valid_start"]) if old_row["valid_start"] is not None else now
        )
        valid_end = (
            float(old_row["valid_end"]) if old_row["valid_end"] is not None else None
        )
        tier_val = (
            old_row["tier"] if old_row["tier"] else MemoryTier.WORKING.value
        )
        salience = (
            float(old_row["salience"]) if old_row["salience"] is not None else 1.0
        )
        access_count = (
            int(old_row["access_count"]) if old_row["access_count"] is not None else 0
        )
        reinforcement_count = (
            int(old_row["reinforcement_count"])
            if "reinforcement_count" in old_row.keys() and old_row["reinforcement_count"] is not None
            else 0
        )
        meta_json = (
            old_row["metadata_json"]
            if "metadata_json" in old_row.keys() and old_row["metadata_json"]
            else "{}"
        )

        try:
            resolved_metadata = json.loads(meta_json)
        except Exception:
            resolved_metadata = {}
        if metadata:
            resolved_metadata.update(metadata)

        try:
            fact_tier = MemoryTier(tier_val)
        except ValueError:
            fact_tier = MemoryTier.WORKING

        old_source_type = (
            old_row["source_type"]
            if "source_type" in old_row.keys() and old_row["source_type"]
            else "agent"
        )
        old_source_ref = (
            old_row["source_ref"] if "source_ref" in old_row.keys() else None
        )
        old_confidence = (
            float(old_row["confidence"])
            if "confidence" in old_row.keys() and old_row["confidence"] is not None
            else 1.0
        )

        resolved_source = source_type if source_type != "agent" else old_source_type
        resolved_ref = source_ref if source_ref is not None else old_source_ref
        resolved_conf = confidence if confidence != 1.0 else old_confidence

        if fact_tier == MemoryTier.CORE and (resolved_source == "tool_output" or resolved_conf < 0.8):
            fact_tier = MemoryTier.WORKING

        from mnemo.storage.fts_store import extract_search_tokens

        search_tokens = extract_search_tokens(new_text)

        fact = Fact(
            text=new_text,
            category=category or (old_row["category"] if "category" in old_row.keys() else "general"),
            salience=salience,
            access_count=access_count,
            tier=fact_tier,
            last_accessed_at=now,
            last_accessed_tick=current_tick,
            reinforcement_count=reinforcement_count,
            valid_start=valid_start,
            valid_end=valid_end,
            ingest_start=now,
            ingest_end=None,
            metadata=resolved_metadata,
            source_type=resolved_source,
            source_ref=resolved_ref,
            confidence=resolved_conf,
            search_tokens=search_tokens,
        )

        embedding = self._vs.embed_text(new_text)

        conn.execute(
            "INSERT INTO facts "
            "(id, text, category, salience, access_count, tier, last_accessed_at, "
            " last_accessed_tick, reinforcement_count, "
            " valid_start, valid_end, ingest_start, ingest_end, embedding_blob, metadata_json, "
            " source_type, source_ref, confidence, search_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id,
                fact.text,
                fact.category,
                fact.salience,
                fact.access_count,
                fact.tier.value,
                now,
                current_tick,
                fact.reinforcement_count,
                fact.valid_start,
                fact.valid_end,
                fact.ingest_start,
                fact.ingest_end,
                _vec_to_blob(embedding),
                json.dumps(fact.metadata),
                fact.source_type,
                fact.source_ref,
                fact.confidence,
                fact.search_tokens,
            ),
        )
        fact.embedding = embedding.tolist()

        from mnemo.engine.scanner import link_fact_to_entities

        link_fact_to_entities(fact.id, fact.text, conn, now=now)
        return fact

    def execute_delete(
        self,
        fact_id: str,
        conn: sqlite3.Connection,
    ) -> None:
        """Close a fact's ``valid_end`` (soft delete)."""
        now = time.time()
        conn.execute(
            "UPDATE facts SET valid_end = ? WHERE id = ? AND valid_end IS NULL",
            (now, fact_id),
        )

    def execute_purge(
        self,
        fact_id: str,
        conn: sqlite3.Connection,
        reason: str = "security_redaction",
    ) -> str:
        """Physically and irrevocably purge a fact, its links, vectors, and FTS entries.

        Creates an administrative audit tombstone with sha256 of the redacted fact text.
        Returns the purged fact_id.
        """
        import hashlib

        row = conn.execute("SELECT text FROM facts WHERE id = ?", (fact_id,)).fetchone()
        fact_text = row["text"] if row is not None else ""
        fact_hash = (
            hashlib.sha256(fact_text.encode("utf-8")).hexdigest()
            if fact_text
            else "unknown"
        )

        now = time.time()
        # 1. Delete links
        conn.execute("DELETE FROM fact_entity_links WHERE fact_id = ?", (fact_id,))
        # 2. Delete from FTS5
        conn.execute("DELETE FROM facts_fts WHERE id = ?", (fact_id,))
        # 3. Delete from facts (which also permanently removes text and embedding_blob)
        conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        # 4. Insert audit tombstone
        tombstone_id = f"tomb_{fact_id}"
        conn.execute(
            "INSERT OR REPLACE INTO audit_tombstones (id, fact_hash, reason, purged_at) "
            "VALUES (?, ?, ?, ?)",
            (tombstone_id, fact_hash, reason, now),
        )
        return fact_id

    def execute_noop(
        self,
        fact_id: str,
        conn: sqlite3.Connection,
        boost: float = 0.10,
    ) -> None:
        """Reinforce an existing fact (NOOP / duplicate detected)."""
        now = time.time()
        current_tick = get_activity_tick(conn)
        conn.execute(
            "UPDATE facts SET access_count = access_count + 1, "
            "reinforcement_count = reinforcement_count + 1, "
            "last_accessed_at = ?, last_accessed_tick = ?, "
            "salience = MIN(1.0, salience + ?) "
            "WHERE id = ?",
            (now, current_tick, boost, fact_id),
        )
