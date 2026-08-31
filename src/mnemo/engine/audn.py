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

import sqlite3
import time

from mnemo.core.models import AUDNOperation, Fact, MemoryTier
from mnemo.storage.vector_store import VectorStore, cosine_similarity

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
    ) -> tuple[AUDNOperation, str | None]:
        """Classify incoming text and return ``(operation, existing_fact_id)``.

        Args:
            text: Incoming fact text.
            category: Fact category.
            conn: Active SQLite connection.
            force_op: If set, skip classification and use this operation.

        Returns:
            ``(AUDNOperation, existing_fact_id_or_None)``.
        """
        if force_op == AUDNOperation.DELETE:
            # For DELETE we need the caller to supply the target id
            return AUDNOperation.DELETE, None
        if force_op is not None:
            return force_op, None

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
    ) -> Fact:
        """Insert a new fact (ADD operation)."""
        fact = Fact(text=text, category=category, tier=tier, metadata=metadata or {})
        now = time.time()
        embedding = self._vs.embed_text(text)
        from mnemo.storage.vector_store import _vec_to_blob

        conn.execute(
            "INSERT INTO facts "
            "(id, text, category, salience, access_count, tier, last_accessed_at, "
            " valid_start, valid_end, ingest_start, ingest_end, embedding_blob, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id,
                fact.text,
                fact.category,
                fact.salience,
                fact.access_count,
                fact.tier.value,
                now,
                now,
                None,
                now,
                None,
                _vec_to_blob(embedding),
                "{}",
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
    ) -> Fact:
        """Close the old fact's ``valid_end`` and insert a replacement."""
        now = time.time()
        # Close old record
        conn.execute(
            "UPDATE facts SET valid_end = ? WHERE id = ?",
            (now, old_fact_id),
        )
        # Insert new
        return self.execute_add(new_text, category, conn)

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

    def execute_noop(
        self,
        fact_id: str,
        conn: sqlite3.Connection,
        boost: float = 0.10,
    ) -> None:
        """Reinforce an existing fact (NOOP / duplicate detected)."""
        now = time.time()
        conn.execute(
            "UPDATE facts SET access_count = access_count + 1, "
            "last_accessed_at = ?, salience = MIN(1.0, salience + ?) "
            "WHERE id = ?",
            (now, boost, fact_id),
        )
