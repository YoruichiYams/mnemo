"""Pydantic v2 domain models for the Mnemo bitemporal memory system.

Every record carries four temporal coordinates (TemporalWindow):
  ingest_start / ingest_end  — when the system learned about the fact
  valid_start  / valid_end   — when the fact was true in the real world

Invalidation and updates close ``valid_end``; physical deletion never occurs.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _epoch_now() -> float:
    """Return the current Unix epoch in seconds (monotonic-safe)."""
    return time.time()


def _new_id() -> str:
    """Generate a new UUID-4 string identifier."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemoryTier(StrEnum):
    """Salience-based memory retention tiers.

    Thresholds (salience S):
      core       — S ≥ 0.90  (architectural rules, persona, permanent facts)
      working    — S ≥ 0.70  (active sprint context, current tasks)
      peripheral — S ≥ 0.40  (background / episodic memory)
      archived   — S < 0.40  (cold storage, eviction candidate)
    """

    CORE = "core"
    WORKING = "working"
    PERIPHERAL = "peripheral"
    ARCHIVED = "archived"

    @classmethod
    def from_salience(cls, salience: float) -> MemoryTier:
        """Derive the tier from a salience score."""
        if salience >= 0.90:
            return cls.CORE
        if salience >= 0.70:
            return cls.WORKING
        if salience >= 0.40:
            return cls.PERIPHERAL
        return cls.ARCHIVED


class SourceType(StrEnum):
    """Origin / provenance category of a memory fact."""

    AGENT = "agent"
    HUMAN_DEVELOPER = "human_developer"
    GIT_COMMIT = "git_commit"
    DOCUMENTATION = "documentation"
    TOOL_OUTPUT = "tool_output"


class AUDNOperation(StrEnum):
    """AUDN pipeline operations (Add / Update / Delete / Noop / Correct / Purge)."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    CORRECT = "correct"
    PURGE = "purge"


# ---------------------------------------------------------------------------
# Temporal window
# ---------------------------------------------------------------------------


class TemporalWindow(BaseModel):
    """Four-coordinate bitemporal window.

    * ``ingest_start`` / ``ingest_end`` — system (transaction) time.
    * ``valid_start``  / ``valid_end``  — real-world validity interval.

    Open intervals are represented by ``None`` in the ``*_end`` fields.
    """

    ingest_start: float = Field(default_factory=_epoch_now)
    ingest_end: float | None = Field(default=None)
    valid_start: float = Field(default_factory=_epoch_now)
    valid_end: float | None = Field(default=None)

    @property
    def valid_from(self) -> float:
        return self.valid_start

    @property
    def valid_to(self) -> float | None:
        return self.valid_end

    @property
    def system_created_at(self) -> float:
        return self.ingest_start

    @property
    def system_expired_at(self) -> float | None:
        return self.ingest_end

    @property
    def is_active(self) -> bool:
        """``True`` when both intervals are still open (current system record)."""
        return self.ingest_end is None and self.valid_end is None

    def is_valid_at(
        self,
        valid_time: float | None = None,
        ingest_time: float | None = None,
    ) -> bool:
        """Bitemporal point-in-time containment check.

        Args:
            valid_time:  Real-world instant to test.  Defaults to *now*.
            ingest_time: System-time instant to test.  Defaults to *now*.

        Returns:
            ``True`` when the point falls inside both intervals.
        """
        now = _epoch_now()
        vt = valid_time if valid_time is not None else now
        it = ingest_time if ingest_time is not None else now

        v_ok = self.valid_start <= vt and (self.valid_end is None or self.valid_end > vt)
        i_ok = self.ingest_start <= it and (self.ingest_end is None or self.ingest_end > it)
        return v_ok and i_ok

    def close_valid(self, at: float | None = None) -> None:
        """Close the real-world validity interval (soft-delete)."""
        self.valid_end = at if at is not None else _epoch_now()

    def close_ingest(self, at: float | None = None) -> None:
        """Close the system/transaction interval (superseded by correction)."""
        self.ingest_end = at if at is not None else _epoch_now()


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """A named node in the knowledge graph."""

    id: str = Field(default_factory=_new_id)
    name: str
    entity_type: str = Field(default="concept", alias="type")
    properties: dict[str, Any] = Field(default_factory=dict)
    salience: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: float = Field(default_factory=_epoch_now)
    temporal: TemporalWindow = Field(default_factory=TemporalWindow)

    model_config = {"populate_by_name": True}

    @property
    def is_active(self) -> bool:
        return self.temporal.is_active

    @property
    def is_currently_valid(self) -> bool:
        return self.temporal.is_valid_at()


class Relation(BaseModel):
    """A directed, weighted edge between two entities."""

    id: str = Field(default_factory=_new_id)
    source_id: str
    target_id: str
    relation_type: str
    weight: float = Field(default=1.0, ge=0.0)
    temporal: TemporalWindow = Field(default_factory=TemporalWindow)

    @property
    def is_active(self) -> bool:
        return self.temporal.is_active

    @property
    def is_currently_valid(self) -> bool:
        return self.temporal.is_valid_at()


class Fact(BaseModel):
    """Atomic unit of bitemporal memory.

    Fields mirror the ``facts`` SQL table.
    ``salience`` decays over activity ticks via the Ebbinghaus formula in ``decay.py``.
    """

    id: str = Field(default_factory=_new_id)
    text: str
    category: str = Field(default="general")
    salience: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    tier: MemoryTier = Field(default=MemoryTier.WORKING)
    last_accessed_at: float = Field(default_factory=_epoch_now)
    last_accessed_tick: int = Field(default=0, ge=0)
    reinforcement_count: int = Field(default=0, ge=0)
    # Bitemporal coordinates stored inline (also in TemporalWindow for convenience)
    valid_start: float = Field(default_factory=_epoch_now)
    valid_end: float | None = Field(default=None)
    ingest_start: float = Field(default_factory=_epoch_now)
    ingest_end: float | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_stale: bool = Field(default=False)
    # Provenance and code search tokens
    source_type: str = Field(default=SourceType.AGENT.value)
    source_ref: str | None = Field(default=None)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    search_tokens: str = Field(default="")

    @property
    def valid_from(self) -> float:
        return self.valid_start

    @property
    def valid_to(self) -> float | None:
        return self.valid_end

    @property
    def system_created_at(self) -> float:
        return self.ingest_start

    @property
    def system_expired_at(self) -> float | None:
        return self.ingest_end

    @property
    def is_active(self) -> bool:
        """``True`` when the fact is currently valid and not superseded."""
        return self.ingest_end is None and self.valid_end is None

    @property
    def is_currently_valid(self) -> bool:
        """Bitemporal point-in-time check at current time."""
        now = _epoch_now()
        v_ok = self.valid_start <= now and (self.valid_end is None or self.valid_end > now)
        i_ok = self.ingest_start <= now and (self.ingest_end is None or self.ingest_end > now)
        return v_ok and i_ok

    def close_valid(self, at: float | None = None) -> None:
        """Soft-invalidate the fact in the real-world dimension."""
        self.valid_end = at if at is not None else _epoch_now()

    def close_ingest(self, at: float | None = None) -> None:
        """Mark this system version as superseded."""
        self.ingest_end = at if at is not None else _epoch_now()

    def reinforce(self, boost: float = 0.10, tick: int | None = None) -> None:
        """Bump access statistics and salience on retrieval / explicit reinforcement."""
        self.access_count += 1
        self.reinforcement_count += 1
        self.last_accessed_at = _epoch_now()
        if tick is not None:
            self.last_accessed_tick = tick
        self.salience = min(1.0, self.salience + boost)
        self.tier = MemoryTier.from_salience(self.salience)


class FactEntityLink(BaseModel):
    """Association between a memory fact and an AST entity with node version hash."""

    fact_id: str
    entity_id: str
    entity_hash_at_link: str
    created_at: str


# ---------------------------------------------------------------------------
# Search / retrieval results
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single ranked hit returned by the hybrid retriever."""

    fact: Fact
    score: float = Field(default=0.0, ge=0.0)
    channel: Literal["vector", "fts5", "graph", "rrf"] = Field(default="rrf")


# ---------------------------------------------------------------------------
# Debt ledger
# ---------------------------------------------------------------------------


class DebtLedgerItem(BaseModel):
    """An item of architectural / knowledge debt detected in the store.

    Examples: contradictions between facts, decaying core-tier salience,
    orphaned entities with no relations, stale facts without recent access.
    """

    id: str = Field(default_factory=_new_id)
    ceiling: str = Field(
        description=("Maximum tier this item could affect: 'critical', 'high', 'medium', 'low'"),
    )
    trigger: str = Field(
        description="Condition that surfaced this debt item",
    )
    code_context: str = Field(
        default="",
        description="Relevant fact / entity IDs or short code reference",
    )
    created_at: float = Field(default_factory=_epoch_now)
