"""Unit tests for Pydantic v2 core domain models and bitemporal window methods."""

from __future__ import annotations

import time

from mnemo.core.models import (
    DebtLedgerItem,
    Entity,
    Fact,
    MemoryTier,
    Relation,
    SearchResult,
    TemporalWindow,
)


class TestModels:
    """Test suite for domain models and temporal window logic."""

    def test_temporal_window_lifecycle(self) -> None:
        """TemporalWindow accurately checks point-in-time validity and closes intervals."""
        now = time.time()
        win = TemporalWindow(
            ingest_start=now,
            ingest_end=None,
            valid_start=now,
            valid_end=None,
        )

        assert win.is_active is True
        assert win.is_valid_at(now, now) is True

        # Point in past is before valid_start
        assert win.is_valid_at(now - 10.0, now) is False

        # Close valid interval (soft-delete)
        win.close_valid(now + 100.0)
        assert win.is_active is False
        assert win.is_valid_at(now + 50.0) is True
        assert win.is_valid_at(now + 150.0) is False

        # Close ingest interval
        win.close_ingest(now + 200.0)
        assert win.is_valid_at(now + 50.0, now + 250.0) is False

    def test_fact_reinforce_and_bitemporal_methods(self) -> None:
        """Fact model supports reinforcement and closing timestamps."""
        fact = Fact(text="Test fact", category="general", salience=0.75, tier=MemoryTier.WORKING)
        assert fact.is_active is True

        # Reinforce
        fact.reinforce(boost=0.20)
        assert fact.access_count == 1
        assert fact.salience == 0.95
        assert fact.tier == MemoryTier.CORE

        # Invalidate
        fact.close_valid()
        assert fact.valid_end is not None
        assert fact.is_active is False

        fact.close_ingest()
        assert fact.ingest_end is not None

    def test_entity_and_relation_models(self) -> None:
        """Entity and Relation models instantiate with default temporal windows."""
        e1 = Entity(name="Entity1", entity_type="concept")
        e2 = Entity(name="Entity2", entity_type="concept")
        rel = Relation(source_id=e1.id, target_id=e2.id, relation_type="relates_to", weight=0.8)

        assert e1.name == "Entity1"
        assert e1.entity_type == "concept"
        assert rel.weight == 0.8
        assert rel.is_currently_valid is True

    def test_debt_ledger_and_search_result_models(self) -> None:
        """DebtLedgerItem and SearchResult models validate required fields."""
        fact = Fact(text="Content", category="general")
        sr = SearchResult(fact=fact, score=0.033, channel="rrf")
        assert sr.score == 0.033
        assert sr.channel == "rrf"

        debt = DebtLedgerItem(
            ceiling="high",
            trigger="test_trigger",
            code_context="test_context",
        )
        assert debt.ceiling == "high"
        assert debt.trigger == "test_trigger"
