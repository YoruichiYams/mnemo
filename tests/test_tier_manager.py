"""Unit tests for the TierManager and Debt detection engine."""

from __future__ import annotations

import time

from mnemo.engine.tier_manager import TierManager
from mnemo.storage.connection import Database


class TestTierManager:
    """Test suite for batch decay, tier migration, and debt ledger detection."""

    def test_decay_all_migrates_working_to_archived(
        self, in_memory_db: Database, tier_manager: TierManager
    ) -> None:
        """Old facts without access migrate from Working tier to Peripheral and Archived."""
        now = time.time()
        with in_memory_db.session() as conn:
            # 1. Fact accessed 10 days ago (240 hours)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start, metadata_json) "
                "VALUES ('f_old', 'Old fact from 10 days ago', 'test', 0.80, 0, 'working', ?, ?, ?, '{}')",
                (now - 10 * 86400.0, now - 10 * 86400.0, now - 10 * 86400.0),
            )
            # 2. Pinned Core fact (should NOT decay)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start, metadata_json) "
                "VALUES ('f_core', 'Permanent rule', 'test', 1.0, 0, 'core', ?, ?, ?, '{\"pinned\": true}')",
                (now - 10 * 86400.0, now - 10 * 86400.0, now - 10 * 86400.0),
            )
            # 3. Unpinned Core fact (should decay over time)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start, metadata_json) "
                "VALUES ('f_unpinned_core', 'Unpinned core fact', 'test', 1.0, 0, 'core', ?, ?, ?, '{}')",
                (now - 10 * 86400.0, now - 10 * 86400.0, now - 10 * 86400.0),
            )

            result = tier_manager.decay_all(conn, now=now)
            assert result["processed"] >= 2
            assert result["migrated"] >= 2

            # Check old fact was demoted
            row_old = conn.execute(
                "SELECT salience, tier, last_accessed_at FROM facts WHERE id = 'f_old'"
            ).fetchone()
            assert row_old["tier"] in ("peripheral", "archived")
            assert row_old["salience"] < 0.80
            assert row_old["last_accessed_at"] == now

            # Check pinned core fact remained Core
            row_core = conn.execute(
                "SELECT salience, tier FROM facts WHERE id = 'f_core'"
            ).fetchone()
            assert row_core["tier"] == "core"
            assert row_core["salience"] == 1.0

            # Check unpinned core fact decayed
            row_unpinned = conn.execute(
                "SELECT salience, tier FROM facts WHERE id = 'f_unpinned_core'"
            ).fetchone()
            assert row_unpinned["tier"] in ("peripheral", "archived")
            assert row_unpinned["salience"] < 0.90

    def test_detect_and_persist_debt(
        self, in_memory_db: Database, tier_manager: TierManager
    ) -> None:
        """TierManager detects decaying working facts, stale core facts, and orphaned entities."""
        now = time.time()
        with in_memory_db.session() as conn:
            # 1. At-risk working fact (salience = 0.72 < 0.75)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start) "
                "VALUES ('f_at_risk', 'At risk working fact', 'test', 0.72, 0, 'working', ?, ?, ?)",
                (now, now, now),
            )
            # 2. Stale core fact (>30 days without access)
            conn.execute(
                "INSERT INTO facts (id, text, category, salience, access_count, tier, last_accessed_at, valid_start, ingest_start) "
                "VALUES ('f_stale_core', 'Stale core fact', 'test', 1.0, 0, 'core', ?, ?, ?)",
                (now - 35 * 86400.0, now - 35 * 86400.0, now - 35 * 86400.0),
            )
            # 3. Orphaned entity (no relations)
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, valid_start, ingest_start, last_accessed_at) "
                "VALUES ('e_orphan', 'OrphanedConcept', 'concept', ?, ?, ?)",
                (now, now, now),
            )

            # Detect debt
            debt_items = tier_manager.detect_debt(conn, now=now)
            assert len(debt_items) >= 3

            triggers = {item.trigger for item in debt_items}
            assert "decaying_working" in triggers
            assert "stale_core" in triggers
            assert "orphaned_entity" in triggers

            # Persist debt
            count = tier_manager.persist_debt(debt_items, conn)
            assert count == len(debt_items)

            rows = conn.execute("SELECT COUNT(*) AS c FROM debt_ledger").fetchone()
            assert rows["c"] >= 3
