"""Comprehensive test suite for Stage 2: Activity-Based Decay, Spaced Repetition, and Bitemporal Search (as_of).

Tests verify:
1. Activity Ticks vs Wall-clock: Memory does not decay during idle time without activity.
2. Spaced Repetition Stability: Multiplicative growth R = R_0 * (1 + alpha)^n.
3. Search Idempotency: Searches do NOT auto-reinforce (prevents rich-get-richer loop).
4. Explicit Reinforcement & Acknowledgment: mnemo_acknowledge_usage and mnemo_reinforce.
5. Retrospective Bitemporal Querying: as_of parameter supporting RFC3339 strings and epoch floats.
6. Physical Correction vs Logical Invalidation: Ingest closing vs validity closing.
7. Archived Tier Fallback: Surfacing cold facts with archived_warning to avoid blind spots.
8. AST Drift on Pinned Facts: Stale pinned facts lose protection and decay fast.
"""

from __future__ import annotations

import time

from mnemo.core.decay import calculate_salience
from mnemo.core.models import MemoryTier
from mnemo.engine.audn import AUDNClassifier
from mnemo.engine.retriever import HybridRetriever
from mnemo.engine.tier_manager import TierManager
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.state import (
    get_activity_tick,
    increment_activity_tick,
    parse_as_of,
)
from mnemo.storage.vector_store import VectorStore


def test_parse_as_of_formats() -> None:
    """parse_as_of parses RFC3339, ISO8601 strings and numeric timestamps."""
    assert parse_as_of(None) is None
    assert parse_as_of(1700000000.5) == 1700000000.5
    assert parse_as_of(1700000000) == 1700000000.0
    assert parse_as_of("1700000000.0") == 1700000000.0

    # RFC3339 UTC string
    parsed_utc = parse_as_of("2026-03-29T12:00:00Z")
    assert parsed_utc is not None
    assert isinstance(parsed_utc, float)

    # ISO format with timezone offset
    parsed_tz = parse_as_of("2026-03-29T15:00:00+03:00")
    assert parsed_tz == parsed_utc


def test_activity_ticks_advance_on_mutations(in_memory_db: Database) -> None:
    """Activity ticks increment on mutations and remain unchanged on reads."""
    with in_memory_db.session() as conn:
        assert get_activity_tick(conn) == 0

        t1 = increment_activity_tick(conn)
        assert t1 == 1
        assert get_activity_tick(conn) == 1

        t2 = increment_activity_tick(conn, 3)
        assert t2 == 4
        assert get_activity_tick(conn) == 4


def test_idle_time_does_not_decay_without_ticks(
    in_memory_db: Database,
    tier_manager: TierManager,
    audn_classifier: AUDNClassifier,
) -> None:
    """A fact does not decay from idle time when activity ticks do not advance."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add("System uses SQLite", "architecture", conn)
        initial_salience = fact.salience
        tick_at_add = fact.last_accessed_tick
        assert tick_at_add == 1

        # Simulate passage of 30 wall-clock days without any project activity ticks
        future_now = time.time() + 30 * 86400.0
        res = tier_manager.decay_all(conn, now=future_now)
        assert res["processed"] >= 1

        row = conn.execute("SELECT salience, tier FROM facts WHERE id = ?", (fact.id,)).fetchone()
        assert row["salience"] == initial_salience
        assert row["tier"] == MemoryTier.CORE.value


def test_activity_ticks_cause_decay(
    in_memory_db: Database,
    tier_manager: TierManager,
    audn_classifier: AUDNClassifier,
) -> None:
    """When activity ticks accumulate, facts experience salience decay."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add("Temporary deployment flag", "ops", conn)
        assert get_activity_tick(conn) == 1

        # Advance activity ticks by 50 units (e.g. 50 commits/edits/mutations)
        increment_activity_tick(conn, 50)
        assert get_activity_tick(conn) == 51

        tier_manager.decay_all(conn)

        row = conn.execute("SELECT salience, tier FROM facts WHERE id = ?", (fact.id,)).fetchone()
        assert row["salience"] < 1.0
        # Expected salience: 1.0 * exp(-0.01 * 50 / 1.0) = exp(-0.5) ≈ 0.6065
        assert 0.55 < row["salience"] < 0.65
        assert row["tier"] == MemoryTier.PERIPHERAL.value


def test_spaced_repetition_multiplicative_stability() -> None:
    """Stability factor grows multiplicatively R = R_0 * (1 + alpha)^n."""
    alpha = 0.5
    r0 = 1.0
    delta_ticks = 40
    lambda_param = 0.05

    # n = 0: R = 1.0
    s_n0 = calculate_salience(1.0, delta_ticks=delta_ticks, reinforcement_count=0, lambda_param=lambda_param, alpha=alpha, r0=r0)
    # n = 1: R = 1.5
    s_n1 = calculate_salience(1.0, delta_ticks=delta_ticks, reinforcement_count=1, lambda_param=lambda_param, alpha=alpha, r0=r0)
    # n = 2: R = 2.25
    s_n2 = calculate_salience(1.0, delta_ticks=delta_ticks, reinforcement_count=2, lambda_param=lambda_param, alpha=alpha, r0=r0)
    # n = 3: R = 3.375
    s_n3 = calculate_salience(1.0, delta_ticks=delta_ticks, reinforcement_count=3, lambda_param=lambda_param, alpha=alpha, r0=r0)

    assert s_n0 < s_n1 < s_n2 < s_n3
    # Significant retention difference over 40 ticks
    assert s_n3 > 0.50
    assert s_n0 < 0.20


def test_search_is_idempotent_no_auto_reinforcement(
    in_memory_db: Database,
    audn_classifier: AUDNClassifier,
    hybrid_retriever: HybridRetriever,
) -> None:
    """Repeated searches do NOT bump access_count, reinforcement_count, or salience."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add("PostgreSQL connection pooling configuration", "db", conn)
        fid = fact.id

        # Search 10 times
        for _ in range(10):
            hits = hybrid_retriever.search("PostgreSQL connection", conn, limit=5)
            assert len(hits) >= 1

        row = conn.execute(
            "SELECT access_count, reinforcement_count, salience FROM facts WHERE id = ?",
            (fid,),
        ).fetchone()

        assert row["access_count"] == 0
        assert row["reinforcement_count"] == 0
        assert row["salience"] == 1.0


def test_explicit_acknowledgment_and_reinforcement(
    in_memory_db: Database,
    audn_classifier: AUDNClassifier,
) -> None:
    """Explicit reinforcement increases reinforcement_count, access_count, and sets last_accessed_tick."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add("Rust FFI safety contract", "ffi", conn)
        fid = fact.id
        increment_activity_tick(conn, 10)
        cur_tick = get_activity_tick(conn)

        audn_classifier.execute_noop(fid, conn, boost=0.05)

        row = conn.execute(
            "SELECT access_count, reinforcement_count, last_accessed_tick, salience FROM facts WHERE id = ?",
            (fid,),
        ).fetchone()

        assert row["access_count"] == 1
        assert row["reinforcement_count"] == 1
        assert row["last_accessed_tick"] == cur_tick


def test_retrospective_bitemporal_as_of_query(
    in_memory_db: Database,
    vector_store: VectorStore,
    fts_store: FTSStore,
) -> None:
    """as_of returns point-in-time state of the memory accurately."""
    t0 = 1000.0
    t1 = 2000.0
    t2 = 3000.0

    with in_memory_db.session() as conn:
        # Insert fact valid and ingested at t1
        conn.execute(
            """
            INSERT INTO facts (id, text, category, salience, access_count, tier,
                               last_accessed_at, valid_start, valid_end, ingest_start, ingest_end)
            VALUES ('f_hist', 'Historical architecture decision', 'arch', 1.0, 0, 'working',
                    ?, ?, ?, ?, ?);
            """,
            (t1, t1, t2, t1, None),
        )
        conn.execute(
            "INSERT INTO facts_fts (id, text, category) VALUES ('f_hist', 'Historical architecture decision', 'arch');"
        )
        vec = vector_store.embed_text("Historical architecture decision")
        vector_store.store_embedding(conn, "f_hist", vec)

        # 1. Query at t0 (before the fact existed) -> should return 0 hits
        v_t0 = vector_store.search("Historical architecture", conn, as_of=t0)
        assert len(v_t0) == 0
        f_t0 = fts_store.search("Historical architecture", conn, as_of=t0)
        assert len(f_t0) == 0

        # 2. Query at t1 + 500 (while the fact was valid and ingested) -> returns hit
        v_t1 = vector_store.search("Historical architecture", conn, as_of=t1 + 500.0)
        assert len(v_t1) == 1
        assert v_t1[0][0] == "f_hist"

        f_t1 = fts_store.search("Historical architecture", conn, as_of=t1 + 500.0)
        assert len(f_t1) == 1
        assert f_t1[0][0] == "f_hist"

        # 3. Query at t2 + 100 (after valid_end expired) -> returns 0 hits
        v_t2 = vector_store.search("Historical architecture", conn, as_of=t2 + 100.0)
        assert len(v_t2) == 0


def test_physical_correction_vs_logical_invalidation(
    in_memory_db: Database,
    audn_classifier: AUDNClassifier,
    vector_store: VectorStore,
) -> None:
    """Correction closes ingest_end but keeps valid_end open; preserves historical bitemporality."""
    with in_memory_db.session() as conn:
        fact1 = audn_classifier.execute_add("System uses Redis 6.0", "cache", conn)
        f1_id = fact1.id

        time.sleep(0.02)
        t_before_correct = time.time()
        time.sleep(0.02)

        # User realizes typo: it was Redis 7.0 all along
        corrected_fact = audn_classifier.execute_correct(f1_id, "System uses Redis 7.0", "cache", conn)

        time.sleep(0.02)
        t_after_correct = time.time()

        # Check old fact state
        old_row = conn.execute("SELECT * FROM facts WHERE id = ?", (f1_id,)).fetchone()
        assert old_row["ingest_end"] is not None
        assert old_row["valid_end"] is None  # valid_end untouched!

        # Check new corrected fact state
        new_row = conn.execute("SELECT * FROM facts WHERE id = ?", (corrected_fact.id,)).fetchone()
        assert new_row["ingest_start"] > t_before_correct
        assert new_row["ingest_end"] is None
        assert new_row["valid_start"] == old_row["valid_start"]  # Preserved original validity!
        assert new_row["valid_end"] is None

        # Point-in-time search before correction in system time:
        hits_past = vector_store.search("System uses Redis", conn, as_of=t_before_correct)
        assert len(hits_past) == 1
        assert hits_past[0][0] == f1_id

        # Point-in-time search after correction:
        hits_now = vector_store.search("System uses Redis", conn, as_of=t_after_correct)
        assert len(hits_now) == 1
        assert hits_now[0][0] == corrected_fact.id


def test_archived_tier_fallback_with_warning(
    in_memory_db: Database,
    audn_classifier: AUDNClassifier,
    hybrid_retriever: HybridRetriever,
) -> None:
    """Retriever falls back to ARCHIVED facts when working tier is empty, attaching archived_warning."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add("Archived legacy compiler flag -O3", "compiler", conn)
        # Demote fact directly to ARCHIVED tier with low salience
        conn.execute("UPDATE facts SET tier = 'archived', salience = 0.20 WHERE id = ?", (fact.id,))

        # Search with min_tier=WORKING and include_archived=True
        results = hybrid_retriever.search(
            "compiler flag",
            conn,
            limit=5,
            min_tier=MemoryTier.WORKING,
            include_archived=True,
        )

        assert len(results) == 1
        res = results[0]
        assert res.fact.id == fact.id
        assert res.fact.tier == MemoryTier.ARCHIVED
        assert "archived_warning" in res.fact.metadata
        assert "архиве" in res.fact.metadata["archived_warning"]


def test_stale_ast_drift_penalizes_pinned_facts(
    in_memory_db: Database,
    tier_manager: TierManager,
    audn_classifier: AUDNClassifier,
) -> None:
    """If a pinned fact becomes stale (is_stale=1) due to AST code drift, pinning is ignored and it decays."""
    with in_memory_db.session() as conn:
        fact = audn_classifier.execute_add(
            "Defines function process_data",
            "code",
            conn,
            tier=MemoryTier.CORE,
            metadata={"pinned": True},
        )
        fid = fact.id

        # Mark fact stale due to code drift
        conn.execute("UPDATE facts SET is_stale = 1 WHERE id = ?", (fid,))

        # Advance activity ticks by 30
        increment_activity_tick(conn, 30)
        tier_manager.decay_all(conn)

        row = conn.execute("SELECT salience, tier FROM facts WHERE id = ?", (fid,)).fetchone()
        # Stale facts decay with lambda * 3 = 0.03
        # Salience: 1.0 * exp(-0.03 * 30) = exp(-0.9) ≈ 0.4065 -> demoted out of core!
        assert row["salience"] < 0.60
        assert row["tier"] != MemoryTier.CORE.value
