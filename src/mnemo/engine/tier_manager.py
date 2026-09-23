"""Tier Manager: salience recomputation and automatic tier migration.

Scans all active facts, recomputes their salience via the Ebbinghaus
decay formula, and migrates them between tiers:

    Core       (S >= 0.90)  -- non-decaying unless explicitly downgraded
    Working    (S >= 0.70)
    Peripheral (S >= 0.40)
    Archived   (S <  0.40)
"""

from __future__ import annotations

import sqlite3
import time

from mnemo.core.decay import calculate_salience, salience_to_tier
from mnemo.core.models import DebtLedgerItem
from mnemo.storage.state import get_activity_tick


class TierManager:
    """Recomputes salience and migrates facts between tiers."""

    def __init__(
        self,
        lambda_param: float = 0.01,
        gamma: float = 0.2,
    ) -> None:
        self._lambda = lambda_param
        self._gamma = gamma

    def decay_all(
        self,
        conn: sqlite3.Connection,
        *,
        now: float | None = None,
        current_tick: int | None = None,
    ) -> dict[str, int]:
        """Recompute salience for every active fact and update tiers.

        Args:
            conn: Active SQLite connection.
            now: Reference timestamp (epoch seconds). Defaults to ``time.time()``.
            current_tick: Optional explicit current activity tick. Defaults to ``get_activity_tick(conn)``.

        Returns:
            ``{"processed": N, "migrated": M}`` counts.
        """
        import json

        current = now if now is not None else time.time()
        cur_tick = current_tick if current_tick is not None else get_activity_tick(conn)

        rows = conn.execute(
            "SELECT id, salience, access_count, last_accessed_at, last_accessed_tick, "
            "reinforcement_count, tier, metadata_json, is_stale, source_type, confidence "
            "FROM facts "
            "WHERE ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)",
            (current,),
        ).fetchall()

        processed = 0
        migrated = 0

        for row in rows:
            old_tier = str(row["tier"])
            metadata = {}
            if "metadata_json" in row.keys() and row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except Exception:
                    pass

            source_type = (
                str(row["source_type"])
                if "source_type" in row.keys() and row["source_type"]
                else "agent"
            )
            confidence = (
                float(row["confidence"])
                if "confidence" in row.keys() and row["confidence"] is not None
                else 1.0
            )
            is_trusted = (source_type != "tool_output") and (confidence >= 0.8)

            is_stale = bool(row["is_stale"]) if "is_stale" in row.keys() and row["is_stale"] else False
            is_pinned = bool(metadata.get("pinned") or metadata.get("permanent"))

            # Pinning protection: untrusted facts or drift-stale facts are not exempt from decay
            if is_pinned and not is_stale and is_trusted:
                continue

            last_accessed = (
                float(row["last_accessed_at"]) if row["last_accessed_at"] is not None else current
            )
            last_tick = (
                int(row["last_accessed_tick"])
                if "last_accessed_tick" in row.keys() and row["last_accessed_tick"] is not None
                else 0
            )
            reinf = (
                0
                if is_stale
                else (
                    int(row["reinforcement_count"])
                    if "reinforcement_count" in row.keys() and row["reinforcement_count"] is not None
                    else 0
                )
            )
            lam = self._lambda * 3.0 if is_stale else self._lambda

            # Activity ticks or fallback to elapsed seconds
            if current_tick is not None or cur_tick > 0 or last_tick > 0:
                delta_ticks = max(0, cur_tick - last_tick)
                new_salience = calculate_salience(
                    s0=float(row["salience"]),
                    delta_ticks=delta_ticks,
                    reinforcement_count=reinf,
                    lambda_param=lam,
                    access_count=int(row["access_count"]),
                )
            else:
                elapsed = max(0.0, current - last_accessed)
                new_salience = calculate_salience(
                    s0=float(row["salience"]),
                    elapsed_seconds=elapsed,
                    access_count=int(row["access_count"]),
                    lambda_param=lam,
                    gamma=self._gamma,
                )

            new_tier = salience_to_tier(new_salience).value
            # Core Tier protection: untrusted facts cannot be in or enter Core Tier
            if not is_trusted and new_tier == "core":
                new_tier = "working"

            conn.execute(
                "UPDATE facts SET salience = ?, tier = ?, last_accessed_at = ?, last_accessed_tick = ? WHERE id = ?",
                (new_salience, new_tier, current, cur_tick, str(row["id"])),
            )
            processed += 1
            if new_tier != old_tier:
                migrated += 1

        return {"processed": processed, "migrated": migrated}

    def detect_debt(
        self,
        conn: sqlite3.Connection,
        *,
        now: float | None = None,
    ) -> list[DebtLedgerItem]:
        """Scan for knowledge / architectural debt.

        Detected patterns:
          1. **decaying_working**: Working-tier facts whose salience < 0.75
             (at risk of demotion).
          2. **stale_core**: Core-tier facts not accessed in > 30 days.
          3. **orphaned_entity**: Entities with zero active relations.
        """
        current = now if now is not None else time.time()
        items: list[DebtLedgerItem] = []

        # 1. Working facts close to demotion
        risky = conn.execute(
            "SELECT id, salience FROM facts "
            "WHERE tier = 'working' AND salience < 0.75 "
            "AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)",
            (current,),
        ).fetchall()
        for r in risky:
            items.append(
                DebtLedgerItem(
                    ceiling="medium",
                    trigger="decaying_working",
                    code_context=f"fact:{r['id']} s={r['salience']:.3f}",
                    created_at=current,
                )
            )

        # 2. Core facts without recent access (>30 days)
        stale_threshold = current - 30 * 86400
        stale = conn.execute(
            "SELECT id, last_accessed_at FROM facts "
            "WHERE tier = 'core' AND last_accessed_at < ? "
            "AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)",
            (stale_threshold, current),
        ).fetchall()
        for r in stale:
            items.append(
                DebtLedgerItem(
                    ceiling="low",
                    trigger="stale_core",
                    code_context=f"fact:{r['id']}",
                    created_at=current,
                )
            )

        # 3. Orphaned entities (no active relations)
        orphans = conn.execute(
            """
            SELECT e.id, e.name FROM entities e
            WHERE e.ingest_end IS NULL
              AND (e.valid_end IS NULL OR e.valid_end > ?)
              AND NOT EXISTS (
                  SELECT 1 FROM relations r
                  WHERE (r.source_id = e.id OR r.target_id = e.id)
                    AND r.ingest_end IS NULL
                    AND (r.valid_end IS NULL OR r.valid_end > ?)
              )
            """,
            (current, current),
        ).fetchall()
        for r in orphans:
            items.append(
                DebtLedgerItem(
                    ceiling="low",
                    trigger="orphaned_entity",
                    code_context=f"entity:{r['id']} name={r['name']}",
                    created_at=current,
                )
            )

        # 4. Stale facts due to AST drift / code changes
        stale_facts = conn.execute(
            """
            SELECT f.id, fel.entity_id, e.name AS entity_name
            FROM facts f
            LEFT JOIN fact_entity_links fel ON fel.fact_id = f.id
            LEFT JOIN entities e ON e.id = fel.entity_id
            WHERE f.is_stale = 1
              AND f.ingest_end IS NULL
              AND (f.valid_end IS NULL OR f.valid_end > ?);
            """,
            (current,),
        ).fetchall()

        seen_stale_facts: set[str] = set()
        for sf in stale_facts:
            fid = str(sf["id"])
            if fid in seen_stale_facts:
                continue
            seen_stale_facts.add(fid)
            ent_name = sf["entity_name"] or "code_entity"
            items.append(
                DebtLedgerItem(
                    ceiling="high",
                    trigger="stale_fact_code_drift",
                    code_context=f"fact:{fid} entity:{ent_name}",
                    created_at=current,
                )
            )

        return items

    def persist_debt(
        self,
        items: list[DebtLedgerItem],
        conn: sqlite3.Connection,
    ) -> int:
        """Write debt items to the ``debt_ledger`` table.

        Returns the number of items inserted.
        """
        for item in items:
            conn.execute(
                "INSERT OR REPLACE INTO debt_ledger (id, ceiling, trigger, code_context, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (item.id, item.ceiling, item.trigger, item.code_context, item.created_at),
            )
        return len(items)
