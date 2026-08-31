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
from mnemo.core.models import DebtLedgerItem, MemoryTier


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
    ) -> dict[str, int]:
        """Recompute salience for every active fact and update tiers.

        Core-tier facts are **not** decayed unless they are explicitly
        demoted first (they represent permanent architectural rules).

        Args:
            conn: Active SQLite connection.
            now: Reference timestamp (epoch seconds). Defaults to ``time.time()``.

        Returns:
            ``{"processed": N, "migrated": M}`` counts.
        """
        current = now if now is not None else time.time()

        rows = conn.execute(
            "SELECT id, salience, access_count, last_accessed_at, tier "
            "FROM facts "
            "WHERE ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)",
            (current,),
        ).fetchall()

        processed = 0
        migrated = 0

        for row in rows:
            old_tier = str(row["tier"])
            if old_tier == MemoryTier.CORE.value:
                continue  # core facts don't decay

            elapsed = max(0.0, current - float(row["last_accessed_at"]))
            new_salience = calculate_salience(
                s0=float(row["salience"]),
                elapsed_seconds=elapsed,
                access_count=int(row["access_count"]),
                lambda_param=self._lambda,
                gamma=self._gamma,
            )
            new_tier = salience_to_tier(new_salience).value

            conn.execute(
                "UPDATE facts SET salience = ?, tier = ? WHERE id = ?",
                (new_salience, new_tier, str(row["id"])),
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
