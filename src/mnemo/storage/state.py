"""Project state management for Activity Ticks and temporal helpers."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime


def get_activity_tick(conn: sqlite3.Connection) -> int:
    """Retrieve the current global activity tick of the project.

    Args:
        conn: Active SQLite connection.

    Returns:
        The current integer activity tick (defaults to 0).
    """
    try:
        row = conn.execute("SELECT activity_tick FROM project_state WHERE id = 1;").fetchone()
        if row is None:
            return 0
        return int(row["activity_tick"]) if isinstance(row, sqlite3.Row) else int(row[0])
    except Exception:
        return 0


def increment_activity_tick(conn: sqlite3.Connection, delta: int = 1) -> int:
    """Increment the global activity tick counter atomically.

    Args:
        conn: Active SQLite connection.
        delta: Amount to increment (default 1).

    Returns:
        The new activity tick value after increment.
    """
    now = time.time()
    try:
        conn.execute(
            """
            INSERT INTO project_state (id, activity_tick, updated_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                activity_tick = project_state.activity_tick + excluded.activity_tick,
                updated_at = excluded.updated_at;
            """,
            (delta, now),
        )
    except Exception:
        # Table might not exist yet during initial boot before migrations
        pass
    return get_activity_tick(conn)


def parse_as_of(as_of: str | float | int | None) -> float | None:
    """Parse an as_of parameter (RFC3339 string, ISO format, or epoch timestamp).

    Args:
        as_of: Point-in-time representation or None.

    Returns:
        Unix timestamp in epoch seconds (float) or None.
    """
    if as_of is None:
        return None
    if isinstance(as_of, (int, float)):
        return float(as_of)
    if isinstance(as_of, str):
        as_of_str = as_of.strip()
        if not as_of_str:
            return None
        try:
            return float(as_of_str)
        except ValueError:
            pass
        clean = as_of_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(clean)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None
    return None
