"""SQLite connection factory and schema bootstrapper.

Usage::

    from mnemo.storage.connection import Database

    db = Database()                      # ~/.mnemo/memory.db
    db = Database(db_path=":memory:")     # ephemeral for tests

    with db.session() as conn:
        conn.execute("SELECT 1")
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DIR: Final[Path] = Path.home() / ".mnemo"
_SCHEMA_FILE: Final[Path] = Path(__file__).parent / "schema.sql"


def default_db_path() -> Path:
    """Return ``~/.mnemo/memory.db``, creating the directory if needed."""
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DIR / "memory.db"


# ---------------------------------------------------------------------------
# Low-level connection
# ---------------------------------------------------------------------------


def _apply_pragmas(conn: sqlite3.Connection, *, wal: bool = True) -> None:
    """Apply performance and correctness PRAGMAs.

    WAL mode is skipped for ``:memory:`` databases (not applicable).
    """
    if wal:
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA synchronous  = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA cache_size   = -64000;")  # 64 MiB


def create_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a new SQLite connection with WAL, FK enforcement, and Row factory.

    Args:
        db_path: Filesystem path or ``":memory:"``.

    Returns:
        Configured ``sqlite3.Connection``.
    """
    path_str = str(db_path)
    is_memory = path_str == ":memory:"

    if not is_memory:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        path_str,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        timeout=10.0,
    )
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn, wal=not is_memory)
    return conn


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def _load_schema_sql() -> str:
    """Read the bundled ``schema.sql`` file and return its contents."""
    if not _SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Schema file not found: {_SCHEMA_FILE}. "
            "Ensure the mnemo package is installed correctly."
        )
    return _SCHEMA_FILE.read_text(encoding="utf-8")


def init_db(db_path: str | Path | None = None) -> None:
    """Create all tables, FTS virtual tables, triggers, and indexes via migrations.

    Safe to call multiple times — every migration is idempotent and tracked
    via ``PRAGMA user_version``.

    Args:
        db_path: Target database path.  ``None`` → default path.
    """
    from mnemo.storage.migrations import apply_migrations

    target = str(db_path) if db_path is not None else str(default_db_path())
    conn = create_connection(target)
    try:
        apply_migrations(conn)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# High-level Database manager
# ---------------------------------------------------------------------------


class Database:
    """Convenience wrapper: owns a single connection for ``:memory:`` or
    creates short-lived connections for file-backed databases.

    Attributes:
        db_path: Resolved path (or ``":memory:"``).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        from mnemo.storage.migrations import apply_migrations

        resolved = str(db_path) if db_path is not None else str(default_db_path())
        self.db_path: str = resolved
        self._is_memory: bool = resolved == ":memory:"

        # For in-memory DBs we keep a single persistent connection so the
        # schema (and data) survive across ``session()`` calls.
        self._persistent_conn: sqlite3.Connection | None = None

        if self._is_memory:
            self._persistent_conn = create_connection(":memory:")
            apply_migrations(self._persistent_conn)
            self._persistent_conn.commit()
        else:
            init_db(resolved)

    # -- public API --------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return the active connection (in-memory) or open a new one."""
        if self._is_memory:
            if self._persistent_conn is None:
                raise RuntimeError("Database has been closed.")
            return self._persistent_conn
        return create_connection(self.db_path)

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager: yields a connection, commits on success,
        rolls back on exception.

        For ``:memory:`` databases the persistent connection is reused.
        For file-backed databases a fresh connection is opened and closed.
        """
        if self._is_memory:
            conn = self.get_connection()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        else:
            conn = create_connection(self.db_path)
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def check_integrity(self) -> dict[str, Any]:
        """Run self-diagnostic checks on database health and schema integrity."""
        from mnemo.storage.migrations import check_database_integrity

        with self.session() as conn:
            return check_database_integrity(conn)

    def close(self) -> None:
        """Release the persistent connection (if any)."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None
