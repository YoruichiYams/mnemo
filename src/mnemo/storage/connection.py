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
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, TypeVar

T = TypeVar("T")
_BUSY_DELAYS: Final[tuple[float, ...]] = (0.05, 0.1, 0.2, 0.4, 0.8)

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

    for i, delay in enumerate(_BUSY_DELAYS):
        try:
            conn = sqlite3.connect(
                path_str,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=10.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            _apply_pragmas(conn, wal=not is_memory)
            return conn
        except sqlite3.OperationalError as e:
            err = str(e).lower()
            if ("locked" in err or "busy" in err) and i < len(_BUSY_DELAYS) - 1:
                time.sleep(delay)
                continue
            raise

    # Fallback attempt
    conn = sqlite3.connect(
        path_str,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        timeout=10.0,
        check_same_thread=False,
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
        self._lock = threading.Lock()

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

    @staticmethod
    def _commit_with_retry(conn: sqlite3.Connection) -> None:
        """Commit active transaction with exponential backoff on busy/locked errors."""
        for i, delay in enumerate(_BUSY_DELAYS):
            try:
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                err = str(e).lower()
                if ("locked" in err or "busy" in err) and i < len(_BUSY_DELAYS) - 1:
                    time.sleep(delay)
                    continue
                raise

    @staticmethod
    def _begin_immediate_with_retry(conn: sqlite3.Connection) -> None:
        """Begin immediate transaction with exponential backoff on busy/locked errors."""
        for i, delay in enumerate(_BUSY_DELAYS):
            try:
                conn.execute("BEGIN IMMEDIATE;")
                return
            except sqlite3.OperationalError as e:
                err = str(e).lower()
                if ("locked" in err or "busy" in err) and i < len(_BUSY_DELAYS) - 1:
                    time.sleep(delay)
                    continue
                raise
        conn.execute("BEGIN IMMEDIATE;")

    @contextmanager
    def session(self, *, readonly: bool = False) -> Generator[sqlite3.Connection, None, None]:
        """Context manager: yields a connection, commits on success,
        rolls back on exception.

        For ``:memory:`` databases the persistent connection is reused under a thread lock.
        For file-backed databases a fresh connection is opened. Writing sessions are synchronized
        via ``BEGIN IMMEDIATE`` with exponential backoff to eliminate SHARED->RESERVED deadlocks.
        Read-only sessions skip transaction acquisition, allowing concurrent non-blocking reads.
        """
        if self._is_memory:
            with self._lock:
                conn = self.get_connection()
                try:
                    yield conn
                    if not readonly:
                        self._commit_with_retry(conn)
                except BaseException:
                    conn.rollback()
                    raise
        else:
            conn = create_connection(self.db_path)
            try:
                if not readonly:
                    self._begin_immediate_with_retry(conn)
                yield conn
                if not readonly:
                    self._commit_with_retry(conn)
            except BaseException:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def execute_with_retry(self, fn: Callable[[sqlite3.Connection], T], max_retries: int = 5) -> T:
        """Execute a callback in a session with full transaction retry on database lock."""
        for attempt in range(max_retries):
            try:
                with self.session() as conn:
                    return fn(conn)
            except sqlite3.OperationalError as e:
                err = str(e).lower()
                if ("locked" in err or "busy" in err) and attempt < max_retries - 1:
                    delay = _BUSY_DELAYS[min(attempt, len(_BUSY_DELAYS) - 1)]
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

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
