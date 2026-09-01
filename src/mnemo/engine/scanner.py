"""Autonomous AST project scanner and entity linker for Mnemo.

Parses Python AST (and JS/TS structures) to automatically discover modules,
classes, internal imports, and external dependencies. Caches file hashes
for sub-second incremental scanning and automatically links memory facts
to architectural entities.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

# Ignored directories for repository scanning
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".agents",
    ".gemini",
    ".idea",
    ".vscode",
    ".tox",
}

# Standard library modules to distinguish from third-party dependencies
_STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "glob",
    "hashlib",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "queue",
    "re",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "weakref",
    "zipfile",
}


def _compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hex digest of file contents."""
    return hashlib.sha256(content).hexdigest()


class ProjectScanner:
    """Lightweight incremental AST code scanner for project repositories."""

    def __init__(self, root_path: str | Path | None = None) -> None:
        self.root_path: Path = Path(root_path).resolve() if root_path else Path.cwd().resolve()

    def scan(
        self,
        conn: sqlite3.Connection,
        *,
        target_dir: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Perform an incremental AST scan and update knowledge graph entities and relations.

        Args:
            conn: Active SQLite connection.
            target_dir: Directory to scan (defaults to project root).
            force: If True, re-scans all files regardless of cache.

        Returns:
            Dictionary with scan metrics (scanned, skipped, entities, relations).
        """
        scan_root = Path(target_dir).resolve() if target_dir else self.root_path
        if not scan_root.exists() or not scan_root.is_dir():
            return {
                "scanned_files": 0,
                "skipped_files": 0,
                "entities_added": 0,
                "relations_added": 0,
            }

        scanned_count = 0
        skipped_count = 0
        entities_added = 0
        relations_added = 0
        now = time.time()

        # 1. Load existing file cache
        cache_rows = conn.execute(
            "SELECT file_path, mtime, sha256 FROM file_scan_cache;"
        ).fetchall()
        scan_cache: dict[str, tuple[float, str]] = {
            r["file_path"]: (float(r["mtime"]), str(r["sha256"])) for r in cache_rows
        }

        # 2. Walk directory
        seen_rel_paths: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(scan_root):
            # Prune excluded directories in-place (case-insensitive)
            dirnames[:] = [
                d
                for d in dirnames
                if d.lower() not in DEFAULT_EXCLUDED_DIRS and not d.startswith(".")
            ]

            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                    continue

                full_path = Path(dirpath) / fname
                rel_path = str(full_path.relative_to(scan_root)).replace("\\", "/")
                seen_rel_paths.add(rel_path)

                try:
                    stat = full_path.stat()
                    file_mtime = stat.st_mtime
                    file_bytes = full_path.read_bytes()
                    file_sha = _compute_sha256(file_bytes)
                except Exception:
                    continue

                # Check incremental cache
                if not force and rel_path in scan_cache:
                    cached_mtime, cached_sha = scan_cache[rel_path]
                    if cached_mtime == file_mtime and cached_sha == file_sha:
                        skipped_count += 1
                        continue

                # Parse file
                discovered_entities, discovered_relations = self._parse_file(
                    full_path, rel_path, file_bytes, ext
                )

                # Persist discovered entities and relations
                ent_map: dict[str, str] = {}
                for ent_name, ent_type in discovered_entities:
                    eid = self._upsert_entity(conn, ent_name, ent_type, now)
                    ent_map[ent_name] = eid
                    entities_added += 1

                for src_name, tgt_name, rel_type in discovered_relations:
                    src_id = ent_map.get(src_name) or self._upsert_entity(
                        conn, src_name, "module", now
                    )
                    tgt_id = ent_map.get(tgt_name) or self._upsert_entity(
                        conn, tgt_name, "external_dep", now
                    )
                    if self._insert_relation_if_new(conn, src_id, tgt_id, rel_type, now):
                        relations_added += 1

                # Update file cache
                conn.execute(
                    """
                    INSERT INTO file_scan_cache (file_path, mtime, sha256, entity_count, relation_count, scanned_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        mtime = excluded.mtime,
                        sha256 = excluded.sha256,
                        entity_count = excluded.entity_count,
                        relation_count = excluded.relation_count,
                        scanned_at = excluded.scanned_at;
                    """,
                    (
                        rel_path,
                        file_mtime,
                        file_sha,
                        len(discovered_entities),
                        len(discovered_relations),
                        now,
                    ),
                )
                scanned_count += 1

        # 3. Deleted files reconciliation
        deleted_paths = set(scan_cache.keys()) - seen_rel_paths
        for del_path in deleted_paths:
            del_parts = list(Path(del_path).with_suffix("").parts)
            if del_parts and del_parts[0] in {"src", "lib", "app"}:
                del_parts = del_parts[1:]
            del_module = ".".join(del_parts) if del_parts else del_path

            # Soft delete entities for deleted files
            conn.execute(
                """
                UPDATE entities SET valid_end = ?
                WHERE (name = ? OR name LIKE ?) AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?);
                """,
                (now, del_module, f"{del_module}.%", now),
            )
            # Remove from cache
            conn.execute("DELETE FROM file_scan_cache WHERE file_path = ?;", (del_path,))

        return {
            "scanned_files": scanned_count,
            "skipped_files": skipped_count,
            "entities_added": entities_added,
            "relations_added": relations_added,
        }

    def _parse_file(
        self, file_path: Path, rel_path: str, content_bytes: bytes, ext: str
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        """Parse source code and extract entities and relations."""
        entities: list[tuple[str, str]] = []
        relations: list[tuple[str, str, str]] = []

        # Derive module name from relative path
        parts = list(Path(rel_path).with_suffix("").parts)
        if parts and parts[0] in {"src", "lib", "app"}:
            parts = parts[1:]
        module_name = ".".join(parts) if parts else rel_path

        entities.append((module_name, "module"))

        if ext == ".py":
            try:
                tree = ast.parse(
                    content_bytes.decode("utf-8", errors="replace"), filename=str(file_path)
                )
            except SyntaxError:
                return entities, relations

            for node in ast.walk(tree):
                # 1. Classes
                if isinstance(node, ast.ClassDef):
                    class_name = f"{module_name}.{node.name}"
                    entities.append((class_name, "class"))
                    relations.append((module_name, class_name, "CONTAINS"))

                # 2. Imports (import foo, import foo.bar as bar)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        top_pkg = alias.name.split(".")[0]
                        ent_type = (
                            "module"
                            if top_pkg in parts
                            else ("stdlib" if top_pkg in _STDLIB_MODULES else "external_dep")
                        )
                        entities.append((alias.name, ent_type))
                        relations.append((module_name, alias.name, "IMPORTS"))

                # 3. ImportFrom (from foo import bar, from . import bar, from ..pkg import bar)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        # Relative import
                        lvl = node.level
                        base_parts = parts[:-lvl] if len(parts) >= lvl else []
                        if node.module:
                            rel_target = (
                                ".".join(base_parts + [node.module]) if base_parts else node.module
                            )
                        else:
                            rel_target = ".".join(base_parts) if base_parts else module_name

                        entities.append((rel_target, "module"))
                        relations.append((module_name, rel_target, "DEPENDS_ON"))
                        if not node.module:
                            for alias in node.names:
                                sub_name = (
                                    f"{rel_target}.{alias.name}" if rel_target else alias.name
                                )
                                entities.append((sub_name, "module"))
                                relations.append((module_name, sub_name, "IMPORTS"))
                    elif node.module:
                        top_pkg = node.module.split(".")[0]
                        ent_type = (
                            "module"
                            if top_pkg in parts
                            else ("stdlib" if top_pkg in _STDLIB_MODULES else "external_dep")
                        )
                        entities.append((node.module, ent_type))
                        relations.append((module_name, node.module, "DEPENDS_ON"))

        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            # Lightweight regex extraction for JS/TS
            text = content_bytes.decode("utf-8", errors="replace")

            # Extract imports: import ... from 'pkg' or require('pkg')
            import_matches = re.findall(
                r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))", text
            )
            for m1, m2 in import_matches:
                dep = m1 or m2
                if dep:
                    dep_type = "module" if dep.startswith(".") else "external_dep"
                    entities.append((dep, dep_type))
                    relations.append((module_name, dep, "DEPENDS_ON"))

            # Extract classes
            class_matches = re.findall(r"\bclass\s+([A-Za-z0-9_$]+)", text)
            for cls in class_matches:
                class_name = f"{module_name}.{cls}"
                entities.append((class_name, "class"))
                relations.append((module_name, class_name, "CONTAINS"))

        return entities, relations

    def _upsert_entity(
        self, conn: sqlite3.Connection, name: str, entity_type: str, now: float
    ) -> str:
        """Find existing active entity or create a new one."""
        row = conn.execute(
            """
            SELECT id FROM entities
            WHERE name = ? AND entity_type = ? AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)
            LIMIT 1;
            """,
            (name, entity_type, now),
        ).fetchone()

        if row:
            conn.execute(
                "UPDATE entities SET last_accessed_at = ? WHERE id = ?;",
                (now, row["id"]),
            )
            return str(row["id"])

        new_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO entities (
                id, name, entity_type, properties_json, salience, access_count,
                last_accessed_at, valid_start, valid_end, ingest_start, ingest_end
            ) VALUES (?, ?, ?, '{}', 1.0, 0, ?, ?, NULL, ?, NULL);
            """,
            (new_id, name, entity_type, now, now, now),
        )
        return new_id

    def _insert_relation_if_new(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        target_id: str,
        relation_type: str,
        now: float,
    ) -> bool:
        """Insert directed relation edge if an active edge does not already exist."""
        if source_id == target_id:
            return False

        row = conn.execute(
            """
            SELECT id FROM relations
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
              AND ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?)
            LIMIT 1;
            """,
            (source_id, target_id, relation_type, now),
        ).fetchone()

        if row:
            return False

        rel_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO relations (
                id, source_id, target_id, relation_type, weight,
                valid_start, valid_end, ingest_start, ingest_end
            ) VALUES (?, ?, ?, ?, 1.0, ?, NULL, ?, NULL);
            """,
            (rel_id, source_id, target_id, relation_type, now, now),
        )
        return True


# ---------------------------------------------------------------------------
# Auto-Linking helper for facts
# ---------------------------------------------------------------------------


def link_fact_to_entities(
    fact_id: str,
    fact_text: str,
    conn: sqlite3.Connection,
    *,
    now: float | None = None,
) -> list[str]:
    """Scan active entities and link any mentioned entities to the given fact.

    Args:
        fact_id: ID of the inserted/updated fact.
        fact_text: Text of the fact.
        conn: Active SQLite connection.
        now: Reference timestamp.

    Returns:
        List of linked entity IDs.
    """
    current_time = now if now is not None else time.time()
    text_lower = fact_text.lower()

    # Find active entities
    entities = conn.execute(
        """
        SELECT id, name FROM entities
        WHERE ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?);
        """,
        (current_time,),
    ).fetchall()

    linked_ids: list[str] = []
    for ent in entities:
        ent_name = str(ent["name"])
        # Match if entity name or short name is mentioned in the text
        base_name = ent_name.split(".")[-1].lower()
        if len(base_name) >= 3 and (base_name in text_lower or ent_name.lower() in text_lower):
            eid = str(ent["id"])
            linked_ids.append(eid)
            # Create link relation (Entity -> Fact)
            existing = conn.execute(
                """
                SELECT id FROM relations
                WHERE source_id = ? AND target_id = ? AND relation_type = 'MENTIONS'
                  AND ingest_end IS NULL;
                """,
                (eid, fact_id),
            ).fetchone()

            if not existing:
                conn.execute(
                    """
                    INSERT INTO relations (
                        id, source_id, target_id, relation_type, weight,
                        valid_start, valid_end, ingest_start, ingest_end
                    ) VALUES (?, ?, ?, 'MENTIONS', 1.0, ?, NULL, ?, NULL);
                    """,
                    (str(uuid.uuid4()), eid, fact_id, current_time, current_time),
                )

    return linked_ids
