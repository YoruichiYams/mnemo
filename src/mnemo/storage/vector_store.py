"""Vector embedding store with cosine similarity search.

Embedding backends (tried in order):
  1. ``fastembed`` (high-quality sentence embeddings) — if installed.
  2. Deterministic character-ngram hashing via numpy (always available).

The fallback produces fixed-dimension vectors from text deterministically,
so it works offline without downloading model weights.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Embedding protocol
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Any object that can turn text into a float vector."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]: ...


# ---------------------------------------------------------------------------
# Fallback: deterministic char-ngram hash embedder
# ---------------------------------------------------------------------------

_FALLBACK_DIM = 128


class _HashEmbedder:
    """Deterministic fixed-dimension embedder using character n-gram hashing.

    Each text is converted into a unit-normalised float32 vector of size
    ``dimension`` by hashing overlapping character 3-grams into random
    buckets and normalising.  The result is deterministic and adequate for
    cosine-similarity deduplication but NOT for semantic search.
    """

    def __init__(self, dimension: int = _FALLBACK_DIM) -> None:
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return f"deterministic-hash-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self._dim, dtype=np.float32)
        lower = text.lower().strip()
        if not lower:
            return vec
        for i in range(max(1, len(lower) - 2)):
            gram = lower[i : i + 3]
            h = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# FastEmbed wrapper (optional, high-quality)
# ---------------------------------------------------------------------------


class _FastEmbedEmbedder:
    """Wrapper around ``fastembed.TextEmbedding``."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe dimension
        probe = list(self._model.embed(["hello"]))
        self._dim = len(probe[0])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        raw = list(self._model.embed(texts))
        return [np.asarray(v, dtype=np.float32) for v in raw]


# ---------------------------------------------------------------------------
# Public: build the best available embedder
# ---------------------------------------------------------------------------


def create_embedder(model_name: str | None = None) -> Embedder:
    """Return the best available embedding backend.

    Tries ``fastembed`` first; falls back to deterministic hash embedder.
    """
    try:
        if model_name:
            return _FastEmbedEmbedder(model_name=model_name)  # type: ignore[return-value]
        return _FastEmbedEmbedder()  # type: ignore[return-value]
    except Exception:
        return _HashEmbedder()  # type: ignore[return-value]


def sync_embedding_metadata(
    conn: sqlite3.Connection,
    model_name: str,
    dimension: int,
    *,
    force: bool = False,
) -> None:
    """Record current embedding model and dimension in project_state."""
    row = conn.execute(
        "SELECT embedding_model, embedding_dimension FROM project_state WHERE id = 1"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO project_state (id, embedding_model, embedding_dimension) "
            "VALUES (1, ?, ?)",
            (model_name, dimension),
        )
    elif force or not row["embedding_model"] or row["embedding_model"] in ("deterministic-hash-64", "unknown"):
        conn.execute(
            "UPDATE project_state SET embedding_model = ?, embedding_dimension = ? WHERE id = 1",
            (model_name, dimension),
        )


def get_embedding_metadata(
    conn: sqlite3.Connection,
) -> tuple[str | None, int | None]:
    """Retrieve recorded embedding model and dimension from project_state."""
    row = conn.execute(
        "SELECT embedding_model, embedding_dimension FROM project_state WHERE id = 1"
    ).fetchone()
    if row is None:
        return None, None
    m = row["embedding_model"]
    d = row["embedding_dimension"]
    return (str(m) if m is not None else None, int(d) if d is not None else None)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    if a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Blob serialisation helpers  (float32 array <-> bytes)
# ---------------------------------------------------------------------------


def _vec_to_blob(vec: NDArray[np.float32]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec.tolist())


def _blob_to_vec(blob: bytes) -> NDArray[np.float32]:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """Embedding-based vector search over the ``facts`` table.

    Embeddings are stored as BLOBs in ``facts.embedding_blob``.
    Search is brute-force cosine similarity with bitemporal validity filter.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder or create_embedder()  # type: ignore[assignment]

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return getattr(self._embedder, "model_name", "unknown")

    def delete_embedding(self, conn: sqlite3.Connection, fact_id: str) -> None:
        """Clear the embedding blob for a fact."""
        conn.execute("UPDATE facts SET embedding_blob = NULL WHERE id = ?", (fact_id,))

    # -- write -------------------------------------------------------------

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string."""
        return self._embedder.embed([text])[0]

    def embed_texts(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Embed multiple text strings."""
        return self._embedder.embed(texts)

    def store_embedding(
        self,
        conn: sqlite3.Connection,
        fact_id: str,
        embedding: NDArray[np.float32],
    ) -> None:
        """Persist an embedding blob for an existing fact row."""
        blob = _vec_to_blob(embedding)
        conn.execute(
            "UPDATE facts SET embedding_blob = ? WHERE id = ?",
            (blob, fact_id),
        )

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
        valid_at: float | None = None,
        as_of: str | float | None = None,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine-similarity search with bitemporal filter.

        Args:
            query: Natural-language query to embed.
            conn: Active SQLite connection.
            limit: Maximum results.
            valid_at: Deprecated point-in-time validity (epoch seconds). Defaults to *now*.
            as_of: Point-in-time for bitemporal point-in-time search (RFC3339 or epoch).

        Returns:
            ``[(fact_id, cosine_score), ...]`` ordered best-first.
        """
        from mnemo.storage.state import parse_as_of

        target_time = parse_as_of(as_of) if as_of is not None else (valid_at if valid_at is not None else None)
        query_vec = self.embed_text(query)

        if target_time is not None:
            sql = """
                SELECT id, embedding_blob FROM facts
                WHERE embedding_blob IS NOT NULL
                  AND valid_start  <= ?
                  AND (valid_end   IS NULL OR valid_end  > ?)
                  AND ingest_start <= ?
                  AND (ingest_end   IS NULL OR ingest_end > ?);
            """
            rows = conn.execute(sql, (target_time, target_time, target_time, target_time)).fetchall()
        else:
            now = time.time()
            sql = """
                SELECT id, embedding_blob FROM facts
                WHERE embedding_blob IS NOT NULL
                  AND valid_start  <= ?
                  AND (valid_end   IS NULL OR valid_end  > ?)
                  AND ingest_end   IS NULL;
            """
            rows = conn.execute(sql, (now, now)).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            row_vec = _blob_to_vec(row["embedding_blob"])
            sim = cosine_similarity(query_vec, row_vec)
            if sim > 0.0:
                scored.append((str(row["id"]), round(sim, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
