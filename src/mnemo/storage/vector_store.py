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

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]

        self._model = TextEmbedding(model_name=model_name)
        # Probe dimension
        probe = list(self._model.embed(["hello"]))
        self._dim = len(probe[0])

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        raw = list(self._model.embed(texts))
        return [np.asarray(v, dtype=np.float32) for v in raw]


# ---------------------------------------------------------------------------
# Public: build the best available embedder
# ---------------------------------------------------------------------------


def create_embedder() -> Embedder:
    """Return the best available embedding backend.

    Tries ``fastembed`` first; falls back to deterministic hash embedder.
    """
    try:
        return _FastEmbedEmbedder()  # type: ignore[return-value]
    except Exception:
        return _HashEmbedder()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
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
    ) -> list[tuple[str, float]]:
        """Brute-force cosine-similarity search with bitemporal filter.

        Args:
            query: Natural-language query to embed.
            conn: Active SQLite connection.
            limit: Maximum results.
            valid_at: Point-in-time validity (epoch seconds). Defaults to *now*.

        Returns:
            ``[(fact_id, cosine_score), ...]`` ordered best-first.
        """
        now = valid_at if valid_at is not None else time.time()
        query_vec = self.embed_text(query)

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
