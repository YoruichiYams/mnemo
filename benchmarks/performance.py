"""Experiment 3: Latency & Scalability Profiling.

Measures execution latency percentiles (p50, p95, p99 in milliseconds) for:
  - mnemo_remember / AUDN classification & insertion
  - FTS5 BM25 search
  - Vector cosine search
  - Graph traversal CTE
  - Full mnemo_search (3 channels + RRF)
  - mnemo_purge (physical destruction & audit tombstone)
  - Full project scan and incremental diff scan
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.dataset_generator import BenchmarkDataset
from mnemo.engine.audn import AUDNClassifier
from mnemo.engine.retriever import HybridRetriever
from mnemo.engine.scanner import ProjectScanner
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, create_embedder


def _percentiles(samples: list[float]) -> dict[str, float]:
    """Calculate p50, p95, p99, mean, and throughput from a list of millisecond latencies."""
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "ops_sec": 0.0}

    sorted_s = sorted(samples)
    n = len(sorted_s)

    def _perc(p: float) -> float:
        idx = max(0, min(n - 1, math.ceil(p * n) - 1))
        return sorted_s[idx]

    p50 = _perc(0.50)
    p95 = _perc(0.95)
    p99 = _perc(0.99)
    mean_val = sum(sorted_s) / n
    ops_sec = (1000.0 / mean_val) if mean_val > 0 else 0.0

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "mean": round(mean_val, 3),
        "ops_sec": round(ops_sec, 1),
    }


def run_performance_benchmarks(
    dataset: BenchmarkDataset,
    iterations: int = 50,
) -> dict[str, Any]:
    """Profile latency across core operations on the benchmark dataset."""
    vs = VectorStore(create_embedder())
    fts = FTSStore()
    graph = GraphStore()
    retriever = HybridRetriever(vs, fts, graph)
    audn = AUDNClassifier(vs)

    latencies: dict[str, list[float]] = {
        "remember_insert": [],
        "fts_search": [],
        "vector_search": [],
        "graph_cte": [],
        "full_hybrid_search": [],
        "purge_redaction": [],
    }

    test_queries = [q.query for q in dataset.queries[:10]]
    if not test_queries:
        test_queries = ["database connection timeout", "OrderWorkflowService"]

    with dataset.db.session() as conn:
        # 1. Profile FTS5 BM25 Search
        for i in range(iterations):
            q = test_queries[i % len(test_queries)]
            t_start = time.perf_counter()
            fts.search(q, conn, limit=10)
            latencies["fts_search"].append((time.perf_counter() - t_start) * 1000)

        # 2. Profile Vector Cosine Search
        for i in range(iterations):
            q = test_queries[i % len(test_queries)]
            t_start = time.perf_counter()
            vs.search(q, conn, limit=10)
            latencies["vector_search"].append((time.perf_counter() - t_start) * 1000)

        # 3. Profile Graph Traversal CTE
        for i in range(iterations):
            q = test_queries[i % len(test_queries)]
            t_start = time.perf_counter()
            graph.find_related_facts(q, conn, limit=10)
            latencies["graph_cte"].append((time.perf_counter() - t_start) * 1000)

        # 4. Profile Full Hybrid Search (3 Channels + RRF)
        for i in range(iterations):
            q = test_queries[i % len(test_queries)]
            t_start = time.perf_counter()
            retriever.search(q, conn, limit=10)
            latencies["full_hybrid_search"].append((time.perf_counter() - t_start) * 1000)

        # 5. Profile mnemo_remember (AUDN classify + add)
        purgeable_fact_ids: list[str] = []
        for i in range(iterations):
            t_start = time.perf_counter()
            f = audn.execute_add(
                f"Synthetic latency benchmark fact number {i} with unique payload token",
                "perf",
                conn,
            )
            latencies["remember_insert"].append((time.perf_counter() - t_start) * 1000)
            purgeable_fact_ids.append(f.id)

        # 6. Profile mnemo_purge
        for fid in purgeable_fact_ids:
            t_start = time.perf_counter()
            audn.execute_purge(fid, conn, reason="latency_benchmark")
            latencies["purge_redaction"].append((time.perf_counter() - t_start) * 1000)

    # 7. Profile Project Scanning (Full vs Incremental)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create 5 synthetic python modules
        for i in range(10):
            p = tmp_path / f"service_{i}.py"
            p.write_text(
                f"class Service{i}:\n"
                f"    def do_action_{i}(self):\n"
                f"        return {i} * 2\n",
                encoding="utf-8",
            )

        scanner = ProjectScanner(root_path=str(tmp_path))
        with dataset.db.session() as conn:
            # Full scan
            t_start = time.perf_counter()
            scanner.scan(conn, force=True)
            full_scan_ms = (time.perf_counter() - t_start) * 1000

            # Incremental scan (no changes)
            t_start = time.perf_counter()
            scanner.scan(conn, force=False)
            inc_scan_ms = (time.perf_counter() - t_start) * 1000

    profile_summary = {op: _percentiles(samples) for op, samples in latencies.items()}
    profile_summary["scan_full_ms"] = round(full_scan_ms, 2)
    profile_summary["scan_incremental_ms"] = round(inc_scan_ms, 2)
    profile_summary["scan_speedup"] = (
        round(full_scan_ms / max(0.001, inc_scan_ms), 1)
        if inc_scan_ms > 0
        else 1.0
    )

    return {
        "scale": dataset.scale,
        "facts_count": dataset.facts_count,
        "entities_count": dataset.entities_count,
        "relations_count": dataset.relations_count,
        "iterations": iterations,
        "latencies": profile_summary,
    }
