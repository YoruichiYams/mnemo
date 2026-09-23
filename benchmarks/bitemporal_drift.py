"""Experiment 2: Bitemporal Accuracy & AST Code Drift Benchmark.

Evaluates:
  1. Point-in-time historical reconstruction accuracy (as_of)
  2. AST Code Drift detection rate and warning propagation
"""

from __future__ import annotations

from typing import Any

from benchmarks.dataset_generator import BenchmarkDataset
from mnemo.engine.retriever import HybridRetriever
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, create_embedder


def run_bitemporal_and_drift_benchmark(dataset: BenchmarkDataset) -> dict[str, Any]:
    """Execute Bitemporal and AST drift evaluation."""
    vs = VectorStore(create_embedder())
    fts = FTSStore()
    graph = GraphStore()
    retriever = HybridRetriever(vs, fts, graph)

    bitemporal_queries = [q for q in dataset.queries if q.group == "bitemporal" and q.as_of is not None]
    drift_queries = [q for q in dataset.queries if q.expected_stale]

    correct_reconstructions = 0
    total_bitemporal = len(bitemporal_queries)

    stale_detected = 0
    warning_emitted = 0
    total_drift = len(drift_queries)

    with dataset.db.session() as conn:
        # 1. Bitemporal reconstruction evaluation
        reconstruction_details: list[dict[str, Any]] = []
        for q in bitemporal_queries:
            hits = retriever.search(q.query, conn, limit=3, as_of=q.as_of)
            top_id = hits[0].fact.id if hits else None
            is_match = top_id in q.target_fact_ids
            if is_match:
                correct_reconstructions += 1

            reconstruction_details.append(
                {
                    "query": q.query,
                    "as_of": q.as_of,
                    "target_id": q.target_fact_ids[0],
                    "returned_id": top_id,
                    "is_correct": is_match,
                }
            )

        # 2. AST Drift evaluation
        drift_details: list[dict[str, Any]] = []
        for q in drift_queries:
            hits = retriever.search(q.query, conn, limit=3)
            found_stale = False
            found_warning = False

            for h in hits:
                if h.fact.id in q.target_fact_ids:
                    if h.fact.is_stale:
                        found_stale = True
                    if h.fact.metadata.get("warning") == "stale_code_drift":
                        found_warning = True

            if found_stale:
                stale_detected += 1
            if found_warning:
                warning_emitted += 1

            drift_details.append(
                {
                    "query": q.query,
                    "target_id": q.target_fact_ids[0],
                    "stale_flagged": found_stale,
                    "warning_emitted": found_warning,
                }
            )

        # Check raw database state for drift
        stale_rows_count = conn.execute("SELECT COUNT(*) FROM facts WHERE is_stale = 1").fetchone()[0]

    bitemporal_accuracy = (
        round((correct_reconstructions / total_bitemporal) * 100, 2)
        if total_bitemporal > 0
        else 100.0
    )
    stale_detection_rate = (
        round((stale_detected / total_drift) * 100, 2)
        if total_drift > 0
        else 100.0
    )
    warning_emission_rate = (
        round((warning_emitted / total_drift) * 100, 2)
        if total_drift > 0
        else 100.0
    )

    return {
        "bitemporal_accuracy_pct": bitemporal_accuracy,
        "total_bitemporal_queries": total_bitemporal,
        "correct_bitemporal_queries": correct_reconstructions,
        "reconstruction_details": reconstruction_details,
        "stale_detection_rate_pct": stale_detection_rate,
        "warning_emission_rate_pct": warning_emission_rate,
        "total_drift_queries": total_drift,
        "stale_facts_in_db": stale_rows_count,
        "drift_details": drift_details,
    }
