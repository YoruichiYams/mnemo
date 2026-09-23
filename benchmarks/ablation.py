"""Experiment 1: Channel Ablation Study (Dense, Sparse, Graph, RRF).

Compares retrieval performance across 5 configurations:
  Config A: Dense Only (Vector)
  Config B: Sparse Only (FTS5 BM25 with code tokenization)
  Config C: Graph Only (AST neighbourhood CTE via fact_entity_links)
  Config D: Dense + Sparse (Standard Hybrid RAG)
  Config E: Mnemo Full (Dense + Sparse + Graph + RRF)
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

from benchmarks.dataset_generator import BenchmarkDataset
from benchmarks.metrics import aggregate_metrics, evaluate_ranking
from mnemo.engine.retriever import HybridRetriever
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, create_embedder


@dataclasses.dataclass
class AblationConfig:
    name: str
    label: str
    channels: tuple[str, ...]


CONFIGS: list[AblationConfig] = [
    AblationConfig(name="dense_only", label="Config A (Dense Only)", channels=("vector",)),
    AblationConfig(name="sparse_only", label="Config B (Sparse Only)", channels=("fts",)),
    AblationConfig(name="graph_only", label="Config C (Graph Only)", channels=("graph",)),
    AblationConfig(name="dense_sparse", label="Config D (Dense + Sparse)", channels=("vector", "fts")),
    AblationConfig(name="mnemo_full", label="Config E (Mnemo Full: 3-Channel + RRF)", channels=("vector", "fts", "graph")),
]


def run_ablation_study(dataset: BenchmarkDataset) -> dict[str, Any]:
    """Execute ablation study comparing the 5 retrieval configurations."""
    vs = VectorStore(create_embedder())
    fts = FTSStore()
    graph = GraphStore()
    retriever = HybridRetriever(vs, fts, graph)

    results: dict[str, Any] = {}

    with dataset.db.session() as conn:
        for cfg in CONFIGS:
            group_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
            all_metrics: list[dict[str, float]] = []

            for q in dataset.queries:
                # Execute search for this configuration
                hits = retriever.search(
                    q.query,
                    conn,
                    limit=5,
                    as_of=q.as_of,
                    channels=cfg.channels,
                )
                ranked_ids = [h.fact.id for h in hits]
                m = evaluate_ranking(ranked_ids, q.target_fact_ids)

                group_metrics[q.group].append(m)
                all_metrics.append(m)

            aggregated_by_group = {grp: aggregate_metrics(metrics) for grp, metrics in group_metrics.items()}
            aggregated_overall = aggregate_metrics(all_metrics)

            results[cfg.name] = {
                "label": cfg.label,
                "channels": cfg.channels,
                "overall": aggregated_overall,
                "by_group": aggregated_by_group,
                "query_count": len(all_metrics),
            }

    return results
