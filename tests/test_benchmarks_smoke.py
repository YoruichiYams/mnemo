"""Smoke and regression tests for Mnemo Benchmark Suite."""

from __future__ import annotations

from benchmarks.ablation import run_ablation_study
from benchmarks.bitemporal_drift import run_bitemporal_and_drift_benchmark
from benchmarks.dataset_generator import generate_dataset
from benchmarks.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    evaluate_ranking,
)
from benchmarks.performance import run_performance_benchmarks
from benchmarks.run_benchmarks import generate_markdown_report
from mnemo.storage.connection import Database


class TestBenchmarkMetrics:
    """Validate ranking evaluation metrics."""

    def test_hit_at_k(self) -> None:
        ranked = ["f1", "f2", "f3", "f4", "f5"]
        gt = {"f3"}

        assert calculate_hit_at_k(ranked, gt, k=1) == 0.0
        assert calculate_hit_at_k(ranked, gt, k=2) == 0.0
        assert calculate_hit_at_k(ranked, gt, k=3) == 1.0
        assert calculate_hit_at_k(ranked, gt, k=5) == 1.0

    def test_mrr(self) -> None:
        ranked = ["f1", "f2", "f3"]
        assert calculate_mrr(ranked, {"f1"}) == 1.0
        assert calculate_mrr(ranked, {"f2"}) == 0.5
        assert calculate_mrr(ranked, {"f3"}) == round(1.0 / 3, 4) or abs(calculate_mrr(ranked, {"f3"}) - 1.0 / 3) < 1e-4
        assert calculate_mrr(ranked, {"f99"}) == 0.0

    def test_ndcg_at_k(self) -> None:
        ranked = ["f1", "f2", "f3", "f4", "f5"]
        # Perfect ranking: relevant item at rank 1
        assert calculate_ndcg_at_k(ranked, {"f1"}, k=5) == 1.0
        # Lower rank item
        ndcg_f2 = calculate_ndcg_at_k(ranked, {"f2"}, k=5)
        assert 0.0 < ndcg_f2 < 1.0

    def test_evaluate_ranking_bundle(self) -> None:
        res = evaluate_ranking(["f1", "f2", "f3"], {"f2"})
        assert "hit@1" in res
        assert "hit@3" in res
        assert "hit@5" in res
        assert "mrr" in res
        assert "ndcg@5" in res
        assert res["hit@1"] == 0.0
        assert res["hit@3"] == 1.0
        assert res["mrr"] == 0.5


class TestBenchmarkPipelineSmoke:
    """Smoke test full benchmark pipeline execution on ephemeral in-memory DB."""

    def test_dataset_generation_and_ablation(self, in_memory_db: Database) -> None:
        dataset = generate_dataset(in_memory_db, scale="small")
        assert dataset.facts_count >= 100
        assert dataset.entities_count >= 30
        assert dataset.relations_count >= 30
        assert len(dataset.queries) >= 30

        # Run ablation study
        ablation = run_ablation_study(dataset)
        assert "mnemo_full" in ablation
        assert "dense_only" in ablation
        assert "sparse_only" in ablation
        assert "graph_only" in ablation
        assert "dense_sparse" in ablation

        full_mrr = ablation["mnemo_full"]["overall"]["mrr"]
        assert full_mrr > 0.0

    def test_bitemporal_and_drift_benchmark(self, in_memory_db: Database) -> None:
        dataset = generate_dataset(in_memory_db, scale="small")
        res = run_bitemporal_and_drift_benchmark(dataset)

        assert res["bitemporal_accuracy_pct"] == 100.0
        assert res["stale_detection_rate_pct"] == 100.0
        assert res["warning_emission_rate_pct"] == 100.0

    def test_performance_benchmarks(self, in_memory_db: Database) -> None:
        dataset = generate_dataset(in_memory_db, scale="small")
        perf = run_performance_benchmarks(dataset, iterations=5)

        assert "full_hybrid_search" in perf["latencies"]
        assert "remember_insert" in perf["latencies"]
        assert "purge_redaction" in perf["latencies"]
        assert perf["latencies"]["full_hybrid_search"]["p50"] >= 0.0

    def test_report_generation(self, in_memory_db: Database) -> None:
        dataset = generate_dataset(in_memory_db, scale="small")
        ablation = run_ablation_study(dataset)
        bitemporal = run_bitemporal_and_drift_benchmark(dataset)
        perf = run_performance_benchmarks(dataset, iterations=2)

        report = generate_markdown_report(
            scale="small",
            dataset_stats={"facts": 500, "entities": 100, "relations": 100},
            ablation_results=ablation,
            bitemporal_results=bitemporal,
            perf_results=perf,
            elapsed_total_sec=1.23,
        )

        assert "# Mnemo Benchmark Report" in report
        assert "Experiment 1: Channel Ablation Study" in report
        assert "Experiment 2: Bitemporal Precision & AST Code Drift" in report
        assert "Experiment 3: Latency & Scalability Profile" in report
        assert "Hyperparameter Tuning & Recommendations" in report
