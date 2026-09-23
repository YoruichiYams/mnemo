"""Ranking and retrieval evaluation metrics.

Calculates Hit@K (1, 3, 5), MRR (Mean Reciprocal Rank), and NDCG@5.
"""

from __future__ import annotations

import math
from collections.abc import Collection


def calculate_hit_at_k(
    ranked_ids: list[str],
    ground_truth_ids: Collection[str],
    k: int,
) -> float:
    """Return 1.0 if any ground truth ID is in ranked_ids[:k], else 0.0."""
    gt_set = set(ground_truth_ids)
    for fid in ranked_ids[:k]:
        if fid in gt_set:
            return 1.0
    return 0.0


def calculate_mrr(
    ranked_ids: list[str],
    ground_truth_ids: Collection[str],
) -> float:
    """Calculate Reciprocal Rank (1/rank of first relevant item, or 0.0)."""
    gt_set = set(ground_truth_ids)
    for rank, fid in enumerate(ranked_ids, start=1):
        if fid in gt_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(
    ranked_ids: list[str],
    ground_truth_ids: Collection[str],
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (NDCG@k)."""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0

    dcg = 0.0
    for i, fid in enumerate(ranked_ids[:k]):
        if fid in gt_set:
            # binary relevance: rel_i = 1 -> 2^1 - 1 = 1
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: top items are all relevant up to min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gt_set))))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(
    ranked_ids: list[str],
    ground_truth_ids: Collection[str],
) -> dict[str, float]:
    """Compute standard ranking metrics for a single query."""
    return {
        "hit@1": calculate_hit_at_k(ranked_ids, ground_truth_ids, k=1),
        "hit@3": calculate_hit_at_k(ranked_ids, ground_truth_ids, k=3),
        "hit@5": calculate_hit_at_k(ranked_ids, ground_truth_ids, k=5),
        "mrr": calculate_mrr(ranked_ids, ground_truth_ids),
        "ndcg@5": calculate_ndcg_at_k(ranked_ids, ground_truth_ids, k=5),
    }


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Compute arithmetic mean across a collection of per-query metric dicts."""
    if not metrics_list:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}

    keys = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"]
    n = len(metrics_list)
    return {k: round(sum(m[k] for m in metrics_list) / n, 4) for k in keys}
