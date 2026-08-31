"""Reciprocal Rank Fusion (RRF) for multi-channel hybrid retrieval.

Given *N* ranked lists of item IDs (one per retrieval channel), RRF
produces a single merged ranking using the formula:

    RRF(d) = Σ_{r ∈ rankings}  1 / (k + rank_r(d))

where ``k`` is a smoothing constant (default 60) and ``rank_r(d)`` is
the 1-based position of document *d* in ranked list *r* (∞ if absent).
"""

from __future__ import annotations

DEFAULT_K: int = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = DEFAULT_K,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Args:
        ranked_lists: One list per retrieval channel.  Each inner list
            contains item IDs ordered from most to least relevant
            (index 0 = rank 1).
        k: Smoothing constant.  Higher values dampen the influence
            of top-ranked positions.  Standard default is 60.

    Returns:
        Merged ``(item_id, rrf_score)`` tuples sorted descending by score.

    Examples:
        >>> reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
        [('b', 0.03268698), ('a', 0.03268698), ('c', 0.01587302)]
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_0, item_id in enumerate(ranked_list):
            rank_1 = rank_0 + 1  # 1-based rank
            contribution = 1.0 / (k + rank_1)
            scores[item_id] = scores.get(item_id, 0.0) + contribution

    # Stable sort: primary by score (desc), secondary by item_id (asc)
    merged = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item_id, round(score, 8)) for item_id, score in merged]


def reciprocal_rank_fusion_weighted(
    ranked_lists: dict[str, list[str]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Weighted variant — each channel can have a custom multiplier.

    Args:
        ranked_lists: ``{channel_name: [id, …]}``.
        k: Smoothing constant.
        weights: ``{channel_name: weight}``.  Missing keys default to 1.0.

    Returns:
        Merged ``(item_id, rrf_score)`` tuples sorted descending by score.
    """
    w = weights or {}
    scores: dict[str, float] = {}

    for channel, ids in ranked_lists.items():
        channel_weight = w.get(channel, 1.0)
        if channel_weight <= 0.0:
            continue
        for rank_0, item_id in enumerate(ids):
            rank_1 = rank_0 + 1
            contribution = channel_weight / (k + rank_1)
            scores[item_id] = scores.get(item_id, 0.0) + contribution

    merged = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item_id, round(score, 8)) for item_id, score in merged]
