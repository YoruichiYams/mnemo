"""Unit tests for Reciprocal Rank Fusion (RRF) algorithm."""

from __future__ import annotations

from mnemo.core.rrf import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_weighted,
)


class TestReciprocalRankFusion:
    """Test suite for Reciprocal Rank Fusion ranking and merging."""

    def test_empty_input_returns_empty_list(self) -> None:
        """Empty rankings or empty lists return empty merged list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_ranked_list(self) -> None:
        """Single channel ranking preserves relative ordering with 1/(k+rank) scores."""
        items = ["doc1", "doc2", "doc3"]
        merged = reciprocal_rank_fusion([items], k=60)

        assert len(merged) == 3
        assert [item_id for item_id, _ in merged] == items
        # Rank 1: 1 / 61 = 0.01639344
        assert merged[0][1] == round(1.0 / 61.0, 8)
        # Rank 2: 1 / 62 = 0.01612903
        assert merged[1][1] == round(1.0 / 62.0, 8)
        # Rank 3: 1 / 63 = 0.01587302
        assert merged[2][1] == round(1.0 / 63.0, 8)

    def test_multi_channel_intersection_boost(self) -> None:
        """Items appearing in multiple channels receive higher combined scores."""
        channel_vec = ["common_doc", "vec_only"]
        channel_fts = ["fts_only", "common_doc"]
        channel_graph = ["common_doc", "graph_only"]

        merged = reciprocal_rank_fusion([channel_vec, channel_fts, channel_graph], k=60)
        top_item, top_score = merged[0]

        assert top_item == "common_doc"
        # common_doc ranks: (pos 1 in vec) + (pos 2 in fts) + (pos 1 in graph)
        # 1/61 + 1/62 + 1/61 = 0.01639344 + 0.01612903 + 0.01639344 = 0.04891592
        expected_score = round(1.0 / 61 + 1.0 / 62 + 1.0 / 61, 8)
        assert top_score == expected_score

    def test_custom_k_parameter(self) -> None:
        """Different k smoothing factors alter score distributions properly."""
        items = [["a", "b"]]
        merged_k10 = reciprocal_rank_fusion(items, k=10)
        merged_k100 = reciprocal_rank_fusion(items, k=100)

        # k=10: rank 1 is 1/11 ~ 0.0909
        assert merged_k10[0][1] == round(1.0 / 11.0, 8)
        # k=100: rank 1 is 1/101 ~ 0.0099
        assert merged_k100[0][1] == round(1.0 / 101.0, 8)

    def test_weighted_rrf(self) -> None:
        """Weighted variant scales channel contributions correctly."""
        rankings = {
            "vector": ["doc_v"],
            "fts": ["doc_f"],
        }
        weights = {"vector": 2.0, "fts": 1.0}
        merged = reciprocal_rank_fusion_weighted(rankings, k=60, weights=weights)

        assert merged[0][0] == "doc_v"
        assert merged[1][0] == "doc_f"
        assert merged[0][1] == round(2.0 / 61.0, 8)
        assert merged[1][1] == round(1.0 / 61.0, 8)

    def test_weighted_rrf_zero_or_negative_weight_ignored(self) -> None:
        """Channels with weight <= 0 are excluded from results."""
        rankings = {
            "active": ["doc1"],
            "ignored": ["doc2"],
        }
        weights = {"active": 1.0, "ignored": 0.0}
        merged = reciprocal_rank_fusion_weighted(rankings, k=60, weights=weights)

        assert len(merged) == 1
        assert merged[0][0] == "doc1"
