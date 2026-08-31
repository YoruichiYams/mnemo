"""Unit tests for Ebbinghaus Salience Decay engine."""

from __future__ import annotations

import pytest

from mnemo.core.decay import calculate_salience, salience_to_tier
from mnemo.core.models import MemoryTier


class TestCalculateSalience:
    """Test suite for calculate_salience mathematical properties and edge cases."""

    def test_zero_elapsed_preserves_initial_salience(self) -> None:
        """At t=0 (no elapsed time), salience remains equal to s0."""
        assert calculate_salience(1.0, 0.0) == 1.0
        assert calculate_salience(0.75, 0.0) == 0.75
        assert calculate_salience(0.42, 0.0) == 0.42

    def test_negative_elapsed_time_clamped_to_zero(self) -> None:
        """Negative elapsed time (e.g. clock skew) is safely clamped to 0."""
        assert calculate_salience(1.0, -100.0) == 1.0
        assert calculate_salience(0.8, -1.0) == 0.8

    def test_zero_or_negative_initial_salience(self) -> None:
        """Initial salience <= 0 always returns 0.0."""
        assert calculate_salience(0.0, 100.0) == 0.0
        assert calculate_salience(-0.5, 0.0) == 0.0

    def test_oversized_initial_salience_clamped_to_one(self) -> None:
        """Initial salience > 1.0 is clamped to 1.0."""
        assert calculate_salience(1.5, 0.0) == 1.0
        # Over 1 day with s0=2.0 clamped to 1.0
        s = calculate_salience(2.0, 86400.0, lambda_param=0.01)
        assert 0.75 < s < 0.80

    def test_gradual_decay_over_time(self) -> None:
        """Salience decreases monotonically as elapsed time increases."""
        s_0 = calculate_salience(1.0, 0.0)
        s_1h = calculate_salience(1.0, 3600.0)
        s_1d = calculate_salience(1.0, 86400.0)
        s_7d = calculate_salience(1.0, 7 * 86400.0)
        s_14d = calculate_salience(1.0, 14 * 86400.0)

        assert s_0 == 1.0
        assert s_0 > s_1h > s_1d > s_7d > s_14d
        assert s_1d > 0.75  # Still active in working tier after 1 day
        assert s_7d < 0.25  # Decayed to peripheral/archived after 7 days

    def test_reinforcement_slows_decay(self) -> None:
        """Higher access count (f) significantly slows down forgetting."""
        elapsed_7d = 7 * 86400.0

        s_no_access = calculate_salience(1.0, elapsed_7d, access_count=0)
        s_access_5 = calculate_salience(1.0, elapsed_7d, access_count=5)
        s_access_20 = calculate_salience(1.0, elapsed_7d, access_count=20)

        assert s_access_20 > s_access_5 > s_no_access
        # With 20 accesses, retention after 7 days remains high
        assert s_access_20 > 0.70

    def test_extreme_time_overflow_protection(self) -> None:
        """Massive elapsed time does not crash and returns 0.0 cleanly."""
        extreme_time = 1e12  # billions of seconds
        assert calculate_salience(1.0, extreme_time) == 0.0

    def test_custom_parameters(self) -> None:
        """Custom lambda and gamma parameters behave predictably."""
        # Fast decay rate
        fast = calculate_salience(1.0, 3600.0, lambda_param=1.0)
        # Slow decay rate
        slow = calculate_salience(1.0, 3600.0, lambda_param=0.001)
        assert slow > fast


class TestSalienceToTier:
    """Test suite for mapping salience values to MemoryTier enum."""

    @pytest.mark.parametrize(
        ("salience", "expected_tier"),
        [
            (1.0, MemoryTier.CORE),
            (0.95, MemoryTier.CORE),
            (0.90, MemoryTier.CORE),
            (0.89999, MemoryTier.WORKING),
            (0.75, MemoryTier.WORKING),
            (0.70, MemoryTier.WORKING),
            (0.69999, MemoryTier.PERIPHERAL),
            (0.50, MemoryTier.PERIPHERAL),
            (0.40, MemoryTier.PERIPHERAL),
            (0.39999, MemoryTier.ARCHIVED),
            (0.10, MemoryTier.ARCHIVED),
            (0.0, MemoryTier.ARCHIVED),
        ],
    )
    def test_tier_threshold_mapping(self, salience: float, expected_tier: MemoryTier) -> None:
        """Verify exact boundary values for all 4 memory tiers."""
        assert salience_to_tier(salience) == expected_tier
        assert MemoryTier.from_salience(salience) == expected_tier
