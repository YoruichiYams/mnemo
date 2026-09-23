"""Activity-based salience decay and Spaced Repetition stability engine for Mnemo.

The core forgetting curve operates on **Activity Ticks** (or hours when time is passed):

    R(n) = R_0 * (1 + alpha) ^ n
    S(tick) = s0 * exp( -lambda * Delta_tick / R )

where:
    s0                   -- initial salience score in [0.0, 1.0]
    Delta_tick           -- ticks elapsed since last access (current_tick - last_accessed_tick)
    n                    -- cumulative reinforcement count (>= 0)
    R_0                  -- initial stability factor (default 1.0)
    alpha                -- multiplicative stability growth factor (default 0.5)
    lambda               -- activity-based decay rate (default 0.05)

Multiplicative stability progression (alpha=0.5):
    n=0: R = 1.000
    n=1: R = 1.500
    n=2: R = 2.250
    n=3: R = 3.375
    n=4: R = 5.063
"""

from __future__ import annotations

import math

from mnemo.core.models import MemoryTier

DEFAULT_LAMBDA: float = 0.01
DEFAULT_ALPHA: float = 0.5
DEFAULT_R0: float = 1.0


def calculate_salience(
    s0: float,
    time_or_ticks: float | int = 0.0,
    reinforcement_count: int = 0,
    lambda_param: float = DEFAULT_LAMBDA,
    alpha: float = DEFAULT_ALPHA,
    r0: float = DEFAULT_R0,
    *,
    delta_ticks: float | int | None = None,
    elapsed_seconds: float | None = None,
    access_count: int | None = None,
    gamma: float | None = None,
) -> float:
    """Compute decayed salience using Spaced Repetition stability over Activity Ticks.

    Args:
        s0: Base / initial salience in ``[0.0, 1.0]``.
        time_or_ticks: Elapsed activity ticks (or elapsed seconds if >= 3600.0).
        reinforcement_count: Number of prior explicit reinforcements.
        lambda_param: Base decay rate (must be > 0).
        alpha: Multiplicative stability factor per reinforcement (default 0.5).
        r0: Initial stability factor (default 1.0).
        delta_ticks: Explicit elapsed activity ticks.
        elapsed_seconds: Explicit elapsed seconds (converted to hours for backwards compatibility).
        access_count: Alias for reinforcement count.
        gamma: Deprecated legacy reinforcement parameter (if provided without alpha).

    Returns:
        Decayed salience clamped to ``[0.0, 1.0]``, rounded to 6 decimals.
    """
    if s0 <= 0.0:
        return 0.0
    s0 = min(1.0, max(0.0, s0))

    # Determine delta and mode
    if delta_ticks is not None:
        delta = max(0.0, float(delta_ticks))
        is_seconds = False
    elif elapsed_seconds is not None:
        delta = max(0.0, float(elapsed_seconds)) / 3600.0
        is_seconds = True
    else:
        val = max(0.0, float(time_or_ticks))
        if val >= 3600.0:
            delta = val / 3600.0
            is_seconds = True
        else:
            delta = val
            is_seconds = False

    # Effective reinforcement count
    n = max(0, reinforcement_count if reinforcement_count != 0 else (access_count or 0))

    # Base decay rate
    lam = max(1e-15, lambda_param if lambda_param is not None else DEFAULT_LAMBDA)

    # Calculate stability R
    if gamma is not None and is_seconds:
        # Legacy linear denominator support for pure elapsed_seconds tests
        r = max(1.0, 1.0 + gamma * float(min(n, 1000)))
    else:
        # Multiplicative Spaced Repetition stability: R = R0 * (1 + alpha)^n (clamped to prevent float overflow)
        growth = max(0.0, 1.0 + alpha)
        n_clamped = min(n, 100)
        r = max(1e-9, min(1e9, r0 * (growth**n_clamped)))

    exponent = (lam * delta) / r

    if exponent > 700.0:
        return 0.0

    result = s0 * math.exp(-exponent)
    return round(min(1.0, max(0.0, result)), 6)


def salience_to_tier(salience: float) -> MemoryTier:
    """Map a numeric salience to its corresponding ``MemoryTier``."""
    return MemoryTier.from_salience(salience)

