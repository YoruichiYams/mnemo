"""Ebbinghaus-inspired salience decay for memory retention.

The core formula (operating in **hours**):

    S(t) = s0 * exp( -lambda * elapsed_hours / (1 + gamma * f) )

where:
    s0             -- initial salience score in [0, 1]
    elapsed_hours  -- hours since last access  (elapsed_seconds / 3600)
    f              -- cumulative access / reinforcement count (>= 0)
    lambda         -- base decay rate   (default 0.01  =>  ~7-14 day half-life)
    gamma          -- reinforcement resistance (higher => slower decay per access)

Typical behaviour with defaults (lambda=0.01, gamma=0.2, no reinforcement):
    1 day  (24 h):   S = exp(-0.24)  ~ 0.787
    7 days (168 h):  S = exp(-1.68)  ~ 0.186
    14 days (336 h): S = exp(-3.36)  ~ 0.035

With 10 accesses (gamma=0.2 => denominator=3.0):
    1 day:   S = exp(-0.08)  ~ 0.923
    7 days:  S = exp(-0.56)  ~ 0.571
    14 days: S = exp(-1.12)  ~ 0.326
"""

from __future__ import annotations

import math

from mnemo.core.models import MemoryTier


def calculate_salience(
    s0: float,
    elapsed_seconds: float,
    access_count: int = 0,
    lambda_param: float = 0.01,
    gamma: float = 0.2,
) -> float:
    """Compute decayed salience using the Ebbinghaus forgetting curve.

    Internally converts *elapsed_seconds* to hours so that the default
    ``lambda_param=0.01`` gives a gentle 7-14 day decay window.

    Args:
        s0: Base / initial salience in ``[0.0, 1.0]``.
        elapsed_seconds: Seconds since the last access or creation.
        access_count: Total number of prior retrievals (reinforcements).
        lambda_param: Hourly decay rate (must be > 0).
        gamma: Reinforcement resistance (must be >= 0).

    Returns:
        Decayed salience clamped to ``[0.0, 1.0]``, rounded to 6 decimals.

    Examples:
        >>> calculate_salience(1.0, 0.0)
        1.0
        >>> calculate_salience(1.0, 86400)          # 1 day, no reinforcement
        0.786628
        >>> calculate_salience(1.0, 86400, access_count=10)
        0.923116
    """
    # --- Input sanitisation ---------------------------------------------------
    if s0 <= 0.0:
        return 0.0
    s0 = min(1.0, max(0.0, s0))

    # Negative elapsed time is physically impossible -- clamp to 0.
    dt_seconds = max(0.0, elapsed_seconds)

    # Convert to hours for human-friendly lambda scaling.
    elapsed_hours = dt_seconds / 3600.0

    # Ensure positive decay rate.
    lam = max(1e-15, lambda_param)

    # Ensure non-negative reinforcement resistance.
    gam = max(0.0, gamma)

    # Ensure non-negative access count.
    f = max(0, access_count)

    # --- Core formula ---------------------------------------------------------
    denominator = 1.0 + gam * float(f)  # always >= 1.0
    exponent = (lam * elapsed_hours) / denominator

    # Guard exponential overflow (exp(-700) ~ 0).
    if exponent > 700.0:
        return 0.0

    result = s0 * math.exp(-exponent)
    return round(min(1.0, max(0.0, result)), 6)


def salience_to_tier(salience: float) -> MemoryTier:
    """Map a numeric salience to its corresponding ``MemoryTier``."""
    return MemoryTier.from_salience(salience)
