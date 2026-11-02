"""Binomial score intervals shared by detection-rate reports."""

from __future__ import annotations

import math
import operator

import numpy as np


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return Wilson score bounds, or (nan, nan) for an empty sample.

    Counts must be exact integers with 0 <= successes <= trials. The default
    z gives a 95% interval; z=0 gives the observed proportion itself.
    """
    counts = []
    for name, value in (("successes", successes), ("trials", trials)):
        message = f"{name} must be a nonnegative integer"
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        try:
            count = operator.index(value)
        except TypeError as exc:
            raise ValueError(message) from exc
        if count < 0:
            raise ValueError(message)
        counts.append(count)
    k, n = counts
    if k > n:
        raise ValueError("successes must not exceed trials")
    try:
        critical = float(z)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("z must be finite and nonnegative") from exc
    if not math.isfinite(critical) or critical < 0:
        raise ValueError("z must be finite and nonnegative")
    if n == 0:
        return math.nan, math.nan
    p = k / n
    if critical == 0:
        return p, p
    try:
        root_n = math.sqrt(n)
    except OverflowError as exc:
        raise ValueError("trials exceeds the floating-point numerical range") from exc

    # Scale by sqrt(n + z^2) without forming z^2 or products of n. The
    # factors are bounded even when the unscaled intermediates overflow.
    scale = math.hypot(root_n, critical)
    a, b = root_n / scale, critical / scale
    if k == 0:
        return 0.0, b * b
    if k == n:
        return a * a, 1.0
    center = a * a * p + b * b / 2
    radius = b * math.hypot(a * math.sqrt(p * (1-p)), b / 2)
    upper = min(1.0, center + radius)
    # The roots' product is a^2*p^2. Use it to avoid subtracting nearly
    # equal center/radius terms in a small lower tail.
    lower = (a * p) * ((a * p) / upper) if upper > 0 else 0.0
    return lower, upper
