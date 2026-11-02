"""Per-frame scores of the narrowband-marker detector.

This module computes the exact Q16 keep boundaries for each background rank of
the order-statistic constant false alarm rate (OS-CFAR) test, and the fixed
rank. For one frame, the fine statistic ``T[f] = 2 S_0[f] / (S_1[f] + S_2[f])``
of the marker bins ``D`` and of the bulk ``B`` is ranked in exact rational
arithmetic. For each one-based background rank ``rho``, the smallest Q16
multiplier ``eta_q16`` that satisfies ``max_D T <= (eta_q16 / 2^16) T_(rho)``
is known as the frame's *keep boundary* (:func:`required_eta_q16_by_rank`). A
frame is kept when the multiplier is at least its boundary. A frame that no
deployable multiplier keeps is given :data:`ALWAYS_MASKED_Q16`. This arithmetic
is authoritative for the release and must equal the detector's own fine decision
(``pilot_proxy.fine_decision.fine_required_multiplier_q16``).

The rank is the median of the intended bulk geometry
(:func:`candidate_rho_values`).
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    MAX_MULTIPLIER_Q16,
    MULTIPLIER_ONE as Q16_SCALE,
    designated_bins,
)
from pilot_proxy.products.reader import FINE_BINS
from pilot_proxy.rank_scores import required_multipliers_by_rank


class ScoreRefused(RuntimeError):
    """Raised when a score bundle cannot be computed exactly."""


def _declared_bins(anchor_bin, designated_half_width) -> np.ndarray:
    try:
        return designated_bins(anchor_bin, designated_half_width)
    except (TypeError, ValueError) as exc:
        raise ScoreRefused(str(exc)) from exc


def _boolean_mask(values, size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(bool) or result.shape != (size,):
        raise ScoreRefused(
            f"{name} must be an explicit boolean vector of length {size}")
    return np.ascontiguousarray(result)


def required_eta_q16_by_rank(
        fine_power_u64,
        *,
        anchor_bin: int,
        designated_half_width: int,
        bulk_mask,
) -> tuple[int, ...]:
    """Return the exact Q16 keep boundary of one frame for each one-based rank rho."""
    powers = np.asarray(fine_power_u64)
    if powers.dtype != np.dtype(np.uint64) or powers.shape != (3, FINE_BINS):
        raise ScoreRefused(
            "fine_power_u64 must have uint64 dtype and shape (3, 256)")
    designated = _declared_bins(anchor_bin, designated_half_width)
    bulk = _boolean_mask(bulk_mask, FINE_BINS, "bulk_mask")
    if not bulk.any():
        raise ScoreRefused("bulk_mask must select at least one bin")
    if bulk[designated].any():
        raise ScoreRefused("bulk_mask must exclude every designated bin")

    try:
        return required_multipliers_by_rank(powers, designated=designated, bulk_mask=bulk)
    except ValueError as exc:
        raise ScoreRefused(str(exc)) from exc


def rank_from_index(cfar_rank: int) -> int:
    """Convert the detector's zero-based rank to one-based rho."""
    if isinstance(cfar_rank, bool) or not isinstance(cfar_rank, Integral):
        raise TypeError("cfar_rank must be an integer")
    rank = int(cfar_rank)
    if rank < 0:
        raise ValueError("cfar_rank must be non-negative")
    return rank + 1


def candidate_rho_values(intended_bulk_size: int) -> tuple[int, ...]:
    """Return the fixed median rank, before invalid bins are removed."""
    if (isinstance(intended_bulk_size, bool)
            or not isinstance(intended_bulk_size, Integral)):
        raise TypeError("intended_bulk_size must be an integer")
    size = int(intended_bulk_size)
    if size <= 0:
        raise ValueError("intended_bulk_size must be positive")
    return ((size + 1) // 2,)


__all__ = ["ALWAYS_MASKED_Q16", "MAX_MULTIPLIER_Q16", "Q16_SCALE", "ScoreRefused", "candidate_rho_values",
           "rank_from_index", "required_eta_q16_by_rank"]
