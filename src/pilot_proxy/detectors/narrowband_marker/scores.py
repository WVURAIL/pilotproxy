"""The narrowband-marker detector's per-frame scores: exact Q16 keep
boundaries per background rank (the OS-CFAR statistic), and the fixed rank.

For one frame, the fine statistic ``T[f] = 2 S_0[f] / (S_1[f] + S_2[f])`` of
the marker bins ``D`` and of the bulk ``B`` is ranked in exact rational
arithmetic; for every one-based background rank ``rho`` the frame's *keep
boundary* is the smallest Q16 multiplier ``eta_q16`` at which
``max_D T <= (eta_q16 / 2^16) T_(rho)`` (:func:`required_eta_q16_by_rank`).
A frame is kept when the multiplier is at least its boundary; a frame that no
deployable multiplier keeps carries :data:`ALWAYS_MASKED_Q16`. The arithmetic
is release-authoritative and must equal the detector's own fine decision
(``pilot_proxy.fine_decision.fine_required_multiplier_q16``).

The rank is the median of the intended bulk geometry
(:func:`candidate_rho_values`).
"""
from __future__ import annotations

from functools import cmp_to_key
from numbers import Integral

import numpy as np

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    MAX_MULTIPLIER_Q16,
    MULTIPLIER_ONE as Q16_SCALE,
)
from pilot_proxy.products.reader import FINE_BINS


class ScoreRefused(RuntimeError):
    """Raised when a score bundle cannot be established exactly."""


def _declared_bins(anchor_bin, designated_half_width) -> np.ndarray:
    for value, name in ((anchor_bin, "anchor_bin"),
                        (designated_half_width, "designated_half_width")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ScoreRefused(f"{name} must be an integer")
    anchor = int(anchor_bin)
    width = int(designated_half_width)
    if not 0 <= anchor < FINE_BINS:
        raise ScoreRefused("anchor_bin must be in [0, 256)")
    if not 0 <= width < FINE_BINS // 2:
        raise ScoreRefused("designated_half_width must be in [0, 128)")
    return np.asarray([(anchor + offset) % FINE_BINS
                       for offset in range(-width, width + 1)], dtype="<i8")


def _boolean_mask(values, size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(bool) or result.shape != (size,):
        raise ScoreRefused(
            f"{name} must be an explicit boolean vector of length {size}")
    return np.ascontiguousarray(result)


def _rational_order(left, right) -> int:
    left_num, left_den, left_bin = left
    right_num, right_den, right_bin = right
    first = left_num * right_den
    second = right_num * left_den
    if first < second:
        return -1
    if first > second:
        return 1
    return left_bin - right_bin


def required_eta_q16_by_rank(
        fine_power_u64,
        *,
        anchor_bin: int,
        designated_half_width: int,
        bulk_mask,
) -> tuple[int, ...]:
    """Return every exact one-based-rho Q16 boundary for one frame."""
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

    numerator = [2 * int(value) for value in powers[0]]
    denominator = [int(lower) + int(upper)
                   for lower, upper in zip(powers[1], powers[2])]
    ranked = [(numerator[bin_index], denominator[bin_index], bin_index)
              for bin_index in np.flatnonzero(bulk)
              if denominator[bin_index] > 0]
    if not ranked:
        raise ScoreRefused(
            "declared bulk has no bin with a positive reference denominator")
    ranked.sort(key=cmp_to_key(_rational_order))

    designated_values = [
        (numerator[int(bin_index)], denominator[int(bin_index)])
        for bin_index in designated
        if denominator[int(bin_index)] > 0 and numerator[int(bin_index)] > 0
    ]
    designated_maximum = None
    for value in designated_values:
        if (designated_maximum is None
                or value[0] * designated_maximum[1]
                > designated_maximum[0] * value[1]):
            designated_maximum = value
    boundaries = []
    for rank_num, rank_den, _ in ranked:
        if designated_maximum is None:
            boundaries.append(1)
            continue
        if rank_num == 0:
            boundaries.append(ALWAYS_MASKED_Q16)
            continue
        designated_num, designated_den = designated_maximum
        numerator_product = designated_num * Q16_SCALE * rank_den
        denominator_product = rank_num * designated_den
        boundary = ((numerator_product + denominator_product - 1)
                    // denominator_product)
        boundaries.append(
            ALWAYS_MASKED_Q16 if boundary > MAX_MULTIPLIER_Q16
            else max(1, boundary))
    return tuple(boundaries)


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
