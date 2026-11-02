"""Exact Q16 keep boundaries for marker-to-background power ratios.

The archive scorer and the synthetic frontier studies use this sorted-rank
implementation. ``fine_decision`` holds the independent counting reference
that is used to check it and the CUDA decision.
"""
from __future__ import annotations

from functools import cmp_to_key
from numbers import Integral
from typing import Sequence

import numpy as np

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    FINE_BINS,
    MAX_MULTIPLIER_Q16,
    MULTIPLIER_ONE,
)


def _compare_ratios(left, right) -> int:
    left_num, left_den, left_bin = left
    right_num, right_den, right_bin = right
    first = left_num * right_den
    second = right_num * left_den
    if first < second:
        return -1
    if first > second:
        return 1
    return left_bin - right_bin


def required_multipliers_by_rank(
    fine_powers: np.ndarray,
    *,
    designated: Sequence[int],
    bulk_mask: Sequence[bool],
) -> tuple[int, ...]:
    """Return the smallest keeping Q16 multiplier at every usable bulk rank.

    Each fine bin b has the ratio 2 S_t[b] / (S_l[b] + S_u[b]), formed from
    rows 0, 1 and 2 of ``fine_powers``. The bulk bins with a positive
    denominator are sorted by this ratio, and ties are broken by bin order.
    Ranks are one-based in this order. For each rank, the boundary is the
    smallest Q16 multiplier m such that (m / 2**16) times the rank's ratio is
    at least the largest designated-bin ratio. ``ALWAYS_MASKED_Q16`` marks a
    valid rank where no uint64 multiplier keeps the frame. When no
    designated bin has a positive ratio, every boundary is 1.

    Parameters
    ----------
    fine_powers : ndarray of uint64, shape (3, 256)
        Exact target, lower-reference and upper-reference fine powers.
    designated : sequence of int
        Designated (marker) bins in [0, 256).
    bulk_mask : ndarray of bool, shape (256,)
        Bulk reference bins. Must exclude every designated bin.
    """
    powers = np.asarray(fine_powers)
    if powers.shape != (3, FINE_BINS) or powers.dtype != np.dtype(np.uint64):
        raise ValueError("fine_powers must be exact uint64 with shape [3, 256].")
    mask = np.asarray(bulk_mask)
    if mask.shape != (FINE_BINS,) or mask.dtype != np.dtype(bool):
        raise ValueError("bulk_mask must be an explicit boolean vector of length 256.")
    bins = list(designated)
    if not bins:
        raise ValueError("the designated set must not be empty.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in bins):
        raise ValueError("designated bins must be integers.")
    bins = [int(value) for value in bins]
    if any(not 0 <= value < FINE_BINS for value in bins):
        raise ValueError("designated bins must lie in [0, 256).")
    if mask[bins].any():
        raise ValueError("the bulk mask must exclude every designated bin.")

    numerator = [2 * int(value) for value in powers[0]]
    denominator = [int(lower) + int(upper)
                   for lower, upper in zip(powers[1], powers[2])]
    ranked = [(numerator[index], denominator[index], int(index))
              for index in np.flatnonzero(mask) if denominator[index] > 0]
    if not ranked:
        raise ValueError("the declared bulk has no bin with a positive denominator.")
    ranked.sort(key=cmp_to_key(_compare_ratios))

    maximum = None
    for index in bins:
        if denominator[index] <= 0 or numerator[index] <= 0:
            continue
        value = (numerator[index], denominator[index])
        if maximum is None or value[0] * maximum[1] > maximum[0] * value[1]:
            maximum = value

    boundaries = []
    for rank_num, rank_den, _ in ranked:
        if maximum is None:
            boundaries.append(1)
        elif rank_num == 0:
            boundaries.append(ALWAYS_MASKED_Q16)
        else:
            product_num = maximum[0] * MULTIPLIER_ONE * rank_den
            product_den = rank_num * maximum[1]
            boundary = (product_num + product_den - 1) // product_den
            boundaries.append(ALWAYS_MASKED_Q16 if boundary > MAX_MULTIPLIER_Q16
                              else max(1, boundary))
    return tuple(boundaries)
