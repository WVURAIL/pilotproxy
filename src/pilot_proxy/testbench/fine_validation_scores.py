"""Full-power score reduction for frozen fine-detector validation studies.

No thresholds are fitted here, and every input frame is kept in the output.
Each exact fixed decision uses one rational sort per frame in
arbitrary-precision integers. The conversion of the resulting powers to float
is a different ablation and is never used for the Q16 decision. The fine
ratio of the designated maximum to the bulk rank cancels a common weight-norm
correction. Coarse ratios require an explicit positive ``mu0`` from the
caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from numbers import Integral, Real

import numpy as np

from pilot_proxy.detectors.narrowband_marker.scores import candidate_rho_values
from pilot_proxy.fine_reduction import independent_bin_mask

FINE_BINS = 256
Q16_ONE = 1 << 16
MAX_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class FineScoreGeometry:
    anchor_bin: int
    designated_bins: np.ndarray
    bulk_mask: np.ndarray
    ranks: np.ndarray
    n_bulk: int
    rank_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def construct_geometry(anchor):
    """Return the anchor +/-2 padded bins and a bulk with only the anchor guarded by one native bin.

    The independent grid is the even padded bins. An even anchor leaves 125
    bulk bins and an odd anchor leaves 126. The ranks are fixed by this
    intended bulk size and are never recalculated from the
    positive-denominator subset of each frame.
    """
    anchor = _integer(anchor, "anchor", 0, FINE_BINS - 1)
    designated = (anchor + np.arange(-2, 3, dtype=np.int64)) % FINE_BINS
    bulk = independent_bin_mask(
        FINE_BINS, pad_factor=2, designated_bins=[anchor], guard_fine_bins=1
    )
    bulk[designated] = False
    n = int(np.count_nonzero(bulk))
    ranks = np.array([(q * n + 3) // 4 - 1 for q in (1, 2, 3, 4)], dtype=np.int64)
    ranks[1] = candidate_rho_values(n)[0] - 1
    for values in (designated, bulk, ranks):
        values.setflags(write=False)
    return FineScoreGeometry(anchor, designated, bulk, ranks, n)


def _geometry(geometry):
    if not isinstance(geometry, FineScoreGeometry):
        raise TypeError("geometry must come from construct_geometry")
    expected = construct_geometry(geometry.anchor_bin)
    if (
        geometry.n_bulk != expected.n_bulk
        or geometry.rank_fractions != expected.rank_fractions
        or not np.array_equal(geometry.designated_bins, expected.designated_bins)
        or not np.array_equal(geometry.bulk_mask, expected.bulk_mask)
        or not np.array_equal(geometry.ranks, expected.ranks)
    ):
        raise ValueError("geometry differs from the declared anchor-only guard contract")
    return geometry


def _powers(values, *, fixed):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (3, FINE_BINS):
        raise ValueError("powers must have shape [frames,3,256]")
    if fixed:
        if values.dtype != np.dtype(np.uint64):
            raise TypeError("fixed powers must have exact uint64 dtype")
    else:
        if values.dtype.kind not in "fiu":
            raise TypeError("floating powers must contain real numeric values")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("powers must be finite and nonnegative; no rows are dropped")
    return values


def _float_ratios(values):
    """Float power-ratio arithmetic without overflowing reference addition."""
    target, lower, upper = (values[:, k].astype(np.float64) for k in range(3))
    maximum_reference = np.maximum(lower, upper)
    positive = maximum_reference > 0
    # The three powers are scaled together by the larger reference. This keeps
    # the ratio and avoids the uint64 sum wrap and float overflow at large
    # finite powers.
    ratio = np.zeros_like(target)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = np.divide(lower, maximum_reference, where=positive,
                                out=np.zeros_like(lower))
        denominator += np.divide(upper, maximum_reference, where=positive,
                                 out=np.zeros_like(upper))
        numerator = np.divide(target, maximum_reference, where=positive,
                              out=np.zeros_like(target))
        np.divide(numerator, denominator / 2.0, where=positive, out=ratio)
    return ratio, positive


def _coarse(values, n, mu0, *, fixed):
    if isinstance(mu0, (bool, np.bool_)) or not isinstance(mu0, Real):
        raise TypeError("mu0 must be real")
    mu0 = float(mu0)
    if not np.isfinite(mu0) or mu0 <= 0:
        raise ValueError("mu0 must be finite and positive")
    if values is None:
        return {}
    values = np.asarray(values)
    if values.shape != (n, 3):
        raise ValueError("coarse_powers must have shape [frames,3]")
    if fixed:
        if values.dtype != np.dtype(np.uint64):
            raise TypeError("fixed coarse powers must have exact uint64 dtype")
    elif values.dtype.kind not in "fiu":
        raise TypeError("coarse powers must contain real numeric values")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("coarse powers must be finite and nonnegative")
    ratio, valid = _float_ratios(values[:, :, None])
    return {"coarse_ratio": ratio[:, 0] / mu0, "coarse_valid": valid[:, 0], "mu0": mu0}


def float_scores(powers, geometry, *, coarse_powers=None, mu0=1.0):
    """Return floating fine responses [frames,4] and explicit validity.

    The usable bulk bins are defined only by a positive reference
    denominator, as in the frozen decision. An invalid rank has a NaN
    response. A valid rank whose bulk ratio is zero returns infinity when any
    usable designated power is positive, and zero otherwise, since the strict
    comparison never fires. These states are kept. Floating ties for
    ``rank_bin`` are broken by stable ascending bin order.
    """
    geometry = _geometry(geometry)
    powers = _powers(powers, fixed=False)
    ratios, positive = _float_ratios(powers)
    n = len(powers)
    bulk_bins = np.flatnonzero(geometry.bulk_mask)
    usable = positive[:, bulk_bins]
    bulk = np.where(usable, ratios[:, bulk_bins], np.inf)
    # Invalid bins sort after usable bins even when a valid floating ratio
    # overflows to infinity. No invalid reference can occupy a usable rank.
    order = np.lexsort((bulk, ~usable), axis=1)
    n_bulk = np.count_nonzero(usable, axis=1)
    valid = geometry.ranks[None, :] < n_bulk[:, None]
    selected = order[:, geometry.ranks]
    rank_values = np.take_along_axis(bulk, selected, axis=1)
    rank_bins = bulk_bins[selected]
    rank_bins = np.where(valid, rank_bins, -1)
    designated = ratios[:, geometry.designated_bins]
    designated_max = np.max(designated, axis=1)
    response = np.full((n, len(geometry.ranks)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(designated_max[:, None], rank_values, out=response, where=valid)
    zero_rank = valid & (rank_values == 0)
    response[zero_rank & (designated_max[:, None] == 0)] = 0.0
    return {
        "response": response,
        "valid": valid,
        "n_bulk": n_bulk,
        "rank_bin": rank_bins,
        "zero_reference_bins": np.count_nonzero(~positive, axis=1),
        "designated_max": designated_max,
        "rank_value": np.where(valid, rank_values, np.nan),
        **_coarse(coarse_powers, n, mu0, fixed=False),
    }


def fixed_scores(powers, geometry, *, coarse_powers=None, mu0=1.0):
    """Reduce exact uint64 powers to all four minimal deployable Q16 bounds.

    ``required_q16`` [frames,4] stores 1..2**64-1 for a finite valid boundary.
    Zero encodes either an invalid rank (valid=False, always_masked=False) or
    logical 2**64 (valid=True, always_masked=True), so the flags must be kept
    with it. ``fixed_float_response`` is a float ablation of the same powers
    and is reported on its own. In the exact branch, each usable rational
    bulk is sorted only once in Python arbitrary-precision integers, which
    also hold reference sums above the uint64 range.
    """
    geometry = _geometry(geometry)
    powers = _powers(powers, fixed=True)
    n = len(powers)
    shape = (n, len(geometry.ranks))
    required = np.zeros(shape, dtype=np.uint64)
    always = np.zeros(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    rank_bins = np.full(shape, -1, dtype=np.int64)
    n_bulk = np.zeros(n, dtype=np.int64)
    zero_reference = np.zeros(n, dtype=np.int64)
    bulk_bins = np.flatnonzero(geometry.bulk_mask).tolist()
    designated_bins = geometry.designated_bins.tolist()
    for row_index, row in enumerate(powers):
        num = [2 * int(v) for v in row[0]]
        den = [int(a) + int(b) for a, b in zip(row[1], row[2])]
        usable = [b for b in bulk_bins if den[b] > 0]
        n_bulk[row_index] = len(usable)
        zero_reference[row_index] = den.count(0)

        def compare(i, j):
            difference = num[i] * den[j] - num[j] * den[i]
            return (difference > 0) - (difference < 0)

        ordered = sorted(usable, key=cmp_to_key(compare))
        designated = [b for b in designated_bins if den[b] > 0 and num[b] > 0]
        if designated:
            limiting = max(designated, key=cmp_to_key(compare))
        else:
            limiting = None
        for column, rank in enumerate(geometry.ranks):
            if rank >= len(ordered):
                continue
            rank_bin = ordered[rank]
            valid[row_index, column] = True
            rank_bins[row_index, column] = rank_bin
            boundary = 1
            if limiting is not None:
                if num[rank_bin] == 0:
                    boundary = ALWAYS_MASKED_Q16
                else:
                    numerator = num[limiting] * Q16_ONE * den[rank_bin]
                    denominator = den[limiting] * num[rank_bin]
                    boundary = max(1, (numerator + denominator - 1) // denominator)
            if boundary > MAX_Q16:
                always[row_index, column] = True
            else:
                required[row_index, column] = boundary
    floating = float_scores(powers, geometry)
    if not np.array_equal(valid, floating["valid"]):
        raise AssertionError("float and exact positive-denominator validity differ")
    return {
        "required_q16": required,
        "always_masked": always,
        "valid": valid,
        "n_bulk": n_bulk,
        "rank_bin": rank_bins,
        "zero_reference_bins": zero_reference,
        "fixed_float_response": floating["response"],
        "fixed_float_rank_bin": floating["rank_bin"],
        **_coarse(coarse_powers, n, mu0, fixed=True),
    }


def decode_requirements(required_q16, always_masked, valid):
    """Return the logical requirements as object integers of the same shape, and refuse invalid rows."""
    encoded = np.asarray(required_q16)
    always = np.asarray(always_masked)
    usable = np.asarray(valid)
    if encoded.dtype != np.dtype(np.uint64):
        raise TypeError("encoded requirements must have uint64 dtype")
    if always.dtype != np.dtype(bool) or usable.dtype != np.dtype(bool):
        raise TypeError("requirement flags must be Boolean")
    if encoded.shape != always.shape or encoded.shape != usable.shape:
        raise ValueError("requirement arrays must have matching shapes")
    if not usable.all():
        raise ValueError("invalid fine ranks must be reported before calibration; no rows are dropped")
    if np.any((encoded == 0) != always):
        raise ValueError("sentinel flag must correspond exactly to encoded zero")
    result = encoded.astype(object)
    result[always] = ALWAYS_MASKED_Q16
    return result
