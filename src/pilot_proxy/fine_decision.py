# coding=utf-8
"""Compare fine-bin power ratios using exact integer arithmetic.

This is the Python reference for fine decision v1 in the fused kernel
epilogue (kernel core 4.0.0, PilotProxy_Compute_FusedFineMask_U64). For
the same fine powers and calibration constants, the CUDA implementation
must return the same mask bit. Python's arbitrary-precision integers
provide a reference for the kernel's 128/192-bit products.

First define the measured ratios
-------------------------------

The input S[term, bin] contains exact uint64 fine powers: frozen fxfft256
v1 magnitudes summed over feeds. At each bin b, write the ratio without
dividing:

    F2[b] = num[b] / den[b]
    num[b] = 2 * S[target, b]
    den[b] = S[ref_lower, b] + S[ref_upper, b]

The runtime bundle supplies the comparison geometry and threshold:

* anchor_bin: measured pilot-line bin, from 0 through 255.
* designated_half_width: examine anchor +/- w, wrapping modulo 256.
* bulk_mask: a 256-bit mask of independent background bins, taking every
  pad_factor-th bin and excluding guard and census bins. Bundle export
  builds it with fine_reduction.independent_bin_mask.
* cfar_rank: zero-based rank in the ascending background ratios.
* multiplier_q16: threshold multiplier on a Q16 grid.

Then choose the background reference
------------------------------------

Keep background bins with bulk_mask[b] and den[b] > 0. If cfar_rank is
not smaller than their count, the frame is invalid and its mask is zero.
Otherwise, select F2_r at that rank by exact comparisons. Counting ranks
makes the selected value unique even when multiple bins tie.

Reject the frame (mask = 1) if any designated bin with den[b] > 0 satisfies::

    num[b] * 2**16 * den_r > multiplier_q16 * num_r * den[b]

This is the strict comparison F2[b] > (multiplier_q16 / 2**16) * F2_r,
written as integer products. A bin with zero reference power cannot fire.
A frame with insufficient background support also returns mask zero;
its invalid status must therefore be checked separately from the mask.

fine_required_multiplier_q16 reverses the comparison to find the smallest
integer multiplier that keeps a frame. This gives exact change points
without scanning a floating-point threshold grid.

Why use a rank and an integer multiplier?
----------------------------------------

The survey's float calibration (fine_reduction.calibrate_cfar) uses
medians and interpolated quantiles. Those operations do not define a
bit-exact result across NumPy, NVCC, and compiler implementations. A rank
reference and Q16 multiplier make the comparison exact while reducing
sensitivity to contamination in the upper background tail.

This module and the kernel fix the arithmetic. The bundle supplies the
rank and multiplier. The current rank is the bulk median;
characterization.threshold_rule selects the channel's knee or the
loosest word below it that meets the science and time limits.
runtime_calibration separately proposes commissioning thresholds from
declared calibration populations.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)

FINE_DECISION_VERSION = "fine_decision_v1"
FINE_BINS = 256
MULTIPLIER_Q = 16
MULTIPLIER_ONE = 1 << MULTIPLIER_Q
MAX_MULTIPLIER_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64


def _exact_integer(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc


def designated_bins(anchor_bin: int, half_width: int) -> np.ndarray:
    """Designated set: anchor +/- half_width, modulo 256 (survey rule)."""
    a = _exact_integer(anchor_bin, field="anchor_bin")
    w = _exact_integer(half_width, field="designated_half_width")
    if not 0 <= a < FINE_BINS:
        raise ValueError("anchor_bin must be in [0, 256).")
    if not 0 <= w < FINE_BINS // 2:
        raise ValueError("designated_half_width must be in [0, 128).")
    return np.array(
        [(a + k) % FINE_BINS for k in range(-w, w + 1)], dtype=np.int64
    )


def pack_bulk_mask(mask: Sequence[bool]) -> tuple[int, int, int, int]:
    """Pack a 256-entry boolean mask into 4 uint64 words (bin b ->
    word b // 64, bit b % 64)."""
    m = np.asarray(mask)
    if m.shape != (FINE_BINS,):
        raise ValueError("bulk mask must have exactly 256 entries.")
    if m.dtype != np.dtype(bool):
        raise TypeError("bulk mask entries must be booleans.")
    words = [0, 0, 0, 0]
    for b in np.flatnonzero(m):
        words[int(b) >> 6] |= 1 << (int(b) & 63)
    return tuple(words)  # type: ignore[return-value]


def unpack_bulk_mask(words: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`pack_bulk_mask`."""
    w = [
        _exact_integer(value, field=f"bulk_mask_words[{index}]")
        for index, value in enumerate(words)
    ]
    if len(w) != 4 or any(not 0 <= x < (1 << 64) for x in w):
        raise ValueError("bulk mask words must be 4 uint64 values.")
    return np.array(
        [bool((w[b >> 6] >> (b & 63)) & 1) for b in range(FINE_BINS)],
        dtype=bool,
    )


@dataclass(frozen=True)
class FineDecision:
    """Decision outcome plus the diagnostics the tests pin down."""

    mask: int          # 1 = reject (pilot present), 0 = keep
    valid: bool        # False when the bulk was degenerate for the rank
    n_bulk: int        # usable bulk bins (mask bit set and den > 0)
    rank_bin: int      # representative bin of the rank value (-1 invalid)
    fired_bin: int     # first designated bin that fired (-1 when mask=0)


@dataclass(frozen=True)
class FineRequiredMultiplier:
    """Exact Q16 keep boundary for one frame and rank."""

    valid: bool
    n_bulk: int
    rank_bin: int
    limiting_bin: int
    multiplier_q16: int | None


def _f2_less(num_i: int, den_i: int, num_j: int, den_j: int) -> bool:
    """Exact ``F2_i < F2_j`` by cross multiplication (dens > 0)."""
    return num_i * den_j < num_j * den_i


def _rank_value(num, den, mask_arr, rank):
    bulk = [b for b in range(FINE_BINS) if mask_arr[b] and den[b] > 0]
    if rank >= len(bulk):
        return bulk, -1
    for i in bulk:
        c_lt = 0
        c_eq = 0
        for j in bulk:
            if _f2_less(num[j], den[j], num[i], den[i]):
                c_lt += 1
            elif not _f2_less(num[i], den[i], num[j], den[j]):
                c_eq += 1
        if c_lt <= rank < c_lt + c_eq:
            return bulk, i
    raise AssertionError("rank selection failed")


def fine_required_multiplier_q16(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Sequence[bool] | Sequence[int],
    cfar_rank: int,
) -> FineRequiredMultiplier:
    """Return the exact Q16 boundary for one frame and rank.

    ``ALWAYS_MASKED_Q16`` marks a valid frame that no deployable multiplier
    can keep. ``None`` marks an invalid rank because too few bulk bins have a
    positive reference denominator.
    """
    S = np.asarray(fine_powers)
    if S.shape != (3, FINE_BINS):
        raise ValueError("fine_powers must have shape [3, 256].")
    if S.dtype != np.dtype(np.uint64):
        raise TypeError("fine_powers must have exact uint64 dtype.")
    rank = _exact_integer(cfar_rank, field="cfar_rank")
    if not 0 <= rank < FINE_BINS:
        raise ValueError(f"cfar_rank must be in [0, {FINE_BINS - 1}].")
    designated = designated_bins(anchor_bin, designated_half_width)
    if len(bulk_mask) == 4:
        mask_arr = unpack_bulk_mask(bulk_mask)
    else:
        mask_arr = np.asarray(bulk_mask)
        if mask_arr.dtype != np.dtype(bool):
            raise TypeError("bulk mask entries must be booleans.")
    if mask_arr.shape != (FINE_BINS,):
        raise ValueError("bulk mask must describe exactly 256 bins.")

    num = [2 * int(S[WEIGHT_TERM_TARGET, b]) for b in range(FINE_BINS)]
    den = [
        int(S[WEIGHT_TERM_REF_LOWER, b]) + int(S[WEIGHT_TERM_REF_UPPER, b])
        for b in range(FINE_BINS)
    ]
    bulk, rank_bin = _rank_value(num, den, mask_arr, rank)
    if rank_bin < 0:
        return FineRequiredMultiplier(
            valid=False, n_bulk=len(bulk), rank_bin=-1, limiting_bin=-1,
            multiplier_q16=None)

    num_r = num[rank_bin]
    den_r = den[rank_bin]
    required = 1
    limiting_bin = -1
    for raw_bin in designated:
        bin_index = int(raw_bin)
        if den[bin_index] <= 0 or num[bin_index] == 0:
            continue
        if num_r == 0:
            return FineRequiredMultiplier(
                valid=True, n_bulk=len(bulk), rank_bin=rank_bin,
                limiting_bin=bin_index,
                multiplier_q16=ALWAYS_MASKED_Q16)
        numerator = num[bin_index] * MULTIPLIER_ONE * den_r
        denominator = num_r * den[bin_index]
        boundary = (numerator + denominator - 1) // denominator
        if boundary > required:
            required = boundary
            limiting_bin = bin_index
    if required > MAX_MULTIPLIER_Q16:
        required = ALWAYS_MASKED_Q16
    return FineRequiredMultiplier(
        valid=True, n_bulk=len(bulk), rank_bin=rank_bin,
        limiting_bin=limiting_bin, multiplier_q16=required)


def fine_mask_decision(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Sequence[bool] | Sequence[int],
    cfar_rank: int,
    multiplier_q16: int,
) -> FineDecision:
    """Frozen fine decision v1 over one frame's exact fine powers.

    ``fine_powers``: integer array ``[3, 256]`` (target, lower ref,
    upper ref) --- the fused kernel's exact uint64 ``FinePowers`` for
    one batch entry. ``bulk_mask`` accepts either 256 booleans or the
    4-word packed form. Returns a :class:`FineDecision`.
    """
    S = np.asarray(fine_powers)
    if S.shape != (3, FINE_BINS):
        raise ValueError("fine_powers must have shape [3, 256].")
    if S.dtype != np.dtype(np.uint64):
        raise TypeError("fine_powers must have exact uint64 dtype.")
    rank = _exact_integer(cfar_rank, field="cfar_rank")
    if not 0 <= rank < FINE_BINS:
        raise ValueError(f"cfar_rank must be in [0, {FINE_BINS - 1}].")
    mult = _exact_integer(multiplier_q16, field="multiplier_q16")
    if not 0 < mult < (1 << 64):
        raise ValueError("multiplier_q16 must be in [1, 2**64 - 1].")
    designated = designated_bins(anchor_bin, designated_half_width)

    if len(bulk_mask) == 4:
        mask_arr = unpack_bulk_mask(bulk_mask)
    else:
        mask_arr = np.asarray(bulk_mask)
        if mask_arr.dtype != np.dtype(bool):
            raise TypeError("bulk mask entries must be booleans.")
    if mask_arr.shape != (FINE_BINS,):
        raise ValueError("bulk mask must describe exactly 256 bins.")

    num = [2 * int(S[WEIGHT_TERM_TARGET, b]) for b in range(FINE_BINS)]
    den = [
        int(S[WEIGHT_TERM_REF_LOWER, b]) + int(S[WEIGHT_TERM_REF_UPPER, b])
        for b in range(FINE_BINS)
    ]

    bulk, rank_bin = _rank_value(num, den, mask_arr, rank)
    n_bulk = len(bulk)
    if rank_bin < 0:
        return FineDecision(
            mask=0, valid=False, n_bulk=n_bulk, rank_bin=-1, fired_bin=-1
        )

    num_r = num[rank_bin]
    den_r = den[rank_bin]

    fired_bin = -1
    for b in designated:
        bi = int(b)
        if den[bi] <= 0:
            continue  # zero-reference forced 0, per bin
        if num[bi] * MULTIPLIER_ONE * den_r > mult * num_r * den[bi]:
            fired_bin = bi
            break
    return FineDecision(
        mask=1 if fired_bin >= 0 else 0,
        valid=True,
        n_bulk=n_bulk,
        rank_bin=rank_bin,
        fired_bin=fired_bin,
    )
