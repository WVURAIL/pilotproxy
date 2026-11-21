# coding=utf-8
"""Compare fine-bin power ratios using exact integer arithmetic.

This module is the Python reference for fine decision v1 in the fused
kernel epilogue (kernel core 4.0.0, PilotProxy_Compute_FusedFineMask_U64).
For the same fine powers and calibration constants, the CUDA
implementation must return the same mask bit. Python's arbitrary-precision
integers give the reference for the 128-bit and 192-bit products of the
kernel.

Measured ratios
---------------

The input S[term, bin] holds exact uint64 fine powers, which are the
frozen fxfft256 v1 magnitudes summed over feeds. At each bin b, the ratio
is written without a division:

    F2[b] = num[b] / den[b]
    num[b] = 2 * S[target, b]
    den[b] = S[ref_lower, b] + S[ref_upper, b]

The runtime bundle supplies the bin geometry and threshold:

* anchor_bin: measured pilot-line bin, from 0 through 255.
* designated_half_width: half-width w of the designated bins anchor +/- w,
  wrapped modulo 256.
* bulk_mask: 256-bit mask of independent background bins. Every
  pad_factor-th bin is taken, and guard and census bins are excluded.
  Bundle export builds it with fine_reduction.independent_bin_mask.
* cfar_rank: zero-based rank in the ascending background ratios.
* multiplier_q16: threshold multiplier on a Q16 grid.

Background reference
--------------------

Background bins are the bins with bulk_mask[b] set and den[b] > 0.
When cfar_rank is equal to or larger than their count, the frame is
invalid and its mask is zero. Otherwise, the ratio F2_r at that rank is
selected by exact integer cross products. Ranks are counted, so the
selected value is unique even when several bins tie.

The frame is rejected (mask = 1) when any designated bin with den[b] > 0
satisfies::

    num[b] * 2**16 * den_r > multiplier_q16 * num_r * den[b]

This is the strict test F2[b] > (multiplier_q16 / 2**16) * F2_r,
written as integer products. A bin with zero reference power cannot fire.
A frame with too few background bins also returns mask zero, so the
valid flag must be read along with the mask.

fine_required_multiplier_q16 inverts the test to find the smallest
integer multiplier that keeps a frame. Exact change points are found this
way, with no scan over a floating-point threshold grid.

Rank and integer multiplier
---------------------------

The float calibration of the survey (fine_reduction.calibrate_cfar) uses
medians and interpolated quantiles. These operations do not define a
bit-exact result across NumPy, NVCC, and compiler implementations. With a
rank reference and a Q16 multiplier, the test is exact, and the
sensitivity to contamination in the upper background tail is reduced.

This module and the kernel fix the arithmetic, and the bundle supplies
the rank and the multiplier. The rank in use is the bulk median.
characterization.threshold_rule selects the supported word with the greatest
retained fraction among those meeting the bias tolerance.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pilot_proxy.fine_reduction import (
    FINE_PAD_FACTOR,
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
    """Return the designated bins anchor +/- half_width, modulo 256 (survey rule)."""
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
    """Pack a 256-entry boolean mask into 4 uint64 words.

    Bin b is stored in word b // 64 at bit b % 64.
    """
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
    """Decision outcome and the diagnostics fixed by the tests."""

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
    """Return the frozen fine decision v1 for the exact fine powers of one frame.

    ``fine_powers`` is an integer array ``[3, 256]`` (target, lower ref,
    upper ref) that holds the exact uint64 ``FinePowers`` of the fused
    kernel for one batch entry. ``bulk_mask`` accepts 256 booleans or the
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
            continue  # a bin with zero reference power cannot fire
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


# Fine decision v2
# ----------------
#
# Version 2 reads the target alone in the designated bins and takes the
# reference level from the reference powers of the background bins:
#
#     num[b] * 2**16 > multiplier_q16 * den_r
#
# where den_r is the value of zero-based rank cfar_rank in the ascending
# den[b] of the background bins (bulk_mask[b] set and den[b] > 0). The frame
# is invalid, with mask zero, when fewer than cfar_rank + 1 such bins exist.
# Version 1 above is unchanged.

FINE_DECISION_V2_VERSION = "fine_decision_v2"


def _v2_inputs(fine_powers, anchor_bin, designated_half_width, bulk_mask, cfar_rank):
    S = np.asarray(fine_powers)
    if S.shape != (3, FINE_BINS):
        raise ValueError("fine_powers must have shape [3, 256].")
    if S.dtype != np.dtype(np.uint64):
        raise TypeError("fine_powers must have exact uint64 dtype.")
    rank = _exact_integer(cfar_rank, field="cfar_rank")
    if not 0 <= rank < FINE_BINS:
        raise ValueError(f"cfar_rank must be in [0, {FINE_BINS - 1}].")
    designated = [int(b) for b in designated_bins(anchor_bin, designated_half_width)]
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
    bulk = [b for b in range(FINE_BINS) if mask_arr[b] and den[b] > 0]
    if rank >= len(bulk):
        return num, den, designated, bulk, -1
    value = sorted(den[b] for b in bulk)[rank]
    # The smallest bin that holds the rank value represents it, as in v1.
    return num, den, designated, bulk, next(b for b in bulk if den[b] == value)


def fine_mask_decision_v2(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Sequence[bool] | Sequence[int],
    cfar_rank: int,
    multiplier_q16: int,
) -> FineDecision:
    """Return fine decision v2 for the exact fine powers of one frame.

    The frame is rejected (mask = 1) when any designated bin satisfies
    ``num[b] * 2**16 > multiplier_q16 * den_r``, where ``den_r`` is the
    reference power of rank ``cfar_rank`` over the background bins.
    """
    mult = _exact_integer(multiplier_q16, field="multiplier_q16")
    if not 0 < mult < (1 << 64):
        raise ValueError("multiplier_q16 must be in [1, 2**64 - 1].")
    num, den, designated, bulk, rank_bin = _v2_inputs(
        fine_powers, anchor_bin, designated_half_width, bulk_mask, cfar_rank)
    if rank_bin < 0:
        return FineDecision(
            mask=0, valid=False, n_bulk=len(bulk), rank_bin=-1, fired_bin=-1)
    den_r = den[rank_bin]
    fired_bin = next(
        (b for b in designated if num[b] * MULTIPLIER_ONE > mult * den_r), -1)
    return FineDecision(
        mask=1 if fired_bin >= 0 else 0, valid=True, n_bulk=len(bulk),
        rank_bin=rank_bin, fired_bin=fired_bin)


def fine_required_multiplier_q16_v2(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Sequence[bool] | Sequence[int],
    cfar_rank: int,
) -> FineRequiredMultiplier:
    """Return the exact Q16 keep boundary of fine decision v2 for one frame.

    The boundary is the smallest multiplier at which no designated bin
    fires. ``ALWAYS_MASKED_Q16`` marks a valid frame that no uint64
    multiplier keeps, and ``None`` an invalid rank.
    """
    num, den, designated, bulk, rank_bin = _v2_inputs(
        fine_powers, anchor_bin, designated_half_width, bulk_mask, cfar_rank)
    if rank_bin < 0:
        return FineRequiredMultiplier(
            valid=False, n_bulk=len(bulk), rank_bin=-1, limiting_bin=-1,
            multiplier_q16=None)
    den_r = den[rank_bin]
    required = 1
    limiting_bin = -1
    for b in designated:
        boundary = (num[b] * MULTIPLIER_ONE + den_r - 1) // den_r
        if boundary > required:
            required = boundary
            limiting_bin = b
    if required > MAX_MULTIPLIER_Q16:
        required = ALWAYS_MASKED_Q16
    return FineRequiredMultiplier(
        valid=True, n_bulk=len(bulk), rank_bin=rank_bin,
        limiting_bin=limiting_bin, multiplier_q16=required)


FINE_DECISION_V3_VERSION = "fine_decision_v3"
# Fine decision v3 takes the reference median over the even bins, which zero
# padding leaves as the independent samples of the unpadded transform.
V3_REFERENCE_BINS = tuple(range(0, FINE_BINS, FINE_PAD_FACTOR))
V3_RANK = (len(V3_REFERENCE_BINS) - 1) // 2


@dataclass(frozen=True)
class FineStat:
    """The statistic of fine decision v3 for one frame (kernel d_stat_out)."""

    n_peak: int        # largest num[b] = 2 S_target[b] over the designated set
    d_rho: int         # den of rank V3_RANK over the even bins, zeros included
    peak_bin: int      # first designated bin holding n_peak
    rank_bin: int      # lowest even bin holding d_rho


def fine_stat_v3(fine_powers: Any, *, anchor_bin: int, designated_half_width: int) -> FineStat:
    """Return the fine decision v3 statistic of one frame's exact fine powers.

    The GPU entry PilotProxy_Compute_FineStatV3_U64 must return the same
    ``[n_peak, d_rho]``. ``d_rho`` is zero when at least half of the even bins
    have no reference power, and the frame then has no decision.
    """
    S = np.asarray(fine_powers)
    if S.shape != (3, FINE_BINS):
        raise ValueError("fine_powers must have shape [3, 256].")
    if S.dtype != np.dtype(np.uint64):
        raise TypeError("fine_powers must have exact uint64 dtype.")
    designated = [int(b) for b in designated_bins(anchor_bin, designated_half_width)]
    num = [2 * int(S[WEIGHT_TERM_TARGET, b]) for b in range(FINE_BINS)]
    den = [int(S[WEIGHT_TERM_REF_LOWER, b]) + int(S[WEIGHT_TERM_REF_UPPER, b]) for b in range(FINE_BINS)]
    d_rho = sorted(den[b] for b in V3_REFERENCE_BINS)[V3_RANK]
    n_peak = max(num[b] for b in designated)
    return FineStat(n_peak=n_peak, d_rho=d_rho, peak_bin=next(b for b in designated if num[b] == n_peak),
                    rank_bin=next(b for b in V3_REFERENCE_BINS if den[b] == d_rho))


def fine_keep_v3(n_peak: int, d_rho: int, multiplier_q16: int) -> int | None:
    """Return 1 to keep, 0 to flag, or None when ``d_rho`` is zero (no decision).

    The frame is kept when ``n_peak * 2**16 <= multiplier_q16 * d_rho``. As in
    Kotekan's RFI mask, 1 keeps. This is pilotproxy_v3_keep in
    cuda/pilotproxy_decision.h.
    """
    mult = _exact_integer(multiplier_q16, field="multiplier_q16")
    if not 0 < mult < (1 << 64):
        raise ValueError("multiplier_q16 must be in [1, 2**64 - 1].")
    n, d = _exact_integer(n_peak, field="n_peak"), _exact_integer(d_rho, field="d_rho")
    if d == 0:
        return None
    return int(n * MULTIPLIER_ONE <= mult * d)


def fine_keep_word_v3(n_peak: int, d_rho: int) -> int | None:
    """Return the smallest Q16 word that keeps the frame.

    ``ALWAYS_MASKED_Q16`` marks a frame that no uint64 word keeps, and ``None``
    a frame with no decision (``d_rho`` zero).
    """
    n, d = _exact_integer(n_peak, field="n_peak"), _exact_integer(d_rho, field="d_rho")
    if d == 0:
        return None
    word = max(1, (n * MULTIPLIER_ONE + d - 1) // d)
    return ALWAYS_MASKED_Q16 if word > MAX_MULTIPLIER_Q16 else word


def fine_mask_decision_v3(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    multiplier_q16: int,
    bulk_mask: Any = None,
    cfar_rank: Any = None,
) -> FineDecision:
    """Return fine decision v3 in the form of the v1 and v2 decisions (``mask`` = 1 rejects).

    ``bulk_mask`` and ``cfar_rank`` are accepted for a common call and not used:
    v3 fixes the even bins and rank 63. A frame with ``d_rho = 0`` is invalid.
    """
    stat = fine_stat_v3(fine_powers, anchor_bin=anchor_bin, designated_half_width=designated_half_width)
    keep = fine_keep_v3(stat.n_peak, stat.d_rho, multiplier_q16)
    n_bulk = len(V3_REFERENCE_BINS)
    if keep is None:
        return FineDecision(mask=0, valid=False, n_bulk=n_bulk, rank_bin=-1, fired_bin=-1)
    return FineDecision(mask=1 - keep, valid=True, n_bulk=n_bulk, rank_bin=stat.rank_bin,
                        fired_bin=-1 if keep else stat.peak_bin)


def fine_required_multiplier_q16_v3(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Any = None,
    cfar_rank: Any = None,
) -> FineRequiredMultiplier:
    """Return the exact Q16 keep boundary of fine decision v3 for one frame.

    As for v2, ``ALWAYS_MASKED_Q16`` marks a frame no uint64 word keeps and
    ``None`` a frame with no decision. ``bulk_mask`` and ``cfar_rank`` are not used.
    """
    stat = fine_stat_v3(fine_powers, anchor_bin=anchor_bin, designated_half_width=designated_half_width)
    word = fine_keep_word_v3(stat.n_peak, stat.d_rho)
    n_bulk = len(V3_REFERENCE_BINS)
    if word is None:
        return FineRequiredMultiplier(valid=False, n_bulk=n_bulk, rank_bin=-1, limiting_bin=-1, multiplier_q16=None)
    return FineRequiredMultiplier(valid=True, n_bulk=n_bulk, rank_bin=stat.rank_bin,
                                  limiting_bin=stat.peak_bin if word > 1 else -1, multiplier_q16=word)


def v3_geometry() -> tuple[np.ndarray, int]:
    """Return the reference bins of fine decision v3 as a 256-bin mask, and its rank."""
    mask = np.zeros(FINE_BINS, dtype=bool)
    mask[list(V3_REFERENCE_BINS)] = True
    return mask, V3_RANK


def calibration_geometry(calibration: dict) -> tuple[np.ndarray, int]:
    """Return the reference-bin mask and rank of a bundle's fine_calibration block.

    Versions 1 and 2 carry them as ``bulk_mask_words_hex`` and ``cfar_rank``.
    Version 3 carries neither, and uses the even bins at rank 63.
    """
    if calibration.get("decision_version") == FINE_DECISION_V3_VERSION:
        return v3_geometry()
    words = calibration["bulk_mask_words_hex"]
    return unpack_bulk_mask([int(w, 16) for w in words]), _exact_integer(calibration["cfar_rank"], field="cfar_rank")


FINE_DECISION_VERSIONS = (FINE_DECISION_VERSION, FINE_DECISION_V2_VERSION, FINE_DECISION_V3_VERSION)
# New calibrations write version 2 until Kotekan runs version 3; a bundle of any version stays valid.
CALIBRATION_DECISION_VERSION = FINE_DECISION_V2_VERSION


def decision_functions(version: str):
    """Return the mask decision and the exact keep boundary of a decision version."""
    if version == FINE_DECISION_VERSION:
        return fine_mask_decision, fine_required_multiplier_q16
    if version == FINE_DECISION_V2_VERSION:
        return fine_mask_decision_v2, fine_required_multiplier_q16_v2
    if version == FINE_DECISION_V3_VERSION:
        return fine_mask_decision_v3, fine_required_multiplier_q16_v3
    raise ValueError(f"unknown decision_version {version!r}, expected one of {FINE_DECISION_VERSIONS}")
