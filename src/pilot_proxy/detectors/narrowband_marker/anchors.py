"""Locate the pilot in a stable era and define the reference bins around it.

The definitions below are quoted from the dissertation.

ch08 ``sec:calibration:anchors``, eq:calibration:anchor: "For fine bin f, let
T_f^on and T_f^quiet denote the sets of recorded fine statistics in
coarse-detected and coarse-quiet frames of the current era. The primary anchor
estimator is f_a = argmax_f [median(T_f^on) - median(T_f^quiet)]. Static
structure common to both cohorts cancels, so the contrast isolates the feature
associated with the transmitter state rather than simply returning the largest
fixed instrumental bin. A channel with no defensible quiet cohort uses a
plain-median fallback and is labelled as such; that fallback is adequate to
place a designated window around an always-present line but does not create a
transmitter-off null."

ch06 ``sec:detection:rule``: "Let f_a be the measured pilot anchor. The
designated set spans the padded main lobe, D = {(f_a + k) mod L_F : |k| <= 2}.
The usable null bulk B contains alternate bins from the independent subset
..., excludes D and any census-marked persistent features, and omits bins with
zero reference denominator. An even-parity anchor leaves 125 bulk bins and an
odd-parity anchor leaves 126." The zero-denominator omission is made per
frame by the frame scorer (:mod:`.scores`). The bulk mask in this module is
the static part of B.

The inputs are a :class:`~pilot_proxy.products.reader.Product` and a boolean
mask of the frames of the era under study. The mask is supplied by the
caller and can be ``product.selected`` for the whole archive, an era mask from
``characterization.eras``, or a block mask from ``characterization.blocks``.
A previous era's anchor is found with the same function on a second mask. The
per-frame ``T_f`` is ``products.reader.fine_power_ratio`` of the exact fine
terms, which are read once per channel and released.

The two cohorts are ``on = mask & valid & rejected``, where ``rejected`` is
the survey flag ``F > mu_0``, and ``quiet = mask & valid & ~rejected``. The
dissertation names ``valid`` only for the quiet cohort, but it is applied to
both cohorts here. This has no effect on the products, whose masks are
subsets of ``valid``. The estimator of record (``method = 'on_minus_quiet'``)
requires ``MIN_COHORT_FRAMES`` (30) frames in each cohort. When the quiet
cohort is smaller, the contrast is the plain median over the on frames. When
fewer than 30 frames are coarse-detected, the contrast is the plain median
over all masked frames. Both cases are labeled ``method = 'median_fallback'``,
with ``fallback_cohort`` set to 'on' or 'all'.

The anchor is the argmax of the contrast over all 256 bins (design section
6). A second argmax, over ``nominal_fine_bin +- WINDOW_HALF_WIDTH`` (30 bins,
the scan's acquisition window), is reported beside it, and
``aliased_out_of_window`` is set when the two differ. This flag records only
the disagreement. The out-of-window argmax can be the alias of a feature off
the fine span, such as channel 33's co-channel carrier at -3.7 kHz, folded by
one coarse bin to -674 Hz. It can also be a lobe in the fine span but past the
window, such as the lobe of channel 23 at 33 bins. Which case applies is
determined from the per-frame spectra by ``archive.psd``. Ties resolve to the
lowest bin. Since the runner-up is the best bin not in D(f_a), the padded
shoulders of the lobe itself are not counted as competitors. ``separation`` is
the anchor contrast minus the runner-up contrast.

Each median is ``blocks.weighted_quantile`` at ``q = 0.5``, which is the
lower middle value for an even count. It is evaluated column-wise in one pass
by :class:`SortedCohort`, so the point estimate and every bootstrap replicate
use the same statistic. The tests check the vectorized form against
``blocks.weighted_quantile``.

The anchor is reported in four coordinates (design section 2).
``anchor_bin`` is the padded fine bin, and ``anchor_fine_hz`` is its
fine-axis offset (``products.fine_hz_of_bin``). ``anchor_offset_bins`` is
the signed offset from the nominal bin, unwrapped to [-128, 128) and positive
toward higher padded bins. ``anchor_rf_offset_hz`` is the RF offset from the
nominal pilot, given by ``products.fine_offset_to_rf_hz`` with the geometry's
grid residual and sense. It is positive toward higher RF, so under the
campaign's ``sense = -1`` a positive bin offset is a negative RF offset.

The spread of the anchor is estimated with ``blocks.block_bootstrap``, which
resamples whole acquisitions. It uses ``BOOTSTRAP_REPLICATES`` (1000)
replicates and seed ``BOOTSTRAP_SEED`` (20260907). The replicate statistic is
the argmax of the weighted-median contrast as an unwrapped offset from the
nominal bin, so its quantiles do not straddle the wrap. Its 2.5, 16, 84, and
97.5 percentiles are reported in bins from nominal and in RF Hz (the same
replicates converted), along with the modal replicate bin, its probability
mass, and the mass at the point estimate. Both the method and its cohorts are
fixed by the point estimate. A replicate whose cohort draws no frame is NaN
and is not counted in ``boot_replicates_used``.

The bulk is ``B = independent_bin_mask(256, pad_factor, designated_bins=(f_a,),
guard_fine_bins, census_excluded_bins)`` with pilot-proxy's semantics. That
is, B is every ``pad_factor``-th bin, minus the ``f_a +- guard * pad`` padded
bins and the census-excluded bins. D is removed as well, so the bundle's
contract holds for any guard. With the campaign's pad factor of 2 and guard
of 1, the guard removes exactly D, giving 125 bulk bins for an even anchor
and 126 for an odd one.

Two calibration options of detector v2 change the marker geometry. With
``marker_guard_bins = g``, the ``g`` padded bins on each side of D are also
removed from the bulk. These are ``{(f_a + k) mod L_F : h < |k| <= h + g}``
for half-width ``h``, so a line just past the markers cannot enter the
reference rank (Paper I finds such lines on channels 22 and 28).
``designated_half_width`` widens the markers themselves. The defaults
(``g = 0``, ``h = 2``) are the calibration of record. A nonzero guard is
recorded in the bundle as ``fine_calibration.marker_guard_bins``, and the
anchor table's ``bulk_size`` reflects the guard band.

The anchor table (``ANCHOR_COLUMNS``) has one row per channel per mask label
and the columns ``channel, freq_id, label, status, frames_masked, frames_on,
frames_quiet, min_cohort_frames, method, fallback_cohort, nominal_fine_bin,
grid_residual_hz, sense, anchor_bin, anchor_fine_hz, anchor_offset_bins,
anchor_rf_offset_hz, anchor_contrast, runner_up_bin, runner_up_contrast,
separation, window_half_width, window_anchor_bin, window_anchor_offset_bins,
window_anchor_rf_offset_hz, window_anchor_contrast, aliased_out_of_window,
designated_half_width, designated_bins, bulk_size, bulk_pad_factor,
bulk_guard_bins, census_excluded_count, boot_status, boot_replicates,
boot_replicates_used, boot_seed, boot_blocks, boot_offset_bins_q025,
boot_offset_bins_q16, boot_offset_bins_q84, boot_offset_bins_q975,
boot_rf_hz_q025, boot_rf_hz_q16, boot_rf_hz_q84, boot_rf_hz_q975,
boot_mode_bin, boot_mode_mass, boot_mass_at_anchor``. In the table,
``designated_bins`` is joined with ``;``, booleans are written as ``True`` or
``False``, floats as their ``repr()``, and NaN as a blank cell.

The file written by :func:`write_contrast_curves` has one row per bin per
channel per label, for the appendix plates, and the columns ``channel, label,
bin, offset_bins, fine_hz, rf_offset_hz, median_on, median_quiet, contrast,
in_window, in_designated, in_bulk``.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np

from pilot_proxy.characterization import blocks
from pilot_proxy.products.reader import (
    FINE_BIN_HZ,
    FINE_BINS,
    Geometry,
    Product,
    fine_hz_of_bin,
    fine_power_ratio,
)

MIN_COHORT_FRAMES = 30
WINDOW_HALF_WIDTH = 30
DESIGNATED_HALF_WIDTH = 2
MEDIAN_QUANTILE = 0.5
BOOTSTRAP_REPLICATES = blocks.DEFAULT_REPLICATES
BOOTSTRAP_SEED = blocks.DEFAULT_SEED
BOOTSTRAP_QUANTILES = (0.025, 0.16, 0.84, 0.975)
MIN_BOOTSTRAP_BLOCKS = blocks.MIN_BLOCKS_PER_HALF

METHOD_ON_MINUS_QUIET = "on_minus_quiet"
METHOD_MEDIAN_FALLBACK = "median_fallback"
METHOD_NONE = "none"


# ------------------------------------------------------------- bin algebra
def unwrap_bins(bin_index, reference_bin: int, fine_bins: int = FINE_BINS):
    """Signed offset of padded bins from a reference bin, unwrapped to [-L/2, L/2)."""
    b = np.asarray(bin_index)
    return (b - int(reference_bin) + fine_bins // 2) % fine_bins - fine_bins // 2


def rf_offset_of_bin(bin_index, geometry: Geometry):
    """RF offset from the nominal pilot, in Hz, of padded fine bins.

    The bin offset is unwrapped about the nominal bin, which is the same wrap
    :func:`unwrap_bins` gives ``anchor_offset_bins``. The two coordinates
    therefore agree at every offset, even -128. A wrap about the
    real-valued grid residual would put that one bin a full coarse bin away
    whenever the residual lies above the center of the nominal bin.
    """
    nominal = int(geometry.nominal_fine_bin)
    fine_hz = unwrap_bins(bin_index, nominal) * FINE_BIN_HZ + fine_hz_of_bin(nominal)
    return float(np.sign(geometry.sense) or 1) * (fine_hz - geometry.grid_residual_hz)


def designated_set(anchor_bin: int, half_width: int = DESIGNATED_HALF_WIDTH,
                   fine_bins: int = FINE_BINS) -> tuple[int, ...]:
    """``D = {(f_a + k) mod L_F : |k| <= half_width}`` in k order."""
    return tuple(int((int(anchor_bin) + k) % fine_bins) for k in range(-int(half_width), int(half_width) + 1))


def window_bins(centre_bin: int, half_width: int = WINDOW_HALF_WIDTH, fine_bins: int = FINE_BINS) -> np.ndarray:
    """The bins ``centre_bin +- half_width`` (mod L_F), in k order."""
    return (int(centre_bin) + np.arange(-int(half_width), int(half_width) + 1)) % fine_bins


def independent_bin_mask(num_bins: int, *, pad_factor: int, designated_bins: Sequence[int] = (),
                         guard_fine_bins: int, census_excluded_bins: Sequence[int] = ()) -> np.ndarray:
    """Return the same mask as ``pilot_proxy.fine_reduction.independent_bin_mask``.

    The mask is every ``pad_factor``-th bin, minus each designated bin with
    ``guard_fine_bins * pad_factor`` padded bins on either side, and minus the
    census-excluded bins.
    """
    n = int(num_bins)
    mask = np.zeros(n, dtype=bool)
    mask[:: int(pad_factor)] = True
    guard = int(guard_fine_bins) * int(pad_factor)
    for b in designated_bins:
        mask[np.arange(int(b) - guard, int(b) + guard + 1) % n] = False
    for b in census_excluded_bins:
        mask[int(b) % n] = False
    return mask


def bulk_mask(anchor_bin: int, *, pad_factor: int, guard_fine_bins: int, census_excluded_bins: Sequence[int] = (),
              designated_half_width: int = DESIGNATED_HALF_WIDTH, marker_guard_bins: int = 0,
              fine_bins: int = FINE_BINS) -> np.ndarray:
    """The static bulk ``B`` for a measured anchor, minus any marker guard band (module docstring)."""
    guard = int(marker_guard_bins)
    if guard != marker_guard_bins or guard < 0 or int(designated_half_width) + guard >= fine_bins // 2:
        raise ValueError("marker_guard_bins must be a nonnegative integer with half width + guard below L_F / 2")
    mask = independent_bin_mask(fine_bins, pad_factor=pad_factor, designated_bins=(int(anchor_bin),),
                                guard_fine_bins=guard_fine_bins, census_excluded_bins=census_excluded_bins)
    mask[list(designated_set(anchor_bin, int(designated_half_width) + guard, fine_bins))] = False
    return mask


# ------------------------------------------------------- weighted medians
class SortedCohort:
    """One cohort's ``T`` (frames x bins), column-sorted once, so the weighted
    median of every bin under new frame weights is a single pass.

    ``values`` must be finite (``fine_power_ratio`` is). ``rows`` are the frame
    indices of the cohort on the full frame axis. ``weighted_median`` takes the
    full-axis integer weights of ``blocks.block_bootstrap`` and returns the
    ``blocks.weighted_quantile`` value per bin (NaN where the cohort's total
    weight is zero).
    """

    def __init__(self, values: np.ndarray, rows: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != self.rows.size:
            raise ValueError("cohort values must be (frames, bins) with one row per cohort frame")
        if not np.isfinite(values).all():
            raise ValueError("cohort values must be finite")
        self.bins = values.shape[1]
        # bins x frames, so the per-replicate cumulative sums run along contiguous memory
        transposed = np.ascontiguousarray(values.T)
        self.order = np.argsort(transposed, axis=1, kind="stable").astype(np.int32)
        self.sorted = np.take_along_axis(transposed, self.order.astype(np.int64), axis=1)

    @property
    def frames(self) -> int:
        return int(self.rows.size)

    def weighted_median(self, weights: np.ndarray, q: float = MEDIAN_QUANTILE) -> np.ndarray:
        if self.rows.size == 0:
            return np.full(self.bins, np.nan)
        w = np.asarray(weights)[self.rows]
        total = int(w.sum())
        if total <= 0:
            return np.full(self.bins, np.nan)
        cdf = np.cumsum(w.astype(np.int32)[self.order], axis=1, dtype=np.float64) / float(total)
        index = np.count_nonzero(cdf < q, axis=1).clip(0, self.rows.size - 1)
        return self.sorted[np.arange(self.bins), index]


@dataclass(frozen=True)
class ContrastEstimator:
    """The contrast curve of one era as a function of frame weights.

    ``method`` and ``fallback_cohort`` are fixed from the unweighted cohort
    sizes. ``contrast(weights)`` returns ``median_on - median_quiet`` for the
    estimator of record and the plain weighted median of the fallback cohort
    otherwise.
    """

    on: SortedCohort
    quiet: SortedCohort
    fallback: SortedCohort | None
    method: str
    fallback_cohort: str
    nominal_fine_bin: int

    def contrast(self, weights: np.ndarray) -> np.ndarray:
        if self.method == METHOD_ON_MINUS_QUIET:
            return self.on.weighted_median(weights) - self.quiet.weighted_median(weights)
        if self.fallback is None:
            return np.full(self.on.bins, np.nan)
        return self.fallback.weighted_median(weights)

    def anchor_bin(self, weights: np.ndarray) -> int:
        """Argmax of the contrast over all bins (lowest bin on ties), or -1 when undefined."""
        c = self.contrast(weights)
        return int(np.argmax(c)) if np.isfinite(c).any() else -1

    def anchor_offset(self, weights: np.ndarray) -> float:
        """The bootstrap statistic: the argmax as an unwrapped offset from the nominal bin."""
        b = self.anchor_bin(weights)
        return float(unwrap_bins(b, self.nominal_fine_bin)) if b >= 0 else float("nan")


def build_estimator(ratio: np.ndarray, *, mask, valid, rejected, nominal_fine_bin: int,
                    min_cohort_frames: int = MIN_COHORT_FRAMES, quiet_usable: bool = True) -> ContrastEstimator:
    """Build the contrast estimator from the per-frame ratio ``(N, 256)`` and the frame flags.

    ``quiet_usable`` is set to False by the caller when the coarse-quiet
    frames are known not to be a null population. This is the case when the
    block's bulk is far above ``mu_0``. The frames below ``mu_0`` are then
    the lower tail of the carrier-on distribution, and an on-minus-quiet
    contrast would compare the carrier with itself. The plain median of the
    on cohort is used instead, with the label 'on (quiet cohort is not a
    null)'.
    """
    mask = np.asarray(mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    rejected = np.asarray(rejected, dtype=bool)
    on_rows = np.flatnonzero(mask & valid & rejected)
    quiet_rows = np.flatnonzero(mask & valid & ~rejected)
    all_rows = np.flatnonzero(mask & valid)
    on = SortedCohort(ratio[on_rows], on_rows)
    quiet = SortedCohort(ratio[quiet_rows], quiet_rows)
    if all_rows.size == 0:
        return ContrastEstimator(on, quiet, None, METHOD_NONE, "", int(nominal_fine_bin))
    if on_rows.size >= min_cohort_frames and quiet_rows.size >= min_cohort_frames and quiet_usable:
        return ContrastEstimator(on, quiet, None, METHOD_ON_MINUS_QUIET, "", int(nominal_fine_bin))
    if on_rows.size >= min_cohort_frames:
        label = "on" if quiet_usable else "on (quiet cohort is not a null)"
        return ContrastEstimator(on, quiet, on, METHOD_MEDIAN_FALLBACK, label, int(nominal_fine_bin))
    return ContrastEstimator(on, quiet, SortedCohort(ratio[all_rows], all_rows), METHOD_MEDIAN_FALLBACK, "all",
                             int(nominal_fine_bin))


# ------------------------------------------------------------------ result
@dataclass(frozen=True)
class AnchorResult:
    """One channel's anchor on one frame mask, with every column of the table and the curves."""

    channel: int
    freq_id: int
    label: str
    status: str                         # 'ok' | 'empty' (no masked valid frame)
    frames_masked: int                  # masked & valid
    frames_on: int                      # masked & valid & rejected
    frames_quiet: int                   # masked & valid & ~rejected
    min_cohort_frames: int
    method: str                         # 'on_minus_quiet' | 'median_fallback' | 'none'
    fallback_cohort: str                # '' | 'on' | 'all'
    nominal_fine_bin: int
    grid_residual_hz: float
    sense: int
    anchor_bin: int                     # padded fine bin of the argmax over all bins, -1 when empty
    anchor_fine_hz: float               # its fine-axis offset
    anchor_offset_bins: int             # unwrapped signed offset from the nominal bin (fine-array direction)
    anchor_rf_offset_hz: float          # RF offset from the nominal pilot, positive toward higher RF
    anchor_contrast: float              # the contrast curve at the anchor
    runner_up_bin: int                  # best bin not in D(f_a)
    runner_up_contrast: float
    separation: float                   # anchor_contrast - runner_up_contrast
    window_half_width: int
    window_anchor_bin: int              # argmax over nominal +- window_half_width
    window_anchor_offset_bins: int
    window_anchor_rf_offset_hz: float
    window_anchor_contrast: float
    aliased_out_of_window: bool         # anchor_bin != window_anchor_bin
    designated_half_width: int
    designated_bins: tuple[int, ...]    # D(f_a) in k order
    bulk_size: int                      # |B|
    bulk_pad_factor: int
    bulk_guard_bins: int
    census_excluded_count: int
    boot_status: str                    # 'ok' | 'insufficient_blocks' | 'skipped' | 'empty'
    boot_replicates: int
    boot_replicates_used: int           # replicates with a defined anchor
    boot_seed: int
    boot_blocks: int                    # acquisitions resampled
    boot_offset_bins_q025: float        # percentiles of the replicate anchor, offset from nominal, bins
    boot_offset_bins_q16: float
    boot_offset_bins_q84: float
    boot_offset_bins_q975: float
    boot_rf_hz_q025: float              # the same replicates in RF Hz from nominal
    boot_rf_hz_q16: float
    boot_rf_hz_q84: float
    boot_rf_hz_q975: float
    boot_mode_bin: int                  # most frequent replicate anchor (padded bin), -1 when none
    boot_mode_mass: float               # its share of the defined replicates
    boot_mass_at_anchor: float          # share of defined replicates equal to the point estimate
    contrast: np.ndarray = field(repr=False, compare=False)       # (256,) the curve of record
    median_on: np.ndarray = field(repr=False, compare=False)      # (256,) NaN when no on frame
    median_quiet: np.ndarray = field(repr=False, compare=False)   # (256,) NaN when no quiet frame
    bulk: np.ndarray = field(repr=False, compare=False)           # (256,) bool, B
    boot_samples: np.ndarray = field(repr=False, compare=False)   # (replicates,) offsets, NaN where undefined

    @property
    def is_fallback(self) -> bool:
        return self.method == METHOD_MEDIAN_FALLBACK


ANCHOR_COLUMNS = tuple(f.name for f in fields(AnchorResult) if f.name not in
                       ("contrast", "median_on", "median_quiet", "bulk", "boot_samples"))
CONTRAST_COLUMNS = ("channel", "label", "bin", "offset_bins", "fine_hz", "rf_offset_hz",
                    "median_on", "median_quiet", "contrast", "in_window", "in_designated", "in_bulk")


def anchor_shift_bins(current: AnchorResult, previous: AnchorResult) -> int:
    """Unwrapped shift of the anchor between two eras, in padded bins (current minus previous)."""
    if current.anchor_bin < 0 or previous.anchor_bin < 0:
        raise ValueError("both anchors must be defined")
    return int(unwrap_bins(current.anchor_bin, previous.anchor_bin))


# ------------------------------------------------------------- estimation
def fine_ratio(product: Product) -> np.ndarray:
    """``T`` for every frame, ``(N, 256)`` float64, from the exact terms read once and released."""
    terms = product.fine_terms_all()
    try:
        return fine_power_ratio(terms)
    finally:
        del terms


def _nan_row(geometry: Geometry, channel: int, freq_id: int, label: str, frames_masked: int, frames_on: int,
             frames_quiet: int, params: dict) -> AnchorResult:
    nan = float("nan")
    empty = np.full(FINE_BINS, np.nan)
    return AnchorResult(
        channel=channel, freq_id=freq_id, label=label, status="empty",
        frames_masked=frames_masked, frames_on=frames_on, frames_quiet=frames_quiet,
        min_cohort_frames=params["min_cohort_frames"], method=METHOD_NONE, fallback_cohort="",
        nominal_fine_bin=geometry.nominal_fine_bin, grid_residual_hz=geometry.grid_residual_hz, sense=geometry.sense,
        anchor_bin=-1, anchor_fine_hz=nan, anchor_offset_bins=0, anchor_rf_offset_hz=nan, anchor_contrast=nan,
        runner_up_bin=-1, runner_up_contrast=nan, separation=nan,
        window_half_width=params["window_half_width"], window_anchor_bin=-1, window_anchor_offset_bins=0,
        window_anchor_rf_offset_hz=nan, window_anchor_contrast=nan, aliased_out_of_window=False,
        designated_half_width=params["designated_half_width"], designated_bins=(), bulk_size=0,
        bulk_pad_factor=params["pad_factor"], bulk_guard_bins=params["guard_fine_bins"],
        census_excluded_count=len(params["census_excluded_bins"]),
        boot_status="empty", boot_replicates=params["replicates"], boot_replicates_used=0, boot_seed=params["seed"],
        boot_blocks=0, boot_offset_bins_q025=nan, boot_offset_bins_q16=nan, boot_offset_bins_q84=nan,
        boot_offset_bins_q975=nan, boot_rf_hz_q025=nan, boot_rf_hz_q16=nan, boot_rf_hz_q84=nan, boot_rf_hz_q975=nan,
        boot_mode_bin=-1, boot_mode_mass=nan, boot_mass_at_anchor=nan,
        contrast=empty, median_on=empty.copy(), median_quiet=empty.copy(), bulk=np.zeros(FINE_BINS, bool),
        boot_samples=np.empty(0),
    )


def anchor_from_ratio(ratio: np.ndarray, *, mask, valid, rejected, unit_index, geometry: Geometry,
                      channel: int, freq_id: int, label: str, pad_factor: int, guard_fine_bins: int,
                      census_excluded_bins: Sequence[int] = (),
                      min_cohort_frames: int = MIN_COHORT_FRAMES, window_half_width: int = WINDOW_HALF_WIDTH,
                      designated_half_width: int = DESIGNATED_HALF_WIDTH, marker_guard_bins: int = 0,
                      replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
                      min_blocks: int = MIN_BOOTSTRAP_BLOCKS, quiet_usable: bool = True) -> AnchorResult:
    """The anchor from a precomputed per-frame ratio ``(N, 256)`` and the frame flags.

    :func:`anchor` is the wrapper that takes a product. This form allows for
    several masks (eras, blocks) to share one read of the fine terms, and for
    tests to pass arrays.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    if ratio.ndim != 2 or ratio.shape[1] != FINE_BINS:
        raise ValueError(f"ratio must be (frames, {FINE_BINS}); got {ratio.shape}")
    mask = np.asarray(mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    rejected = np.asarray(rejected, dtype=bool)
    if not (mask.shape == valid.shape == rejected.shape == (ratio.shape[0],)):
        raise ValueError("mask, valid and rejected must each be (frames,)")
    census = tuple(int(b) for b in np.asarray(census_excluded_bins, dtype=np.int64).reshape(-1))
    params = dict(min_cohort_frames=int(min_cohort_frames), window_half_width=int(window_half_width),
                  designated_half_width=int(designated_half_width), pad_factor=int(pad_factor),
                  guard_fine_bins=int(guard_fine_bins), census_excluded_bins=census,
                  replicates=int(replicates), seed=int(seed))
    est = build_estimator(ratio, mask=mask, valid=valid, rejected=rejected, quiet_usable=bool(quiet_usable),
                          nominal_fine_bin=geometry.nominal_fine_bin, min_cohort_frames=params["min_cohort_frames"])
    frames_masked = int((mask & valid).sum())
    if est.method == METHOD_NONE:
        return _nan_row(geometry, channel, freq_id, label, frames_masked, est.on.frames, est.quiet.frames, params)

    selected = mask & valid
    unit = np.asarray(unit_index)
    unit_weights = np.zeros(ratio.shape[0], dtype=np.int64)
    unit_weights[selected] = 1

    # point estimate
    contrast = est.contrast(unit_weights)
    median_on = est.on.weighted_median(unit_weights)
    median_quiet = est.quiet.weighted_median(unit_weights)
    anchor_bin = int(np.argmax(contrast))
    designated = designated_set(anchor_bin, params["designated_half_width"])
    outside = np.ones(FINE_BINS, dtype=bool)
    outside[list(designated)] = False
    runner_up = int(np.flatnonzero(outside)[np.argmax(contrast[outside])]) if outside.any() else -1
    window = np.sort(window_bins(geometry.nominal_fine_bin, params["window_half_width"]))   # ties: lowest bin
    window_anchor = int(window[np.argmax(contrast[window])])
    bulk = bulk_mask(anchor_bin, pad_factor=params["pad_factor"], guard_fine_bins=params["guard_fine_bins"],
                     census_excluded_bins=census, designated_half_width=params["designated_half_width"],
                     marker_guard_bins=marker_guard_bins)

    # acquisition block bootstrap of the argmax
    nan = float("nan")
    boot_status, boot_blocks, samples = "skipped", int(np.unique(unit[selected]).size), np.empty(0)
    if params["replicates"] > 0:
        try:
            boot = blocks.block_bootstrap(unit, selected, est.anchor_offset, replicates=params["replicates"],
                                          seed=params["seed"], quantiles=BOOTSTRAP_QUANTILES, min_blocks=min_blocks)
        except ValueError as exc:
            if "at least" not in str(exc):
                raise
            boot_status = "insufficient_blocks"
        else:
            boot_status, boot_blocks, samples = "ok", boot.blocks, np.asarray(boot.samples, dtype=float)
    finite = samples[np.isfinite(samples)]
    if finite.size:
        offsets = np.rint(finite).astype(np.int64)
        replicate_bins = (geometry.nominal_fine_bin + offsets) % FINE_BINS
        rf = np.asarray(rf_offset_of_bin(replicate_bins, geometry), dtype=float)
        q_bins = [float(v) for v in np.quantile(finite, list(BOOTSTRAP_QUANTILES))]
        q_rf = [float(v) for v in np.quantile(rf, list(BOOTSTRAP_QUANTILES))]
        values, counts = np.unique(offsets, return_counts=True)
        mode_offset = int(values[np.argmax(counts)])
        mode_bin = int((geometry.nominal_fine_bin + mode_offset) % FINE_BINS)
        mode_mass = float(counts.max() / finite.size)
        mass_at_anchor = float(np.count_nonzero(replicate_bins == anchor_bin) / finite.size)
    else:
        q_bins, q_rf = [nan] * 4, [nan] * 4
        mode_bin, mode_mass, mass_at_anchor = -1, nan, nan

    return AnchorResult(
        channel=int(channel), freq_id=int(freq_id), label=str(label), status="ok",
        frames_masked=frames_masked, frames_on=est.on.frames, frames_quiet=est.quiet.frames,
        min_cohort_frames=params["min_cohort_frames"], method=est.method, fallback_cohort=est.fallback_cohort,
        nominal_fine_bin=geometry.nominal_fine_bin, grid_residual_hz=geometry.grid_residual_hz, sense=geometry.sense,
        anchor_bin=anchor_bin, anchor_fine_hz=float(fine_hz_of_bin(anchor_bin)),
        anchor_offset_bins=int(unwrap_bins(anchor_bin, geometry.nominal_fine_bin)),
        anchor_rf_offset_hz=float(rf_offset_of_bin(anchor_bin, geometry)),
        anchor_contrast=float(contrast[anchor_bin]),
        runner_up_bin=runner_up, runner_up_contrast=float(contrast[runner_up]) if runner_up >= 0 else nan,
        separation=float(contrast[anchor_bin] - contrast[runner_up]) if runner_up >= 0 else nan,
        window_half_width=params["window_half_width"], window_anchor_bin=window_anchor,
        window_anchor_offset_bins=int(unwrap_bins(window_anchor, geometry.nominal_fine_bin)),
        window_anchor_rf_offset_hz=float(rf_offset_of_bin(window_anchor, geometry)),
        window_anchor_contrast=float(contrast[window_anchor]),
        aliased_out_of_window=bool(window_anchor != anchor_bin),
        designated_half_width=params["designated_half_width"], designated_bins=designated,
        bulk_size=int(bulk.sum()), bulk_pad_factor=params["pad_factor"], bulk_guard_bins=params["guard_fine_bins"],
        census_excluded_count=len(census),
        boot_status=boot_status, boot_replicates=params["replicates"], boot_replicates_used=int(finite.size),
        boot_seed=params["seed"], boot_blocks=boot_blocks,
        boot_offset_bins_q025=q_bins[0], boot_offset_bins_q16=q_bins[1], boot_offset_bins_q84=q_bins[2],
        boot_offset_bins_q975=q_bins[3],
        boot_rf_hz_q025=q_rf[0], boot_rf_hz_q16=q_rf[1], boot_rf_hz_q84=q_rf[2], boot_rf_hz_q975=q_rf[3],
        boot_mode_bin=mode_bin, boot_mode_mass=mode_mass, boot_mass_at_anchor=mass_at_anchor,
        contrast=contrast, median_on=median_on, median_quiet=median_quiet, bulk=bulk, boot_samples=samples,
    )


def anchor(product: Product, mask, label: str = "archive", *, ratio: np.ndarray | None = None,
           **params) -> AnchorResult:
    """The anchor of one product on one frame mask.

    ``ratio`` (from :func:`fine_ratio`) may be passed when several masks share
    a channel. Otherwise the fine terms are read here and released. Keyword
    parameters are those of :func:`anchor_from_ratio` after the geometry and
    product constants, which come from the product.
    """
    if ratio is None:
        ratio = fine_ratio(product)
    g = product.geometry
    return anchor_from_ratio(
        ratio, mask=mask, valid=product.valid, rejected=product.rejected, unit_index=product.frame_unit_index,
        geometry=g, channel=g.physical_channel, freq_id=g.freq_id, label=label,
        pad_factor=product.fine_pad_factor, guard_fine_bins=product.fine_guard_bins,
        census_excluded_bins=product.fine_census_excluded_bins, **params)


# ------------------------------------------------------------------ tables
def _cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, tuple):
        return ";".join(str(int(v)) for v in value)
    return str(value)


def anchor_row(result: AnchorResult) -> dict[str, str]:
    """The table row of one result, every column rendered."""
    return {name: _cell(getattr(result, name)) for name in ANCHOR_COLUMNS}


def _native(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, tuple):
        return ";".join(str(int(v)) for v in value)
    return value


def anchor_record(result: AnchorResult) -> dict:
    """The ledger record of one result as native ints, floats (NaN kept), and bools, with tuples joined with ';'."""
    return {name: _native(getattr(result, name)) for name in ANCHOR_COLUMNS}


def contrast_rows(result: AnchorResult) -> list[dict[str, str]]:
    """The 256 per-bin rows of one result for ``write_contrast_curves``."""
    bins = np.arange(FINE_BINS)
    in_window = np.zeros(FINE_BINS, dtype=bool)
    in_window[window_bins(result.nominal_fine_bin, result.window_half_width)] = True
    in_designated = np.zeros(FINE_BINS, dtype=bool)
    if result.designated_bins:
        in_designated[list(result.designated_bins)] = True
    offsets = unwrap_bins(bins, result.nominal_fine_bin)
    fine_hz = fine_hz_of_bin(bins)
    # same wrap as rf_offset_of_bin, about the nominal bin, so offsets -128 and 127 are one fine bin apart
    rf = float(np.sign(result.sense) or 1) * (offsets * FINE_BIN_HZ + fine_hz_of_bin(result.nominal_fine_bin)
                                              - result.grid_residual_hz)
    rows = []
    for b in bins:
        rows.append({
            "channel": _cell(result.channel), "label": result.label, "bin": _cell(b),
            "offset_bins": _cell(offsets[b]), "fine_hz": _cell(float(fine_hz[b])), "rf_offset_hz": _cell(float(rf[b])),
            "median_on": _cell(float(result.median_on[b])), "median_quiet": _cell(float(result.median_quiet[b])),
            "contrast": _cell(float(result.contrast[b])), "in_window": _cell(in_window[b]),
            "in_designated": _cell(in_designated[b]), "in_bulk": _cell(result.bulk[b]),
        })
    return rows


def write_contrast_curves(results: Sequence[AnchorResult], path: Path | str) -> Path:
    """Per-bin contrast curves, one row per bin per result, columns ``CONTRAST_COLUMNS``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CONTRAST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for result in results:
            for row in contrast_rows(result):
                writer.writerow(row)
    return path
