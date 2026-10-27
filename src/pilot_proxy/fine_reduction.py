# coding=utf-8
"""Time-coherent fine reduction over exact kernel row sums.

Stage 1 (CUDA or reference) emits exact int32 complex row sums
``z[n, m]`` for every weight term ``n`` and detector row ``m``. Rows are
term-major and stream-major, with re and im interleaved (see
``PilotProxy_Compute_RowSums_I32`` in ``cuda/pilotproxy_detector.h``). This
module is stage 2:

1. the rows of each term are reshaped to
   ``[num_streams, windows_per_stream]``, with no gather since the rows
   are stream-major,
2. an FFT is taken along the window axis with ``FINE_PAD_FACTOR``x zero
   padding,
3. the spectra are squared and summed incoherently over streams to give
   ``S[n, b]``,
4. the fine-bin statistic ``F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])`` is
   formed.

In deployment, the transform is the frozen fixed-point ``fxfft256``
(:func:`pilot_proxy.fxfft.fine_power_fx`), and the frame powers are
defined from it as ``P[n] = sum_b S[n, b]`` over the exact uint64 fine
powers (:func:`frame_powers_from_fine_u64`). This sum is the reference for
``PilotProxy_Compute_FinePowers_U64`` summed over bins and for the
``d_power_out`` of the fused entry point. Since the rounded transform does
not conserve energy exactly, ``P[n]`` differs from
``pad * windows * sum_m |z[n, m]|^2``, and the two are never compared.

:func:`fine_reduce` is the float prototype (``numpy.fft`` or
``cupy.fft``), and the deployed statistic is the fixed-point one above.
For the float prototype only, the Parseval identity
``sum_b S[n, b] == pad * windows * sum_m |z[n, m]|^2`` holds, and it is
checked at ULP tolerance in its own tests.

The CFAR prototype uses a pre-registered null-bulk calibration. Here, the
location is the median and the scale is the left-side spread (the median
minus the 15.87th percentile) of the independent, non-designated fine
bins. These are every ``FINE_PAD_FACTOR``-th bin, with the designated
transmitter bins and a guard margin removed, along with any
census-excluded bins. The mean of the full mixture is never used. When
the null bulk itself is contaminated (a flag fraction above
``fallback_flag_fraction``), the calibration falls back to lower
quantiles (location = P25, scale = (P25 - P2.275) / 2) and records the
mode.

Detections are declared on any bin against the null calibration, which
removes the census dependency of designated-only rules. For the same
null-bulk bins, the fraction that exceeds the fitted threshold is reported
as ``null_bulk_exceedance_fraction``. Since this fraction is measured on the
bins that set the threshold, it is an in-sample diagnostic and gives no
independent estimate of the false-alarm rate.

The float prototype runs on either backend, selected with ``xp=numpy``
(default) or ``xp=cupy``. Its integer row energies are computed with
integer dtypes on both backends. Under numpy, the FFTs use complex128
(the prototype and parity reference), and under cupy they use complex64
(production). The GPU result is compared with the prototype at ULP
tolerance in the CUDA test
``test_cupy_fine_reduction_matches_float64_prototype_within_ulp_gate``.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

FINE_PAD_FACTOR = 2
CFAR_LEFT_QUANTILE = 0.1587  # one-sided 1-sigma equivalent
CFAR_FALLBACK_LOCATION_QUANTILE = 0.25
CFAR_FALLBACK_LEFT_QUANTILE = 0.02275  # two-sided 2-sigma equivalent
CFAR_DEFAULT_P_FA = 1.0e-3
CFAR_DEFAULT_GUARD_FINE_BINS = 1
CFAR_DEFAULT_FALLBACK_FLAG_FRACTION = 0.2
CFAR_MODE_MEDIAN_LEFT = "median_left_side_scale"
CFAR_MODE_QUANTILE_FALLBACK = "quantile_fallback"

WEIGHT_TERM_TARGET = 0
WEIGHT_TERM_REF_LOWER = 1
WEIGHT_TERM_REF_UPPER = 2


def p_fa_to_threshold_k(p_fa: float) -> float:
    """Gaussian-consistent one-sided threshold multiplier for a target P_FA."""
    p = float(p_fa)
    if not 0.0 < p < 0.5:
        raise ValueError("p_fa must be in (0, 0.5).")
    return float(NormalDist().inv_cdf(1.0 - p))


def fine_bin_count(windows_per_stream: int, pad_factor: int = FINE_PAD_FACTOR) -> int:
    return int(pad_factor) * int(windows_per_stream)


def fine_bin_frequencies_hz(
    windows_per_stream: int,
    envelope_rate_hz: float,
    pad_factor: int = FINE_PAD_FACTOR,
) -> np.ndarray:
    """Centered fine-bin envelope frequencies (Hz) for the padded FFT."""
    p2 = fine_bin_count(windows_per_stream, pad_factor)
    bins = np.arange(p2)
    centered = ((bins + p2 // 2) % p2) - p2 // 2
    return centered * (float(envelope_rate_hz) / p2)


def independent_bin_mask(
    num_bins: int,
    *,
    pad_factor: int = FINE_PAD_FACTOR,
    designated_bins: Sequence[int] = (),
    guard_fine_bins: int = CFAR_DEFAULT_GUARD_FINE_BINS,
    census_excluded_bins: Sequence[int] = (),
) -> np.ndarray:
    """Return the boolean mask of independent, non-designated, non-excluded fine bins.

    Zero padding correlates adjacent bins, so every ``pad_factor``-th bin is
    taken as an independent sample of the underlying ``windows``-point
    spectrum.
    Designated bins are removed together with ``guard_fine_bins`` unpadded
    fine bins (``guard_fine_bins * pad_factor`` padded bins) on each side.
    """
    n = int(num_bins)
    mask = np.zeros(n, dtype=bool)
    mask[:: int(pad_factor)] = True
    guard = int(guard_fine_bins) * int(pad_factor)
    for b in designated_bins:
        lo = int(b) - guard
        hi = int(b) + guard
        idx = np.arange(lo, hi + 1) % n
        mask[idx] = False
    for b in census_excluded_bins:
        mask[int(b) % n] = False
    return mask


@dataclass
class CfarCalibration:
    location: float
    scale: float
    threshold: float
    threshold_k: float
    p_fa: float
    mode: str
    num_null_bins: int
    null_bulk_exceedance_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": float(self.location),
            "scale": float(self.scale),
            "threshold": float(self.threshold),
            "threshold_k": float(self.threshold_k),
            "p_fa": float(self.p_fa),
            "mode": str(self.mode),
            "num_null_bins": int(self.num_null_bins),
            "null_bulk_exceedance_fraction": float(
                self.null_bulk_exceedance_fraction
            ),
        }


@dataclass
class FineReductionResult:
    """Per-frame products of the float prototype.

    ``coarse_power_by_term`` is the row energy ``sum_m |z[n, m]|^2`` of the
    prototype (exact int64 for integer input). By the Parseval identity of
    the float transform, the float spectra reproduce it as
    ``sum_b fine_power[n, b] / (pad * windows)``. The deployed frame power
    is given by :func:`frame_powers_from_fine_u64`.
    """

    fine_power: np.ndarray  # [num_weight_terms, num_bins] float
    fine_power_ratio: np.ndarray  # [num_bins] float
    coarse_power_by_term: np.ndarray  # [num_weight_terms] int64 row energy
    num_streams: int
    windows_per_stream: int
    pad_factor: int

    @property
    def num_bins(self) -> int:
        return int(self.fine_power_ratio.shape[-1])


def _as_complex_rows(matched_filter_row_projections: Any, num_weight_terms: int, xp: Any) -> Any:
    """Normalize kernel or reference output to complex [terms, rows] or integer [terms, rows, 2]."""
    arr = xp.asarray(matched_filter_row_projections)
    if arr.dtype.kind == "c":
        if arr.ndim != 2 or arr.shape[0] != int(num_weight_terms):
            raise ValueError(
                "complex row sums must have shape [num_weight_terms, rows]."
            )
        return arr
    if arr.ndim == 1:
        if arr.size % (2 * int(num_weight_terms)) != 0:
            raise ValueError("flat row sums size must be terms * rows * 2.")
        arr = arr.reshape(int(num_weight_terms), -1, 2)
    if arr.ndim != 3 or arr.shape[0] != int(num_weight_terms) or arr.shape[-1] != 2:
        raise ValueError(
            "integer row sums must have shape [num_weight_terms, rows, 2]."
        )
    return arr


def _row_energy_by_term(arr: Any, num_weight_terms: int, xp: Any) -> np.ndarray:
    """Return the exact int64 ``sum_m |z[n, m]|^2`` of integer rows for the Parseval check of the float prototype."""
    rows = _as_complex_rows(arr, num_weight_terms, xp)  # [terms, rows, 2] ints
    wide = rows.astype(xp.int64)
    mags = wide[..., 0] * wide[..., 0] + wide[..., 1] * wide[..., 1]
    out = mags.sum(axis=1)
    if hasattr(out, "get"):
        out = out.get()
    return np.asarray(out, dtype=np.int64)


def frame_powers_from_fine_u64(
    matched_filter_row_projections: Any,
    *,
    num_streams: int,
    windows_per_stream: int = 128,
    num_weight_terms: int = 3,
) -> np.ndarray:
    """Sum exact fine powers over bins to obtain the deployed frame powers.

    P[n] = sum_b S[n, b], returned as uint64[num_weight_terms]. S is the
    frozen fxfft256 output from pilot_proxy.fxfft.fine_power_fx, already
    summed over streams. This is the CPU reference for the host sum of
    PilotProxy_Compute_FinePowers_U64 and the fused entry's d_power_out.

    The deployed per-bin bound keeps the 256-bin sum below 2**64. The
    uint64 sum of each term is compared with its Python integer sum, since
    NumPy uint64 addition wraps on overflow with no error.
    """
    from pilot_proxy.fxfft import fine_power_fx

    fine = fine_power_fx(
        matched_filter_row_projections,
        num_streams=int(num_streams),
        windows_per_stream=int(windows_per_stream),
        num_weight_terms=int(num_weight_terms),
    )
    powers = fine.sum(axis=-1, dtype=np.uint64)
    for term in range(fine.shape[0]):
        if sum(int(value) for value in fine[term]) != int(powers[term]):
            raise OverflowError("frame power exceeds uint64")
    return powers


def coarse_power_ratio_from_powers(powers: Sequence[int]) -> float:
    """Return F = 2 P_t / (P_l + P_u), or zero when the denominator vanishes."""
    p = np.asarray(powers, dtype=np.float64)
    den = float(p[WEIGHT_TERM_REF_LOWER] + p[WEIGHT_TERM_REF_UPPER])
    if den <= 0.0:
        return 0.0
    return float(2.0 * p[WEIGHT_TERM_TARGET] / den)


def fine_reduce(
    matched_filter_row_projections: Any,
    *,
    num_streams: int,
    windows_per_stream: int,
    num_weight_terms: int = 3,
    pad_factor: int = FINE_PAD_FACTOR,
    xp: Any = np,
) -> FineReductionResult:
    """Reduce row sums to fine-bin spectra and the fine statistic (float prototype).

    ``matched_filter_row_projections`` accepts the kernel's flat int32
    buffer, an ``[terms, rows, 2]`` integer array, or a complex
    ``[terms, rows]`` array. Integer input records the exact integer row
    energy. Complex input records ``coarse_power_by_term`` from the rounded
    Parseval sum of the float spectra, so integer input is preferred. The
    deployed spectra and frame powers are the fixed-point ones of
    :func:`pilot_proxy.fxfft.fine_power_fx` and
    :func:`frame_powers_from_fine_u64`.
    """
    terms = int(num_weight_terms)
    streams = int(num_streams)
    windows = int(windows_per_stream)
    rows = streams * windows
    p2 = fine_bin_count(windows, pad_factor)

    arr = xp.asarray(matched_filter_row_projections)
    integer_input = arr.dtype.kind != "c"
    norm = _as_complex_rows(arr, terms, xp)
    if integer_input:
        if norm.shape[1] != rows:
            raise ValueError(
                f"row count {norm.shape[1]} != num_streams * windows_per_stream = {rows}."
            )
        row_energy = _row_energy_by_term(arr, terms, xp)
        complex_dtype = xp.complex128 if xp is np else xp.complex64
        z = norm[..., 0].astype(complex_dtype) + 1j * norm[..., 1].astype(
            complex_dtype
        )
    else:
        if norm.shape[1] != rows:
            raise ValueError(
                f"row count {norm.shape[1]} != num_streams * windows_per_stream = {rows}."
            )
        z = norm
        row_energy = None

    z = z.reshape(terms, streams, windows)
    spectra = xp.fft.fft(z, n=p2, axis=-1)
    power = (spectra.real * spectra.real + spectra.imag * spectra.imag).sum(
        axis=1
    )

    if row_energy is None:
        approx = power.sum(axis=-1) / float(p2)
        if hasattr(approx, "get"):
            approx = approx.get()
        row_energy = np.rint(np.asarray(approx, dtype=np.float64)).astype(np.int64)

    s_t = power[WEIGHT_TERM_TARGET]
    s_l = power[WEIGHT_TERM_REF_LOWER]
    s_u = power[WEIGHT_TERM_REF_UPPER]
    den = s_l + s_u
    power_ratio = xp.where(den > 0, 2.0 * s_t / xp.where(den > 0, den, 1.0), 0.0)

    power_host = power.get() if hasattr(power, "get") else power
    pilotproxy_host = power_ratio.get() if hasattr(power_ratio, "get") else power_ratio

    return FineReductionResult(
        fine_power=np.asarray(power_host, dtype=np.float32),
        fine_power_ratio=np.asarray(pilotproxy_host, dtype=np.float32),
        coarse_power_by_term=np.asarray(row_energy, dtype=np.int64),
        num_streams=streams,
        windows_per_stream=windows,
        pad_factor=int(pad_factor),
    )


def calibrate_cfar(
    fine_power_ratio: np.ndarray,
    *,
    designated_bins: Sequence[int] = (),
    census_excluded_bins: Sequence[int] = (),
    guard_fine_bins: int = CFAR_DEFAULT_GUARD_FINE_BINS,
    pad_factor: int = FINE_PAD_FACTOR,
    p_fa: float = CFAR_DEFAULT_P_FA,
    fallback_flag_fraction: float = CFAR_DEFAULT_FALLBACK_FLAG_FRACTION,
) -> CfarCalibration:
    """Return the pre-registered null-bulk calibration of the fine statistic.

    Location and scale come from the independent non-designated bins
    (median + left-side spread), never from the mean of the full mixture.
    A contaminated null bulk (flag fraction above
    ``fallback_flag_fraction``) triggers the lower-quantile fallback.
    """
    f = np.asarray(fine_power_ratio, dtype=np.float64)
    mask = independent_bin_mask(
        f.shape[-1],
        pad_factor=pad_factor,
        designated_bins=designated_bins,
        guard_fine_bins=guard_fine_bins,
        census_excluded_bins=census_excluded_bins,
    )
    null_bins = f[mask]
    if null_bins.size < 8:
        raise ValueError(
            "too few independent null bins for CFAR calibration; check "
            "designated/census exclusions."
        )
    k = p_fa_to_threshold_k(p_fa)

    location = float(np.median(null_bins))
    scale = float(location - np.quantile(null_bins, CFAR_LEFT_QUANTILE))
    mode = CFAR_MODE_MEDIAN_LEFT
    if scale <= 0.0:
        scale = float(np.std(null_bins)) or 1.0
    threshold = location + k * scale
    null_bulk_exceedance_fraction = float(np.mean(null_bins > threshold))

    if null_bulk_exceedance_fraction > float(fallback_flag_fraction):
        location = float(np.quantile(null_bins, CFAR_FALLBACK_LOCATION_QUANTILE))
        left = float(np.quantile(null_bins, CFAR_FALLBACK_LEFT_QUANTILE))
        scale = max((location - left) / 2.0, 1.0e-12)
        threshold = location + k * scale
        null_bulk_exceedance_fraction = float(
            np.mean(null_bins > threshold)
        )
        mode = CFAR_MODE_QUANTILE_FALLBACK

    return CfarCalibration(
        location=location,
        scale=scale,
        threshold=threshold,
        threshold_k=float(k),
        p_fa=float(p_fa),
        mode=mode,
        num_null_bins=int(null_bins.size),
        null_bulk_exceedance_fraction=null_bulk_exceedance_fraction,
    )
