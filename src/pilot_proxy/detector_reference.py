#!/usr/bin/env python3
# coding=utf-8
"""
Pure-NumPy reference implementation for the local-reference power ratio kernel.

The kernel's chain (core 3.0.0) is: packed samples and weights -> exact int32
matched-filter row projections ``z[n, m]`` -> the frozen fxfft256 fine power
spectra ``S[n, f]`` (summed over streams) -> frame powers
``P[n] = sum_f S[n, f]`` -> ``R_coarse = 2 P[0] / (P[1] + P[2])``. The packed
references below compute exactly that chain in integers, so a CPU-only sweep
keeps the kernel's semantics. The float reference
(:func:`coarse_power_ratio_cpu_reference`) works on unquantized complex rows,
where the fixed-point transform does not apply; it is the float prototype of
the same ratio, ``sum_m |z[n, m]|^2`` per term, which an exact transform of
length ``L_F`` scales by ``L_F`` in every term, so the ratio is the same.
"""

from __future__ import annotations

from typing import Any

import numpy as np

INT4_COMPONENT_BITS = 4
INT8_COMPONENT_BITS = 8
REFERENCE_WEIGHT_TERMS = 3
REFERENCE_TARGET_TERM_INDEX = 0
REFERENCE_LOWER_TERM_INDEX = 1
REFERENCE_UPPER_TERM_INDEX = 2
COARSE_POWER_RATIO_SCALE = 2.0
FRAME_POWER_WINDOWS_PER_STREAM = 128  # the frozen fxfft256 input length


def packed_dtype_for_component_bits(bits: int) -> np.dtype:
    """Return a packed integer dtype for a per-component bit depth."""
    bits = int(bits)
    if bits == INT4_COMPONENT_BITS:
        return np.dtype(np.int8)
    if bits == INT8_COMPONENT_BITS:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported component bit depth: {bits}. Expected 4 or 8.")


def quantize_complex_numpy(
    data: np.ndarray,
    bits: int,
    scale: float,
) -> np.ndarray:
    """Quantize complex data to packed integer format (NumPy version)."""
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (M, K). Got shape {data.shape}.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("quantization scale must be positive and finite.")
    if not np.all(np.isfinite(data)):
        raise ValueError("quantization samples must be finite.")
    packed_dtype = packed_dtype_for_component_bits(bits)
    max_int = (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1

    r = np.asarray(
        np.clip(np.round(data.real * scale), -max_int, max_int),
        dtype=np.int32,
    )
    i = np.asarray(
        np.clip(np.round(data.imag * scale), -max_int, max_int),
        dtype=np.int32,
    )
    packed = np.asarray((r << bits) | (i & mask), dtype=packed_dtype)
    return np.ascontiguousarray(packed)


def unpack_packed_complex(
    packed: np.ndarray,
    bits: int,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Unpack packed integer samples to a complex array."""
    packed_dtype = packed_dtype_for_component_bits(bits)
    p = np.asarray(packed, dtype=packed_dtype).astype(np.int32, copy=False)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    real = p >> bits
    imag_raw = p & mask
    imag = np.where(imag_raw & sign_bit, imag_raw - (1 << bits), imag_raw)

    return real.astype(dtype) + 1j * imag.astype(dtype)  # type: ignore[arg-type]


def coarse_power_ratio_cpu_reference(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Float prototype of the local-reference power ratio.

    Returns ``(R, sums)`` with ``sums[n] = sum_m |x_m . conj(w_n)|^2`` in
    float64 over any number of rows. These row-domain sums are not the
    kernel's frame powers (the all-bin sums of the fixed-point fine spectra,
    :func:`frame_powers_cpu_reference_packed`); an exact transform would scale
    every term by the same ``L_F``, so ``R`` is the float prototype of the
    kernel's ratio, which differs from it only by the fixed-point transform's
    rounding.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (M, K). Got shape {samples.shape}.")
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D (N, K). Got shape {weights.shape}.")
    if weights.shape[0] != REFERENCE_WEIGHT_TERMS:
        raise ValueError("weights must have N=3 (target, ref+, ref-).")
    if samples.shape[1] != weights.shape[1]:
        raise ValueError("samples K dimension must match weights K dimension.")

    dots = samples @ np.conjugate(weights).T
    power = np.abs(dots) ** 2
    sums = np.sum(power, axis=0)
    denom = float(sums[REFERENCE_LOWER_TERM_INDEX] + sums[REFERENCE_UPPER_TERM_INDEX])
    eps = np.finfo(np.float64).tiny
    if denom <= eps:
        return 0.0, sums
    power_ratio = float(
        COARSE_POWER_RATIO_SCALE * sums[REFERENCE_TARGET_TERM_INDEX] / denom
    )
    return power_ratio, sums


def frame_powers_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> np.ndarray:
    """The kernel's exact frame powers from packed samples and weights.

    ``P[n] = sum_f S[n, f]``: exact row projections
    (:func:`matched_filter_row_projections_cpu_reference_packed`) through the
    frozen fxfft256 (:func:`pilot_proxy.fine_reduction.frame_powers_from_fine_u64`),
    summed over bins. Returns ``uint64[3]``. Like the kernel entry, it derives
    the stream count as ``rows // 128`` and refuses a row count that is not a
    whole number of 128-window streams.
    """
    from pilot_proxy.fine_reduction import frame_powers_from_fine_u64

    rows = int(np.asarray(packed_samples).shape[0]) if np.ndim(packed_samples) == 2 else -1
    if rows <= 0 or rows % FRAME_POWER_WINDOWS_PER_STREAM != 0:
        raise ValueError(
            "frame powers are defined over whole 128-window streams: the "
            f"packed row count ({rows}) must be a positive multiple of "
            f"{FRAME_POWER_WINDOWS_PER_STREAM}."
        )
    projections = matched_filter_row_projections_cpu_reference_packed(
        packed_samples, packed_weights, bits
    )
    return frame_powers_from_fine_u64(
        projections.astype(np.int64),
        num_streams=rows // FRAME_POWER_WINDOWS_PER_STREAM,
        windows_per_stream=FRAME_POWER_WINDOWS_PER_STREAM,
        num_weight_terms=REFERENCE_WEIGHT_TERMS,
    )


def coarse_power_ratio_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> tuple[float, np.ndarray]:
    """The kernel's local-reference power ratio from packed integers.

    Returns ``(R, powers)`` where ``powers`` are the exact ``uint64[3]`` frame
    powers of :func:`frame_powers_cpu_reference_packed` (the all-bin sums of
    the fixed-point fine spectra) and ``R = 2 P[0] / (P[1] + P[2])`` in
    float64 (0.0 when the reference power is zero).
    """
    powers = frame_powers_cpu_reference_packed(packed_samples, packed_weights, bits)
    denom = int(powers[REFERENCE_LOWER_TERM_INDEX]) + int(powers[REFERENCE_UPPER_TERM_INDEX])
    if denom == 0:
        return 0.0, powers
    power_ratio = float(
        COARSE_POWER_RATIO_SCALE * float(powers[REFERENCE_TARGET_TERM_INDEX]) / float(denom)
    )
    return power_ratio, powers


def matched_filter_row_projections_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Return the kernel's exact integer matched-filter row projections.

    The result has shape ``[terms, rows, 2]`` and dtype ``int32``.  Its
    arithmetic and layout are the CPU specification for
    ``PilotProxy_Compute_RowSums_I32``: packed signed components are sign-extended,
    each row is dotted with the conjugated weight vector, and the real and
    imaginary integer sums are retained separately.  This public reference is
    used by the synthetic sensitivity ablations as the boundary between
    input/weight quantization and the frozen fixed-point fine transform.
    """
    samples = np.asarray(packed_samples)
    weights = np.asarray(packed_weights)
    if samples.ndim != 2:
        raise ValueError(
            f"packed_samples must be 2D (rows, K). Got shape {samples.shape}."
        )
    if weights.ndim != 2:
        raise ValueError(
            f"packed_weights must be 2D (terms, K). Got shape {weights.shape}."
        )
    if weights.shape[0] != REFERENCE_WEIGHT_TERMS:
        raise ValueError("packed_weights must have N=3 (target, ref+, ref-).")
    if samples.shape[1] != weights.shape[1]:
        raise ValueError("packed sample and weight K dimensions must match.")

    x = unpack_packed_complex(samples, bits, dtype=np.int64)
    w = unpack_packed_complex(weights, bits, dtype=np.int64)
    xr = np.asarray(x.real, dtype=np.int64)
    xi = np.asarray(x.imag, dtype=np.int64)
    wr = np.asarray(w.real, dtype=np.int64)
    wi = np.asarray(w.imag, dtype=np.int64)

    # z[n, m] = sum_k x[m, k] * conj(w[n, k]).  Integer inputs keep the
    # einsums exact in int64; the detector's int4/K<=128 bound fits int32.
    real = np.einsum("mk,nk->nm", xr, wr, optimize=True)
    real += np.einsum("mk,nk->nm", xi, wi, optimize=True)
    imag = np.einsum("mk,nk->nm", xi, wr, optimize=True)
    imag -= np.einsum("mk,nk->nm", xr, wi, optimize=True)
    out = np.stack((real, imag), axis=-1)
    info = np.iinfo(np.int32)
    if np.any(out < info.min) or np.any(out > info.max):
        raise OverflowError("matched-filter row projection exceeds int32.")
    return np.ascontiguousarray(out.astype(np.int32))
