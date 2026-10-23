# coding=utf-8
"""GPU gates for the v2 row-sum front end (run on CUDA hosts, e.g. the A100).

Three gates, all pre-registered:
1. exact equality of kernel row projections against the numpy int reference;
2. the frame-power identity of kernel core 3.0.0: the fused ``d_power_out``
   == the host sum over bins of the staged ``FinePowers_U64`` ==
   ``fine_power_fx(rows).sum(axis=-1)``, all exact uint64;
3. ULP-tolerance parity of the cupy complex64 fine reduction against the
   numpy complex128 prototype.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_reduction import fine_reduce
from pilot_proxy.fxfft import fine_power_fx
from gpu import cupy_or_skip
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
STREAMS = 16
WINDOWS = 128
FINE_BINS = 256
ROWS = STREAMS * WINDOWS
RNG_SEED = 20260729
# One float32 ULP at the fine-power scale, with margin for FFT ordering:
# relative gate, pre-registered before A100 runs.
FINE_FFT_RELATIVE_TOLERANCE = 5.0e-6


def _kernel_or_skip() -> PilotProxyKernel:
    try:
        kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libpilotproxy.so is not built")
    if not kernel.supports_row_projections() or not kernel.supports_fused_fine():
        pytest.fail(
            "libpilotproxy.so predates kernel core 2.2.0: rebuild from the "
            "current CUDA sources before relaunching any scan."
        )
    return kernel


def _sign_extend_i4(v: np.ndarray) -> np.ndarray:
    v = v & 0xF
    return (v ^ 0x8) - 0x8


def _numpy_reference_matched_filter_row_projections(
    packed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Exact int reference: unpack int4, conj-multiply, integer sums."""
    xr = _sign_extend_i4((packed.astype(np.int32) >> 4)).astype(np.int64)
    xi = _sign_extend_i4(packed.astype(np.int32)).astype(np.int64)
    wr = _sign_extend_i4((weights.astype(np.int32) >> 4)).astype(np.int64)
    wi = _sign_extend_i4(weights.astype(np.int32)).astype(np.int64)
    # z = sum_k x_k * conj(w_k)
    re = np.einsum("mk,nk->nm", xr, wr) + np.einsum("mk,nk->nm", xi, wi)
    im = np.einsum("mk,nk->nm", xi, wr) - np.einsum("mk,nk->nm", xr, wi)
    out = np.stack([re, im], axis=-1)  # [terms, rows, 2]
    assert np.abs(out).max() < 2**31
    return out.astype(np.int32)


def _random_inputs(rng: np.random.Generator):
    packed = rng.integers(0, 256, size=(ROWS, 128), dtype=np.uint8).astype(
        np.int8
    )
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    return packed, weights


def test_gpu_matched_filter_row_projections_exactly_match_numpy_reference():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    assert kernel.version.as_tuple() >= (3, 0, 0)
    rng = np.random.default_rng(RNG_SEED)
    packed, weights = _random_inputs(rng)

    d_in = cp.asarray(packed)
    handle = kernel.create_raw(ROWS, d_in.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * ROWS * 2, dtype=cp.int32)
    d_fine_staged = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_fine_fused = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_powers = cp.full(TERMS, 123, dtype=cp.uint64)
    try:
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        # Staged fine stage over the same device buffer (the archive path).
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr),
            STREAMS,
            WINDOWS,
            1,
            int(d_fine_staged.data.ptr),
        )
        # Fused entry: fine powers plus the device-reduced frame powers.
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_fused.data.ptr),
            int(d_powers.data.ptr),
            0,
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        got = cp.asnumpy(d_matched_filter_row_projections).reshape(TERMS, ROWS, 2)
        fine_staged = cp.asnumpy(d_fine_staged)
        fine_fused = cp.asnumpy(d_fine_fused)
        powers = cp.asnumpy(d_powers)
    finally:
        kernel.destroy(handle)

    # Gate 1: exact row projections.
    expected = _numpy_reference_matched_filter_row_projections(packed, weights)
    np.testing.assert_array_equal(got, expected)

    # Gate 2: the frame powers are the all-bin sums of the fine spectrum on
    # every path --- fused d_power_out == host sum of the staged fine powers
    # == fine_power_fx over the exact rows, summed. All uint64, no tolerance.
    reference_fine = fine_power_fx(
        expected.astype(np.int64), num_streams=STREAMS, windows_per_stream=WINDOWS
    )
    np.testing.assert_array_equal(fine_staged, reference_fine)
    np.testing.assert_array_equal(fine_fused, reference_fine)
    expected_powers = reference_fine.sum(axis=-1, dtype=np.uint64)
    np.testing.assert_array_equal(
        fine_staged.sum(axis=-1, dtype=np.uint64), expected_powers
    )
    np.testing.assert_array_equal(powers.astype(np.uint64), expected_powers)
    assert (expected_powers > 0).all()


def test_cupy_fine_reduction_matches_float64_prototype_within_ulp_gate():
    cp = cupy_or_skip()
    rng = np.random.default_rng(RNG_SEED + 1)
    matched_filter_row_projections = rng.integers(
        -14336, 14337, size=(TERMS, ROWS, 2), dtype=np.int32
    )

    host = fine_reduce(
        matched_filter_row_projections, num_streams=STREAMS, windows_per_stream=WINDOWS, xp=np
    )
    dev = fine_reduce(
        cp.asarray(matched_filter_row_projections),
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
        xp=cp,
    )

    np.testing.assert_array_equal(dev.coarse_power_by_term, host.coarse_power_by_term)
    denom = np.maximum(np.abs(host.fine_power.astype(np.float64)), 1.0)
    rel = np.abs(
        dev.fine_power.astype(np.float64) - host.fine_power.astype(np.float64)
    ) / denom
    assert float(rel.max()) < FINE_FFT_RELATIVE_TOLERANCE, (
        f"fine-power relative error {rel.max():.3e} exceeds the "
        f"pre-registered ULP gate {FINE_FFT_RELATIVE_TOLERANCE:.1e}"
    )
