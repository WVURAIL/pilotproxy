# coding=utf-8
"""Smoke test: the staged GPU library against the exact CPU reference chain.

Runs the frozen chain once on prebuilt DTV weights and a quantized
noise-plus-tone frame: GPU row projections -> GPU fine powers -> frame
powers (the all-bin sums, kernel core 3.0.0), and checks every integer
against the CPU chain ``matched_filter_row_projections_cpu_reference_packed``
-> ``fine_power_fx`` -> sum, plus the fused entry's device-reduced
``d_power_out``. Exact uint64 equality, no tolerance. The packed CPU reference
``coarse_power_ratio_cpu_reference_packed`` is that same exact chain. The
float prototype ``coarse_power_ratio_cpu_reference`` (the sum of |z|^2 over
rows of the unpacked frame) is compared on the same frame only to document
the Parseval-level proximity of the fixed-point definition; it is bounded by
the frozen transform's drift gate, not asserted equal.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_reference import (
    coarse_power_ratio_cpu_reference,
    coarse_power_ratio_cpu_reference_packed,
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.fxfft import fine_power_fx
from gpu import cuda_available
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH

TARGET_FREQ_MHZ = 470.309441
REFERENCE_BANDWIDTH_MHZ = 400.0
REFERENCE_NUM_CHANNELS = 1024.0
HALF_CHANNEL_WIDTH = 2.0
KERNEL_SMOKE_RNG_SEED = 1234
TERMS = 3
WINDOWS = 128
FINE_BINS = 256
KERNEL_SMOKE_STREAMS = 1
KERNEL_SMOKE_ROWS = KERNEL_SMOKE_STREAMS * WINDOWS
KERNEL_SMOKE_NOISE_SIGMA = 1.0
QUANTIZER_CLIP_SIGMA = 3.0
COMPLEX_TONE_CYCLES = 2.0
# The frozen transform's Parseval departure is bounded at 5e-3 on the golden
# vectors (tests/core/test_fxfft256.py); this frame sits far inside it.
PARSEVAL_RELATIVE_BOUND = 5.0e-3


def _import_cupy_or_skip():
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy not available: {exc}")
    return cp


def _run_kernel(
    cp,
    kernel: PilotProxyKernel,
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
):
    """GPU chain on one frame: rows, staged fine, fused fine, frame powers."""
    rows = int(packed_samples.shape[0])
    streams = rows // WINDOWS
    d_in = cp.asarray(packed_samples)
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_rows = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_fine_fused = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_powers = cp.full(TERMS, 123, dtype=cp.uint64)
    try:
        kernel.compute_row_projections_i32(
            handle, packed_weights.ctypes.data, d_rows.data.ptr
        )
        kernel.compute_fine_powers_u64(
            int(d_rows.data.ptr), streams, WINDOWS, 1, int(d_fine.data.ptr)
        )
        kernel.compute_fused_fine_u64(
            handle,
            packed_weights.ctypes.data,
            int(d_fine_fused.data.ptr),
            int(d_powers.data.ptr),
            0,
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        return (
            cp.asnumpy(d_rows).reshape(TERMS, rows, 2),
            cp.asnumpy(d_fine),
            cp.asnumpy(d_fine_fused),
            cp.asnumpy(d_powers),
        )
    finally:
        kernel.destroy(handle)


def _power_ratio(powers: np.ndarray) -> float:
    p = [int(v) for v in powers]
    return 2.0 * p[0] / (p[1] + p[2])


def test_kernel_cpu_gpu_smoke_matches_prebuilt_dtv_weights() -> None:
    cp = _import_cupy_or_skip()
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(f"CUDA not available: {reason}")
    if not DEFAULT_LIB_PATH.exists():
        pytest.skip(f"CUDA kernel library not built: {DEFAULT_LIB_PATH}")
    if not DEFAULT_WEIGHTS_PATH.exists():
        pytest.skip(f"Prebuilt weights not found: {DEFAULT_WEIGHTS_PATH}")

    kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    assert kernel.version.as_tuple() >= (3, 0, 0)
    wb = DetectorWeightBank(
        explicit_path=DEFAULT_WEIGHTS_PATH,
        expected_kernel=kernel.specs,
    )
    weights, valid = wb.get_weights_for_pilot_frequency(TARGET_FREQ_MHZ)
    assert valid and weights is not None

    channel_width_mhz = REFERENCE_BANDWIDTH_MHZ / REFERENCE_NUM_CHANNELS
    chan_idx = int(np.argmin(np.abs(wb.reference_freqs - TARGET_FREQ_MHZ)))
    chan_start = (
        float(wb.reference_freqs[chan_idx])
        - channel_width_mhz / HALF_CHANNEL_WIDTH
    )
    norm_target = (TARGET_FREQ_MHZ - chan_start) / channel_width_mhz

    rng = np.random.default_rng(KERNEL_SMOKE_RNG_SEED)
    rows = KERNEL_SMOKE_ROWS
    noise_sigma = KERNEL_SMOKE_NOISE_SIGMA
    noise = (
        rng.standard_normal((rows, kernel.specs.K))
        + 1j * rng.standard_normal((rows, kernel.specs.K))
    ) * noise_sigma
    k = np.arange(kernel.specs.K, dtype=np.float64)
    tone = np.exp(-1j * COMPLEX_TONE_CYCLES * np.pi * norm_target * k)[None, :]
    samples = noise + tone
    scale = float((1 << (kernel.specs.bits - 1)) - 1) / (
        QUANTIZER_CLIP_SIGMA * noise_sigma
    )
    packed = quantize_complex_numpy(samples, kernel.specs.bits, scale)

    gpu_rows, gpu_fine, gpu_fine_fused, gpu_powers = _run_kernel(
        cp, kernel, packed, weights
    )

    # Exact CPU chain: rows -> frozen transform -> fine powers -> bin sums.
    cpu_rows = matched_filter_row_projections_cpu_reference_packed(
        packed, weights, kernel.specs.bits
    )
    cpu_fine = fine_power_fx(
        cpu_rows.astype(np.int64), num_streams=KERNEL_SMOKE_STREAMS
    )
    cpu_powers = cpu_fine.sum(axis=-1, dtype=np.uint64)

    np.testing.assert_array_equal(gpu_rows, cpu_rows)
    np.testing.assert_array_equal(gpu_fine, cpu_fine)
    np.testing.assert_array_equal(gpu_fine_fused, cpu_fine)
    np.testing.assert_array_equal(gpu_powers, cpu_powers)
    np.testing.assert_array_equal(gpu_powers, gpu_fine.sum(axis=-1, dtype=np.uint64))
    assert (cpu_powers > 0).all()
    assert _power_ratio(gpu_powers) == _power_ratio(cpu_powers)

    # The packed CPU reference is the same exact chain.
    _, packed_powers = coarse_power_ratio_cpu_reference_packed(
        packed, weights, kernel.specs.bits
    )
    np.testing.assert_array_equal(packed_powers, cpu_powers)

    # Parseval proximity to the float |z|^2 sum: documented, not an identity.
    _, float_sums = coarse_power_ratio_cpu_reference(
        unpack_packed_complex(packed, kernel.specs.bits),
        unpack_packed_complex(weights, kernel.specs.bits),
    )
    parseval = FINE_BINS * np.asarray(float_sums, dtype=np.float64)
    drift = np.abs(gpu_powers.astype(np.float64) - parseval) / parseval
    assert float(drift.max()) <= PARSEVAL_RELATIVE_BOUND, drift
