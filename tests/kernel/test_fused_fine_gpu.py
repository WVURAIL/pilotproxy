# coding=utf-8
"""GPU gates for the fused fine kernel (kernel core 2.2.0 / 3.0.0).

``PilotProxy_Compute_FusedFine_U64`` is the fused form of the deployed fine
path: one launch from packed int4 samples to exact uint64 fine powers,
with row projections held in shared memory and materialized to global
only through the optional debug tap, followed by the stream-ordered
reduction of the finalized fine sums to the three frame powers
``P_r = sum_f S_r[f]`` (kernel core 3.0.0; ``d_power_out``).

Acceptance is bit-equality --- no tolerances:

1. single block: fused fine powers == composed
   (``RowSums_I32 -> FinePowers_U64``); fused ``d_power_out`` == the host
   sum over bins of the composed fine powers == the host sum of the fused
   fine powers == ``fine_power_fx(rows).sum(axis=-1)``; tap == the
   composed row-projection buffer; fine == the fxfft reference chain;
2. batch case: same equalities per batch entry with a batched handle;
3. production form: tap pointer NULL, fine + frame-power outputs unchanged
   (the tap is genuinely optional; row projections never touch global
   memory);
4. an impulse frame: the frozen transform of a unit impulse is exactly one
   in every bin, so the fused kernel must report ``S_target[f] = 1`` and
   ``P_target = 256`` with both reference terms zero --- device exactness
   with no reference involved.

The timing report (printed rather than asserted) gives the deployment-footprint
numbers: fused without tap (production), fused with tap, and the composed
chain (two launches plus the device-side bin sum), against the ~41.9 ms
CHIME frame cadence.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from pilot_proxy.fxfft import fine_power_fx
from gpu import cupy_or_skip
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
WINDOWS = 128
FINE_BINS = 256
RNG_SEED = 20260804


def _kernel_or_skip() -> PilotProxyKernel:
    try:
        kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libpilotproxy.so is not built")
    if not kernel.supports_fused_fine():
        pytest.skip(
            "libpilotproxy.so predates kernel core 2.2.0 (no fused fine "
            "kernel): rebuild from the current CUDA sources."
        )
    if kernel.version.as_tuple() < (3, 0, 0):
        pytest.fail(
            "libpilotproxy.so predates kernel core 3.0.0: its d_power_out is "
            "not the fine-bin sum. Rebuild from the current CUDA sources."
        )
    return kernel


def _random_frames(rng, batch: int, rows: int):
    packed = rng.integers(
        0, 256, size=(batch, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    return packed, weights


def _reference_chain(matched_filter_row_projections: np.ndarray, streams: int):
    """fine powers + frame powers from an int32 [terms, rows, 2] buffer."""
    fine = fine_power_fx(
        matched_filter_row_projections.astype(np.int64),
        num_streams=streams,
        windows_per_stream=WINDOWS,
    )
    # Kernel core 3.0.0: P_r is the exact all-bin sum of S_r.
    return fine, fine.sum(axis=-1, dtype=np.uint64)


@pytest.mark.cuda
def test_fused_matches_composed_path_bit_exact():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    streams = 16
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine_composed = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_fine_fused = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_fused = cp.full(TERMS, 123, dtype=cp.uint64)
    d_tap = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    try:
        # Composed path: the validated 2.0.0/2.1.0 chain.
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr), streams, WINDOWS, 1,
            int(d_fine_composed.data.ptr),
        )
        # Fused path: one launch, tap bound.
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_fused.data.ptr),
            int(d_pow_fused.data.ptr),
            int(d_tap.data.ptr),
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        composed_rows = cp.asnumpy(d_matched_filter_row_projections).reshape(TERMS, rows, 2)
        fine_c = cp.asnumpy(d_fine_composed)
        fine_f = cp.asnumpy(d_fine_fused)
        pow_f = cp.asnumpy(d_pow_fused)
        tap = cp.asnumpy(d_tap).reshape(TERMS, rows, 2)
    finally:
        kernel.destroy(handle)

    # Fused == composed, bit for bit.
    np.testing.assert_array_equal(tap, composed_rows)
    np.testing.assert_array_equal(fine_f, fine_c)
    # The frame-power identity: device reduction == host sum over bins of
    # either fine buffer.
    np.testing.assert_array_equal(pow_f, fine_c.sum(axis=-1, dtype=np.uint64))
    np.testing.assert_array_equal(pow_f, fine_f.sum(axis=-1, dtype=np.uint64))
    # Both == the frozen reference chain.
    fine_ref, pow_ref = _reference_chain(composed_rows, streams)
    np.testing.assert_array_equal(fine_f, fine_ref)
    np.testing.assert_array_equal(pow_f, pow_ref)
    assert (pow_ref > 0).all()


@pytest.mark.cuda
def test_fused_batch_case():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    if not kernel.supports_batch:
        pytest.skip("libpilotproxy.so does not expose PilotProxy_Create_Batch")
    rng = np.random.default_rng(RNG_SEED + 1)
    streams, batch = 8, 3
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, batch, rows)

    d_in = cp.asarray(packed)
    handle = kernel.create_raw_batch(rows, batch, d_in.data.ptr)
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow = cp.full((batch, TERMS), 123, dtype=cp.uint64)
    d_tap = cp.zeros(batch * TERMS * rows * 2, dtype=cp.int32)
    try:
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine.data.ptr),
            int(d_pow.data.ptr),
            int(d_tap.data.ptr),
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        fine = cp.asnumpy(d_fine)
        pow_ = cp.asnumpy(d_pow)
        tap = cp.asnumpy(d_tap).reshape(batch, TERMS, rows, 2)
    finally:
        kernel.destroy(handle)

    np.testing.assert_array_equal(pow_, fine.sum(axis=-1, dtype=np.uint64))
    for b in range(batch):
        fine_ref, pow_ref = _reference_chain(tap[b], streams)
        np.testing.assert_array_equal(fine[b], fine_ref)
        np.testing.assert_array_equal(pow_[b], pow_ref)


@pytest.mark.cuda
def test_fused_production_form_without_tap():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 2)
    streams = 8
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_fine_tap = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_tap = cp.zeros(TERMS, dtype=cp.uint64)
    d_fine_notap = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_notap = cp.zeros(TERMS, dtype=cp.uint64)
    d_tap = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    try:
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_tap.data.ptr),
            int(d_pow_tap.data.ptr),
            int(d_tap.data.ptr),
        )
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_notap.data.ptr),
            int(d_pow_notap.data.ptr),
            0,
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        fine_notap = cp.asnumpy(d_fine_notap)
        pow_notap = cp.asnumpy(d_pow_notap)
        np.testing.assert_array_equal(fine_notap, cp.asnumpy(d_fine_tap))
        np.testing.assert_array_equal(pow_notap, cp.asnumpy(d_pow_tap))
        np.testing.assert_array_equal(
            pow_notap, fine_notap.sum(axis=-1, dtype=np.uint64)
        )
    finally:
        kernel.destroy(handle)


@pytest.mark.cuda
def test_fused_impulse_frame_is_exact_without_a_reference():
    """One nonzero sample against a unit target weight: z_target = (1, 0) in
    one row, zero elsewhere. The impulse only ever enters a butterfly as the
    top leg, every bottom leg is an exact zero, so the frozen transform is
    exactly one in every bin: S_target[f] = 1, P_target = 256, references 0."""
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rows = WINDOWS
    packed = np.zeros((rows, 128), dtype=np.int8)
    weights = np.zeros((TERMS, 128), dtype=np.int8)
    packed[0, 0] = 0x10   # real 1, imag 0
    weights[0, 0] = 0x10  # target weight: real 1, imag 0

    d_in = cp.asarray(packed)
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_fine = cp.full((TERMS, FINE_BINS), 123, dtype=cp.uint64)
    d_pow = cp.full(TERMS, 123, dtype=cp.uint64)
    try:
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data, int(d_fine.data.ptr), int(d_pow.data.ptr), 0
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        fine = cp.asnumpy(d_fine)
        pow_ = cp.asnumpy(d_pow)
    finally:
        kernel.destroy(handle)

    np.testing.assert_array_equal(fine[0], np.ones(FINE_BINS, dtype=np.uint64))
    np.testing.assert_array_equal(fine[1:], np.zeros((2, FINE_BINS), dtype=np.uint64))
    np.testing.assert_array_equal(pow_, np.array([FINE_BINS, 0, 0], dtype=np.uint64))


@pytest.mark.cuda
def test_fused_rate_margin_report():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 3)
    streams = 2048
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow = cp.zeros(TERMS, dtype=cp.uint64)

    def fused_notap():
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data,
            int(d_fine.data.ptr), int(d_pow.data.ptr), 0,
        )

    def fused_tap():
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data,
            int(d_fine.data.ptr), int(d_pow.data.ptr),
            int(d_matched_filter_row_projections.data.ptr),
        )

    def composed():
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr), streams, WINDOWS, 1,
            int(d_fine.data.ptr),
        )
        d_fine.sum(axis=-1, out=d_pow)

    def measure(fn, n=50):
        fn()  # warmup + JIT/caches
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    try:
        ms_fused = measure(fused_notap)
        ms_tap = measure(fused_tap)
        ms_composed = measure(composed)
    finally:
        kernel.destroy(handle)

    frame_ms = 16384.0 / 390625.0 * 1000.0
    print(
        f"\nfused fine kernel, 2048-stream frame "
        f"(cadence {frame_ms:.1f} ms):\n"
        f"  fused, no tap (production): {ms_fused:.3f} ms "
        f"(margin x{frame_ms / ms_fused:.0f})\n"
        f"  fused, tap bound (debug):   {ms_tap:.3f} ms\n"
        f"  composed two launches + bin sum: {ms_composed:.3f} ms "
        f"(fused saves {ms_composed - ms_fused:.3f} ms)"
    )
    assert kernel.last_error() == ""
