# coding=utf-8
"""GPU gates for the decision epilogue (kernel core 2.3.0).

``PilotProxy_Compute_FusedFineMask_U64`` is the deployed form of the
detector. Packed samples and the per-channel bundle constants are passed in,
and one mask bit is returned for each aligned frame. On every frame, the mask
of the epilogue must be bit-identical to the frozen Python reference
``pilot_proxy.fine_decision.fine_mask_decision`` applied to the same exact
fine powers, with no tolerance.

1. The mask must equal the reference on noise frames and on frames that carry
   an injected pilot line, so that both mask outcomes are exercised across a
   batch. In the same launch, the fine accumulator is compared bit for bit
   with the fxfft reference chain through the row-projection tap. On every
   launch that binds the optional frame-power tap, the tap must equal the
   all-bin sum of the fine accumulator (``d_power_out`` equal to
   ``fine.sum(axis=-1)``, kernel core 3.0.0). An unbound tap must be left
   untouched.
2. Degenerate rules are tested on the device. An empty bulk mask forces mask
   0, and a designated set whose denominators are live fires only where the
   reference fires.
3. The production form, with no debug taps, must emit the same mask.
4. Twenty repeated launches must produce the same mask and fine-power bits.
   This exercises the completion-counter epilogue of the last block under
   real grid scheduling, which the CPU emulation cannot probe.
5. The rank-support entry must return the same fine powers, mask, and tap as
   the legacy entry, and its rank-valid flag and bulk count must equal the
   reference on each frame. The cases are an ordinary bulk, an empty bulk,
   the rank boundary, the last usable rank, a zero target, and zero
   references.
6. In a library built with a delay in the second warp before its final power
   additions, the mask must still equal the reference. This checks that the
   epilogue waits for every power writer.

The rate report is printed only. It compares the deployed mask form with the
fused-powers form at 2048 streams.

The CPU-side verification (threaded emulation, a pthread barrier, and a
2M-trial fuzz of the 128-bit and 192-bit compares against Python big
integers) was run at integration time. This file holds the gates on the GPU.
"""
from __future__ import annotations

import time
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pilot_proxy.fine_decision import fine_mask_decision, pack_bulk_mask
from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.fxfft import fine_power_fx
from gpu import cupy_or_skip
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
WINDOWS = 128
FINE_BINS = 256
RNG_SEED = 20260806
ANCHOR = 62
HALF_WIDTH = 2
MULT_Q16 = int(1.5 * 65536)
UNWRITTEN_POWER = 123  # sentinel an unbound frame-power tap must keep


def _kernel_or_skip() -> PilotProxyKernel:
    try:
        kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libpilotproxy.so is not built")
    if not kernel.supports_fused_fine_mask():
        pytest.skip(
            "libpilotproxy.so predates kernel core 2.3.0 (no decision "
            "epilogue): rebuild from the current CUDA sources."
        )
    return kernel


def _pack_c(re, im):
    return (
        ((np.asarray(re) & 0xF) << 4) | (np.asarray(im) & 0xF)
    ).astype(np.uint8).astype(np.int8)


def _unpack_c(p):
    b = p.astype(np.int32) & 0xFF

    def se(v):
        return ((v & 0xF) ^ 0x8) - 0x8

    return se(b >> 4), se(b)


def _inject_line(packed, weights, streams, rng, anchor=ANCHOR):
    """Return a copy of one frame with a quantized pilot line at the anchor fine bin.

    Each sample x[m] is set proportional to w_target rotated by the
    window-axis phase 2*pi*anchor*m/256, with a random starting phase for
    each stream.
    """
    out = packed.copy()
    wt_re, wt_im = _unpack_c(weights[0])
    m_idx = np.arange(WINDOWS)
    for s in range(streams):
        phase = np.exp(
            1j * (2 * np.pi * anchor * m_idx / 256 + rng.uniform(0, 2 * np.pi))
        )
        for m in range(WINDOWS):
            c = phase[m]
            xr = np.clip(
                np.round(wt_re * c.real - wt_im * c.imag), -8, 7
            ).astype(np.int64)
            xi = np.clip(
                np.round(wt_im * c.real + wt_re * c.imag), -8, 7
            ).astype(np.int64)
            out[s * WINDOWS + m] = _pack_c(xr, xi)
    return out


def _bulk_and_rank(anchor=ANCHOR):
    bulk = independent_bin_mask(FINE_BINS, designated_bins=[anchor])
    return bulk, int(np.count_nonzero(bulk) // 2)


def _run_mask(cp, kernel, packed, weights, *, anchor, half_width, bulk,
              rank, mult, batch, streams, with_taps=True, with_support=False):
    rows = streams * WINDOWS
    d_in = cp.asarray(packed)
    if batch > 1:
        handle = kernel.create_raw_batch(rows, batch, d_in.data.ptr)
    else:
        handle = kernel.create_raw(rows, d_in.data.ptr)
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_mask = cp.zeros(batch, dtype=cp.int32)
    d_pow = cp.full((batch, TERMS), UNWRITTEN_POWER, dtype=cp.uint64)
    d_tap = cp.zeros(batch * TERMS * rows * 2, dtype=cp.int32)
    d_rank_valid = cp.full(batch, 77, dtype=cp.int32)
    d_n_bulk = cp.full(batch, 77, dtype=cp.int32)
    try:
        compute = (kernel.compute_fused_fine_mask_with_support_u64 if with_support
                   else kernel.compute_fused_fine_mask_u64)
        compute(
            handle,
            weights.ctypes.data,
            anchor,
            half_width,
            pack_bulk_mask(bulk),
            rank,
            mult,
            int(d_fine.data.ptr),
            int(d_mask.data.ptr),
            *((int(d_rank_valid.data.ptr), int(d_n_bulk.data.ptr)) if with_support else ()),
            int(d_pow.data.ptr) if with_taps else 0,
            int(d_tap.data.ptr) if with_taps else 0,
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        fine = cp.asnumpy(d_fine)
        mask = cp.asnumpy(d_mask)
        powers = cp.asnumpy(d_pow)
        tap = (
            cp.asnumpy(d_tap).reshape(batch, TERMS, rows, 2)
            if with_taps
            else None
        )
    finally:
        kernel.destroy(handle)
    if with_taps:
        # the frame-power tap must hold the all-bin sum of the fine
        # accumulator (kernel core 3.0.0)
        np.testing.assert_array_equal(powers, fine.sum(axis=-1, dtype=np.uint64))
    else:
        assert (powers == UNWRITTEN_POWER).all(), "unbound frame-power tap was written"
    if with_support:
        return fine, mask, tap, cp.asnumpy(d_rank_valid), cp.asnumpy(d_n_bulk)
    return fine, mask, tap


@pytest.mark.cuda
@pytest.mark.parametrize("case", ["ordinary", "empty_bulk", "rank_boundary", "last_usable_rank", "zero_target", "zero_references"])
@pytest.mark.parametrize("with_taps", [False, True])
def test_full_rate_rank_support_matches_reference_and_legacy(case, with_taps):
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    assert kernel.supports_fused_fine_mask_with_support()
    assert kernel.version.as_tuple() >= (3, 0, 0)
    rng = np.random.default_rng(RNG_SEED + 11)
    streams, batch = 4, 3
    window = kernel.specs.detector_window_samples
    packed = rng.integers(0, 256, size=(batch, streams * WINDOWS, window), dtype=np.uint8).view(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, window), dtype=np.uint8).view(np.int8)
    packed[1] = _inject_line(packed[1], weights, streams, rng)
    packed[2] = 0  # signed complex zero, so every reference denominator vanishes
    bulk, rank = _bulk_and_rank()
    if case == "empty_bulk":
        bulk[:] = False
        rank = 0
    elif case in ("rank_boundary", "last_usable_rank"):
        bulk[:] = False
        bulk[[20, 30, 40, 50]] = True
        rank = 4 if case == "rank_boundary" else 3
    elif case == "zero_target":
        weights[0] = 0
    elif case == "zero_references":
        weights[1:] = 0
    options = dict(anchor=ANCHOR, half_width=HALF_WIDTH, bulk=bulk, rank=rank,
                   mult=MULT_Q16, batch=batch, streams=streams, with_taps=with_taps)
    legacy = _run_mask(cp, kernel, packed, weights, **options)
    fine, mask, tap, valid, counts = _run_mask(
        cp, kernel, packed, weights, with_support=True, **options)
    np.testing.assert_array_equal(fine, legacy[0])
    np.testing.assert_array_equal(mask, legacy[1])
    if with_taps:
        np.testing.assert_array_equal(tap, legacy[2])
    for index in range(batch):
        ref = fine_mask_decision(fine[index], anchor_bin=ANCHOR,
                                designated_half_width=HALF_WIDTH, bulk_mask=bulk,
                                cfar_rank=rank, multiplier_q16=MULT_Q16)
        assert (mask[index], valid[index], counts[index]) == (ref.mask, int(ref.valid), ref.n_bulk)
    assert valid[2] == 0 and counts[2] == 0 and mask[2] == 0
    if case == "zero_target":
        assert valid[0] == 1 and mask[0] == 0  # a zero score is valid and keeps the frame
    if case == "last_usable_rank":
        assert valid[0] == 1 and counts[0] == rank + 1


@pytest.mark.cuda
def test_mask_matches_reference_noise_and_injected():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    streams, batch = 8, 3
    rows = streams * WINDOWS
    packed = rng.integers(
        0, 256, size=(batch, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    packed[1] = _inject_line(packed[1], weights, streams, rng)
    bulk, rank = _bulk_and_rank()

    fine, mask, tap = _run_mask(
        cp, kernel, packed, weights, anchor=ANCHOR, half_width=HALF_WIDTH,
        bulk=bulk, rank=rank, mult=MULT_Q16, batch=batch, streams=streams,
    )
    for b in range(batch):
        # fine accumulator bit-exact against the fxfft chain
        ref_fine = fine_power_fx(
            tap[b].astype(np.int64), num_streams=streams
        )
        np.testing.assert_array_equal(fine[b], ref_fine)
        # mask bit-exact against the frozen decision reference
        ref = fine_mask_decision(
            fine[b],
            anchor_bin=ANCHOR,
            designated_half_width=HALF_WIDTH,
            bulk_mask=bulk,
            cfar_rank=rank,
            multiplier_q16=MULT_Q16,
        )
        assert mask[b] == ref.mask, (b, mask[b], ref.mask)
    # the injected frame must exercise the reject outcome
    assert mask[1] == 1


@pytest.mark.cuda
def test_mask_degenerate_bulk_forced_zero_on_device():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 1)
    streams = 4
    rows = streams * WINDOWS
    packed = rng.integers(
        0, 256, size=(1, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    packed[0] = _inject_line(packed[0], weights, streams, rng)
    empty = np.zeros(FINE_BINS, dtype=bool)
    fine, mask, _ = _run_mask(
        cp, kernel, packed, weights, anchor=ANCHOR, half_width=HALF_WIDTH,
        bulk=empty, rank=0, mult=MULT_Q16, batch=1, streams=streams,
    )
    assert mask.tolist() == [0]
    ref = fine_mask_decision(
        fine[0], anchor_bin=ANCHOR, designated_half_width=HALF_WIDTH,
        bulk_mask=empty, cfar_rank=0, multiplier_q16=MULT_Q16,
    )
    assert ref.mask == 0 and not ref.valid


@pytest.mark.cuda
def test_mask_production_form_without_taps():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 2)
    streams, batch = 8, 2
    rows = streams * WINDOWS
    packed = rng.integers(
        0, 256, size=(batch, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    packed[0] = _inject_line(packed[0], weights, streams, rng)
    bulk, rank = _bulk_and_rank()
    _, mask_taps, _ = _run_mask(
        cp, kernel, packed, weights, anchor=ANCHOR, half_width=HALF_WIDTH,
        bulk=bulk, rank=rank, mult=MULT_Q16, batch=batch, streams=streams,
        with_taps=True,
    )
    _, mask_prod, _ = _run_mask(
        cp, kernel, packed, weights, anchor=ANCHOR, half_width=HALF_WIDTH,
        bulk=bulk, rank=rank, mult=MULT_Q16, batch=batch, streams=streams,
        with_taps=False,
    )
    np.testing.assert_array_equal(mask_prod, mask_taps)


@pytest.mark.cuda
def test_mask_last_block_determinism_over_repeats():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 3)
    streams, batch = 64, 2
    rows = streams * WINDOWS
    packed = rng.integers(
        0, 256, size=(batch, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    packed[1] = _inject_line(packed[1], weights, streams, rng)
    bulk, rank = _bulk_and_rank()
    baseline = None
    for _ in range(20):
        fine, mask, _ = _run_mask(
            cp, kernel, packed, weights, anchor=ANCHOR,
            half_width=HALF_WIDTH, bulk=bulk, rank=rank, mult=MULT_Q16,
            batch=batch, streams=streams, with_taps=False,
        )
        current = (fine.tobytes(), mask.tobytes())
        if baseline is None:
            baseline = current
        assert current == baseline


@pytest.mark.cuda
@pytest.mark.parametrize("window_samples", [64, 128])
def test_mask_waits_for_delayed_power_writers(tmp_path, window_samples):
    cp = cupy_or_skip()
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc is None:
        pytest.skip("nvcc is required to build the scheduling regression")
    cuda_dir = Path(__file__).resolve().parents[2] / "cuda"
    source = (cuda_dir / "pilotproxy_detector.cu").read_text()
    write = "            atomicAdd(&fine[i], static_cast<unsigned long long>(re * re + im * im));"
    assert source.count(write) == 1
    # Delay the second warp before its final power additions, without changing them.
    delay = """            if (MaskOut != NULL && n == PILOTPROXY_NUM_WEIGHT_TERMS - 1
                    && s == 0 && tid >= 32) {
                const unsigned long long start = clock64();
                while (clock64() - start < 5000000ULL) { }
            }
"""
    instrumented = tmp_path / "pilotproxy_detector.cu"
    instrumented.write_text(source.replace(write, delay + write))
    library = tmp_path / "libpilotproxy.so"
    result = subprocess.run(
        [nvcc, "-std=c++14", "-O3", "-DNDEBUG", "-Xcompiler=-fPIC", "--shared",
         f"-arch=sm_{cp.cuda.Device().compute_capability}",
         f"-DPILOTPROXY_DETECTOR_WINDOW_SAMPLES={window_samples}",
         "-DPILOTPROXY_BLOCK_THREADS=64", "-I", str(cuda_dir), str(instrumented),
         "-o", str(library)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    kernel = PilotProxyKernel(library)
    streams, batch, anchor = 2, 64, 40
    rng = np.random.default_rng(20260908)
    packed = rng.integers(0, 256, size=(batch, streams * WINDOWS, window_samples),
                          dtype=np.uint8).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, window_samples),
                           dtype=np.uint8).astype(np.int8)
    bulk, rank = _bulk_and_rank(anchor)
    fine, mask, _ = _run_mask(
        cp, kernel, packed, weights, anchor=anchor, half_width=0,
        bulk=bulk, rank=rank, mult=MULT_Q16, batch=batch, streams=streams,
        with_taps=False,
    )
    expected = [fine_mask_decision(
        frame, anchor_bin=anchor, designated_half_width=0, bulk_mask=bulk,
        cfar_rank=rank, multiplier_q16=MULT_Q16,
    ).mask for frame in fine]
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.cuda
def test_mask_rate_margin_report():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 4)
    streams = 2048
    rows = streams * WINDOWS
    packed = rng.integers(0, 256, size=(rows, 128), dtype=np.uint8).astype(
        np.int8
    )
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    bulk, rank = _bulk_and_rank()
    words = pack_bulk_mask(bulk)

    d_in = cp.asarray(packed)
    handle = kernel.create_raw(rows, d_in.data.ptr)
    d_fine = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_mask = cp.zeros(1, dtype=cp.int32)
    d_pow = cp.zeros(TERMS, dtype=cp.uint64)

    def mask_form():
        kernel.compute_fused_fine_mask_u64(
            handle, weights.ctypes.data, ANCHOR, HALF_WIDTH, words, rank,
            MULT_Q16, int(d_fine.data.ptr), int(d_mask.data.ptr), 0, 0,
        )

    def powers_form():
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data, int(d_fine.data.ptr),
            int(d_pow.data.ptr), 0,
        )

    def measure(fn, n=50):
        fn()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    try:
        ms_mask = measure(mask_form)
        ms_powers = measure(powers_form)
    finally:
        kernel.destroy(handle)

    frame_ms = 16384.0 / 390625.0 * 1000.0
    print(
        f"\ndeployed mask form, 2048-stream frame "
        f"(cadence {frame_ms:.1f} ms):\n"
        f"  fused + decision epilogue (production): {ms_mask:.3f} ms "
        f"(margin x{frame_ms / ms_mask:.0f})\n"
        f"  fused powers only (2.2.0 form):         {ms_powers:.3f} ms "
        f"(epilogue cost {ms_mask - ms_powers:+.3f} ms)"
    )
    assert kernel.last_error() == ""
