# coding=utf-8
"""GPU test of the fine decision v3 statistic against the Python reference.

The fused datapath is the one of fine decision v1, so the fine powers must be
those of the v1 call. The pair ``[n_peak, d_rho]`` must equal
``pilot_proxy.fine_decision.fine_stat_v3`` over the same exact powers, and its
decision at any word must equal the v2 kernel mask over the even bins at rank 63.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    V3_RANK,
    fine_keep_word_v3,
    fine_keep_v3,
    fine_stat_v3,
    pack_bulk_mask,
)
from gpu import cupy_or_skip
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS, WINDOWS, FINE_BINS = 3, 128, 256
ANCHOR, HALF_WIDTH = 141, 32
EVEN = np.arange(FINE_BINS) % 2 == 0


def _kernel_or_skip() -> PilotProxyKernel:
    try:
        kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libpilotproxy.so is not built")
    if not kernel.supports_fine_stat_v3():
        pytest.skip("libpilotproxy.so has no fine decision v3: rebuild from the current CUDA sources.")
    return kernel


def _frames(rng, streams, batch):
    packed = rng.integers(0, 256, size=(batch, streams * WINDOWS, 128), dtype=np.uint8).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(np.int8)
    return packed, weights


def _stat(cp, kernel, packed, weights, streams):
    batch = packed.shape[0]
    d_in = cp.asarray(packed)
    handle = kernel.create_raw_batch(streams * WINDOWS, batch, d_in.data.ptr)
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_stat = cp.full((batch, 2), 77, dtype=cp.uint64)
    try:
        kernel.compute_fine_stat_v3_u64(handle, weights.ctypes.data, ANCHOR, HALF_WIDTH,
                                        int(d_fine.data.ptr), int(d_stat.data.ptr))
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        return cp.asnumpy(d_fine), cp.asnumpy(d_stat)
    finally:
        kernel.destroy(handle)


def _mask_v2(cp, kernel, packed, weights, streams, mult):
    """The v2 kernel mask and fine powers, with the even bins as the background at rank 63."""
    batch = packed.shape[0]
    d_in = cp.asarray(packed)
    handle = kernel.create_raw_batch(streams * WINDOWS, batch, d_in.data.ptr)
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_mask = cp.zeros(batch, dtype=cp.int32)
    try:
        kernel.compute_fused_fine_mask_v2_u64(handle, weights.ctypes.data, ANCHOR, HALF_WIDTH,
                                              pack_bulk_mask(EVEN), V3_RANK, mult,
                                              int(d_fine.data.ptr), int(d_mask.data.ptr))
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        return cp.asnumpy(d_fine), cp.asnumpy(d_mask)
    finally:
        kernel.destroy(handle)


def test_v3_stat_matches_reference():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(20261121)
    for streams, batch in ((8, 6), (4, 3), (16, 2)):
        packed, weights = _frames(rng, streams, batch)
        fine, stat = _stat(cp, kernel, packed, weights, streams)
        fine1, _ = _mask_v2(cp, kernel, packed, weights, streams, 65536)
        np.testing.assert_array_equal(fine, fine1)                       # same datapath
        for b in range(batch):
            ref = fine_stat_v3(fine[b], anchor_bin=ANCHOR, designated_half_width=HALF_WIDTH)
            assert (int(stat[b, 0]), int(stat[b, 1])) == (ref.n_peak, ref.d_rho), (streams, b)


def test_v3_decision_equals_v2_mask_over_the_even_bins():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(11)
    streams, batch = 8, 6
    packed, weights = _frames(rng, streams, batch)
    _, stat = _stat(cp, kernel, packed, weights, streams)
    words = [fine_keep_word_v3(int(n), int(d)) for n, d in stat]
    assert all(1 < w < (1 << 64) for w in words)
    seen = set()
    for mult in sorted({w + delta for w in words for delta in (-1, 0)} | {1, 65536, 3 * 65536}):
        _, mask = _mask_v2(cp, kernel, packed, weights, streams, mult)
        for b in range(batch):
            keep = fine_keep_v3(int(stat[b, 0]), int(stat[b, 1]), mult)
            assert keep == 1 - int(mask[b]) == int(mult >= words[b]), (mult, b)   # 1 keeps
            seen.add(keep)
    assert seen == {0, 1}


def test_v3_zero_input_has_no_decision():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    streams = 4
    packed = np.zeros((2, streams * WINDOWS, 128), dtype=np.int8)
    weights = np.random.default_rng(3).integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(np.int8)
    _, stat = _stat(cp, kernel, packed, weights, streams)
    assert stat.tolist() == [[0, 0], [0, 0]]
    assert fine_keep_v3(0, 0, 65536) is None
