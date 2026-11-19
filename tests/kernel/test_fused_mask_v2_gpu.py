# coding=utf-8
"""GPU test of the fine decision v2 entry points against the Python reference.

The fused datapath is the one of fine decision v1, so the fine powers must be
those of the v1 call. The mask must equal
``pilot_proxy.fine_decision.fine_mask_decision_v2`` over the same exact powers,
at the keep boundary of every frame and one word below it, and with too few
background bins.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    fine_mask_decision_v2,
    fine_required_multiplier_q16_v2,
    pack_bulk_mask,
)
from gpu import cupy_or_skip
from pilot_proxy.kernel import PilotProxyKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS, WINDOWS, FINE_BINS = 3, 128, 256
ANCHOR, HALF_WIDTH = 141, 32


def _kernel_or_skip() -> PilotProxyKernel:
    try:
        kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libpilotproxy.so is not built")
    if not kernel.supports_fused_fine_mask_v2():
        pytest.skip("libpilotproxy.so has no fine decision v2: rebuild from the current CUDA sources.")
    return kernel


def _geometry():
    b = np.arange(FINE_BINS)
    distance = np.minimum((b - ANCHOR) % FINE_BINS, (ANCHOR - b) % FINE_BINS)
    bulk = (b % 2 == 0) & (distance > HALF_WIDTH)
    return bulk, (int(bulk.sum()) - 1) // 2


def _run(cp, kernel, packed, weights, *, bulk, rank, mult, version, streams):
    batch = packed.shape[0]
    d_in = cp.asarray(packed)
    handle = kernel.create_raw_batch(streams * WINDOWS, batch, d_in.data.ptr)
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_mask = cp.zeros(batch, dtype=cp.int32)
    d_valid = cp.full(batch, 77, dtype=cp.int32)
    d_count = cp.full(batch, 77, dtype=cp.int32)
    try:
        compute = (kernel.compute_fused_fine_mask_v2_with_support_u64 if version == 2
                   else kernel.compute_fused_fine_mask_with_support_u64)
        compute(handle, weights.ctypes.data, ANCHOR, HALF_WIDTH, pack_bulk_mask(bulk), rank, mult,
                int(d_fine.data.ptr), int(d_mask.data.ptr), int(d_valid.data.ptr), int(d_count.data.ptr))
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        return cp.asnumpy(d_fine), cp.asnumpy(d_mask), cp.asnumpy(d_valid), cp.asnumpy(d_count)
    finally:
        kernel.destroy(handle)


def _frames(rng, streams, batch):
    packed = rng.integers(0, 256, size=(batch, streams * WINDOWS, 128), dtype=np.uint8).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(np.int8)
    return packed, weights


def test_v2_mask_matches_reference_at_every_keep_boundary():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(20261118)
    streams, batch = 8, 6
    packed, weights = _frames(rng, streams, batch)
    bulk, rank = _geometry()
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF_WIDTH, bulk_mask=bulk, cfar_rank=rank)
    fine1, *_ = _run(cp, kernel, packed, weights, bulk=bulk, rank=rank, mult=65536, version=1, streams=streams)
    words = [fine_required_multiplier_q16_v2(f, **kw).multiplier_q16 for f in fine1]
    assert all(1 < w < ALWAYS_MASKED_Q16 for w in words)
    seen = set()
    for mult in sorted({w + d for w in words for d in (-1, 0)} | {1, 65536, 3 * 65536}):
        fine, mask, valid, count = _run(cp, kernel, packed, weights, bulk=bulk, rank=rank, mult=mult,
                                        version=2, streams=streams)
        np.testing.assert_array_equal(fine, fine1)                 # same datapath as v1
        for b in range(batch):
            ref = fine_mask_decision_v2(fine[b], multiplier_q16=mult, **kw)
            assert (mask[b], valid[b], count[b]) == (ref.mask, int(ref.valid), ref.n_bulk), (mult, b)
            assert mask[b] == int(mult < words[b])                 # the keep word is the change point
            seen.add(int(mask[b]))
    assert seen == {0, 1}


def test_v2_differs_from_v1_only_in_the_decision():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(7)
    streams, batch = 4, 4
    packed, weights = _frames(rng, streams, batch)
    bulk, rank = _geometry()
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF_WIDTH, bulk_mask=bulk, cfar_rank=rank)
    f1, m1, v1, c1 = _run(cp, kernel, packed, weights, bulk=bulk, rank=rank, mult=70000, version=1, streams=streams)
    f2, m2, v2, c2 = _run(cp, kernel, packed, weights, bulk=bulk, rank=rank, mult=70000, version=2, streams=streams)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(c1, c2)
    for b in range(batch):
        assert m2[b] == fine_mask_decision_v2(f2[b], multiplier_q16=70000, **kw).mask


def test_v2_rank_past_the_background_is_invalid():
    cp = cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(11)
    streams = 4
    packed, weights = _frames(rng, streams, 2)
    bulk, _ = _geometry()
    rank = int(bulk.sum())                                         # one past the last usable rank
    _, mask, valid, count = _run(cp, kernel, packed, weights, bulk=bulk, rank=rank, mult=1, version=2, streams=streams)
    assert mask.tolist() == [0, 0] and valid.tolist() == [0, 0] and count.tolist() == [rank, rank]
