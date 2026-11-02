# coding=utf-8
"""Tests that the scan keeps the exact fine-power terms.

The v1 product stored only ``fine_power_ratio``, one float32 per bin formed
with a floating-point FFT. Two pieces of information were lost at write time
and cannot be recovered. First, the three uint64 terms of each bin were reduced
to their ratio. Second, the ratio was computed with ``numpy.fft`` instead of
the frozen fixed-point fxfft256 deployed in the kernel, so it is a different
number.

The frozen fine decision v1, ``fine_mask_decision``, requires
``uint64 [3, 256]`` input. A product that stores only the ratio therefore
cannot replay the deployed decision at any threshold. Since a rebuild of the
archive takes weeks, these properties are checked by the tests below.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_decision import FINE_BINS, fine_mask_decision
from pilot_proxy.fine_reduction import fine_reduce
from pilot_proxy.fxfft import fine_power_fx

TERMS = 3
STREAMS = 8
WINDOWS = 128
ROW_ABS_MAX = 14336  # kernel contract: |row-sum component| <= 14336


def _row_projections(seed: int = 20260821) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        -ROW_ABS_MAX, ROW_ABS_MAX + 1, size=(TERMS, STREAMS * WINDOWS, 2)
    ).astype(np.int32)


def _exact(rs: np.ndarray) -> np.ndarray:
    return fine_power_fx(rs, num_streams=STREAMS, windows_per_stream=WINDOWS)


def test_exact_fine_powers_are_uint64_with_the_decision_shape() -> None:
    S = _exact(_row_projections())
    assert S.dtype == np.dtype(np.uint64)
    assert S.shape == (TERMS, FINE_BINS)


def test_frozen_decision_accepts_the_retained_terms() -> None:
    """The retained array is exactly what the deployed decision consumes."""
    S = _exact(_row_projections())
    decision = fine_mask_decision(
        S,
        anchor_bin=0,
        designated_half_width=1,
        bulk_mask=[True] * FINE_BINS,
        cfar_rank=128,
        multiplier_q16=1 << 16,
    )
    assert decision.mask in (0, 1)
    assert decision.n_bulk > 0


def test_float_ratio_cannot_drive_the_frozen_decision() -> None:
    """Check that the frozen decision rejects the float ratio."""
    rs = _row_projections()
    ratio = np.asarray(
        fine_reduce(
            rs, num_streams=STREAMS, windows_per_stream=WINDOWS
        ).fine_power_ratio,
        dtype=np.float32,
    )
    assert ratio.shape == (FINE_BINS,)
    # Even broadcast to the right shape, the dtype contract rejects it.
    with pytest.raises(TypeError):
        fine_mask_decision(
            np.broadcast_to(ratio, (TERMS, FINE_BINS)),
            anchor_bin=0,
            designated_half_width=1,
            bulk_mask=[True] * FINE_BINS,
            cfar_rank=128,
            multiplier_q16=1 << 16,
        )


def test_float_reduction_is_not_the_deployed_statistic() -> None:
    """Check the second loss, a float FFT that disagrees with the frozen transform."""
    rs = _row_projections()
    exact = _exact(rs).astype(np.float64)
    approx = np.asarray(
        fine_reduce(rs, num_streams=STREAMS, windows_per_stream=WINDOWS).fine_power,
        dtype=np.float64,
    )
    rel = np.abs(approx - exact) / np.maximum(exact, 1.0)
    # The largest relative difference is below 1e-3 and above 0. The float
    # powers are close to the exact integers, and no later float computation
    # can recover them.
    assert rel.max() > 0.0
    assert rel.max() < 1e-3


def test_retained_terms_survive_the_product_round_trip(tmp_path) -> None:
    """Check that a save and load round trip keeps the uint64 terms unchanged."""
    frames = 4
    stack = np.stack([_exact(_row_projections(seed)) for seed in range(frames)])
    assert stack.shape == (frames, TERMS, FINE_BINS)

    path = tmp_path / "product.npz"
    np.savez_compressed(path, fine_power_u64=stack)
    with np.load(path) as data:
        restored = np.asarray(data["fine_power_u64"])

    assert restored.dtype == np.dtype(np.uint64)
    assert np.array_equal(restored, stack)

    # The decision must also replay from the restored bytes.
    for i in range(frames):
        fine_mask_decision(
            np.ascontiguousarray(restored[i]),
            anchor_bin=0,
            designated_half_width=1,
            bulk_mask=[True] * FINE_BINS,
            cfar_rank=128,
            multiplier_q16=1 << 16,
        )


def test_decision_is_threshold_replayable_from_retained_terms() -> None:
    """Check that eta can be swept from the kept terms, without the raw data."""
    S = _exact(_row_projections())
    outcomes = {
        q: fine_mask_decision(
            S,
            anchor_bin=0,
            designated_half_width=1,
            bulk_mask=[True] * FINE_BINS,
            cfar_rank=128,
            multiplier_q16=q,
        ).mask
        for q in (1 << 14, 1 << 16, 1 << 18, 1 << 20)
    }
    # The family is monotone, since a larger multiplier can only make firing harder.
    masks = [outcomes[q] for q in sorted(outcomes)]
    assert masks == sorted(masks, reverse=True)
