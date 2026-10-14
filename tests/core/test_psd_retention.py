# coding=utf-8
"""The per-frame PSD must survive its int16 dB encoding.

The analyzer already ran a 16384-point FFT over every feed each frame and
power-summed it; v1 and v2 folded that into two accumulators and dropped the
per-frame array. archive_health's SPECTRAL_LIMITATION names the consequence:
the gate "cannot apply an arbitrary new frame mask, FFT window, or threshold to
the archived spectra".

Retaining it is only useful if the encoding is faithful, so the round trip is
pinned here rather than left to review.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.product_contract import (
    PSD_DB_INVALID as _PSD_DB_INVALID,
    PSD_DB_MAX as _PSD_DB_MAX,
    PSD_DB_MIN as _PSD_DB_MIN,
)

STEP_PER_CODE = 0.01  # dB


def _encode(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of the analyzer: each frame encoded about its own reference.

    Encoding on arrival is what keeps residency at int16 rather than holding
    float64 for a whole channel, so the reference is per frame rather than per
    product.
    """
    codes = np.empty(stack.shape, dtype=np.int16)
    refs = np.empty(stack.shape[0], dtype=np.float64)
    for i, frame in enumerate(stack):
        finite = frame[np.isfinite(frame) & (frame > 0.0)]
        reference = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(reference) or reference <= 0.0:
            reference = 1.0
        refs[i] = reference
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(frame / reference)
        codes[i] = np.where(
            np.isfinite(db),
            np.clip(np.rint(db * 100.0), _PSD_DB_MIN, _PSD_DB_MAX),
            _PSD_DB_INVALID,
        ).astype(np.int16)
    return codes, refs


def _decode(codes: np.ndarray, refs: np.ndarray) -> np.ndarray:
    out = np.asarray(refs)[:, None] * 10.0 ** (codes.astype(np.float64) / 1000.0)
    out[codes == _PSD_DB_INVALID] = np.nan
    return out


def _realistic(n_frames: int = 6, n_bins: int = 16384) -> np.ndarray:
    """Broadband floor with a few strong lines, like a real coarse channel."""
    rng = np.random.default_rng(20260822)
    base = rng.gamma(shape=2048.0, scale=7.0e4, size=(n_frames, n_bins))
    for b in (137, 4096, 9001):
        base[:, b] *= 40.0
    return base


def test_round_trip_error_is_within_half_a_step() -> None:
    stack = _realistic()
    codes, refs = _encode(stack)
    back = _decode(codes, refs)
    err_db = np.abs(10.0 * np.log10(back / stack))
    # Rounding to the nearest code cannot exceed half a step.
    assert err_db.max() <= STEP_PER_CODE / 2.0 + 1e-9


def test_strong_lines_survive_the_encoding() -> None:
    """A 40x line must still stand ~16 dB above the floor after decoding."""
    stack = _realistic()
    codes, refs = _encode(stack)
    back = _decode(codes, refs)
    floor = np.median(back, axis=1)
    for b in (137, 4096, 9001):
        excess_db = 10.0 * np.log10(back[:, b] / floor)
        assert np.all(excess_db > 15.0), b


def test_invalid_frames_decode_to_nan_not_a_power() -> None:
    """A frame that never reached the transform must not look like data."""
    stack = _realistic(n_frames=3)
    stack[1, :] = np.nan
    codes, refs = _encode(stack)
    back = _decode(codes, refs)
    assert np.all(codes[1] == _PSD_DB_INVALID)
    assert np.all(np.isnan(back[1]))
    assert np.all(np.isfinite(back[0])) and np.all(np.isfinite(back[2]))


def test_the_sentinel_is_not_a_reachable_measurement() -> None:
    """_PSD_DB_INVALID must sit outside the clipped code range."""
    assert _PSD_DB_INVALID < _PSD_DB_MIN <= _PSD_DB_MAX
    assert _PSD_DB_MIN == -32767 and _PSD_DB_MAX == 32767


def test_summed_frames_reproduce_an_integrated_spectrum() -> None:
    """The accumulators the product still carries are derivable from the frames."""
    stack = _realistic()
    codes, refs = _encode(stack)
    back = _decode(codes, refs)
    direct = stack.sum(axis=0)
    summed = back.sum(axis=0)
    rel = np.abs(summed - direct) / direct
    assert rel.max() < 2e-3


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e12])
def test_reference_normalisation_is_scale_free(scale: float) -> None:
    """Encoding is relative, so an overall gain change must not lose precision."""
    stack = _realistic(n_frames=3) * scale
    codes, refs = _encode(stack)
    back = _decode(codes, refs)
    err_db = np.abs(10.0 * np.log10(back / stack))
    assert err_db.max() <= STEP_PER_CODE / 2.0 + 1e-9
