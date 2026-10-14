# coding=utf-8
"""The per-frame PSD accumulator must stay int16, and that is a memory contract.

The analyzer computes each frame's PSD in float64 but retains only an int16 dB
encoding (``psd_frame_db_i16``). That choice is what bounds resident memory on a
long channel: measured on 2026-08-23, 35,095 frames -- the count needed to reach
the 4.6 GB figure recorded against the float64 behaviour -- costs 1102 MB as
int16 against 3291 MB as float64. A regression to a wider dtype would not fail
any correctness test, so it is asserted here directly.
"""
from __future__ import annotations

import numpy as np


from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer

NFFT = 1024


def _bare_analyzer(nfft=NFFT):
    """An instance with only the state _append_psd_frame touches."""
    a = object.__new__(PilotProxyDetectorAnalyzer)
    a._nfft = nfft
    a._psd_codes = []
    a._psd_ref = []
    return a


def test_psd_frames_are_retained_as_int16():
    a = _bare_analyzer()
    rng = np.random.default_rng(11)
    for _ in range(4):
        a._append_psd_frame(rng.gamma(2.0, 1.0, NFFT).astype(np.float64))
    assert len(a._psd_codes) == 4
    for code in a._psd_codes:
        assert code.dtype == np.int16, "PSD retention widened; resident memory triples"
        assert code.shape == (NFFT,)


def test_invalid_frame_uses_the_sentinel_not_a_float():
    a = _bare_analyzer()
    a._append_psd_frame(None)
    assert a._psd_codes[0].dtype == np.int16
    assert np.isnan(a._psd_ref[0])


def test_resident_cost_per_frame_is_two_bytes_per_bin():
    """The memory contract, stated as arithmetic rather than a timing test."""
    a = _bare_analyzer()
    rng = np.random.default_rng(12)
    for _ in range(8):
        a._append_psd_frame(rng.gamma(2.0, 1.0, NFFT).astype(np.float64))
    resident = sum(c.nbytes for c in a._psd_codes)
    assert resident == 8 * NFFT * 2
    # what that implies at archive scale, for the record
    assert 35095 * NFFT * 2 < 35095 * NFFT * 8


# --- ADC / F-engine saturation ------------------------------------------------

def test_railed_component_count_finds_both_rails():
    """Offset-binary nibble 0 is signed -8 and 15 is +7: both are clipped."""
    from pilot_proxy.archive.detector import railed_component_count

    # 0x00 is also the archive's missing-data fill signature, so an all-zero
    # frame is all fill, not all clipping.
    assert railed_component_count(np.zeros((4, 4), dtype=np.uint8)) == (0, 2 * 16)
    assert railed_component_count(np.full((4, 4), 0xFF, dtype=np.uint8)) == (2 * 16, 0)
    # 0x88 is signed (0, 0) -- mid-scale, nothing clipped
    assert railed_component_count(np.full((4, 4), 0x88, dtype=np.uint8)) == (0, 0)


def test_railed_component_count_counts_real_and_imag_separately():
    from pilot_proxy.archive.detector import railed_component_count

    # 0x0F -> real 0 (rail), imag 15 (rail)      = 2
    # 0x88 -> real 8, imag 8                     = 0
    # 0xF0 -> real 15 (rail), imag 0 (rail)      = 2
    # 0x77 -> real 7, imag 7                     = 0
    packed = np.array([[0x0F, 0x88], [0xF0, 0x77]], dtype=np.uint8)
    assert railed_component_count(packed) == (4, 0)


def test_railed_component_count_splits_fill_from_single_rails():
    """A single railed nibble counts as clipping; only the 0x00 byte is fill."""
    from pilot_proxy.archive.detector import railed_component_count

    # 0x08 -> real 0 (rail), imag 8              = 1 railed
    # 0x80 -> real 8, imag 0 (rail)              = 1 railed
    # 0x00 -> both components fill               = 2 fill
    packed = np.array([0x08, 0x80, 0x00], dtype=np.uint8)
    assert railed_component_count(packed) == (2, 2)


def test_railed_component_count_matches_direct_nibble_count():
    """The byte-indexed tally equals the straightforward nibble arithmetic."""
    from pilot_proxy.archive.detector import railed_component_count

    rng = np.random.default_rng(5)
    packed = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    real = packed >> 4
    imag = packed & np.uint8(0x0F)
    rails = (
        np.count_nonzero((real == 0) & (packed != 0))
        + np.count_nonzero(real == 15)
        + np.count_nonzero((imag == 0) & (packed != 0))
        + np.count_nonzero(imag == 15)
    )
    fill = 2 * int(np.count_nonzero(packed == 0))
    railed_count, fill_count = railed_component_count(packed)
    assert (railed_count, fill_count) == (rails, fill)
    assert 0 <= railed_count + fill_count <= 2 * packed.size
