"""CHIME packing preserves the quantization contract."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
from pilot_proxy.chime.hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    COMPLEX_FLOAT,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
)


def _pack(block, *, encoding=COMPLEX_FLOAT, **quantization):
    return pack_chime_block_for_detector(
        block, frame_size_samples=8, detector_window_samples=4,
        nyquist_zone=1, frames_in_chunk=1, sample_encoding=encoding,
        selected_coarse_channel=843, physical_channel=14, **quantization)


@pytest.mark.parametrize("bits", [4.5, 4.0, "4", True])
def test_float_adapter_does_not_coerce_invalid_bit_depth(bits):
    values = np.tile([-2 + 1j, 2 - 1j], 4).reshape(1, 1, 8)
    with pytest.raises((TypeError, ValueError), match="bit"):
        _pack(values, bits_per_component=bits)


@pytest.mark.parametrize("clip_sigma", [True, "3.0", 0.0, -1.0, np.nan, np.inf])
def test_float_adapter_preserves_invalid_automatic_scale_settings(clip_sigma):
    values = np.tile([-2 + 1j, 2 - 1j], 4).reshape(1, 1, 8)
    with pytest.raises((TypeError, ValueError), match="clip_sigma"):
        _pack(values, clip_sigma=clip_sigma)


@pytest.mark.parametrize("encoding", [
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
])
@pytest.mark.parametrize("bits", [8, 4.5, True])
def test_native_int4_cannot_be_labeled_with_another_bit_depth(encoding, bits):
    with pytest.raises((TypeError, ValueError), match="bit"):
        _pack(np.zeros((1, 1, 8), dtype=np.uint8), encoding=encoding,
              bits_per_component=bits)


@pytest.mark.parametrize("bits,dtype,pair", [
    (np.int64(4), np.int8, [-62, 78]),
    (np.int64(8), np.int16, [-16352, 16608]),
])
def test_float_adapter_uses_the_estimated_scale_and_matching_packed_dtype(bits, dtype, pair):
    # Real samples +/-2 have sigma=2; clip_sigma=2 gives scales 7/4 and 127/4.
    values = np.tile([-2 + 1j, 2 - 1j], 4).reshape(1, 1, 8)
    result = _pack(values, bits_per_component=bits, clip_sigma=2)
    assert result.packed.dtype == dtype
    np.testing.assert_array_equal(result.packed, np.tile(pair, 4).reshape(1, 2, 4))
    np.testing.assert_allclose(result.baseband_power_linear, [5], rtol=5e-16, atol=0)
    assert result.quantization["bits_per_component"] == bits
