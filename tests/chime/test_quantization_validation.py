"""CHIME calibration and packing preserve the quantization contract."""
from dataclasses import replace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
from pilot_proxy.chime.hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    COMPLEX_FLOAT,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    ChimePilotDataset,
    ChimeSegment,
)
from pilot_proxy.chime.runner import _calibration_scale_for_dataset


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


@pytest.fixture
def calibration_dataset(tmp_path):
    # Successive blocks have known real spreads 2, 4 and 8.
    values = np.concatenate([np.tile([-amplitude, amplitude], 4)
                             for amplitude in (2, 4, 8)]).astype(np.complex128)
    values += 3j
    path = tmp_path / "samples.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("baseband", data=values[:, None])
    segment = ChimeSegment(
        path=path, physical_channel=14, pilot_frequency_hz=470.31e6,
        dataset_path="baseband", num_time_samples=24, shape=(24, 1),
        dtype="complex128", sample_encoding=COMPLEX_FLOAT)
    return ChimePilotDataset(
        physical_channel=14, pilot_frequency_hz=470.31e6,
        segments=[segment], dataset_path="baseband", time_axis=0,
        stream_axis=1, sample_encoding=COMPLEX_FLOAT)


@pytest.mark.parametrize("seconds,variance", [(0, 4), (16 / 390625, 10), (10, 28)])
def test_calibration_window_respects_one_frame_and_available_samples(calibration_dataset, seconds, variance):
    scale = _calibration_scale_for_dataset(
        calibration_dataset, frame_size_samples=8, calibration_seconds=seconds,
        bits_per_component=4, clip_sigma=2)
    assert scale == pytest.approx(7 / (2 * np.sqrt(variance)))


@pytest.mark.parametrize("settings", [
    {"bits_per_component": 4.5}, {"clip_sigma": True}, {"clip_sigma": "3.0"},
])
def test_dataset_calibration_does_not_coerce_invalid_settings(calibration_dataset, settings):
    options = {"bits_per_component": 4, "clip_sigma": 3, **settings}
    with pytest.raises((TypeError, ValueError), match="bit|clip_sigma"):
        _calibration_scale_for_dataset(
            calibration_dataset, frame_size_samples=8, calibration_seconds=0,
            **options)


@pytest.mark.parametrize("encoding", [
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
])
def test_native_calibration_does_not_read_or_rescale_samples(calibration_dataset, encoding):
    calibration_dataset.segments[0].path.unlink()
    native = replace(calibration_dataset, sample_encoding=encoding)
    assert _calibration_scale_for_dataset(
        native, frame_size_samples=8, calibration_seconds=1,
        bits_per_component=4, clip_sigma=3) is None
