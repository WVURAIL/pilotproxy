# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.frame_adapter import (
    baseband_power_by_frame,
    pack_chime_block_for_detector,
    repack_chime_offset_binary_i4_to_twos_complement,
    unpack_chime_offset_binary_i4_to_complex,
    unpack_twos_complement_i4_to_complex,
)
from pilot_proxy.chime.hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
)


def _encode_offset_binary(real: np.ndarray, imag: np.ndarray | int = 0) -> np.ndarray:
    r = (np.asarray(real, dtype=np.int16) + 8) & 0x0F
    i = (np.asarray(imag, dtype=np.int16) + 8) & 0x0F
    packed = np.asarray((r << 4) | i, dtype=np.uint8)
    return packed.astype(np.uint8, copy=False)


def test_native_packed_chunk_shape_and_row_order() -> None:
    real = np.asarray(
        [
            [-3, -2, -1, 0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0, -1, -2, -3],
        ],
        dtype=np.int16,
    )
    block = _encode_offset_binary(real)[:, np.newaxis, :]

    packed = pack_chime_block_for_detector(
        block,
        frame_size_samples=8,
        detector_window_samples=4,
        nyquist_zone=1,
        frames_in_chunk=1,
        sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        selected_coarse_channel=843,
        physical_channel=14,
    )

    assert packed.packed.shape == (1, 4, 4)
    assert packed.input_layout["detector_rows_per_frame"] == 4
    assert set(packed.input_layout) == {
        "schema_version",
        "frame_size_samples",
        "detector_window_samples",
        "windows_per_stream",
        "num_input_streams",
        "num_selected_channels",
        "num_streams",
        "detector_rows_per_frame",
        "samples_per_result",
        "combine_mode",
    }
    expected = np.stack(
        [
            repack_chime_offset_binary_i4_to_twos_complement(block[0, 0, 0:4]),
            repack_chime_offset_binary_i4_to_twos_complement(block[0, 0, 4:8]),
            repack_chime_offset_binary_i4_to_twos_complement(block[1, 0, 0:4]),
            repack_chime_offset_binary_i4_to_twos_complement(block[1, 0, 4:8]),
        ]
    )
    np.testing.assert_array_equal(packed.packed[0], expected)
    assert packed.quantization["source"] == "native_chime"


def test_row_count_is_streams_times_windows() -> None:
    block = _encode_offset_binary(np.zeros((3, 8), dtype=np.int16))[:, np.newaxis, :]

    packed = pack_chime_block_for_detector(
        block,
        frame_size_samples=8,
        detector_window_samples=2,
        nyquist_zone=1,
        frames_in_chunk=1,
        sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        selected_coarse_channel=843,
        physical_channel=14,
    )

    assert packed.packed.shape == (1, 12, 2)


@pytest.mark.parametrize("encoding,decode,levels", [
    (CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
     unpack_chime_offset_binary_i4_to_complex, tuple(range(-8, 8))),
    (PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
     unpack_twos_complement_i4_to_complex,
     (0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1)),
])
def test_every_packed_byte_decodes_and_contributes_exact_frame_power(
        encoding, decode, levels) -> None:
    """Check every real/imag nibble pair against signed integer levels."""
    raw = np.arange(256, dtype=np.uint8).reshape(2, 1, 128)
    expected = np.asarray([
        complex(levels[real], levels[imag])
        for real in range(16) for imag in range(16)
    ], dtype=np.complex64).reshape(raw.shape)

    decoded = decode(raw)
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == np.complex64

    powers = baseband_power_by_frame(
        raw, frame_size_samples=64, frames_in_chunk=2,
        sample_encoding=encoding)
    expected_power = [
        sum(int(value.real) ** 2 + int(value.imag) ** 2
            for value in expected[:, :, start:start + 64].flat) / 128
        for start in (0, 64)
    ]
    np.testing.assert_array_equal(powers, expected_power)

    if encoding == CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4:
        repacked = repack_chime_offset_binary_i4_to_twos_complement(raw)
        np.testing.assert_array_equal(repacked.view(np.uint8), raw ^ 0x88)
        np.testing.assert_array_equal(
            unpack_twos_complement_i4_to_complex(repacked), expected)
