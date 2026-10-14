# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.frame_adapter import (
    pack_chime_block_for_detector,
    repack_chime_offset_binary_i4_to_twos_complement,
)
from pilot_proxy.chime.hdf5_input import CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4


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
        spectral_sense="normal",
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
        spectral_sense="normal",
        frames_in_chunk=1,
        sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        selected_coarse_channel=843,
        physical_channel=14,
    )

    assert packed.packed.shape == (1, 12, 2)
