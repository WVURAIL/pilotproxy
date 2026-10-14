# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime.hdf5_input import (
    discover_chime_pilot_datasets,
    nearest_atsc_physical_channel,
    read_complex_window,
)
from pilot_proxy.chime.segmented_input import available_frames, iter_frame_chunks
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz


def _write_segment(path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=data)
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)


def test_multiple_segment_files_concatenate_in_sorted_file_order(tmp_path) -> None:
    first = np.asarray(
        [
            [10, 20],
            [11, 21],
            [12, 22],
            [13, 23],
        ],
        dtype=np.uint8,
    )
    second = np.asarray(
        [
            [30, 40],
            [31, 41],
            [32, 42],
            [33, 43],
        ],
        dtype=np.uint8,
    )
    _write_segment(tmp_path / "ch0844" / "002.h5", second)
    _write_segment(tmp_path / "ch0844" / "001.h5", first)

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]
    block = read_complex_window(dataset, start_sample=2, stop_sample=6)

    assert dataset.total_time_samples == 8
    assert block.shape == (2, 1, 4)
    np.testing.assert_array_equal(block[0, 0], [12, 13, 30, 31])
    np.testing.assert_array_equal(block[1, 0], [22, 23, 40, 41])


def test_absolute_time_is_not_required_for_frame_chunks(tmp_path) -> None:
    _write_segment(tmp_path / "ch0844" / "001.h5", np.zeros((10, 2), dtype=np.uint8))
    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]

    assert available_frames(dataset, frame_size_samples=4) == 2
    chunks = list(
        iter_frame_chunks(
            dataset,
            frame_size_samples=4,
            frames_per_chunk=1,
        )
    )

    assert [chunk.start_sample for chunk in chunks] == [0, 4]
    assert [chunk.stop_sample for chunk in chunks] == [4, 8]


def test_adjacent_coarse_channel_is_not_mapped_to_same_atsc_pilot() -> None:
    assert nearest_atsc_physical_channel(470_312_500.0) == 14
    assert nearest_atsc_physical_channel(470_703_125.0) is None


def test_discovery_refuses_adjacent_freq_id_with_explicit_physical_channel(tmp_path) -> None:
    for freq_id, centre in ((844, 470.3125), (843, 470.703125)):
        path = tmp_path / str(freq_id) / "001.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as h5:
            h5.attrs["physical_channel"] = 14
            h5.attrs["freq"] = centre
            h5.attrs["freq_id"] = freq_id
            ds = h5.create_dataset(
                "baseband", data=np.zeros((4, 2), dtype=np.uint8)
            )
            ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)

    with pytest.raises(ValueError, match="not within half a coarse bin"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_discovery_refuses_mixed_dtype_within_coarse_channel(tmp_path) -> None:
    _write_segment(tmp_path / "ch0844" / "001.h5", np.zeros((4, 2), np.uint8))
    _write_segment(tmp_path / "ch0844" / "002.h5", np.zeros((4, 2), np.int8))

    with pytest.raises(ValueError, match="inconsistent dtypes"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_discovery_refuses_mixed_rank_within_coarse_channel(tmp_path) -> None:
    _write_segment(tmp_path / "ch0844" / "001.h5", np.zeros((4, 2), np.uint8))
    _write_segment(
        tmp_path / "ch0844" / "002.h5", np.zeros((4, 2, 2), np.uint8)
    )

    with pytest.raises(ValueError, match="inconsistent ranks"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_discovery_refuses_single_dataset_with_extra_axis(tmp_path) -> None:
    _write_segment(tmp_path / "ch0844" / "001.h5", np.zeros((4, 2, 1), np.uint8))

    with pytest.raises(ValueError, match="exactly time and input axes"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_discovery_classifies_explicit_uint8_real_imag_axis(tmp_path) -> None:
    path = tmp_path / "ch0844" / "001.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=np.zeros((4, 2, 2), np.uint8))
        ds.attrs["axis"] = np.asarray(["time", "input", "complex"], dtype=object)

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]

    assert dataset.complex_axis == 2
    assert dataset.sample_encoding == "real_imag_last_axis"


def test_unlabelled_real_imag_array_uses_time_major_convention(tmp_path) -> None:
    path = tmp_path / "ch0844" / "001.h5"
    path.parent.mkdir(parents=True)
    raw = np.zeros((4, 2, 2), np.uint8)
    raw[:, 0, 0] = np.arange(4, dtype=np.uint8)
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        h5.create_dataset("baseband", data=raw)

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]
    block = read_complex_window(dataset, start_sample=0, stop_sample=4)

    assert dataset.time_axis == 0
    assert dataset.stream_axis == 1
    assert dataset.total_time_samples == 4
    assert dataset.num_input_streams == 2
    assert block.shape == (2, 1, 4)


def test_discovery_refuses_extra_axis_on_complex_dtype(tmp_path) -> None:
    _write_segment(
        tmp_path / "ch0844" / "001.h5", np.zeros((4, 2, 2), np.complex64)
    )

    with pytest.raises(ValueError, match="exactly time and input axes"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def _write_identity_only_segment(path, **attrs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        ds = h5.create_dataset("baseband", data=np.zeros((4, 2), dtype=np.uint8))
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)


def test_freq_id_only_metadata_derives_coarse_center(tmp_path) -> None:
    _write_identity_only_segment(
        tmp_path / "844.h5", freq_id=844, physical_channel=14
    )

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]

    assert dataset.freq_id == 844
    assert dataset.coarse_channel_center_hz == pytest.approx(470_312_500.0)


def test_freq_id_only_metadata_validates_explicit_physical_channel(tmp_path) -> None:
    _write_identity_only_segment(
        tmp_path / "843.h5", freq_id=843, physical_channel=14
    )

    with pytest.raises(ValueError, match="not within half a coarse bin"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_nonintegral_freq_id_is_rejected(tmp_path) -> None:
    _write_identity_only_segment(
        tmp_path / "bad.h5", freq_id=843.5, physical_channel=14
    )

    with pytest.raises(ValueError, match="'freq_id' must be integral"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_missing_receiver_channel_identity_is_rejected(tmp_path) -> None:
    _write_identity_only_segment(tmp_path / "unknown.h5", physical_channel=14)

    with pytest.raises(ValueError, match="identity is unverifiable"):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)


def test_pilot_and_coarse_frequency_aliases_are_separate_and_consistent(
    tmp_path,
) -> None:
    pilot_hz = physical_channel_to_pilot_hz(14)
    _write_identity_only_segment(
        tmp_path / "aliases.h5",
        freq_id=844,
        physical_channel=14,
        dtv_physical_channel=14,
        pilot_frequency_hz=pilot_hz,
        dtv_pilot_hz=pilot_hz,
        coarse_channel_center_hz=470_312_500.0,
        frequency_hz=470_312_500.0,
        freq=470.3125,
    )

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]

    assert dataset.pilot_frequency_hz == pytest.approx(pilot_hz)
    assert dataset.coarse_channel_center_hz == pytest.approx(470_312_500.0)


@pytest.mark.parametrize(
    ("attrs", "message"),
    (
        (
            {
                "freq_id": 844,
                "pilot_frequency_hz": physical_channel_to_pilot_hz(14),
                "dtv_pilot_hz": physical_channel_to_pilot_hz(14) + 10.0,
            },
            "contradictory pilot-frequency",
        ),
        (
            {
                "freq_id": 844,
                "frequency_hz": 470_312_500.0,
                "freq": 470.5,
            },
            "contradictory coarse-centre",
        ),
        (
            {
                "freq_id": 844,
                "physical_channel": 14,
                "dtv_physical_channel": 15,
            },
            "contradictory physical-channel",
        ),
    ),
)
def test_contradictory_hdf_identity_aliases_fail_closed(
    tmp_path, attrs, message
) -> None:
    _write_identity_only_segment(tmp_path / "contradiction.h5", **attrs)

    with pytest.raises(ValueError, match=message):
        discover_chime_pilot_datasets(tmp_path, dataset_path=None)
