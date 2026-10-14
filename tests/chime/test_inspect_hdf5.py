# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime.hdf5_input import discover_chime_pilot_datasets
from pilot_proxy.chime.inspect import inspect_file


def test_inspect_reports_dataset_shape_and_dtype(tmp_path, capsys) -> None:
    path = tmp_path / "ch0844" / "001.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=np.zeros((4, 2), dtype=np.uint8))
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)

    inspect_file(path)
    output = capsys.readouterr().out

    assert "FILE" in output
    assert "/baseband shape=(4, 2) dtype=uint8" in output
    assert "attr axis" in output

    inspect_file(path, dataset_path="baseband")
    output = capsys.readouterr().out
    assert "/baseband shape=(4, 2) dtype=uint8" in output


def test_hdf5_reader_detects_shape_dtype_and_axes(tmp_path) -> None:
    path = tmp_path / "ch0844" / "001.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=np.zeros((8, 3), dtype=np.uint8))
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)

    datasets = discover_chime_pilot_datasets(tmp_path, dataset_path=None)
    dataset = datasets[14]

    assert dataset.dataset_path == "baseband"
    assert dataset.time_axis == 0
    assert dataset.stream_axis == 1
    assert dataset.num_input_streams == 3
    assert dataset.segments[0].shape == (8, 3)
    assert dataset.segments[0].dtype == "uint8"
