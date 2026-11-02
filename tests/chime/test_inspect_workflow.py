"""Directory inspection selects actual files and respects the requested limit."""

import h5py
import numpy as np
import pytest

from pilot_proxy import cli
from pilot_proxy.chime import inspect


def _write_inputs(root):
    root.mkdir()
    (root / "nested").mkdir()
    paths = [root / "b.h5", root / "nested" / "a.hdf5"]
    for index, path in enumerate(paths):
        with h5py.File(path, "w") as h5:
            h5.attrs["source_number"] = index
            data = h5.create_dataset("baseband", data=np.zeros((index + 2, 3), dtype=np.uint8))
            data.attrs["units"] = "packed_samples"
            h5.create_dataset("unused", data=np.ones(4))
    (root / "directory.h5").mkdir()
    (root / "directory.hdf5").mkdir()
    (root / "notes.txt").write_text("not an HDF5 input")
    return paths


@pytest.mark.parametrize("limit", [0, 1, 4])
def test_inspection_counts_files_and_limits_dataset_reports(tmp_path, capsys, limit):
    root = tmp_path / "inputs"
    paths = _write_inputs(root)
    inspect.main([str(root), "--max-files", str(limit), "--dataset-path", "baseband"])
    output = capsys.readouterr().out
    assert f"Found 2 HDF5 files under {root}" in output
    assert [line for line in output.splitlines() if line.startswith("FILE ")] == [
        f"FILE {path}" for path in paths[:limit]
    ]
    for index in range(min(limit, 2)):
        assert f"/baseband shape=({index + 2}, 3) dtype=uint8" in output
    assert "unused" not in output
    if limit:
        assert "attr units = 'packed_samples'" in output


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_inspection_refuses_inputs_that_are_not_directories(tmp_path, capsys, kind):
    path = tmp_path / "input"
    if kind == "file":
        path.write_bytes(b"input file")
    with pytest.raises(SystemExit) as error:
        inspect.main([str(path)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "existing directory" in captured.err
    assert captured.out == ""


def test_negative_file_limit_is_rejected_before_inspection(tmp_path, capsys):
    root = tmp_path / "inputs"
    _write_inputs(root)
    with pytest.raises(SystemExit) as error:
        inspect.main([str(root), "--max-files", "-1"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "--max-files must be non-negative" in captured.err
    assert captured.out == ""


def test_public_cli_forwards_pattern_dataset_and_limit_to_real_inspection(tmp_path, capfd):
    root = tmp_path / "inputs"
    paths = _write_inputs(root)
    assert cli.main([
        "chime-inspect", "--input-dir", str(root), "--filename-pattern", "*.hdf5",
        "--dataset-path", "baseband", "--max-files", "1",
    ]) == 0
    output = capfd.readouterr().out
    assert "Found 1 HDF5 files" in output
    assert [line for line in output.splitlines() if line.startswith("FILE ")] == [
        f"FILE {paths[1]}"
    ]
    assert "/baseband shape=(3, 3) dtype=uint8" in output
    assert "unused" not in output
