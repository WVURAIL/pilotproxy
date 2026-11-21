import h5py
import numpy as np
import pytest

from pilot_proxy.bundle_calibration import _expand, run_rows
from pilot_proxy.run_files import combine, iter_windows, read_attrs, split, windows
from pilot_proxy.run_plans import ordered_run_files


def write_window(path, coarse, seq, time_ns, fill):
    data = np.full((4, len(coarse), 3, 256), fill, dtype=np.uint64)
    with h5py.File(path, "w") as handle:
        handle["dtv_fine_powers"] = data
        if any(c >= 0 for c in coarse):           # Kotekan drops an all -1 label
            handle["dtv_fine_powers"].attrs["coarse_freq"] = np.array(coarse, np.int32)
        handle["dtv_fine_powers"].attrs["fpga_seq_num"] = seq
        if time_ns is not None:
            handle["dtv_fine_powers"].attrs["fpga_seq_time_nsec"] = time_ns
        handle["dtv_fine_powers"].attrs["seq_length_nsec"] = 5120


def test_combined_windows_read_as_their_sources(tmp_path):
    acq = tmp_path / "acq_1"
    acq.mkdir()
    # the F-engine restarts after the second window, so fpga_seq_num falls while the time rises
    layout = [(900, 10), (1000, 20), (100, 30), (200, 40)]
    for index, (seq, seconds) in enumerate(layout):
        write_window(acq / f"dtv_fine_powers.{2 * index:08d}.h5", [2715, 3115], seq, seconds * 10**9, index + 1)
        write_window(acq / f"dtv_fine_powers.{2 * index + 1:08d}.h5", [-1, -1], seq, seconds * 10**9, 0)
    summary = combine(acq, tmp_path / "out", "fine_powers")
    files = [r for r in summary if "file" in r]
    assert len(files) == 1 and files[0]["windows"] == 4 and files[0]["verified"]
    assert summary[-1] == {"product": "fine_powers", "skipped_without_coarse_freq": 4}
    path = files[0]["file"]
    assert path.endswith("dtv_fine_powers.2715-3115.combined.h5")
    specs = windows(path)
    assert specs == [f"{path}#{i}" for i in range(4)] and split(specs[2]) == (path, 2)
    assert _expand([str(tmp_path / "out" / "*.h5")]) == specs
    # recording order, and each window's own counter and time
    assert [read_attrs(s)["fpga_seq_num"] for s in specs] == [s for s, _ in layout]
    times = [t for t, _, _ in ordered_run_files(specs)]
    assert times == [s * 10**9 for _, s in layout]
    sources = sorted(acq.glob("dtv_fine_powers.*.h5"))[0::2]
    for spec, source in zip(specs, sources):
        assert np.array_equal(run_rows(spec, 3115), run_rows(source, 3115))
    assert [int(a["fpga_seq_num"]) for a, _ in iter_windows(path)] == [s for s, _ in layout]
    assert read_attrs(specs[0])["seq_length_nsec"] == 5120


def test_combine_refuses_a_window_of_another_shape(tmp_path):
    acq = tmp_path / "acq_1"
    acq.mkdir()
    write_window(acq / "dtv_fine_powers.00000000.h5", [2715, 3115], 0, 10, 1)
    with h5py.File(acq / "dtv_fine_powers.00000001.h5", "w") as handle:
        handle["dtv_fine_powers"] = np.zeros((2, 2, 3, 256), np.uint64)
        handle["dtv_fine_powers"].attrs.update(coarse_freq=np.array([2715, 3115], np.int32), fpga_seq_num=1,
                                               fpga_seq_time_nsec=20)
    with pytest.raises(ValueError, match="shape"):
        combine(acq, tmp_path / "out", "fine_powers")
