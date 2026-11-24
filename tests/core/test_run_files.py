import h5py
import numpy as np
import pytest

from pilot_proxy.bundle_calibration import _expand, run_rows
from pilot_proxy.run_files import combine, iter_windows, read_attrs, split, windows
from pilot_proxy.run_plans import ordered_run_files

from kotekan_files import write_frames_file


def write_window(path, coarse, seq, time_ns, fill):
    data = np.full((4, len(coarse), 3, 256), fill, dtype=np.uint64)
    with h5py.File(path, "w") as handle:
        handle["dtv_fine_powers"] = data
        if any(c >= 0 for c in coarse):           # Kotekan before #1930 drops an all -1 label
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


def test_combine_groups_halves_by_their_bound_channels(tmp_path):
    # kotekan#1930 labels an unused row -2 where earlier builds wrote -1, and keeps an all-unused label
    acq = tmp_path / "acq_1"
    acq.mkdir()
    labels = [[2531, -1], [2531, -2], [-2, -2]]
    for index, coarse in enumerate(labels):
        write_window(acq / f"dtv_fine_powers.{index:08d}.h5", coarse, index, index * 10**9, index + 1)
        with h5py.File(acq / f"dtv_fine_powers.{index:08d}.h5", "a") as handle:
            handle["dtv_fine_powers"].attrs["coarse_freq"] = np.array(coarse, np.int32)
    summary = combine(acq, tmp_path / "out", "fine_powers")
    files = [r for r in summary if "file" in r]
    assert len(files) == 1 and files[0]["windows"] == 2
    assert files[0]["file"].endswith("dtv_fine_powers.2531.combined.h5")
    assert summary[-1] == {"product": "fine_powers", "skipped_without_coarse_freq": 1}


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


def write_half(path, coarse, seq, fill, t0=10**18):
    """One frame of 4 rows of one half, with the attributes recv writes; time is linear in fpga_seq_num."""
    write_window(path, coarse, seq, t0 + seq * 5120, fill)
    with h5py.File(path, "a") as handle:
        handle["dtv_fine_powers"].attrs.update(dim_scalings=np.array([8192, 1, 1, 1]), time_downsampling_fpga=8192)
    return path


def test_several_frame_file_reads_as_its_frames(tmp_path):
    """A kotekan#1925 file: frames of two halves in arrival order, one arriving after a later window's."""
    window = 4 * 8192
    seq = [10**9 + w * window for w in range(3)]
    sources = {}
    for w in range(3):
        sources["a", w] = write_half(tmp_path / f"a{w}.h5", [2715, 3115], seq[w], w + 1)
        sources["b", w] = write_half(tmp_path / f"b{w}.h5", [2869, -1], seq[w], 10 + w)
    order = [("a", 1), ("a", 0), ("b", 1), ("b", 0), ("a", 2), ("b", 2)]
    acq = tmp_path / "acq_1"
    acq.mkdir()
    path = str(write_frames_file([sources[k] for k in order], acq / "dtv_fine_powers.00000000.h5"))
    specs = windows(path)
    assert specs == [f"{path}#{i}" for i in range(6)]
    for spec, key in zip(specs, order):
        source = str(sources[key])
        got, want = read_attrs(spec), read_attrs(source)
        for name in ("fpga_seq_num", "fpga_seq_time_nsec", "time_downsampling_fpga", "seq_length_nsec"):
            assert got[name] == want[name], name
        assert list(got["coarse_freq"]) == list(want["coarse_freq"])
        assert np.array_equal(run_rows(spec, 3115), run_rows(source, 3115))
        assert np.array_equal(run_rows(spec, 2869), run_rows(source, 2869))
    assert [int(a["fpga_seq_num"]) for a, _ in iter_windows(path)] == [seq[w] for _, w in order]
    # recording order is by time, whatever the arrival order
    times = [t for t, _, _ in ordered_run_files([path])]
    assert times == sorted(read_attrs(sources[k])["fpga_seq_time_nsec"] for k in order)
    assert _expand([str(acq / "*.h5")]) == specs
    with pytest.raises(ValueError, match="several frames"):
        read_attrs(path)
    assert combine(acq, tmp_path / "out", "fine_powers") == [
        {"product": "fine_powers", "skipped_several_frame_files": 1}]
    assert not list((tmp_path / "out").glob("*.h5"))


def test_frame_after_an_f_engine_restart_has_no_time(tmp_path):
    first = write_half(tmp_path / "a0.h5", [2715, 3115], 10**9, 1)
    restarted = write_half(tmp_path / "a1.h5", [2715, 3115], 4096, 2, t0=2 * 10**18)
    path = str(write_frames_file([first, restarted], tmp_path / "dtv_fine_powers.00000000.h5"))
    assert "fpga_seq_time_nsec" not in read_attrs(f"{path}#1")
    assert read_attrs(f"{path}#1")["fpga_seq_num"] == 4096
    # a time recorded with each frame is taken as it is
    path = str(write_frames_file([first, restarted], tmp_path / "dtv_fine_powers.00000002.h5", per_frame_time=True))
    assert read_attrs(f"{path}#1")["fpga_seq_time_nsec"] == read_attrs(restarted)["fpga_seq_time_nsec"]


def test_frames_of_an_idle_half_have_no_channels(tmp_path):
    idle = [write_half(tmp_path / f"i{w}.h5", [-1, -1], 10**9 + w * 4 * 8192, 0) for w in range(2)]
    path = str(write_frames_file(idle, tmp_path / "dtv_fine_powers.00000000.h5"))
    assert all("coarse_freq" not in read_attrs(spec) for spec in windows(path))
    assert run_rows(f"{path}#0", 3115).shape == (0, 3, 256)


def test_file_left_open_by_kotekan_is_read(tmp_path):
    """Kotekan writes in SWMR mode; a file it had open when killed opens only in SWMR mode."""
    frames = [write_half(tmp_path / f"a{w}.h5", [2715, 3115], 10**9 + w * 4 * 8192, w + 1) for w in range(2)]
    path = str(write_frames_file(frames, tmp_path / "dtv_fine_powers.00000000.h5", left_open=True))
    with pytest.raises(OSError):
        h5py.File(path, "r")
    assert windows(path) == [f"{path}#0", f"{path}#1"]
    assert np.array_equal(run_rows(f"{path}#1", 3115), run_rows(frames[1], 3115))


def test_frame_whose_rows_are_not_written_yet_is_left_out(tmp_path):
    frames = [write_half(tmp_path / f"a{w}.h5", [2715, 3115], 10**9 + w * 4 * 8192, w + 1) for w in range(2)]
    path = str(write_frames_file(frames, tmp_path / "dtv_fine_powers.00000000.h5"))
    with h5py.File(path, "a") as handle:
        for name in ("dim", "FPGA_SEQ_NUM", "COARSE_FREQ"):
            values = handle["frame_metadata"][name][()]
            del handle["frame_metadata"][name]
            handle["frame_metadata"][name] = np.concatenate([values, values[-1:]])
    assert windows(path) == [f"{path}#0", f"{path}#1"]


def test_combined_v3_stat_reads_as_its_windows(tmp_path):
    """dtv_stat, the [n_peak, d_rho] pair of fine decision v3, combines like the other products."""
    acq = tmp_path / "acq_1"
    acq.mkdir()
    rng = np.random.default_rng(3)
    sources = []
    for w in range(3):
        path = acq / f"dtv_stat.{w:08d}.h5"
        with h5py.File(path, "w") as handle:
            handle["dtv_stat"] = rng.integers(0, 2**40, size=(4, 2, 2), dtype=np.uint64)
            handle["dtv_stat"].attrs.update(coarse_freq=np.array([2715, 3115], np.int32), fpga_seq_num=100 + w,
                                            fpga_seq_time_nsec=10**9 * (w + 1))
        sources.append(path)
    summary = combine(acq, tmp_path / "out", "stat")
    assert [r["windows"] for r in summary] == [3] and summary[0]["verified"]
    combined = list(iter_windows(summary[0]["file"]))
    for (attrs, block), source in zip(combined, sources):
        with h5py.File(source, "r") as handle:
            assert np.array_equal(block, handle["dtv_stat"][()])
        assert attrs["fpga_seq_num"] == read_attrs(source)["fpga_seq_num"]


def test_combine_tool_combines_again_when_windows_are_added(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    tool = Path(__file__).resolve().parents[2] / "tools" / "combine_pathfinder_dtv_run.py"
    acq, out = tmp_path / "acq_1", tmp_path / "out"
    acq.mkdir()

    def run():
        return subprocess.run([sys.executable, str(tool), "--acq", str(acq), "--out", str(out), "--products",
                               "fine_powers", "--settle-minutes", "0"], capture_output=True, text=True)

    def combined():
        return windows(str(out / "acq_1" / "dtv_fine_powers.2715-3115.combined.h5"))

    for i in range(2):
        write_window(acq / f"dtv_fine_powers.{i:08d}.h5", [2715, 3115], 100 * i, 10**9 * (i + 1), i + 1)
    assert run().returncode == 0 and len(combined()) == 2
    assert "already combined" in run().stdout
    # recv adds a window to the same acquisition after a quiet gap
    write_window(acq / "dtv_fine_powers.00000002.h5", [2715, 3115], 200, 3 * 10**9, 3)
    done = run()
    assert done.returncode == 0 and "1 files written since" in done.stdout and len(combined()) == 3
    record = json.loads((out / "acq_1" / "combine.json").read_text())
    assert record["sources"] == [f"dtv_fine_powers.{i:08d}.h5" for i in range(3)]
    # a combined file deleted, then another added: combining again would drop the deleted window
    (acq / "dtv_fine_powers.00000000.h5").unlink()
    write_window(acq / "dtv_fine_powers.00000003.h5", [2715, 3115], 300, 4 * 10**9, 4)
    done = run()
    assert done.returncode == 1 and "would drop their windows" in done.stderr and len(combined()) == 3
    # a record from the version before the file list is left alone
    del record["sources"]
    (out / "acq_1" / "combine.json").write_text(json.dumps(record))
    done = run()
    assert done.returncode == 0 and "kept no file list" in done.stdout and len(combined()) == 3


def test_untimed_windows_after_a_restart_keep_recording_order(tmp_path):
    """The windows after an F-engine restart have no time and an fpga_seq_num near zero, and stay last."""
    window = 4 * 8192
    sources = [write_half(tmp_path / f"a{w}.h5", [2715, 3115], 10**9 + w * window, w + 1) for w in range(2)]
    sources += [write_half(tmp_path / f"a{w}.h5", [2715, 3115], 4096 + (w - 2) * window, w + 1, t0=2 * 10**18)
                for w in (2, 3)]
    acq = tmp_path / "acq_1"
    acq.mkdir()
    path = str(write_frames_file(sources, acq / "dtv_fine_powers.00000000.h5"))
    files = ordered_run_files([path])
    assert [p for _, _, p in files] == [f"{path}#{i}" for i in range(4)]
    assert [t is None for t, _, _ in files] == [False, False, True, True]
