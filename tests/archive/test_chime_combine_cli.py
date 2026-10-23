# coding=utf-8
"""Tests for the standalone `chime-combine` CLI (mid-survey combining)."""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.archive.instruments import load_instrument  # noqa: E402
from pilot_proxy.archive.interfaces import RunContext  # noqa: E402
from pilot_proxy.chime import baseband_format as fmt  # noqa: E402

from pilot_proxy.cli import main as cli_main  # noqa: E402
from pilot_proxy.chime.products import (  # noqa: E402
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_INTEGRATED_SPECTRA_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer  # noqa: E402
from pilot_proxy.archive.packed_reader import (  # noqa: E402
    ChimeBasebandPackedReader,
)
from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz  # noqa: E402

from cpu_detector import exact_cpu_detector_fn  # noqa: E402

K = 128
NFFT = 16384
N_FRAMES = 3
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.3125}
FREQ_IDS = {ch: chime_freq_id_from_hz(mhz * 1e6) for ch, mhz in CHANNELS.items()}


def _stub_kernel(k):
    class _Kernel:
        detector_window_samples = k
    return _Kernel()


def _build_work_dir(tmp_path):
    rng = np.random.default_rng(7)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8) for ch in CHANNELS
    }
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    inst = load_instrument("chime")
    reader = ChimeBasebandPackedReader()
    work = tmp_path / "work"
    work.mkdir()
    for ch, mhz in CHANNELS.items():
        path = input_dir / f"baseband_evt_{FREQ_IDS[ch]}.h5"
        fmt.make_synth_file(str(path), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                            f_center_mhz=mhz, f_tone_bb=1200.0 + 10 * ch, seed=ch)
        with h5py.File(path, "a") as handle:
            handle.attrs["delta_time"] = 1.0 / fmt.FS
            handle.attrs["time0_ctime"] = 1.0e9
        ctx = RunContext(instrument=inst, selection=[FREQ_IDS[ch]], options={
            "detector_fn": exact_cpu_detector_fn,
            "kernel": _stub_kernel(K),
            "weights": weights_by_channel[ch],
        })
        meta = dict(reader.probe(str(path)))
        meta["unit_key"] = str(path)
        meta["scope"] = "local"
        analyzer = PilotProxyDetectorAnalyzer()
        analyzer.begin(ctx, meta)
        analyzer.consume_file(reader.iter_arrays(str(path), ctx), meta)
        analyzer.save(str(work / f"{FREQ_IDS[ch]}.npz"))
    return work


def test_chime_combine_work_dir_mode(tmp_path, capsys) -> None:
    work = _build_work_dir(tmp_path)
    out = tmp_path / "combined"

    rc = cli_main([
        "chime-combine", "--work-dir", str(work), "--output-dir", str(out)
    ])

    assert rc in (0, None)
    for name in (CHIME_DETECTOR_OUTPUTS_FILENAME,
                 CHIME_SPECTROGRAM_CACHE_FILENAME,
                 CHIME_INTEGRATED_SPECTRA_FILENAME):
        assert (out / name).exists(), name
    detector = np.load(out / CHIME_DETECTOR_OUTPUTS_FILENAME)
    assert detector["physical_channel"].tolist() == [14, 15]
    assert detector["valid"].shape == (N_FRAMES, 2)
    assert "Combined 2 pilot product(s)" in capsys.readouterr().out


def test_chime_combine_explicit_products_match_work_dir(tmp_path) -> None:
    work = _build_work_dir(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cli_main(["chime-combine", "--work-dir", str(work),
              "--output-dir", str(out_a)])
    ids = sorted(FREQ_IDS.values())
    cli_main(["chime-combine",
              "--product", str(work / f"{ids[0]}.npz"),
              "--product", str(work / f"{ids[1]}.npz"),
              "--output-dir", str(out_b)])

    a = np.load(out_a / CHIME_DETECTOR_OUTPUTS_FILENAME)
    b = np.load(out_b / CHIME_DETECTOR_OUTPUTS_FILENAME)
    assert set(a.files) == set(b.files)
    for key in a.files:
        assert np.array_equal(a[key], b[key]), key


def test_chime_combine_single_product_is_valid(tmp_path) -> None:
    work = _build_work_dir(tmp_path)
    out = tmp_path / "single"
    one = sorted(FREQ_IDS.values())[0]
    cli_main(["chime-combine", "--product", str(work / f"{one}.npz"),
              "--output-dir", str(out)])
    detector = np.load(out / CHIME_DETECTOR_OUTPUTS_FILENAME)
    assert detector["physical_channel"].shape == (1,)


def test_chime_combine_errors(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no per-pilot products matched"):
        cli_main(["chime-combine", "--work-dir", str(empty),
                  "--output-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="missing product file"):
        cli_main(["chime-combine", "--product", str(tmp_path / "nope.npz"),
                  "--output-dir", str(tmp_path / "y")])
