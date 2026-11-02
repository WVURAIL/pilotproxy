"""Public CHIME commands preserve selected inputs and publish checked outputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy import cli
from pilot_proxy.chime import plots, runner
from pilot_proxy.detector_reference import frame_powers_cpu_reference_packed
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration import load_detector_core_profile, load_receiver_profile
from pilot_proxy.integration.weight_generation import write_weight_bank_from_receiver_profile
from pilot_proxy.paths import DEFAULT_LIB_PATH
from test_product_workflows import _write_run


ROOT = Path(__file__).resolve().parents[2]
FRAME = 16384


def _runner_inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    n = np.arange(3 * FRAME)[:, None]
    feed = np.arange(2)[None, :]
    # The pilot is 3059 Hz below channel center; Nyquist zone 2 records its
    # mirrored positive-frequency tone before per-window time reversal.
    phase = 2 * np.pi * 3059 / 390625 * n + 0.3 * feed
    real = np.rint(4 * np.cos(phase)).astype(np.int8)
    imag = np.rint(4 * np.sin(phase)).astype(np.int8)
    raw = (((real.astype(np.int16) + 8) << 4) | (imag + 8)).astype(np.uint8)
    with h5py.File(source / "capture.hdf5", "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.attrs["freq_id"] = 844
        data = handle.create_dataset("raw/samples", data=raw)
        data.attrs["axis"] = np.asarray(["time", "input"], dtype=object)
    (source / "ignored.h5").write_bytes(b"excluded by --filename-pattern")
    profile = json.loads((ROOT / "configs/receiver_profiles/chime_dtv_fengine.json").read_text())
    profile["input_streams"]["num_input_streams"] = 2
    profile_path = tmp_path / "receiver.json"
    profile_path.write_text(json.dumps(profile))
    stream_path = tmp_path / "streams.json"
    stream_path.write_text(json.dumps({
        "schema_version": "pilotproxy_stream_map_v1",
        "receiver_profile_id": profile["receiver_profile_id"],
        "stream_unit": "feed_polarization", "num_streams": 2,
        "streams": [{"stream_index": 0}, {"stream_index": 1}],
    }))
    args = ["--input-dir", str(source), "--output-dir", str(tmp_path / "run"),
            "--receiver-profile", str(profile_path), "--stream-map", str(stream_path),
            "--filename-pattern", "*.hdf5", "--dataset-path", "raw/samples"]
    return args, real, imag, profile_path


@pytest.mark.cuda
def test_standalone_runner_cli_matches_packed_cpu_reference(tmp_path, capsys):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    if not DEFAULT_LIB_PATH.is_file():
        pytest.skip("compiled detector library is unavailable")
    args, real, imag, profile_path = _runner_inputs(tmp_path)
    weights_path = tmp_path / "weights.bin"
    write_weight_bank_from_receiver_profile(
        output_path=weights_path, profile=load_receiver_profile(profile_path),
        core=load_detector_core_profile(ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"),
        physical_channels=[14],
    )
    inputs = [profile_path, tmp_path / "streams.json", tmp_path / "input/capture.hdf5", weights_path]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    assert runner.main(args + [
        "--weights-path", str(weights_path), "--lib-path", str(DEFAULT_LIB_PATH),
        "--physical-channel-range", "14:14", "--frame-size-samples", str(FRAME),
        "--frames-per-chunk", "1", "--max-frames", "2",
        "--pilot-below-data-db", "13", "--bin-enbw-hz", "1000",
        "--dtv-bandwidth-hz", "5000000", "--pilot-capture-efficiency", "0.75",
        "--calibration-seconds", "0.01",
    ]) == 0
    output = tmp_path / "run"
    bank = DetectorWeightBank(explicit_path=weights_path)
    weights, valid = bank.get_weights_for_physical_channel(14)
    assert valid
    expected = []
    for frame in range(2):
        selection = slice(frame * FRAME, (frame + 1) * FRAME)
        # Native offset-binary samples become signed nibbles. Nyquist zone 2
        # reverses samples within each 128-sample window, preserving row order.
        packed = (((real[selection].astype(np.int16) & 15) << 4)
                  | (imag[selection].astype(np.int16) & 15)).astype(np.uint8).view(np.int8)
        rows = packed.T.reshape(2, 128, 128)[:, :, ::-1].reshape(256, 128)
        expected.append(frame_powers_cpu_reference_packed(rows, weights, 4))
    expected = np.asarray(expected, dtype=np.uint64)
    with np.load(output / "chime_detector_outputs.npz") as product:
        assert product["physical_channel"].tolist() == [14]
        assert product["frame_index"].tolist() == [0, 1]
        for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
            np.testing.assert_array_equal(product[name][:, 0], expected[:, term])
        reference = expected[:, 1] + expected[:, 2]
        norms = (int(product["target_norm_sq"][0]), int(product["reference_norm_sum_sq"][0]))
        wanted_mask = [int(t) * norms[1] > int(r) * norms[0]
                       for t, r in zip(expected[:, 0], reference, strict=True)]
        assert product["mask"][:, 0].tolist() == wanted_mask
        np.testing.assert_array_equal(product["fine_power_u64"].sum(axis=-1)[:, 0], expected)
        excess = expected[:, 0].astype(float) * norms[1] / (reference.astype(float) * norms[0]) - 1
        expected_shelf = np.full(2, np.nan)
        positive = excess > 0
        expected_shelf[positive] = (10 * np.log10(excess[positive]) + 13
                                    - 10 * np.log10(5000000 / 1000)
                                    - 10 * np.log10(0.75))
        assert np.any(positive)
        np.testing.assert_allclose(product["estimated_data_shelf_snr_db"][:, 0], expected_shelf,
                                   rtol=0, atol=1e-12, equal_nan=True)
    with np.load(output / "chime_spectrogram_cache.npz") as cache:
        power = (real.astype(float)**2 + imag.astype(float)**2).reshape(3, FRAME, 2).mean(axis=(1, 2))
        np.testing.assert_allclose(cache["baseband_power_linear"][:, 0], power[:2], rtol=0, atol=0)
    config = json.loads((output / "run_config.json").read_text())
    assert config["frames_per_chunk"] == 1 and config["max_frames"] == 2
    assert config["physical_channels"] == [14]
    assert config["pilot_below_data_db"] == 13
    assert config["bin_enbw_hz"] == 1000
    assert config["dtv_bandwidth_hz"] == 5000000
    assert config["pilot_capture_efficiency"] == 0.75
    assert config["calibration_seconds"] == 0.01
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    assert "detector_outputs:" in capsys.readouterr().out


@pytest.mark.parametrize("selection", [["--physical-channel", "13"], ["--physical-channel", "15"]])
def test_runner_cli_refuses_unavailable_selection_before_native_library(tmp_path, selection):
    args, _, _, _ = _runner_inputs(tmp_path)
    with pytest.raises(ValueError, match="physical channel|not present|not found|missing"):
        runner.main(args + selection + ["--lib-path", str(tmp_path / "missing.so")])
    assert not (tmp_path / "run/chime_detector_outputs.npz").exists()


def test_public_plot_cli_renders_current_artifacts_and_cleans_owned_figures(tmp_path, monkeypatch, capfd):
    pytest.importorskip("matplotlib")
    run = tmp_path / "run with spaces"
    _write_run(run)
    figures = run / "figures"
    figures.mkdir()
    stale = figures / "mask_spectrogram.svg"
    stale.write_text("old render")
    unrelated = figures / "notes.txt"
    unrelated.write_text("preserve")
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv(plots.FIGURE_FORMATS_ENV, "png")
    assert cli.main(["chime-plot", "--run-dir", str(run), "--clean-figures"]) == 0
    assert not stale.exists() and unrelated.read_text() == "preserve"
    assert {path.name for path in figures.glob("*.png")} == plots.KNOWN_CHIME_FIGURES
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures.glob("*.png"))
    with (run / "tables/data_shelf_snr_histogram_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["physical_channel"]) for row in rows] == [30, 14]
    assert [int(row["num_finite_data_shelf_snr_estimates"]) for row in rows] == [2, 1]
    assert [float(row["mean_estimated_data_shelf_snr_db"]) for row in rows] == [-20, -20]
    assert capfd.readouterr().out.count("Wrote ") == 6
