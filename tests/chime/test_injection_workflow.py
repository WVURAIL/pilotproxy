"""Exercise injection commands and recovery on an explicit shared frame population."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy import cli
from pilot_proxy.chime import hdf5_input, injection, injection_recovery
from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)
from pilot_proxy.detector_contract import DETECTOR_CONTRACT_SCHEMA_TOKEN


def _source(path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.arange(32, dtype=np.uint8).reshape(16, 2) * 7
    with h5py.File(path, "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.attrs["collection_server"] = "fixture"
        handle.create_dataset("baseband", data=packed)
        handle.create_dataset("index_map/input", data=np.asarray([7, 9]))
    return packed


def _strict_json(path: Path):
    def reject_constant(value):
        raise AssertionError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=reject_constant)


@pytest.mark.parametrize("entrypoint", ["standalone", "public"])
def test_injection_cli_preserves_control_and_matches_scalar_quantizer(tmp_path, entrypoint):
    source = tmp_path / "input" / "capture_844.h5"
    packed = _source(source)
    runner = injection.main if entrypoint == "standalone" else cli.main
    prefix = [] if entrypoint == "standalone" else ["inject-pilot-tone"]
    for amplitude in (0.0, 2.5):
        output = tmp_path / f"a{amplitude}"
        result = runner(prefix + ["--input", str(source), "--output-dir", str(output),
                                "--amplitude-lsb", str(amplitude),
                                "--baseband-frequency-hz", "-3059", "--phase-seed", "17"])
        assert result in (0, None)
        entry = _strict_json(output / injection.INJECTION_MANIFEST_FILENAME)["files"][0]
        destination = output / source.name
        assert entry["n_time"] == 16 and entry["n_feeds"] == 2
        assert entry["baseband_frequency_hz"] == -3059
        assert entry["phase_seed"] == 17
        if amplitude == 0:
            assert destination.read_bytes() == source.read_bytes()
            assert entry["byte_identical_to_source"] is True
            continue
        phases = 2 * math.pi * np.random.default_rng(17).random(2)
        expected = np.empty_like(packed)
        clipped = 0
        for time in range(16):
            for feed in range(2):
                value = int(packed[time, feed])
                angle = 2 * math.pi * -3059 * time / 390625 + phases[feed]
                real = round((value >> 4) - 8 + amplitude * math.cos(angle))
                imag = round((value & 15) - 8 + amplitude * math.sin(angle))
                clipped += real < -8 or real > 7 or imag < -8 or imag > 7
                expected[time, feed] = ((min(7, max(-8, real)) + 8) << 4) | (min(7, max(-8, imag)) + 8)
        with h5py.File(destination) as handle:
            np.testing.assert_array_equal(handle["baseband"][...], expected)
            np.testing.assert_array_equal(handle["index_map/input"][...], [7, 9])
            assert handle.attrs["collection_server"] == "fixture"
        assert entry["clip_count"] == clipped
        assert entry["clip_fraction"] == clipped / 32


@pytest.mark.parametrize("field,value", [("amplitude_lsb", float("nan")),
                                         ("amplitude_lsb", float("inf")),
                                         ("baseband_frequency_hz", float("nan")),
                                         ("baseband_frequency_hz", float("inf"))])
def test_nonfinite_injection_is_refused_before_copy(tmp_path, field, value):
    source = tmp_path / "capture_844.h5"
    _source(source)
    destination = tmp_path / "output" / source.name
    kwargs = dict(baseband_frequency_hz=0.0, amplitude_lsb=1.0, phase_seed=3)
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite"):
        injection.inject_tone_into_baseband_file(source, destination, **kwargs)
    assert not destination.exists()


def test_large_finite_amplitude_saturates_without_integer_overflow(tmp_path):
    source = tmp_path / "capture_844.h5"
    _source(source)
    destination = tmp_path / "output" / source.name
    entry = injection.inject_tone_into_baseband_file(
        source, destination, baseband_frequency_hz=0.0, amplitude_lsb=1e30, phase_seed=0)
    with h5py.File(destination) as handle:
        # Seed zero gives a negative real/imaginary phase on feed 0 and a
        # negative-real, positive-imaginary phase on feed 1.
        np.testing.assert_array_equal(handle["baseband"][...], np.tile([0x00, 0x0F], (16, 1)))
    assert entry["clip_count"] == 32


def _point(path: Path, amplitude: float):
    path.mkdir()
    # Sixty common finite frames, then one bad value in each statistic and
    # one explicitly invalid frame. None of these last four enters any rate.
    rho = np.r_[2 + 3 * amplitude**2 + np.tile([-1.0, 1.0], 30), np.nan, 1e6, 1e6, 1e6][:, None]
    ratio = np.r_[3 + np.arange(60) + 10 * amplitude**2, 1e6, np.nan, 1e6, 1e6][:, None]
    power = np.r_[100 + np.arange(60) + amplitude**2, 1e6, 1e6, np.nan, 1e6][:, None]
    valid = np.ones((64, 1), dtype=np.uint8)
    valid[-1] = 0
    identity = {
        "frame_index": np.arange(64), "physical_channel": np.asarray([14]),
        "pilot_frequency_hz": np.asarray([470309441.0]),
        "chime_frequency_hz": np.asarray([470312500.0]),
    }
    np.savez(path / CHIME_DETECTOR_OUTPUTS_FILENAME, normalized_pilot_excess=rho,
             coarse_power_ratio=ratio, valid=valid, **identity)
    np.savez(path / CHIME_SPECTROGRAM_CACHE_FILENAME, baseband_power_linear=power,
             valid=valid, **identity)
    (path / "run_config.json").write_text(json.dumps({
        "detector_contract": {"schema_version": DETECTOR_CONTRACT_SCHEMA_TOKEN}}))
    (path / injection.INJECTION_MANIFEST_FILENAME).write_text(json.dumps({
        "schema_version": "pilotproxy_injection_manifest_v1",
        "files": [{"amplitude_lsb": amplitude, "baseband_frequency_hz": -3059.0,
                   "clip_count": int(amplitude)}]}))
    return path


@pytest.mark.parametrize("entrypoint", ["standalone", "public"])
def test_recovery_cli_uses_identical_finite_frames_for_fit_and_both_detectors(tmp_path, entrypoint):
    points = [_point(tmp_path / f"a{i}", a) for i, a in enumerate((0.0, 1.0, 2.0))]
    output = tmp_path / "report"
    argv = ["--output-dir", str(output), "--false-alarm-rate", "0.5"]
    for point in reversed(points):
        argv += ["--point", str(point)]
    if entrypoint == "public":
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["analyze-injection-recovery", *argv])
        assert exit_info.value.code == 0
    else:
        assert injection_recovery.main(argv) == 0
    report = _strict_json(output / injection_recovery.RECOVERY_SUMMARY_FILENAME)
    assert report["false_alarm_rates"] == [0.5]
    assert report["thresholds"] == {"power_ratio": {"0.5": 32.5}, "radiometer": {"0.5": 129.5}}
    assert report["fit"]["floor"] == pytest.approx(2.0)
    assert report["fit"]["gain_per_lsb2"] == pytest.approx(3.0)
    assert report["signal_dominated_log_slope"] is None
    for row, amplitude in zip(report["points"], (0.0, 1.0, 2.0)):
        assert row["amplitude_lsb"] == amplitude
        assert row["n_valid"] == 60
        assert row["rho_hat"] == 2 + 3 * amplitude**2
        assert row["rho_sem"] == pytest.approx(1 / math.sqrt(59))
        for name, count in (("power_ratio", min(60, 30 + 10 * amplitude**2)),
                            ("radiometer", 30 + amplitude**2)):
            rate = count / 60
            key = f"pd_{name}_pfa0.5"
            assert row[key] == rate
            z = 1.959963984540054
            center = (rate + z*z / 120) / (1 + z*z / 60)
            half = z * math.sqrt(rate * (1-rate) / 60 + z*z / (4*60*60)) / (1 + z*z / 60)
            assert row[key + "_wilson95_lo"] == pytest.approx(center - half, abs=1e-15)
            assert row[key + "_wilson95_hi"] == pytest.approx(center + half, abs=1e-15)
    with (output / injection_recovery.RECOVERY_CSV_FILENAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["n_valid"]) for row in rows] == [60, 60, 60]
    assert [float(row["rho_hat"]) for row in rows] == [2, 5, 14]
    for filename in (injection_recovery.RECOVERY_FIGURE, injection_recovery.BASELINE_FIGURE):
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("pfa", [0.0, -0.1, 1.0, 1.1, float("nan"), float("inf")])
def test_recovery_refuses_invalid_false_alarm_rates_before_publication(tmp_path, pfa):
    point = _point(tmp_path / "control", 0.0)
    output = tmp_path / "report"
    with pytest.raises(SystemExit, match="false.alarm.*finite.*between"):
        injection_recovery.main(["--point", str(point), "--output-dir", str(output),
                                 "--false-alarm-rate", str(pfa)])
    assert not output.exists()


def test_nested_hdf5_discovery_skips_incompatible_shapes_and_reads_samples(tmp_path):
    path = tmp_path / "capture.h5"
    samples = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    with h5py.File(path, "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.create_dataset("metadata/scalar", data=42)
        handle.create_dataset("metadata/vector", data=np.arange(100))
        handle.create_dataset("metadata/cube", shape=(8, 8, 8), dtype="f4")
        handle.create_dataset("capture/voltage", data=samples)
    (tmp_path / "directory.h5").mkdir()
    dataset = hdf5_input.discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]
    assert dataset.dataset_path == "capture/voltage"
    assert dataset.num_input_streams == 3 and dataset.total_time_samples == 4
    values = hdf5_input.read_complex_window(dataset, start_sample=1, stop_sample=3)
    np.testing.assert_array_equal(values[:, 0], (samples[1:3, :, 0] + 1j * samples[1:3, :, 1]).T)


def test_nested_hdf5_equal_candidates_require_explicit_path(tmp_path):
    path = tmp_path / "capture.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.create_dataset("a/voltage", data=np.zeros((4, 2), dtype="u1"))
        handle.create_dataset("b/voltage", data=np.full((4, 2), 17, dtype="u1"))
    with pytest.raises(ValueError, match="ambiguous.*dataset"):
        hdf5_input.discover_chime_pilot_datasets(tmp_path, dataset_path=None)
    dataset = hdf5_input.discover_chime_pilot_datasets(tmp_path, dataset_path="a/voltage")[14]
    np.testing.assert_array_equal(hdf5_input.read_complex_window(dataset, start_sample=0, stop_sample=4), np.zeros((2, 1, 4), dtype="u1"))


@pytest.mark.parametrize("field,value", [("amplitude_lsb", -1.0),
                                         ("amplitude_lsb", float("nan")),
                                         ("amplitude_lsb", float("inf")),
                                         ("baseband_frequency_hz", float("nan")),
                                         ("baseband_frequency_hz", float("inf"))])
def test_recovery_refuses_invalid_injection_truth(tmp_path, field, value):
    point = _point(tmp_path / "point", 1.0)
    manifest_path = point / injection.INJECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0][field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="finite.*amplitude.*frequency"):
        injection_recovery.analyze_injection_recovery([point], false_alarm_rates=[0.5])


def test_recovery_refuses_different_tone_frequencies_between_points(tmp_path):
    control = _point(tmp_path / "control", 0.0)
    point = _point(tmp_path / "point", 1.0)
    manifest_path = point / injection.INJECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["baseband_frequency_hz"] = -4000.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="same.*frequency"):
        injection_recovery.analyze_injection_recovery([control, point], false_alarm_rates=[0.5])


def test_recovery_refuses_control_only_before_singular_fit(tmp_path):
    point = _point(tmp_path / "control", 0.0)
    with pytest.raises(SystemExit, match="nonzero.*amplitude"):
        injection_recovery.analyze_injection_recovery([point], false_alarm_rates=[0.5])


def test_recovery_refuses_mismatched_statistic_shapes(tmp_path):
    point = _point(tmp_path / "point", 1.0)
    np.savez(point / CHIME_SPECTROGRAM_CACHE_FILENAME, baseband_power_linear=np.zeros(64))
    with pytest.raises(SystemExit, match="inconsistent statistic shapes"):
        injection_recovery.analyze_injection_recovery([point], false_alarm_rates=[0.5])


def test_injection_cli_nonfinite_frequency_does_not_publish(tmp_path):
    source = tmp_path / "capture_844.h5"
    _source(source)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="frequency.*finite"):
        injection.main(["--input", str(source), "--output-dir", str(output),
                        "--amplitude-lsb", "0", "--baseband-frequency-hz", "nan"])
    assert not (output / source.name).exists()
    assert not (output / injection.INJECTION_MANIFEST_FILENAME).exists()
