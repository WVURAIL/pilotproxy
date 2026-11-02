"""Synthetic native files exercise injection, CUDA detection, and recovery.

The fixture tests the software chain at a supported empirical false-alarm
rate. It does not represent an ATSC waveform or qualify telescope sensitivity.
"""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import injection, injection_recovery, runner
from pilot_proxy.detector_contract import DETECTOR_CONTRACT_SCHEMA_TOKEN
from pilot_proxy.detector_reference import frame_powers_cpu_reference_packed
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration import load_detector_core_profile, load_receiver_profile
from pilot_proxy.integration.weight_generation import write_weight_bank_from_receiver_profile
from pilot_proxy.paths import DEFAULT_LIB_PATH


ROOT = Path(__file__).resolve().parents[2]
FRAME = 16384
FRAMES = 40
FEEDS = 2
SAMPLE_RATE = 390625
PHASE_SEED = 42
PFA = 0.25


@pytest.mark.parametrize("specification", [
    {"physical_channel": 14}, {"pilot_frequency_hz": 470309441},
])
def test_rf_injection_uses_native_mirrored_frequency(specification):
    # Zone 2 obeys f_RF = f_center - f_native. This expectation does not use
    # the resolver or the receiver's frequency-conversion helpers.
    assert injection.resolve_baseband_frequency_hz(
        470312500, sample_rate_hz=SAMPLE_RATE, **specification,
    ) == 3059
    assert injection.resolve_baseband_frequency_hz(
        470312500, sample_rate_hz=SAMPLE_RATE, baseband_frequency_hz=-3059,
    ) == -3059


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_products(path, amplitude):
    path.mkdir()
    identity = {
        "frame_index": np.arange(FRAMES), "physical_channel": np.asarray([14]),
        "pilot_frequency_hz": np.asarray([470309441.0]),
        "chime_frequency_hz": np.asarray([470312500.0]),
        "valid": np.ones((FRAMES, 1), dtype=np.uint8),
    }
    rho = (np.arange(FRAMES, dtype=float) / FRAMES + amplitude ** 2)[:, None]
    np.savez(path / "chime_detector_outputs.npz", normalized_pilot_excess=rho,
             coarse_power_ratio=rho + 1, **identity)
    np.savez(path / "chime_spectrogram_cache.npz", baseband_power_linear=rho + 10,
             **identity)
    (path / "run_config.json").write_text(json.dumps({
        "detector_contract": {"schema_version": DETECTOR_CONTRACT_SCHEMA_TOKEN}}))
    (path / injection.INJECTION_MANIFEST_FILENAME).write_text(json.dumps({
        "schema_version": "pilotproxy_injection_manifest_v1",
        "files": [{"amplitude_lsb": amplitude, "baseband_frequency_hz": 3059,
                   "clip_count": 0}]}))
    return path


@pytest.mark.parametrize("product", ["chime_detector_outputs.npz", "chime_spectrogram_cache.npz"])
@pytest.mark.parametrize("key", ["frame_index", "physical_channel", "pilot_frequency_hz", "chime_frequency_hz", "valid"])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_recovery_refuses_unpaired_product_population(tmp_path, product, key, mutation):
    control = _recovery_products(tmp_path / "control", 0)
    point = _recovery_products(tmp_path / "point", 1)
    path = point / product
    with np.load(path) as stored:
        arrays = dict(stored)
    if mutation == "missing":
        del arrays[key]
    else:
        arrays[key] = arrays[key].copy()
        arrays[key].flat[0] = 0 if key == "valid" else arrays[key].flat[0] + 1
    np.savez(path, **arrays)
    with pytest.raises(SystemExit, match="identit|valid|missing|lacks"):
        injection_recovery.analyze_injection_recovery([control, point], false_alarm_rates=[PFA])


@pytest.mark.parametrize("key", ["frame_index", "physical_channel", "pilot_frequency_hz", "chime_frequency_hz"])
def test_recovery_refuses_identity_shape_even_when_products_agree(tmp_path, key):
    control = _recovery_products(tmp_path / "control", 0)
    for name in ("chime_detector_outputs.npz", "chime_spectrogram_cache.npz"):
        path = control / name
        with np.load(path) as stored:
            arrays = dict(stored)
        arrays[key] = arrays[key][None, :]
        np.savez(path, **arrays)
    with pytest.raises(SystemExit, match="identit.*shape"):
        injection_recovery.analyze_injection_recovery([control], false_alarm_rates=[PFA])


def _native_inputs(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    source = original / "capture_844.h5"
    rng = np.random.default_rng(20261101)
    real = rng.integers(-2, 3, size=(FRAMES * FRAME, FEEDS), dtype=np.int16)
    imag = rng.integers(-2, 3, size=real.shape, dtype=np.int16)
    real[0, 0] = imag[0, 0] = -8
    raw = (((real + 8) << 4) | (imag + 8)).astype(np.uint8)
    with h5py.File(source, "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.attrs["freq_id"] = 844
        handle.attrs["fixture"] = "synthetic native-format noise"
        handle.create_dataset("baseband", data=raw)
        handle.create_dataset("index_map/input", data=np.arange(FEEDS))
        handle.create_dataset("sibling", data=np.arange(7))
    profile = json.loads((ROOT / "configs/receiver_profiles/chime_dtv_fengine.json").read_text())
    assert profile["channelizer"]["frequency_axis"]["nyquist_zone"] == 2
    assert profile["detector_adapter"]["time_reverse_detector_windows_before_kernel"] is True
    profile["input_streams"]["num_input_streams"] = FEEDS
    profile_path = tmp_path / "receiver.json"
    profile_path.write_text(json.dumps(profile))
    stream_path = tmp_path / "streams.json"
    stream_path.write_text(json.dumps({
        "schema_version": "pilotproxy_stream_map_v1",
        "receiver_profile_id": profile["receiver_profile_id"],
        "stream_unit": "feed_polarization", "num_streams": FEEDS,
        "streams": [{"stream_index": index} for index in range(FEEDS)],
    }))
    return source, real, imag, profile_path, stream_path


def _injected_components(real, imag, amplitude):
    phases = 2 * np.pi * np.random.default_rng(PHASE_SEED).random(FEEDS)
    # The CH14 pilot is 3059 Hz below center, hence +3059 Hz in native zone 2.
    angle = 2 * np.pi * 3059 / SAMPLE_RATE * np.arange(len(real))[:, None] + phases
    re = np.rint(real + amplitude * np.cos(angle))
    im = np.rint(imag + amplitude * np.sin(angle))
    clipped = int(np.count_nonzero((re < -8) | (re > 7) | (im < -8) | (im > 7)))
    return np.clip(re, -8, 7).astype(np.int16), np.clip(im, -8, 7).astype(np.int16), clipped


def _frame_expectations(real, imag, weights):
    powers = []
    for frame in range(FRAMES):
        selected = slice(frame * FRAME, (frame + 1) * FRAME)
        # Offset-binary HDF5 becomes signed nibbles with reversed 128-sample
        # windows. Assemble this independently of the production adapter.
        packed = (((real[selected] & 15) << 4) | (imag[selected] & 15)).astype(np.uint8).view(np.int8)
        rows = packed.T.reshape(FEEDS, 128, 128)[:, :, ::-1].reshape(FEEDS * 128, 128)
        powers.append(frame_powers_cpu_reference_packed(rows, weights, 4))
    powers = np.asarray(powers, dtype=np.uint64)
    # This CPU oracle shares the frozen fixed-point FFT specification with the
    # kernel reference. It is not an independent derivation of that transform.
    re = weights.astype(np.int16) >> 4
    im = (weights.astype(np.int16) & 15)
    im = np.where(im >= 8, im - 16, im)
    norms = (re.astype(np.int64) ** 2 + im.astype(np.int64) ** 2).sum(axis=1)
    target_norm, reference_norm = int(norms[0]), int(norms[1] + norms[2])
    reference = powers[:, 1].astype(float) + powers[:, 2].astype(float)
    ratio = 2 * powers[:, 0].astype(float) / reference
    rho = powers[:, 0].astype(float) * reference_norm / (reference * target_norm) - 1
    power = (real.astype(float) ** 2 + imag.astype(float) ** 2).reshape(FRAMES, FRAME, FEEDS).mean(axis=(1, 2))
    return powers, target_norm, reference_norm, ratio, rho, power


def _quantile(values, q):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _wilson(detected, total):
    z = 1.959963984540054
    fraction = detected / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def _scalar_fit(points):
    x = [point["amplitude"] ** 2 for point in points]
    y = [float(point["rho"].mean()) for point in points]
    sem = [float(point["rho"].std(ddof=1) / math.sqrt(FRAMES)) for point in points]
    weights = [1 / value ** 2 for value in sem]
    s = sum(weights)
    sx = sum(w * a for w, a in zip(weights, x, strict=True))
    sxx = sum(w * a * a for w, a in zip(weights, x, strict=True))
    sy = sum(w * b for w, b in zip(weights, y, strict=True))
    sxy = sum(w * a * b for w, a, b in zip(weights, x, y, strict=True))
    determinant = s * sxx - sx * sx
    return {
        "floor": (sxx * sy - sx * sxy) / determinant,
        "gain_per_lsb2": (s * sxy - sx * sy) / determinant,
        "floor_err": math.sqrt(sxx / determinant),
        "gain_err": math.sqrt(s / determinant),
    }


@pytest.mark.cuda
def test_native_injection_detector_recovery_chain(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    if not DEFAULT_LIB_PATH.is_file():
        pytest.skip("compiled detector library is unavailable")
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv("PILOT_PROXY_FIGURE_FORMATS", "png")
    source, original_re, original_im, profile_path, stream_path = _native_inputs(tmp_path)
    weights_path = tmp_path / "weights.bin"
    write_weight_bank_from_receiver_profile(
        output_path=weights_path, profile=load_receiver_profile(profile_path),
        core=load_detector_core_profile(ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"),
        physical_channels=[14],
    )
    protected = {path: _digest(path) for path in (source, profile_path, stream_path, weights_path)}
    weights, valid_weights = DetectorWeightBank(explicit_path=weights_path).get_weights_for_physical_channel(14)
    assert valid_weights
    points = []
    for amplitude in (0, 1, 2):
        inputs = tmp_path / f"injected-{amplitude}"
        output = tmp_path / f"run-{amplitude}"
        assert injection.main([
            "--input", str(source), "--output-dir", str(inputs),
            "--amplitude-lsb", str(amplitude), "--physical-channel", "14",
            "--phase-seed", str(PHASE_SEED),
        ]) == 0
        re, im, clipped = _injected_components(original_re, original_im, amplitude)
        with h5py.File(inputs / source.name) as handle:
            np.testing.assert_array_equal(handle["baseband"], (((re + 8) << 4) | (im + 8)).astype(np.uint8))
            np.testing.assert_array_equal(handle["sibling"], np.arange(7))
            np.testing.assert_array_equal(handle["index_map/input"], np.arange(FEEDS))
            assert handle.attrs["fixture"] == "synthetic native-format noise"
        manifest_path = inputs / injection.INJECTION_MANIFEST_FILENAME
        entry, = json.loads(manifest_path.read_text())["files"]
        assert entry["baseband_frequency_hz"] == 3059
        assert entry["clip_count"] == clipped
        assert entry["clip_fraction"] == clipped / (FRAMES * FRAME * FEEDS)
        assert entry["byte_identical_to_source"] is (amplitude == 0)
        if amplitude == 0:
            assert _digest(inputs / source.name) == protected[source]
        protected.update({path: _digest(path) for path in (inputs / source.name, manifest_path)})
        assert runner.main([
            "--input-dir", str(inputs), "--output-dir", str(output),
            "--receiver-profile", str(profile_path), "--stream-map", str(stream_path),
            "--weights-path", str(weights_path), "--lib-path", str(DEFAULT_LIB_PATH),
            "--physical-channel", "14", "--frame-size-samples", str(FRAME),
            "--frames-per-chunk", "7", "--max-frames", str(FRAMES),
        ]) == 0
        shutil.copy2(manifest_path, output / manifest_path.name)
        powers, target_norm, reference_norm, ratio, rho, power = _frame_expectations(re, im, weights)
        with np.load(output / "chime_detector_outputs.npz") as detector, np.load(output / "chime_spectrogram_cache.npz") as cache:
            for product in (detector, cache):
                np.testing.assert_array_equal(product["frame_index"], np.arange(FRAMES))
                assert product["physical_channel"].tolist() == [14]
                assert product["pilot_frequency_hz"].tolist() == [470309441]
                assert product["chime_frequency_hz"].tolist() == [470312500]
                assert product["valid"].shape == (FRAMES, 1)
                assert product["valid"].all()
            for column, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
                np.testing.assert_array_equal(detector[name][:, 0], powers[:, column])
            np.testing.assert_array_equal(detector["fine_power_u64"].sum(axis=-1)[:, 0], powers)
            assert detector["target_norm_sq"].tolist() == [target_norm]
            assert detector["reference_norm_sum_sq"].tolist() == [reference_norm]
            np.testing.assert_allclose(detector["coarse_power_ratio"][:, 0], ratio, rtol=1e-15)
            np.testing.assert_allclose(detector["normalized_pilot_excess"][:, 0], rho, rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(cache["baseband_power_linear"][:, 0], power)
            expected_mask = [int(t) * reference_norm > (int(lo) + int(hi)) * target_norm
                             for t, lo, hi in powers]
            assert detector["mask"][:, 0].tolist() == expected_mask
        points.append({"amplitude": amplitude, "rho": rho, "power_ratio": ratio,
                       "radiometer": power, "clipped": clipped, "output": output})
    # Forty controls meet the unchanged minimum: 10 expected false alarms at
    # Pfa=0.25. A 1% tail is deliberately unsupported by this bounded fixture.
    assert FRAMES * PFA >= injection_recovery.MIN_FRAMES_PER_FALSE_ALARM
    report_dir = tmp_path / "recovery"
    args = [argument for point in reversed(points) for argument in ("--point", str(point["output"]))]
    assert injection_recovery.main(args + ["--false-alarm-rate", str(PFA),
                                           "--false-alarm-rate", "0.01", "--output-dir", str(report_dir)]) == 0
    report = json.loads((report_dir / injection_recovery.RECOVERY_SUMMARY_FILENAME).read_text())
    assert report["false_alarm_rates"] == [PFA]
    assert report["control_point_dir"] == str(points[0]["output"])
    for row, expected in zip(report["points"], points, strict=True):
        assert row["amplitude_lsb"] == expected["amplitude"]
        assert row["injected_power_lsb2"] == expected["amplitude"] ** 2
        assert row["n_valid"] == FRAMES
        assert row["total_clip_count"] == expected["clipped"]
        assert row["rho_hat"] == pytest.approx(expected["rho"].mean(), abs=1e-12)
        assert row["rho_sem"] == pytest.approx(expected["rho"].std(ddof=1) / math.sqrt(FRAMES), abs=1e-12)
        for statistic in ("power_ratio", "radiometer"):
            threshold = _quantile(points[0][statistic], 1 - PFA)
            assert report["thresholds"][statistic]["0.25"] == threshold
            detected = sum(float(value) > threshold for value in expected[statistic])
            prefix = f"pd_{statistic}_pfa0.25"
            assert row[prefix] == detected / FRAMES
            lo, hi = _wilson(detected, FRAMES)
            assert row[f"{prefix}_wilson95_lo"] == pytest.approx(lo, abs=1e-14)
            assert row[f"{prefix}_wilson95_hi"] == pytest.approx(hi, abs=1e-14)
    fit = _scalar_fit(points)
    assert report["fit"] == pytest.approx(fit, rel=1e-10, abs=1e-12)
    assert fit["gain_per_lsb2"] > 0
    dominated = [point for point in points if point["amplitude"] ** 2 * fit["gain_per_lsb2"] > 3 * abs(fit["floor"])]
    assert len(dominated) == 2
    low, high = dominated
    slope = math.log((high["rho"].mean() - fit["floor"]) / (low["rho"].mean() - fit["floor"])) / math.log(high["amplitude"] ** 2 / low["amplitude"] ** 2)
    assert report["signal_dominated_log_slope"] == pytest.approx(slope, rel=1e-10)
    with (report_dir / injection_recovery.RECOVERY_CSV_FILENAME).open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["amplitude_lsb"]) for row in rows] == [0, 1, 2]
    for name in (injection_recovery.RECOVERY_FIGURE, injection_recovery.BASELINE_FIGURE):
        assert (report_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(_digest(path) == digest for path, digest in protected.items())
