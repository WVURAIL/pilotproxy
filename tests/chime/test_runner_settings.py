"""Reject invalid run settings before publishing scientific products."""

import json

import numpy as np
import pytest

from pilot_proxy.chime.runner import run_chime_analysis
from test_chime_runner_small import _fake_detector, _fake_kernel, _small_input


def _settings(tmp_path):
    source, profile = _small_input(tmp_path)
    return {
        "input_dir": source,
        "output_dir": tmp_path / "run",
        "receiver_profile_path": profile,
        "stream_map_path": None,
        "physical_channels": [14],
        "frame_size_samples": 256,
        "detector_window_samples": 128,
        "frames_per_chunk": 1,
        "max_frames": 2,
        "kernel": _fake_kernel(128),
        "detector_fn": _fake_detector,
        "weights_by_channel": {14: np.ones((3, 128), dtype=np.int8)},
    }


@pytest.mark.parametrize("name", [
    "pilot_below_data_db", "bin_enbw_hz", "dtv_bandwidth_hz",
    "pilot_capture_efficiency", "calibration_seconds",
])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_settings_do_not_publish_native_int4_run(tmp_path, name, value):
    settings = _settings(tmp_path)
    settings[name] = value
    with pytest.raises(ValueError, match=name):
        run_chime_analysis(**settings)
    assert not settings["output_dir"].exists()


@pytest.mark.parametrize("name,value", [
    ("bin_enbw_hz", 0), ("bin_enbw_hz", -1),
    ("dtv_bandwidth_hz", 0), ("dtv_bandwidth_hz", -1),
    ("pilot_capture_efficiency", 0), ("pilot_capture_efficiency", -1),
    ("calibration_seconds", -1),
])
def test_invalid_calibration_domain_precedes_input_access(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        run_chime_analysis(
            input_dir=tmp_path / "missing-input", output_dir=tmp_path / "run",
            receiver_profile_path=tmp_path / "missing-profile", **{name: value},
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", [
    "frame_size_samples", "detector_window_samples", "frames_per_chunk", "max_frames",
])
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, 0, -1])
def test_frame_counts_require_positive_integers_before_input_access(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        run_chime_analysis(
            input_dir=tmp_path / "missing-input", output_dir=tmp_path / "run",
            receiver_profile_path=tmp_path / "missing-profile", **{name: value},
        )
    assert not (tmp_path / "run").exists()


def test_zero_calibration_negative_db_and_numpy_counts_preserve_products(tmp_path):
    settings = _settings(tmp_path)
    settings.update(
        frame_size_samples=np.int64(256), detector_window_samples=np.int32(128),
        frames_per_chunk=np.int64(1), max_frames=None, calibration_seconds=0,
        pilot_below_data_db=-3, bin_enbw_hz=1000, dtv_bandwidth_hz=5000000,
        pilot_capture_efficiency=0.75,
    )
    run_chime_analysis(**settings)
    output = settings["output_dir"]
    config = json.loads((output / "run_config.json").read_text())
    assert config["calibration_seconds"] == 0
    assert config["pilot_below_data_db"] == -3
    assert config["max_frames"] is None
    assert config["frame_size_samples"] == 256
    # Equal weight norms give 2*30/(12+8)-1 = 2 normalized pilot excess.
    expected_db = 10 * np.log10(2) - 3 - 10 * np.log10(5000) - 10 * np.log10(0.75)
    with np.load(output / "chime_detector_outputs.npz") as product:
        assert product["frame_index"].tolist() == [0, 1]
        np.testing.assert_array_equal(product["p_target_u64"], [[30], [30]])
        np.testing.assert_allclose(product["estimated_data_shelf_snr_db"], expected_db,
                                   rtol=0, atol=1e-12)
