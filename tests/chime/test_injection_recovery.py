# coding=utf-8
"""Tests for the injection-recovery / radiometer-baseline analysis."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.injection import INJECTION_MANIFEST_FILENAME
from pilot_proxy.chime.injection_recovery import (
    BASELINE_FIGURE,
    RECOVERY_CSV_FILENAME,
    RECOVERY_FIGURE,
    RECOVERY_SUMMARY_FILENAME,
    analyze_injection_recovery,
    write_outputs,
)
from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)

MU0 = 10.0 / 9.0
GAIN_TRUE = 0.05      # rho per LSB^2
FLOOR_TRUE = 0.02
N_FRAMES = 1500  # >= 10 / P_fa for the 1e-2 empirical quantile
RNG_NOISE = 0.01      # per-frame rho scatter


def _write_point(point_dir, amplitude: float, *, seed: int,
                 radiometer_shift: float = 0.0) -> None:
    point_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rho = (FLOOR_TRUE + GAIN_TRUE * amplitude**2
           + RNG_NOISE * rng.standard_normal((N_FRAMES, 1)))
    fstat = MU0 * (1.0 + rho)
    valid = np.ones((N_FRAMES, 1), dtype=np.uint8)
    detector = {
        "physical_channel": np.asarray([14], dtype=np.int32),
        "coarse_power_ratio": fstat,
        "normalized_pilot_excess": rho,
        "valid": valid,
        "mask": (fstat > MU0).astype(np.uint8),
        "p_target_u64": np.full((N_FRAMES, 1), 1, dtype=np.uint64),
        "p_ref_sum_u64": np.full((N_FRAMES, 1), 2, dtype=np.uint64),
        "target_norm_sq": np.asarray([5], dtype=np.int64),
        "reference_norm_sum_sq": np.asarray([9], dtype=np.int64),
        "null_power_ratio": np.asarray([MU0]),
    }
    np.savez_compressed(point_dir / CHIME_DETECTOR_OUTPUTS_FILENAME, **detector)
    # Radiometer statistic: weakly shifted by the tone (a one-bin tone barely
    # moves total band power), so its Pd must trail the F-statistic's.
    power = 100.0 + radiometer_shift + 1.0 * rng.standard_normal((N_FRAMES, 1))
    np.savez_compressed(
        point_dir / CHIME_SPECTROGRAM_CACHE_FILENAME,
        baseband_power_linear=power, valid=valid,
        mask=detector["mask"],
    )
    manifest = {
        "schema_version": "pilotproxy_injection_manifest_v1",
        "files": [{
            "amplitude_lsb": amplitude,
            "baseband_frequency_hz": -3059.0,
            "clip_count": 0,
        }],
    }
    (point_dir / INJECTION_MANIFEST_FILENAME).write_text(json.dumps(manifest))


def _ladder(tmp_path, amplitudes=(0.0, 1.0, 2.0, 4.0)):
    dirs = []
    for index, amplitude in enumerate(amplitudes):
        point = tmp_path / f"a{index}"
        # Total power shift ~ a^2 * tiny: keeps the radiometer weak.
        _write_point(point, amplitude, seed=100 + index,
                     radiometer_shift=0.05 * amplitude**2)
        dirs.append(point)
    return dirs


def test_recovery_fit_finds_gain_and_floor(tmp_path) -> None:
    report = analyze_injection_recovery(_ladder(tmp_path))

    fit = report["fit"]
    assert fit["gain_per_lsb2"] == pytest.approx(GAIN_TRUE, rel=0.05)
    assert fit["floor"] == pytest.approx(FLOOR_TRUE, abs=3 * fit["floor_err"])
    assert report["signal_dominated_log_slope"] == pytest.approx(1.0, abs=0.1)
    # Points are amplitude-sorted with SEMs attached.
    amplitudes = [row["amplitude_lsb"] for row in report["points"]]
    assert amplitudes == sorted(amplitudes)
    assert all(row["rho_sem"] > 0 for row in report["points"])


def test_fstat_beats_radiometer_at_matched_pfa(tmp_path) -> None:
    report = analyze_injection_recovery(_ladder(tmp_path))

    strongest = report["points"][-1]
    weakest_signal = report["points"][1]
    assert strongest["pd_fstat_pfa0.01"] > 0.99
    assert strongest["pd_fstat_pfa0.01"] >= strongest["pd_radiometer_pfa0.01"]
    # At the weak point the F-statistic already separates; the radiometer
    # stays near its false-alarm rate.
    assert weakest_signal["pd_fstat_pfa0.01"] > 0.5
    assert weakest_signal["pd_radiometer_pfa0.01"] < 0.2
    # Wilson bounds bracket every rate.
    for row in report["points"]:
        for name in ("fstat", "radiometer"):
            lo = row[f"pd_{name}_pfa0.01_wilson95_lo"]
            hi = row[f"pd_{name}_pfa0.01_wilson95_hi"]
            assert lo <= row[f"pd_{name}_pfa0.01"] <= hi


def test_requires_exactly_one_control(tmp_path) -> None:
    with pytest.raises(SystemExit, match="a = 0 control"):
        analyze_injection_recovery(
            _ladder(tmp_path / "none", amplitudes=(1.0, 2.0))
        )
    with pytest.raises(SystemExit, match="a = 0 control"):
        analyze_injection_recovery(
            _ladder(tmp_path / "two", amplitudes=(0.0, 0.0, 2.0))
        )


def test_unsupported_pfa_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="cannot support"):
        analyze_injection_recovery(
            _ladder(tmp_path), false_alarm_rates=[1e-6]
        )


def test_mixed_amplitude_manifest_is_refused(tmp_path) -> None:
    dirs = _ladder(tmp_path)
    manifest_path = dirs[1] / INJECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(dict(manifest["files"][0], amplitude_lsb=9.0))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="mixes amplitudes"):
        analyze_injection_recovery(dirs)


def test_write_outputs_produces_csv_json_and_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    report = analyze_injection_recovery(_ladder(tmp_path))

    written = write_outputs(report, tmp_path / "out")

    names = {path.name for path in written}
    assert names == {
        RECOVERY_CSV_FILENAME, RECOVERY_SUMMARY_FILENAME,
        RECOVERY_FIGURE, BASELINE_FIGURE,
    }
    for path in written:
        assert path.exists() and path.stat().st_size > 0
