# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.dtv_units import DB_POWER_FACTOR
from pilot_proxy.paths import PACKAGE_ROOT
import pilot_proxy.testbench.evaluate_snr as evaluate_snr
# noinspection PyProtectedMember
from pilot_proxy.testbench.evaluate_snr import (
    DEFAULT_SNR_SWEEP_MAX_DB,
    DEFAULT_SNR_SWEEP_MIN_DB,
    DEFAULT_SNR_SWEEP_STEP_DB,
    STANDARD_FREQUENCY_OFFSET_SWEEP_HZ,
    SUPPORTED_SNR_MAX_DB,
    _positive_to_db,
    _frequency_offset_values,
    _requested_snr_shelf_values,
    apply_channel_impairments,
    add_complex_awgn_for_snr,
    add_gnuradio_awgn_for_snr,
    required_iq_samples,
)

IQ_SAMPLE_RATE_HZ = 10.0
ADC_SAMPLE_RATE_HZ = 100.0
OUTPUT_SAMPLES = 5
PFB_TAPS = 4
PFB_FFT_SIZE = 20
REQUIRED_IQ_SAMPLES = 17

RNG_SEED = 1234
AWGN_TEST_SAMPLES = 200_000
BANDWIDTH_AWGN_TEST_SAMPLES = 20_000
GNURADIO_AWGN_TEST_SAMPLES = 50_000
NEGATIVE_SNR_DB = -6.0
ZERO_SNR_DB = 0.0
UNIT_SIGNAL_POWER = 1.0
BANDWIDTH_NOISE_POWER = 2.0
FULL_SAMPLE_RATE_HZ = 10.0
HALF_BANDWIDTH_HZ = 5.0
HALF_BANDWIDTH_RATIO = 0.5
ACTUAL_SNR_TOLERANCE_DB = 0.04
GNURADIO_SNR_TOLERANCE_DB = 0.08
POSITIVE_TO_DB_INPUT = 100.0
POSITIVE_TO_DB_OUTPUT = 20.0

SNR_EXPLICIT_DB = -30.0
SNR_RANGE_START_DB = -20.0
SNR_RANGE_STOP_DB = -10.0
SNR_RANGE_STEP_DB = 5.0
SNR_RANGE_EXPECTED = [-30.0, -20.0, -15.0, -10.0]
DEFAULT_SNR_SWEEP_EXPECTED_COUNT = 23
FREQUENCY_OFFSET_HZ = 1_000.0
CHANNEL_GAIN_DB = 6.0
CHANNEL_PHASE_DEG = 90.0
CHANNEL_EFFECT_SAMPLE_RATE_HZ = 4_000.0
DB_AMPLITUDE_FACTOR = 20.0


def test_required_iq_samples_matches_adc_span() -> None:
    required = required_iq_samples(
        iq_sample_rate_hz=IQ_SAMPLE_RATE_HZ,
        adc_sample_rate_hz=ADC_SAMPLE_RATE_HZ,
        num_output_samples=OUTPUT_SAMPLES,
        pfb_taps=PFB_TAPS,
        pfb_fft_size=PFB_FFT_SIZE,
    )

    # (5 + 4 - 1) * 20 ADC samples, last index 159, at 0.1 IQ samples/ADC.
    assert required == REQUIRED_IQ_SAMPLES


def test_add_complex_awgn_for_snr_hits_requested_power_ratio() -> None:
    rng = np.random.default_rng(RNG_SEED)
    signal = np.ones(AWGN_TEST_SAMPLES, dtype=np.complex64)
    noisy, signal_power, noise_power = add_complex_awgn_for_snr(
        signal,
        snr_db=NEGATIVE_SNR_DB,
        rng=rng,
    )

    actual_noise = noisy - signal
    actual_snr = DB_POWER_FACTOR * math.log10(
        float(np.mean(np.abs(signal) ** 2))
        / float(np.mean(np.abs(actual_noise) ** 2))
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(
        DB_POWER_FACTOR * math.log10(signal_power / noise_power),
        NEGATIVE_SNR_DB,
    )
    assert math.isclose(actual_snr, NEGATIVE_SNR_DB, abs_tol=ACTUAL_SNR_TOLERANCE_DB)


def test_add_complex_awgn_for_snr_accounts_for_noise_bandwidth() -> None:
    rng = np.random.default_rng(RNG_SEED)
    signal = np.ones(BANDWIDTH_AWGN_TEST_SAMPLES, dtype=np.complex64)
    _, signal_power, noise_power = add_complex_awgn_for_snr(
        signal,
        snr_db=ZERO_SNR_DB,
        rng=rng,
        sample_rate_hz=FULL_SAMPLE_RATE_HZ,
        snr_bandwidth_hz=HALF_BANDWIDTH_HZ,
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(noise_power, BANDWIDTH_NOISE_POWER, rel_tol=1e-6)
    in_band_noise_power = noise_power * HALF_BANDWIDTH_RATIO
    assert math.isclose(
        DB_POWER_FACTOR * math.log10(signal_power / in_band_noise_power),
        ZERO_SNR_DB,
    )


def test_target_snr_values_accepts_explicit_values_and_range() -> None:
    args = argparse.Namespace(
        requested_data_shelf_snr_db=[SNR_EXPLICIT_DB],
        snr_start_db=SNR_RANGE_START_DB,
        snr_stop_db=SNR_RANGE_STOP_DB,
        snr_step_db=SNR_RANGE_STEP_DB,
    )

    assert _requested_snr_shelf_values(args) == SNR_RANGE_EXPECTED


def test_target_snr_values_default_to_public_sweep() -> None:
    args = argparse.Namespace(
        requested_data_shelf_snr_db=None,
        snr_start_db=None,
        snr_stop_db=None,
        snr_step_db=None,
    )

    values = _requested_snr_shelf_values(args)

    assert values[0] == DEFAULT_SNR_SWEEP_MIN_DB
    assert values[-1] == DEFAULT_SNR_SWEEP_MAX_DB
    assert len(values) == DEFAULT_SNR_SWEEP_EXPECTED_COUNT
    assert values[1] - values[0] == DEFAULT_SNR_SWEEP_STEP_DB


def test_explicit_snr_supports_high_end_exploration() -> None:
    args = argparse.Namespace(
        requested_data_shelf_snr_db=[SUPPORTED_SNR_MAX_DB],
        snr_start_db=None,
        snr_stop_db=None,
        snr_step_db=None,
    )

    assert _requested_snr_shelf_values(args) == [SUPPORTED_SNR_MAX_DB]


def test_frequency_offsets_support_standard_sweep_and_deduplication() -> None:
    args = argparse.Namespace(
        frequency_offset_hz=[0.0, FREQUENCY_OFFSET_HZ],
        standard_frequency_offset_sweep=True,
    )

    assert _frequency_offset_values(args) == [
        0.0,
        FREQUENCY_OFFSET_HZ,
        STANDARD_FREQUENCY_OFFSET_SWEEP_HZ[0],
    ]


def test_apply_channel_impairments_applies_gain_phase_and_frequency() -> None:
    signal = np.ones(4, dtype=np.complex64)
    shifted = apply_channel_impairments(
        signal,
        sample_rate_hz=CHANNEL_EFFECT_SAMPLE_RATE_HZ,
        frequency_offset_hz=FREQUENCY_OFFSET_HZ,
        gain_db=CHANNEL_GAIN_DB,
        phase_deg=CHANNEL_PHASE_DEG,
    )

    assert shifted.shape == signal.shape
    assert np.abs(shifted[0]) == pytest.approx(
        10.0 ** (CHANNEL_GAIN_DB / DB_AMPLITUDE_FACTOR)
    )
    assert shifted[0].real == pytest.approx(0.0, abs=1e-6)
    assert shifted[0].imag > 0.0
    assert shifted[1].real < 0.0


def test_positive_to_db_converts_float_audit_value() -> None:
    assert math.isclose(_positive_to_db(POSITIVE_TO_DB_INPUT), POSITIVE_TO_DB_OUTPUT)
    assert _positive_to_db(ZERO_SNR_DB) == float("-inf")


def test_gnuradio_awgn_helper_hits_requested_band_snr(tmp_path) -> None:
    pytest.importorskip("gnuradio")
    signal = np.ones(GNURADIO_AWGN_TEST_SAMPLES, dtype=np.complex64)
    input_path = tmp_path / "clean.cfile"
    output_path = tmp_path / "noisy.cfile"
    signal.tofile(input_path)

    noisy, signal_power, noise_power, metadata = add_gnuradio_awgn_for_snr(
        signal,
        input_iq_path=input_path,
        output_iq_path=output_path,
        snr_db=ZERO_SNR_DB,
        seed=RNG_SEED,
        gnuradio_python=sys.executable,
        sample_rate_hz=FULL_SAMPLE_RATE_HZ,
        snr_bandwidth_hz=HALF_BANDWIDTH_HZ,
    )

    realized_noise = noisy - signal
    realized_noise_power = float(np.mean(np.abs(realized_noise) ** 2))
    realized_in_band_noise_power = realized_noise_power * HALF_BANDWIDTH_RATIO
    realized_snr_db = DB_POWER_FACTOR * math.log10(
        float(signal_power) / realized_in_band_noise_power
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(noise_power, BANDWIDTH_NOISE_POWER, rel_tol=1e-6)
    assert metadata["gnuradio_block"] == "analog.noise_source_c"
    assert math.isclose(realized_snr_db, ZERO_SNR_DB, abs_tol=GNURADIO_SNR_TOLERANCE_DB)


def test_gnuradio_helper_resolves_caller_paths_and_uses_package_only_bridge(
    monkeypatch,
    tmp_path: Path,
) -> None:
    signal = np.ones(32, dtype=np.complex64)
    monkeypatch.chdir(tmp_path)
    signal.tofile("relative-clean.cfile")
    monkeypatch.setenv("PYTHONPATH", "secondary-caller-path")
    observed: dict[str, object] = {}

    def fake_run(cmd, *, cwd, env, capture_output, text):
        input_path = Path(cmd[cmd.index("--input-iq") + 1])
        output_path = Path(cmd[cmd.index("--output-iq") + 1])
        metadata_path = Path(cmd[cmd.index("--metadata-json") + 1])
        bridge = Path(env["PYTHONPATH"].split(os.pathsep)[0])
        observed.update(
            {
                "cwd": cwd,
                "input_path": input_path,
                "output_path": output_path,
                "bridge": bridge,
                "bridge_entries": sorted(path.name for path in bridge.iterdir()),
                "package_target": (bridge / "pilot_proxy").resolve(),
                "pythonpath_tail": env["PYTHONPATH"].split(os.pathsep)[1:],
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        signal.tofile(output_path)
        metadata_path.write_text('{"helper": "fake"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(evaluate_snr.subprocess, "run", fake_run)

    noisy, _, _, metadata = add_gnuradio_awgn_for_snr(
        signal,
        input_iq_path=Path("relative-clean.cfile"),
        output_iq_path=Path("relative-output/noisy.cfile"),
        snr_db=ZERO_SNR_DB,
        seed=RNG_SEED,
        gnuradio_python="/custom/secondary-python",
        sample_rate_hz=FULL_SAMPLE_RATE_HZ,
        snr_bandwidth_hz=HALF_BANDWIDTH_HZ,
    )

    np.testing.assert_array_equal(noisy, signal)
    assert metadata == {"helper": "fake"}
    assert observed["cwd"] == tmp_path
    assert observed["input_path"] == (tmp_path / "relative-clean.cfile").resolve()
    assert observed["output_path"] == (
        tmp_path / "relative-output" / "noisy.cfile"
    ).resolve()
    assert observed["bridge_entries"] == ["pilot_proxy"]
    assert observed["package_target"] == PACKAGE_ROOT
    assert observed["pythonpath_tail"] == ["secondary-caller-path"]
    assert not Path(observed["bridge"]).exists()


def test_evaluator_resolves_all_user_paths_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    args = evaluate_snr.build_parser().parse_args(
        [
            "--input-iq",
            "relative-input.cfile",
            "--output-dir",
            "relative-output",
            "--waveform-audit-json",
            "relative-audit.json",
            "--lib-path",
            "relative-kernel.so",
            "--weights-path",
            "relative-weights.bin",
            "--experimental-bits",
            "8",
        ]
    )

    with pytest.raises(SystemExit, match=r"locked 4\+4 bit"):
        evaluate_snr.run(args)

    assert args.input_iq == (tmp_path / "relative-input.cfile").resolve()
    assert args.output_dir == (tmp_path / "relative-output").resolve()
    assert args.waveform_audit_json == (tmp_path / "relative-audit.json").resolve()
    assert args.lib_path == (tmp_path / "relative-kernel.so").resolve()
    assert args.weights_path == (tmp_path / "relative-weights.bin").resolve()


def test_weight_bank_identity_uses_a_portable_default_path() -> None:
    identity = evaluate_snr._artifact_identity(evaluate_snr.DEFAULT_WEIGHTS_PATH)
    manifest_path = Path(f"{evaluate_snr.DEFAULT_WEIGHTS_PATH}.manifest.json")
    manifest_identity = evaluate_snr._artifact_identity(manifest_path)

    assert identity["path"] == "weights/chime_dtv_weights_k128.bin"
    assert identity["sha256"] == hashlib.sha256(
        evaluate_snr.DEFAULT_WEIGHTS_PATH.read_bytes()
    ).hexdigest()
    assert manifest_identity["path"] == (
        "weights/chime_dtv_weights_k128.bin.manifest.json"
    )
    assert manifest_identity["sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_selected_weight_coefficients_identity_binds_shape_dtype_and_bytes() -> None:
    weights = np.arange(12, dtype=np.int8).reshape(3, 4)

    identity = evaluate_snr._weight_coefficients_identity(weights)

    assert identity == {
        "dtype": "int8",
        "shape": [3, 4],
        "sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def test_wilson_interval_closed_form() -> None:
    from pilot_proxy.testbench.evaluate_snr import wilson_interval

    # n=0 -> undefined
    lo, hi = wilson_interval(0, 0)
    assert lo != lo and hi != hi  # NaN
    # symmetric midpoint case, n=100, k=50: lo/hi ~ 0.404 / 0.596
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.40383, abs=2e-4)
    assert hi == pytest.approx(0.59617, abs=2e-4)
    # edge rates stay inside [0, 1] and are non-degenerate
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert 0.65 < lo < 1.0 and hi == 1.0


def test_summary_rows_report_detection_rates_with_wilson_bounds() -> None:
    from pilot_proxy.testbench.evaluate_snr import (
        _summarize_rows,
        wilson_interval,
    )

    def _trial(pe: int, mask: int) -> dict:
        return {
            "requested_data_shelf_snr_db": -30.0,
            "frequency_offset_hz": 0.0,
            "channel_gain_db": 0.0,
            "channel_phase_deg": 0.0,
            "measured_truth_data_shelf_snr_db": -30.0,
            "measured_truth_composite_atsc_snr_db": -20.0,
            "estimated_data_shelf_snr_db": -30.0,
            "snr_error_db": 0.0,
            "coarse_power_ratio": 1.0,
            "normalized_coarse_power_ratio_db": 0.0,
            "pilot_excess_db": 0.0,
            "cpu_float_estimated_data_shelf_snr_db": -30.0,
            "cpu_float_snr_error_db": 0.0,
            "cpu_float_coarse_power_ratio": 1.0,
            "cpu_gpu_abs_diff": 0.0,
            "cpu_float_gpu_snr_diff_db": 0.0,
            "num_input_streams": 4,
            "normalized_positive_excess_decision": pe,
            "mask": mask,
        }

    rows = [_trial(1, 1), _trial(1, 0), _trial(0, 0), _trial(1, 1)]
    summary = _summarize_rows(
        rows,
        requested_values=[-30.0],
        frequency_offset_values=[0.0],
        composite_to_shelf_db=10.0,
        num_input_streams=4,
    )
    assert len(summary) == 1
    row = summary[0]
    assert row["trials"] == 4
    assert row["positive_excess_fraction"] == pytest.approx(0.75)
    assert row["normalized_positive_excess_detection_rate"] == pytest.approx(0.75)
    lo, hi = wilson_interval(3, 4)
    assert row["normalized_positive_excess_detection_rate_wilson95_lo"] == pytest.approx(lo)
    assert row["normalized_positive_excess_detection_rate_wilson95_hi"] == pytest.approx(hi)
    assert row["threshold_detection_rate"] == pytest.approx(0.5)
    lo, hi = wilson_interval(2, 4)
    assert row["threshold_detection_rate_wilson95_lo"] == pytest.approx(lo)
    assert row["threshold_detection_rate_wilson95_hi"] == pytest.approx(hi)


def test_threshold_free_summary_uses_censoring_vocabulary() -> None:
    group = [
        {"normalized_positive_excess_decision": value}
        for value in (1, 0, 1, 1)
    ]
    fields = evaluate_snr._detection_rate_fields(group)

    assert fields["positive_excess_fraction"] == pytest.approx(0.75)
    assert "normalized_positive_excess_detection_rate" not in fields
    assert "threshold_detection_rate" not in fields


def test_summary_rows_pool_backend_powers_before_conversion() -> None:
    from pilot_proxy.dtv_units import pilot_excess_db_to_data_shelf_snr_db
    from pilot_proxy.testbench.evaluate_snr import _summarize_rows

    def _trial(
        *,
        gpu: tuple[int, int, int],
        cpu_float: tuple[float, float, float],
        cpu_packed: tuple[int, int, int],
    ) -> dict:
        return {
            "requested_data_shelf_snr_db": -30.0,
            "frequency_offset_hz": 0.0,
            "channel_gain_db": 0.0,
            "channel_phase_deg": 0.0,
            "measured_truth_data_shelf_snr_db": -30.0,
            "measured_truth_composite_atsc_snr_db": -20.0,
            "estimated_data_shelf_snr_db": -30.0,
            "snr_error_db": 0.0,
            "coarse_power_ratio": 1.0,
            "normalized_coarse_power_ratio_db": 0.0,
            "pilot_excess_db": 0.0,
            "cpu_float_estimated_data_shelf_snr_db": -30.0,
            "cpu_float_snr_error_db": 0.0,
            "cpu_float_coarse_power_ratio": 1.0,
            "cpu_gpu_abs_diff": 0.0,
            "cpu_float_gpu_snr_diff_db": 0.0,
            "p_target_u64": gpu[0],
            "p_ref_lower_u64": gpu[1],
            "p_ref_upper_u64": gpu[2],
            "cpu_float_p_target": cpu_float[0],
            "cpu_float_p_ref_lower": cpu_float[1],
            "cpu_float_p_ref_upper": cpu_float[2],
            "cpu_packed_p_target": cpu_packed[0],
            "cpu_packed_p_ref_lower": cpu_packed[1],
            "cpu_packed_p_ref_upper": cpu_packed[2],
            "target_weight_norm_sq": 2,
            "reference_weight_norm_sum_sq": 8,
            "cpu_float_target_weight_norm_sq": 2.5,
            "cpu_float_reference_weight_norm_sum_sq": 7.5,
        }

    rows = [
        _trial(gpu=(10, 5, 5), cpu_float=(5.0, 5.0, 5.0), cpu_packed=(20, 10, 10)),
        _trial(gpu=(50, 45, 45), cpu_float=(25.0, 25.0, 25.0), cpu_packed=(30, 30, 30)),
    ]
    calibration = {
        "pilot_below_data_db": 11.918446870168612,
        "bin_enbw_hz": 3051.7578125,
        "pilot_capture_efficiency": 0.9,
        "dtv_bandwidth_hz": 6.0e6,
    }
    summary = _summarize_rows(
        rows,
        requested_values=[-30.0],
        frequency_offset_values=[0.0],
        composite_to_shelf_db=10.0,
        num_input_streams=4,
        **calibration,
    )[0]

    assert summary["pilot_below_data_db"] == calibration["pilot_below_data_db"]
    assert summary["gpu_pooled_p_target"] == 60
    assert summary["gpu_pooled_p_ref_sum"] == 100
    assert summary["gpu_pooled_normalized_coarse_power_ratio"] == pytest.approx(2.4)
    assert summary["gpu_pooled_normalized_pilot_excess"] == pytest.approx(1.4)
    assert summary["cpu_float_pooled_p_target"] == pytest.approx(30.0)
    assert summary["cpu_float_pooled_p_ref_sum"] == pytest.approx(60.0)
    assert summary["cpu_float_pooled_normalized_coarse_power_ratio"] == pytest.approx(1.5)
    assert summary["cpu_float_pooled_normalized_pilot_excess"] == pytest.approx(0.5)
    assert summary["cpu_packed_pooled_p_target"] == 50
    assert summary["cpu_packed_pooled_p_ref_sum"] == 80
    assert summary["cpu_packed_pooled_normalized_coarse_power_ratio"] == pytest.approx(2.5)
    assert summary["cpu_packed_pooled_normalized_pilot_excess"] == pytest.approx(1.5)

    gpu_excess_db = 10.0 * math.log10(1.4)
    assert summary["gpu_pooled_pilot_excess_db"] == pytest.approx(gpu_excess_db)
    assert summary["gpu_pooled_estimated_data_shelf_snr_db"] == pytest.approx(
        pilot_excess_db_to_data_shelf_snr_db(
            gpu_excess_db,
            **calibration,
        )
    )
    assert summary["gpu_pooled_snr_error_db"] == pytest.approx(
        summary["gpu_pooled_estimated_data_shelf_snr_db"] + 30.0
    )


def test_summary_rows_omit_detection_rates_for_legacy_trials() -> None:
    from pilot_proxy.testbench.evaluate_snr import _summarize_rows

    legacy = {
        "requested_data_shelf_snr_db": -30.0,
        "frequency_offset_hz": 0.0,
        "channel_gain_db": 0.0,
        "channel_phase_deg": 0.0,
        "measured_truth_data_shelf_snr_db": -30.0,
        "measured_truth_composite_atsc_snr_db": -20.0,
        "estimated_data_shelf_snr_db": -30.0,
        "snr_error_db": 0.0,
        "coarse_power_ratio": 1.0,
        "normalized_coarse_power_ratio_db": 0.0,
        "pilot_excess_db": 0.0,
        "cpu_float_estimated_data_shelf_snr_db": -30.0,
        "cpu_float_snr_error_db": 0.0,
        "cpu_float_coarse_power_ratio": 1.0,
        "cpu_gpu_abs_diff": 0.0,
        "cpu_float_gpu_snr_diff_db": 0.0,
        "num_input_streams": 4,
    }
    summary = _summarize_rows(
        [legacy],
        requested_values=[-30.0],
        frequency_offset_values=[0.0],
        composite_to_shelf_db=10.0,
        num_input_streams=4,
    )
    assert "normalized_positive_excess_detection_rate" not in summary[0]
    assert "threshold_detection_rate" not in summary[0]


def test_cpu_reference_measurements_match_exact_integers() -> None:
    import numpy as np

    from pilot_proxy.detector_contract import (
        normalized_positive_excess,
        weight_term_norms_sq,
    )
    from pilot_proxy.detector_reference import coarse_power_ratio_cpu_reference_packed
    from pilot_proxy.testbench.evaluate_snr import _cpu_reference_measurements

    rng = np.random.default_rng(42)
    packed = rng.integers(-128, 128, size=(64, 128), dtype=np.int16).astype(np.int8)
    weights = rng.integers(-128, 128, size=(3, 128), dtype=np.int16).astype(np.int8)

    calib = dict(pilot_below_data_db=11.3, bin_enbw_hz=3051.7578125,
                 pilot_capture_efficiency=1.0, dtv_bandwidth_hz=6.0e6)
    out = _cpu_reference_measurements(packed=packed, weights=weights, bits=4, **calib)

    fstat, sums = coarse_power_ratio_cpu_reference_packed(packed, weights, 4)
    assert out["p_target_u64"] == int(round(float(sums[0])))
    assert out["p_ref_sum_u64"] == int(round(float(sums[1] + sums[2])))
    assert out["coarse_power_ratio"] == pytest.approx(
        2.0 * out["p_target_u64"] / out["p_ref_sum_u64"]
    )
    assert out["diagnostic_raw_float32"] == pytest.approx(fstat, rel=1e-6)
    nt, nl, nu = weight_term_norms_sq(weights)
    normalized_ratio = (
        out["p_target_u64"]
        * (nl + nu)
        / (out["p_ref_sum_u64"] * nt)
    )
    assert out["normalized_coarse_power_ratio"] == pytest.approx(normalized_ratio)
    assert out["normalized_pilot_excess"] == pytest.approx(
        normalized_ratio - 1.0
    )
    assert out["normalized_positive_excess_decision"] == normalized_positive_excess(
        out["p_target_u64"], out["p_ref_sum_u64"],
        target_norm_sq=nt, reference_norm_sum_sq=nl + nu,
    )
    assert "mask" not in out  # no threshold requested


def test_measurements_expose_signed_normalized_excess() -> None:
    from pilot_proxy.testbench.evaluate_snr import _measurements_from_powers

    out = _measurements_from_powers(
        diagnostic_float=0.5,
        p_target=1,
        p_ref_lower=2,
        p_ref_upper=2,
        weights=np.ones((3, 4), dtype=np.int8),
        pilot_below_data_db=11.3,
        bin_enbw_hz=3051.7578125,
        pilot_capture_efficiency=1.0,
        dtv_bandwidth_hz=6.0e6,
        threshold=None,
        mask=0,
        overflow=0,
    )

    assert out["normalized_coarse_power_ratio"] == pytest.approx(0.5)
    assert out["normalized_pilot_excess"] == pytest.approx(-0.5)
    assert math.isnan(out["pilot_excess_db"])


def test_cpu_reference_threshold_mask_is_exact_at_the_boundary() -> None:
    import numpy as np

    from pilot_proxy.testbench.evaluate_snr import (
        _cpu_reference_measurements,
        _measurements_from_powers,
    )

    calib = dict(pilot_below_data_db=11.3, bin_enbw_hz=3051.7578125,
                 pilot_capture_efficiency=1.0, dtv_bandwidth_hz=6.0e6)
    weights = np.ones((3, 4), dtype=np.int8)

    # Direct rational-half rule: mask iff p_t * den > num * p_ref, strictly.
    def _mask(p_t, p_ref, num, den):
        out = _measurements_from_powers(
            diagnostic_float=1.0, p_target=p_t,
            p_ref_lower=p_ref // 2, p_ref_upper=p_ref - p_ref // 2,
            weights=weights, threshold={"threshold_half_num": num,
                                        "threshold_half_den": den},
            mask=int(p_ref != 0 and p_t * den > num * p_ref),
            overflow=0, **calib)
        return out["mask"]

    assert _mask(50, 90, 5, 9) == 0   # 50*9 == 5*90: equality is no excess
    assert _mask(51, 90, 5, 9) == 1
    assert _mask(10, 0, 5, 9) == 0    # invalid reference floor

    # End-to-end through the CPU backend with a crafted threshold.
    rng = np.random.default_rng(7)
    packed = rng.integers(-128, 128, size=(16, 4), dtype=np.int16).astype(np.int8)
    out = _cpu_reference_measurements(
        packed=packed, weights=weights, bits=4,
        threshold={"threshold_half_num": 1, "threshold_half_den": 2}, **calib)
    expected = int(out["p_ref_sum_u64"] != 0
                   and 2 * out["p_target_u64"] > out["p_ref_sum_u64"])
    assert out["mask"] == expected
    assert out["rational_overflow_count"] == 0


def test_detector_backend_flag_parses() -> None:
    from pilot_proxy.testbench.evaluate_snr import build_parser

    args = build_parser().parse_args(
        ["--input-iq", "x.cfile", "--detector-backend", "cpu-reference"]
    )
    assert args.detector_backend == "cpu-reference"
    assert build_parser().parse_args(["--input-iq", "x.cfile"]).detector_backend == "cuda"


def test_waveform_audit_calibration_flag_parses() -> None:
    args = evaluate_snr.build_parser().parse_args(
        ["--input-iq", "x.cfile", "--pilot-below-data-db-from-audit"]
    )
    assert args.pilot_below_data_db_from_audit is True


def test_defaults_are_the_matched_clean_pilot_configuration() -> None:
    """The parser defaults must be the combination the clean pilot satisfies.

    The archive phase convention's half-rate factor displaces the channelized
    pilot by half a coarse channel relative to the shipped weight layouts, so
    with it applied the detector measures noise at every SNR regardless of
    spectral sense -- with the requested SNR still tracking perfectly, which
    is why it went unnoticed twice. Measured on channel 14: clean normalized
    ratio 121 with these defaults against 1.1 for every other combination,
    and the startup guard now enforces whichever combination is configured.
    """
    args = evaluate_snr.build_parser().parse_args(["--input-iq", "x.cfile"])
    assert args.spectral_sense == "normal"
    assert args.reference_archive_phase is False


def _tone_stream(normalized_frequency: float, *, window: int, windows: int):
    n = np.arange(window * windows, dtype=np.float64)
    return np.exp(2j * np.pi * float(normalized_frequency) * n).astype(
        np.complex64
    ).reshape(1, -1)


def _guard_layout(window: int, target_bin: int) -> dict:
    return {
        "target_normalized_frequency": target_bin / window,
        "lower_reference_normalized_frequency": (target_bin - 2) / window,
        "upper_reference_normalized_frequency": (target_bin + 2) / window,
    }


def test_clean_pilot_guard_accepts_a_line_the_statistic_sees() -> None:
    """A line the coarse statistic resolves onto its target term must pass.

    The weight rows are ``exp(-2j*pi*f*n)`` and the statistic correlates
    against their conjugates, so the passing tone is the one at ``-f`` -- the
    statistic, not an independent FFT, is the arbiter of "on target".
    """
    window, windows, target_bin = 128, 8, 1
    layout = _guard_layout(window, target_bin)
    weights = evaluate_snr._ideal_float_weights_from_layout(
        layout, detector_window_samples=window
    )
    rng = np.random.default_rng(0)
    streams = _tone_stream(-target_bin / window, window=window, windows=windows)
    streams = streams + 0.01 * (
        rng.standard_normal(streams.shape) + 1j * rng.standard_normal(streams.shape)
    ).astype(np.complex64)
    report = evaluate_snr.assert_clean_pilot_lands_on_target(
        streams,
        selected_weight_layout=layout,
        cpu_float_weights=weights,
        detector_window_samples=window,
        frame_size_samples=window * windows,
        spectral_sense="normal",
    )
    assert report["normalized_coarse_power_ratio"] > 8.0
    assert report["observed_statistic_peak_bin"] == target_bin


def test_clean_pilot_guard_rejects_a_mirrored_line() -> None:
    """A line in the conjugate bin must be an error, not a curve made of noise."""
    window, windows, target_bin = 128, 8, 1
    layout = _guard_layout(window, target_bin)
    weights = evaluate_snr._ideal_float_weights_from_layout(
        layout, detector_window_samples=window
    )
    rng = np.random.default_rng(0)
    streams = _tone_stream(target_bin / window, window=window, windows=windows)
    streams = streams + 0.01 * (
        rng.standard_normal(streams.shape) + 1j * rng.standard_normal(streams.shape)
    ).astype(np.complex64)
    with pytest.raises(SystemExit, match="not in the detector's target"):
        evaluate_snr.assert_clean_pilot_lands_on_target(
            streams,
            selected_weight_layout=layout,
            cpu_float_weights=weights,
            detector_window_samples=window,
            frame_size_samples=window * windows,
            spectral_sense="normal",
        )


def test_default_configuration_passes_the_clean_pilot_guard() -> None:
    """The full synthesis chain at the parser defaults satisfies the guard.

    Pins the matched quadrant end-to-end: a clean channel-14 pilot tone
    channelized with the defaults passes, and applying the archive phase
    convention -- whose half-rate factor displaces the pilot by half a coarse
    channel -- must fail the guard rather than sweep noise.
    """
    args = evaluate_snr.build_parser().parse_args(
        [
            "--input-iq",
            "x.cfile",
            "--physical-channel",
            "14",
            "--frame-size-samples",
            "1024",
        ]
    )
    args.dtv_pilot_mhz = (
        evaluate_snr.physical_channel_to_pilot_hz(14) / evaluate_snr.HZ_PER_MHZ
    )
    window = 128
    frame = 1024
    n_blocks = frame + evaluate_snr.REFERENCE_PFB_TAPS - 1
    required = evaluate_snr.required_iq_samples(
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        num_output_samples=frame,
    )
    band_lower_hz = float(args.band_lower_mhz) * evaluate_snr.HZ_PER_MHZ
    rf_center_hz = evaluate_snr._resolve_rf_center_hz(args)
    pilot_offset_hz = (
        float(args.dtv_pilot_mhz) * evaluate_snr.HZ_PER_MHZ - rf_center_hz
    )
    n = np.arange(required, dtype=np.float64)
    clean_iq = np.exp(
        2j * np.pi * pilot_offset_hz * n / float(args.iq_sample_rate_hz)
    ).astype(np.complex64)
    spec = evaluate_snr.ReferenceChannelizerSpec(
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
    )
    channel_index = evaluate_snr.nearest_reference_channel_index(
        float(args.dtv_pilot_mhz) * evaluate_snr.HZ_PER_MHZ, spec
    )
    blocks = evaluate_snr.complex_envelope_to_real_adc_blocks(
        clean_iq,
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        rf_center_hz=rf_center_hz,
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
        n_blocks=n_blocks,
        block_size=evaluate_snr.REFERENCE_PFB_FFT_SIZE,
    )
    channel_streams = evaluate_snr.channelize_real_blocks_to_reference_channels(
        blocks,
        channel_indices=[channel_index],
        spec=spec,
    )
    bank = evaluate_snr.DetectorWeightBank(
        explicit_path=evaluate_snr.DEFAULT_WEIGHTS_PATH,
        expected_kernel=None,
    )
    layout = bank.layout_for_pilot_frequency(float(args.dtv_pilot_mhz))
    weights = evaluate_snr._ideal_float_weights_from_layout(
        layout, detector_window_samples=window
    )
    streams = evaluate_snr.flatten_feed_channel_streams(
        np.stack([channel_streams], axis=0)
    )
    report = evaluate_snr.assert_clean_pilot_lands_on_target(
        streams,
        selected_weight_layout=layout,
        cpu_float_weights=weights,
        detector_window_samples=window,
        frame_size_samples=frame,
        spectral_sense=str(args.spectral_sense),
    )
    assert report["normalized_coarse_power_ratio"] > 8.0

    shifted = evaluate_snr.apply_reference_archive_phase_convention(channel_streams)
    with pytest.raises(SystemExit, match="not in the detector's target"):
        evaluate_snr.assert_clean_pilot_lands_on_target(
            evaluate_snr.flatten_feed_channel_streams(np.stack([shifted], axis=0)),
            selected_weight_layout=layout,
            cpu_float_weights=weights,
            detector_window_samples=window,
            frame_size_samples=frame,
            spectral_sense=str(args.spectral_sense),
        )
