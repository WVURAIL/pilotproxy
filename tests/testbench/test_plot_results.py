# coding=utf-8
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

# noinspection PyProtectedMember
from pilot_proxy.testbench.plot_results import (
    _Calibration,
    _CleanPackedTerms,
    _ConditionedTransfer,
    _bootstrap_pooled_excess_interval,
    _centered_moving_average,
    _conditioned_transfer_db,
    _conditioned_transfer_from_clean_terms,
    _conditioning_weight_bank_path,
    _conditioning_record,
    _control_expected_points,
    _excess_to_shelf_db,
    _normalized_coarse_power_ratio_db_to_estimated_data_shelf_snr_db,
    _estimated_data_shelf_snr_db_to_normalized_coarse_power_ratio_db,
    _reference_transfer_db,
    _plot_calibrations,
    _plot_control_expected,
    _shelf_to_pilot,
    _summary_ratio,
    _try_conditioned_transfer,
    _verify_selected_weight_provenance,
    _weight_coefficients_identity,
    _y_axis_upper,
    build_parser,
    plot_summary,
)

RAW_VALUES = [1.0, 4.0, 7.0, 10.0]
EXPECTED_WINDOW_1 = RAW_VALUES
EXPECTED_WINDOW_3 = [2.5, 4.0, 7.0, 8.5]
VALUES_WITH_NAN = [1.0, math.nan, 7.0]
EXPECTED_NAN_WINDOW_3 = [1.0, 4.0, 7.0]


def _conditioned_model() -> _ConditionedTransfer:
    return _ConditionedTransfer(
        delta=0.5,
        signal_contrast=0.8,
        reference_loading=0.2,
        conversion_offset_db=-21.0,
        clean_scale=2.0,
        clean_p_target=80,
        clean_p_ref_lower=15,
        clean_p_ref_upper=17,
        target_norm_sq=2,
        reference_norm_sum_sq=4,
        clean_target_normalized=10.0,
        clean_reference_normalized=2.0,
        target_noise_response=3.0,
        reference_noise_response=2.0,
        data_shelf_power=5.0,
        frequency_offset_hz=0.0,
        conditioning_rows=2,
        conditioning_snr_min_db=-42.0,
        conditioning_snr_max_db=-30.0,
        metadata_path=Path("run/dtv_snr_eval.json"),
        input_iq=Path("clean.cfile"),
        weights_path=Path("weights.bin"),
    )


def test_centered_moving_average_leaves_window_one_unchanged() -> None:
    assert _centered_moving_average(RAW_VALUES, 1) == EXPECTED_WINDOW_1


def test_centered_moving_average_smooths_with_clipped_edges() -> None:
    assert _centered_moving_average(RAW_VALUES, 3) == EXPECTED_WINDOW_3


def test_centered_moving_average_ignores_nan_values() -> None:
    assert _centered_moving_average(VALUES_WITH_NAN, 3) == EXPECTED_NAN_WINDOW_3


@pytest.mark.parametrize(
    ("x_max", "plotted_y", "expected"),
    [
        (30.0, [-0.1], 1.0),
        (0.0, [-2.9], 1.0),
        (-10.0, [-9.0], -2.0),
        (30.0, [4.5], 5.5),
        (30.0, [math.nan, math.inf], 1.0),
    ],
)
def test_y_axis_upper_keeps_saturation_visible(
    x_max: float, plotted_y: list[float], expected: float
) -> None:
    assert _y_axis_upper(x_max, plotted_y) == expected


def test_snr_shelf_fstat_axis_transform_round_trips() -> None:
    snr_values = [-60.0, -42.0, -26.0, 0.0]

    fstat_values = _estimated_data_shelf_snr_db_to_normalized_coarse_power_ratio_db(snr_values)
    recovered = _normalized_coarse_power_ratio_db_to_estimated_data_shelf_snr_db(fstat_values)

    assert list(recovered) == pytest.approx(snr_values)


def test_reference_transfer_includes_reference_contamination() -> None:
    result = _reference_transfer_db([-42.0, 0.0])

    assert result[0] == pytest.approx(-42.000274, abs=1e-6)
    assert result[1] == pytest.approx(-3.0102999566)


def test_conditioned_transfer_uses_recorded_coefficients() -> None:
    model = _conditioned_model()

    result = _conditioned_transfer_db([0.0], model)

    assert result[0] == pytest.approx(-21.0 + 10.0 * math.log10(1.3 / 1.2))


def test_conditioned_coefficients_use_lower_raw_power_least_squares() -> None:
    clean = _CleanPackedTerms(
        scale=2.0,
        p_target=80,
        p_ref_lower=15,
        p_ref_upper=17,
        target_norm_sq=2,
        reference_norm_sum_sq=4,
        input_iq=Path("clean.cfile"),
        weights_path=Path("weights.bin"),
    )
    rows = []
    for snr, noise in ((-42.0, 1.0), (-30.0, 3.0)):
        target_residual = 3.0 * noise + (1.0 if noise == 1.0 else 0.0)
        rows.append(
            {
                "requested_data_shelf_snr_db": snr,
                "quantization_scale": 2.0,
                "p_target_u64": (10.0 + target_residual) * 2.0 * 4.0,
                "p_ref_sum_u64": (2.0 + 2.0 * noise) * 4.0 * 4.0,
                "measured_in_band_noise_power": noise,
                "measured_data_shelf_power": 5.0,
                "target_weight_norm_sq": 2.0,
                "reference_weight_norm_sum_sq": 4.0,
            }
        )

    model = _conditioned_transfer_from_clean_terms(
        rows,
        _Calibration(),
        clean,
        metadata_path=Path("run/dtv_snr_eval.json"),
        frequency_offset_hz=0.0,
    )

    assert model.target_noise_response == pytest.approx(3.1)
    assert model.delta == pytest.approx(0.55)
    assert model.signal_contrast == pytest.approx(0.8)
    assert model.reference_loading == pytest.approx(0.2)
    assert model.conditioning_rows == 2
    assert model.conditioning_snr_min_db == -42.0
    assert model.conditioning_snr_max_db == -30.0
    record = _conditioning_record(model, [Path("lower_eval.csv")])
    assert record["comparator_label"] == "Waveform-conditioned expected transfer"


def test_explicit_conditioning_requires_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata was not found"):
        _try_conditioned_transfer(
            summary_paths=[tmp_path / "summary.csv"],
            conditioning_paths=[tmp_path / "lower_eval.csv"],
            rows=[{"requested_data_shelf_snr_db": -42.0}],
            calibration=_Calibration(),
        )


def test_conditioning_uses_the_recorded_weight_bank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bank_path = run_dir / "weights.bin"
    manifest_path = run_dir / "weights.bin.manifest.json"
    bank_path.write_bytes(b"recorded bank\n")
    manifest_path.write_bytes(b"recorded manifest\n")
    digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    resolved = _conditioning_weight_bank_path(
        {
            "weight_bank": {"path": "weights.bin", "sha256": digest},
            "weight_manifest": {
                "path": "weights.bin.manifest.json",
                "sha256": manifest_digest,
            },
        },
        run_dir / "dtv_snr_eval.json",
    )

    assert resolved == bank_path.resolve()


def test_conditioning_rejects_a_changed_weight_bank(tmp_path: Path) -> None:
    bank_path = tmp_path / "weights.bin"
    manifest_path = tmp_path / "weights.bin.manifest.json"
    bank_path.write_bytes(b"changed bank\n")
    manifest_path.write_bytes(b"recorded manifest\n")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _conditioning_weight_bank_path(
            {
                "weight_bank": {"path": str(bank_path), "sha256": "0" * 64},
                "weight_manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest_digest,
                },
            },
            tmp_path / "dtv_snr_eval.json",
        )


def test_conditioning_accepts_a_matching_relocated_bank(tmp_path: Path) -> None:
    bank_path = tmp_path / "relocated.bin"
    manifest_path = tmp_path / "relocated.bin.manifest.json"
    bank_path.write_bytes(b"recorded bank\n")
    manifest_path.write_bytes(b"recorded manifest\n")
    digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    resolved = _conditioning_weight_bank_path(
        {
            "weight_bank": {"path": "missing.bin", "sha256": digest},
            "weight_manifest": {
                "path": "missing.bin.manifest.json",
                "sha256": manifest_digest,
            },
        },
        tmp_path / "run" / "dtv_snr_eval.json",
        bank_path,
    )

    assert resolved == bank_path.resolve()


def test_conditioning_rejects_a_changed_weight_manifest(tmp_path: Path) -> None:
    bank_path = tmp_path / "weights.bin"
    manifest_path = tmp_path / "weights.bin.manifest.json"
    bank_path.write_bytes(b"recorded bank\n")
    manifest_path.write_bytes(b"changed manifest\n")
    bank_digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        _conditioning_weight_bank_path(
            {
                "weight_bank": {"path": str(bank_path), "sha256": bank_digest},
                "weight_manifest": {
                    "path": str(manifest_path),
                    "sha256": "0" * 64,
                },
            },
            tmp_path / "dtv_snr_eval.json",
        )


def test_legacy_conditioning_requires_explicit_verified_weights(tmp_path: Path) -> None:
    bank_path = tmp_path / "weights.bin"
    manifest_path = tmp_path / "weights.bin.manifest.json"
    bank_path.write_bytes(b"recorded bank\n")
    manifest_path.write_bytes(b"recorded manifest\n")
    bank_digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Legacy conditioning requires"):
        _conditioning_weight_bank_path(
            {},
            tmp_path / "dtv_snr_eval.json",
        )
    resolved = _conditioning_weight_bank_path(
        {},
        tmp_path / "dtv_snr_eval.json",
        bank_path,
        bank_digest,
        manifest_digest,
    )

    assert resolved == bank_path.resolve()
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        _conditioning_weight_bank_path(
            {},
            tmp_path / "dtv_snr_eval.json",
            bank_path,
            bank_digest,
            "0" * 64,
        )


def test_conditioning_verifies_selected_weight_layout_and_coefficients() -> None:
    weights = np.arange(12, dtype=np.int8).reshape(3, 4)
    metadata = {
        "selected_weight_layout": {"coarse_channel_index": 7},
        "selected_weight_coefficients": _weight_coefficients_identity(weights),
    }

    _verify_selected_weight_provenance(
        metadata,
        {"coarse_channel_index": 7},
        weights,
    )
    with pytest.raises(ValueError, match="layout disagrees"):
        _verify_selected_weight_provenance(
            metadata,
            {"coarse_channel_index": 8},
            weights,
        )
    with pytest.raises(ValueError, match="SHA256 changed"):
        _verify_selected_weight_provenance(
            metadata,
            {"coarse_channel_index": 7},
            weights + np.int8(1),
        )


def test_secondary_coordinate_uses_run_calibration() -> None:
    nominal = _Calibration(pilot_below_data_db=11.3)
    measured = _Calibration(pilot_below_data_db=11.918446870168612)

    nominal_pilot = float(_shelf_to_pilot(-20.0, nominal))
    measured_pilot = float(_shelf_to_pilot(-20.0, measured))

    assert nominal_pilot - measured_pilot == pytest.approx(0.618446870168612)


def test_radio_plot_uses_separate_input_and_output_calibrations() -> None:
    rows = [
        {
            "requested_data_shelf_snr_db": -12.0,
            "gpu_control_expected_data_shelf_snr_db": -11.8,
            "pilot_below_data_db": 11.9,
            "bin_enbw_hz": 3051.0,
            "dtv_bandwidth_hz": 6_000_000.0,
            "pilot_capture_efficiency": 1.0,
            "received_input_pilot_below_data_db": 12.5,
            "detector_output_pilot_below_data_db": 10.5,
            "detector_output_bin_enbw_hz": 4000.0,
            "detector_output_dtv_bandwidth_hz": 5_000_000.0,
            "detector_output_pilot_capture_efficiency": 0.8,
        }
    ]

    radio, received, output = _plot_calibrations(rows, [])

    assert radio
    assert received.pilot_below_data_db == 12.5
    assert output.pilot_below_data_db == 10.5
    assert received.bin_enbw_hz == output.bin_enbw_hz == 4000.0
    assert received.dtv_bandwidth_hz == output.dtv_bandwidth_hz == 5_000_000.0
    assert received.pilot_capture_efficiency == output.pilot_capture_efficiency == 0.8


def test_control_expected_points_keep_nonfinite_gaps() -> None:
    rows = [
        {
            "requested_data_shelf_snr_db": -9.0,
            "received_input_data_shelf_snr_db": -9.2,
            "frequency_offset_hz": 0.0,
            "gpu_control_expected_data_shelf_snr_db": math.nan,
        },
        {
            "requested_data_shelf_snr_db": -12.0,
            "received_input_data_shelf_snr_db": -12.3,
            "frequency_offset_hz": 0.0,
            "gpu_control_expected_data_shelf_snr_db": -11.7,
        },
        {
            "requested_data_shelf_snr_db": -6.0,
            "received_input_data_shelf_snr_db": math.nan,
            "frequency_offset_hz": 1000.0,
            "gpu_control_expected_data_shelf_snr_db": -5.0,
        },
    ]

    x_values, y_values = _control_expected_points(
        rows, frequency_offset_hz=0.0
    )

    assert x_values == [-12.3, -9.2]
    assert y_values[0] == -11.7
    assert math.isnan(y_values[1])


def test_control_expected_plot_marks_nonfinite_points() -> None:
    class Axis:
        def __init__(self) -> None:
            self.lines: list[tuple[tuple, dict]] = []
            self.scatters: list[tuple[tuple, dict]] = []

        def plot(self, *args, **kwargs):
            self.lines.append((args, kwargs))

        def scatter(self, *args, **kwargs):
            self.scatters.append((args, kwargs))

    axis = Axis()
    rows = [
        {
            "requested_data_shelf_snr_db": -12.0,
            "frequency_offset_hz": 0.0,
            "gpu_control_expected_data_shelf_snr_db": -11.5,
        },
        {
            "requested_data_shelf_snr_db": -9.0,
            "frequency_offset_hz": 0.0,
            "gpu_control_expected_data_shelf_snr_db": math.nan,
        },
    ]

    assert _plot_control_expected(axis, rows, offset=0.0, y_floor=-16.0)

    line_args, line_kwargs = axis.lines[0]
    assert line_args[0] == [-12.0, -9.0]
    assert line_args[1][0] == -11.5
    assert math.isnan(line_args[1][1])
    assert line_kwargs["linestyle"] == "--"
    assert line_kwargs["color"] == "black"
    assert line_kwargs["label"] == "Control-conditioned expected transfer"
    scatter_args, _scatter_kwargs = axis.scatters[0]
    assert scatter_args == ([-9.0], [-16.0])


def test_summary_ratio_pools_linear_values_with_reference_power() -> None:
    rows = [
        {
            "gpu_pooled_normalized_coarse_power_ratio": 1.1,
            "gpu_pooled_p_ref_sum": 1.0,
            "estimated_data_shelf_snr_db_mean": 800.0,
        },
        {
            "gpu_pooled_normalized_coarse_power_ratio": 1.3,
            "gpu_pooled_p_ref_sum": 3.0,
            "estimated_data_shelf_snr_db_mean": -800.0,
        },
    ]

    assert _summary_ratio(rows, "gpu") == pytest.approx(1.25)


def test_bootstrap_is_deterministic_and_keeps_signed_excess() -> None:
    rows = [
        {
            "normalized_coarse_power_ratio": ratio,
            "p_ref_sum_u64": weight,
        }
        for ratio, weight in ((0.96, 10.0), (0.98, 20.0), (0.99, 30.0))
    ]

    first = _bootstrap_pooled_excess_interval(rows, "gpu", samples=500, seed=17)
    second = _bootstrap_pooled_excess_interval(rows, "gpu", samples=500, seed=17)

    assert first == second
    assert first[0] < 0.0
    assert first[1] < 0.0
    assert math.isnan(_excess_to_shelf_db(0.0, _Calibration()))
    assert math.isnan(_excess_to_shelf_db(-0.01, _Calibration()))


def test_bootstrap_resamples_complete_passes() -> None:
    clustered = [
        {
            "pass_index": pass_index,
            "normalized_coarse_power_ratio": ratio,
            "p_ref_sum_u64": 10.0,
        }
        for pass_index, ratio in ((1, 0.9), (1, 0.9), (2, 1.3), (2, 1.3))
    ]
    pass_units = [
        {
            "normalized_coarse_power_ratio": ratio,
            "p_ref_sum_u64": 20.0,
        }
        for ratio in (0.9, 1.3)
    ]

    clustered_interval = _bootstrap_pooled_excess_interval(
        clustered, "gpu", samples=500, seed=17
    )
    unit_interval = _bootstrap_pooled_excess_interval(
        pass_units, "gpu", samples=500, seed=17
    )

    assert clustered_interval == unit_interval


def test_bootstrap_needs_two_passes() -> None:
    rows = [
        {
            "pass_index": 1,
            "normalized_coarse_power_ratio": ratio,
            "p_ref_sum_u64": 10.0,
        }
        for ratio in (0.9, 1.3)
    ]

    low, high = _bootstrap_pooled_excess_interval(
        rows, "gpu", samples=500, seed=17
    )

    assert math.isnan(low)
    assert math.isnan(high)


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_accepts_gpu_only_summary_shards(tmp_path: Path) -> None:
    lower = tmp_path / "lower_summary.csv"
    upper = tmp_path / "upper_summary.csv"
    output = tmp_path / "transfer.png"
    _write_rows(
        lower,
        [
            {
                "requested_data_shelf_snr_db": -42.0,
                "frequency_offset_hz": 0.0,
                "pilot_below_data_db": 11.918446870168612,
                "gpu_pooled_normalized_pilot_excess": 0.001,
            }
        ],
    )
    _write_rows(
        upper,
        [
            {
                "requested_data_shelf_snr_db": 0.0,
                "frequency_offset_hz": 0.0,
                "pilot_below_data_db": 11.918446870168612,
                "gpu_pooled_normalized_pilot_excess": 0.1,
            }
        ],
    )

    result = plot_summary(
        input_csv=[lower, upper],
        output_png=output,
        title="Estimator transfer",
        bootstrap_samples=0,
    )

    assert result == output
    assert output.stat().st_size > 0


def test_plot_accepts_control_conditioned_radio_summary(tmp_path: Path) -> None:
    summary = tmp_path / "radio_summary.csv"
    output = tmp_path / "radio_transfer.png"
    common = {
        "frequency_offset_hz": 0.0,
        "pilot_below_data_db": 11.9,
        "received_input_pilot_below_data_db": 12.4,
        "detector_output_pilot_below_data_db": 11.9,
        "detector_output_bin_enbw_hz": 3051.7578125,
        "detector_output_dtv_bandwidth_hz": 6_000_000.0,
        "detector_output_pilot_capture_efficiency": 1.0,
    }
    _write_rows(
        summary,
        [
            {
                **common,
                "requested_data_shelf_snr_db": -12.0,
                "received_input_data_shelf_snr_db": -12.2,
                "gpu_pooled_normalized_pilot_excess": 0.01,
                "gpu_control_expected_data_shelf_snr_db": -11.8,
            },
            {
                **common,
                "requested_data_shelf_snr_db": -9.0,
                "received_input_data_shelf_snr_db": -9.1,
                "gpu_pooled_normalized_pilot_excess": 0.02,
                "gpu_control_expected_data_shelf_snr_db": math.nan,
            },
        ],
    )

    result = plot_summary(
        input_csv=summary,
        output_png=output,
        title="Radio estimator transfer",
        bootstrap_samples=0,
    )

    assert result == output
    assert output.stat().st_size > 0


def test_parser_accepts_repeated_summary_and_trial_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--input-csv",
            "lower_summary.csv",
            "--input-csv",
            "upper_summary.csv",
            "--trial-csv",
            "lower_eval.csv",
            "--trial-csv",
            "upper_eval.csv",
            "--conditioning-trial-csv",
            "lower_eval.csv",
            "--conditioning-json",
            "conditioning.json",
            "--conditioning-weights-path",
            "alternate-weights.bin",
            "--conditioning-weights-sha256",
            "1" * 64,
            "--conditioning-weight-manifest-sha256",
            "2" * 64,
            "--y-min-db",
            "-60",
        ]
    )

    assert args.input_csv == [Path("lower_summary.csv"), Path("upper_summary.csv")]
    assert args.trial_csv == [Path("lower_eval.csv"), Path("upper_eval.csv")]
    assert args.conditioning_trial_csv == [Path("lower_eval.csv")]
    assert args.conditioning_json == Path("conditioning.json")
    assert args.conditioning_weights_path == Path("alternate-weights.bin")
    assert args.conditioning_weights_sha256 == "1" * 64
    assert args.conditioning_weight_manifest_sha256 == "2" * 64
    assert args.y_min_db == -60.0
