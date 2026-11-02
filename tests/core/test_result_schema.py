# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.result_schema import (
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    RESULT_SCHEMA_TOKEN,
    fixed_point_contract,
    result_layout,
    result_schema_object,
)

FRAME_SIZE_SAMPLES = 16_384
DETECTOR_WINDOW_SAMPLES = 128
NUM_INPUT_STREAMS = 2
WINDOWS_PER_STREAM = 128
DETECTOR_ROWS = 256
DTV_BANDWIDTH_HZ = 6_000_000.0
BIN_ENBW_HZ = 3_051.7578125
PILOT_BELOW_DATA_DB = 11.3
PILOT_CAPTURE_EFFICIENCY = 1.0
THRESHOLD = {
    "threshold_data_shelf_snr_db": -26.0,
    "threshold_pilot_excess_db": -4.364,
    "threshold_coarse_power_ratio": 1.366,
    "threshold_half_num": 123,
    "threshold_half_den": 180,
}


def test_result_layout_uses_all_rows_summed_convention() -> None:
    layout = result_layout(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
    )

    assert layout["windows_per_stream"] == WINDOWS_PER_STREAM
    assert layout["detector_rows_per_frame"] == DETECTOR_ROWS
    assert layout["combine_mode"] == COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO


def test_result_schema_object_contains_fixed_contract_and_threshold() -> None:
    schema = result_schema_object(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
        dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
        bin_enbw_hz=BIN_ENBW_HZ,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
        threshold=THRESHOLD,
    )

    assert schema["schema_version"] == RESULT_SCHEMA_TOKEN
    assert schema["layout"]["num_input_streams"] == NUM_INPUT_STREAMS
    assert schema["calibration"]["pilot_below_data_db"] == PILOT_BELOW_DATA_DB
    assert schema["threshold"]["threshold_half_den"] == THRESHOLD[
        "threshold_half_den"
    ]
    assert schema["fixed_point_contract"]["detector_window_samples"] == (
        DETECTOR_WINDOW_SAMPLES
    )
    assert schema["fixed_point_contract"]["power_accumulator"] == "uint64"


@pytest.mark.parametrize("window", [64, 128])
def test_fixed_point_contract_describes_selected_supported_window(window) -> None:
    contract = fixed_point_contract(detector_window_samples=window)

    assert "K=128" in contract["detector_window_support_reason"]
    assert f"selects K={window}" in contract["detector_window_support_reason"]
    assert "k128_lock_reason" not in contract
    assert contract["supported_detector_window_samples"] == [64, 128]
    assert contract["packed_complex_bits"] == 8
    assert contract["sample_bits_per_component"] == 4


@pytest.mark.parametrize("field", [
    "detector_window_samples", "reference_offset_bins",
])
def test_fixed_point_contract_refuses_noninteger_geometry(field):
    settings = dict(detector_window_samples=128, reference_offset_bins=2)
    value = settings[field]
    for invalid in (value + 0.5, float(value), str(value), True, np.bool_(True)):
        with pytest.raises(ValueError, match=field + " must be an integer"):
            fixed_point_contract(**dict(settings, **{field: invalid}))


@pytest.mark.parametrize("field", [
    "frame_size_samples", "num_input_streams", "detector_window_samples",
    "num_selected_channels",
])
def test_result_layout_refuses_noninteger_dimensions(field):
    settings = dict(frame_size_samples=16384, num_input_streams=2,
                    detector_window_samples=128, num_selected_channels=3)
    value = settings[field]
    for invalid in (value + 0.5, float(value), str(value), True, np.bool_(True)):
        with pytest.raises(ValueError, match=field + " must be an integer"):
            result_layout(**dict(settings, **{field: invalid}))


@pytest.mark.parametrize("field", [
    "frame_size_samples", "num_input_streams", "detector_window_samples",
    "num_selected_channels", "reference_offset_bins",
])
def test_result_schema_does_not_coerce_dimensions_before_validation(field):
    settings = dict(
        frame_size_samples=16384, num_input_streams=2,
        detector_window_samples=128, num_selected_channels=3,
        reference_offset_bins=2, dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
        bin_enbw_hz=BIN_ENBW_HZ, pilot_below_data_db=PILOT_BELOW_DATA_DB,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY, threshold=None,
    )
    value = settings[field]
    for invalid in (value + 0.5, float(value), str(value), True, np.bool_(True)):
        with pytest.raises(ValueError, match=field + " must be an integer"):
            result_schema_object(**dict(settings, **{field: invalid}))


def test_result_layout_preserves_generic_windows_and_numpy_integer_counts():
    # Layout arithmetic supports positive windows beyond the compiled 64/128.
    layout = result_layout(
        frame_size_samples=np.int64(96), num_input_streams=np.int32(2),
        detector_window_samples=np.int64(32), num_selected_channels=np.int32(3),
    )
    assert layout["windows_per_stream"] == 3
    assert layout["detector_rows_per_frame"] == 18
    for field in ("frame_size_samples", "num_input_streams",
                  "detector_window_samples", "num_selected_channels"):
        assert type(layout[field]) is int
    assert fixed_point_contract(detector_window_samples=np.int64(64),
                                reference_offset_bins=np.int32(2)) == (
        fixed_point_contract(detector_window_samples=64, reference_offset_bins=2))


@pytest.mark.parametrize("settings, message", [
    ({"num_input_streams": 0}, "positive"),
    ({"frame_size_samples": -128}, "positive"),
    ({"num_selected_channels": 0}, "positive"),
    ({"detector_window_samples": 0}, "positive"),
    ({"frame_size_samples": 129}, "multiple"),
])
def test_result_layout_retains_positive_and_divisible_dimension_contracts(
        settings, message):
    defaults = dict(frame_size_samples=16384, num_input_streams=2,
                    detector_window_samples=128, num_selected_channels=1)
    with pytest.raises(ValueError, match=message):
        result_layout(**(defaults | settings))


def test_statistic_contract_defines_frame_powers_as_fine_power_sums() -> None:
    from pilot_proxy.detector_constants import (
        LOCKED_COARSE_POWER_DEFINITION,
        LOCKED_COARSE_POWER_RATIO_DEFINITION,
    )
    from pilot_proxy.result_schema import statistic_contract

    contract = statistic_contract()
    assert contract["frame_power_definition"].startswith("P_r = sum_f S_r[f]")
    assert "fxfft256_v1" in contract["frame_power_definition"]
    assert contract["coarse_power_ratio_definition"] == LOCKED_COARSE_POWER_RATIO_DEFINITION == (
        "R_coarse = 2 * sum_f S_target[f] / "
        "(sum_f S_ref_lower[f] + sum_f S_ref_upper[f]); "
        "S = fxfft256_v1 fine powers"
    )
    assert contract["all_rows_coarse_power_ratio_definition"] == (
        "R_coarse = 2 * sum_r(sum_f S_target,r[f]) / "
        "(sum_r(sum_f S_ref_lower,r[f]) + sum_r(sum_f S_ref_upper,r[f])); "
        "S = fxfft256_v1 fine powers"
    )
    assert LOCKED_COARSE_POWER_DEFINITION == "sum_of_fxfft256_fine_powers"
