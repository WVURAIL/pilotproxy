# coding=utf-8
"""Stable public result-schema helpers."""

from __future__ import annotations

from typing import Any

from pilot_proxy.detector_constants import (
    DEFAULT_DETECTOR_WINDOW_SAMPLES,
    LOCKED_NUM_WEIGHT_TERMS,
    LOCKED_PACKED_COMPLEX_BITS,
    LOCKED_REFERENCE_OFFSET_BINS,
    LOCKED_SAMPLE_BITS_PER_COMPONENT,
    LOCKED_SKIPPED_GUARD_BINS,
    POWER_SUM_ACCUMULATOR_BITS,
    SUPPORTED_DETECTOR_WINDOW_SAMPLES as _SUPPORTED_DETECTOR_WINDOW_SAMPLES,
)
from pilot_proxy.schema_identity import schema_token

RESULT_SCHEMA_NAME = "pilotproxy_result_schema"
RESULT_SCHEMA_REVISION = 1
RESULT_SCHEMA_TOKEN = schema_token(RESULT_SCHEMA_NAME, RESULT_SCHEMA_REVISION)
COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO = "all_rows_summed_before_ratio"
MASK_CONVENTION_SCHEMA_NAME = "pilotproxy_mask_convention"
MASK_CONVENTION_SCHEMA_REVISION = 1
MASK_CONVENTION_SCHEMA_TOKEN = schema_token(
    MASK_CONVENTION_SCHEMA_NAME, MASK_CONVENTION_SCHEMA_REVISION
)
MASK_VALUE_EXCLUDED = 1
MASK_VALUE_INCLUDED = 0
SUPPORTED_DETECTOR_WINDOW_SAMPLES = tuple(
    sorted(_SUPPORTED_DETECTOR_WINDOW_SAMPLES)
)

COARSE_POWER_RATIO_DEFINITION = (
    "R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)"
)
ALL_ROWS_COARSE_POWER_RATIO_DEFINITION = (
    "R_coarse = 2 * sum_r(P_target,r) / "
    "(sum_r(P_ref_lower,r) + sum_r(P_ref_upper,r))"
)
NULL_POWER_RATIO_DEFINITION = (
    "R_null = 2*target_norm_sq/reference_norm_sum_sq"
)
NORMALIZED_COARSE_POWER_RATIO_DEFINITION = "Q_coarse = R_coarse/R_null"
NORMALIZED_COARSE_POWER_RATIO_DB_DEFINITION = "10*log10(Q_coarse)"
RAW_PILOT_EXCESS_DEFINITION = "rho_raw = R_coarse - 1 (diagnostic only)"
NORMALIZED_PILOT_EXCESS_DEFINITION = "rho = Q_coarse - 1"
PILOT_EXCESS_DB_DEFINITION = "10*log10(rho) for rho > 0"
DATA_SHELF_SNR_DEFINITION = (
    "DTV data-shelf PSD relative to the non-DTV noise floor"
)
MASKED_AVERAGE_DEFINITION = (
    "masked samples are excluded from averages; they are not zero-filled "
    "before averaging"
)
MASKED_AVERAGE_FORMULA = "sum_b(P_b * (1 - M_b)) / sum_b(1 - M_b)"
DETECTOR_WINDOW_SUPPORT_REASON = (
    "With signed 4-bit real/imaginary samples and weights, each real/imaginary "
    "dot-product component grows as 9 + log2(K) bits. K=64 and K=128 are "
    "supported; K=128 is the largest supported power-of-two detector window "
    "that keeps the component accumulator inside signed int16."
)
REFERENCE_OFFSET_LOCK_REASON = (
    "skipped_guard_bins=1 leaves one fine DFT bin between the target bin and "
    "nearest reference bin. This corresponds to reference_offset_bins=2."
)


def statistic_contract() -> dict[str, str]:
    """Return exact definitions for current detector measurements."""
    return {
        "coarse_power_ratio_definition": COARSE_POWER_RATIO_DEFINITION,
        "all_rows_coarse_power_ratio_definition": (
            ALL_ROWS_COARSE_POWER_RATIO_DEFINITION
        ),
        "power_sum_rule": (
            "sum powers first, then form R_coarse; do not average frame ratios"
        ),
        "null_power_ratio_definition": NULL_POWER_RATIO_DEFINITION,
        "normalized_coarse_power_ratio_definition": (
            NORMALIZED_COARSE_POWER_RATIO_DEFINITION
        ),
        "normalized_coarse_power_ratio_db_definition": (
            NORMALIZED_COARSE_POWER_RATIO_DB_DEFINITION
        ),
        "raw_pilot_excess_definition": RAW_PILOT_EXCESS_DEFINITION,
        "normalized_pilot_excess_definition": NORMALIZED_PILOT_EXCESS_DEFINITION,
        "pilot_excess_db_definition": PILOT_EXCESS_DB_DEFINITION,
        "estimated_data_shelf_snr_definition": DATA_SHELF_SNR_DEFINITION,
    }


def fixed_point_contract(
    *,
    detector_window_samples: int = DEFAULT_DETECTOR_WINDOW_SAMPLES,
    reference_offset_bins: int = LOCKED_REFERENCE_OFFSET_BINS,
) -> dict[str, Any]:
    """Return the fixed-point detector contract for one run's geometry."""
    window = int(detector_window_samples)
    if window not in SUPPORTED_DETECTOR_WINDOW_SAMPLES:
        raise ValueError(
            "detector_window_samples must be one of "
            f"{list(SUPPORTED_DETECTOR_WINDOW_SAMPLES)}; got {window}."
        )
    offset = int(reference_offset_bins)
    if offset != LOCKED_REFERENCE_OFFSET_BINS:
        raise ValueError(
            "reference_offset_bins must match the compiled detector contract: "
            f"expected {LOCKED_REFERENCE_OFFSET_BINS}, got {offset}."
        )
    return {
        "detector_window_samples": window,
        "supported_detector_window_samples": list(
            SUPPORTED_DETECTOR_WINDOW_SAMPLES
        ),
        "num_weight_terms": int(LOCKED_NUM_WEIGHT_TERMS),
        "skipped_guard_bins": int(offset - 1),
        "reference_offset_bins": offset,
        "locked_skipped_guard_bins": int(LOCKED_SKIPPED_GUARD_BINS),
        "locked_reference_offset_bins": int(LOCKED_REFERENCE_OFFSET_BINS),
        "packed_complex_bits": int(LOCKED_PACKED_COMPLEX_BITS),
        "sample_bits_per_component": int(LOCKED_SAMPLE_BITS_PER_COMPONENT),
        "input_format": "complex_int4_packed_int8",
        "power_accumulator": "uint64",
        "power_accumulator_bits": int(POWER_SUM_ACCUMULATOR_BITS),
        "host_masking_policy": "normalized_positive_excess_from_uint64_powers",
        "per_frequency_threshold": False,
        "detector_window_support_reason": (
            f"{DETECTOR_WINDOW_SUPPORT_REASON} This run selects K={window}."
        ),
        "reference_offset_lock_reason": REFERENCE_OFFSET_LOCK_REASON,
    }


def result_layout(
    *,
    frame_size_samples: int,
    num_input_streams: int,
    detector_window_samples: int = DEFAULT_DETECTOR_WINDOW_SAMPLES,
    num_selected_channels: int = 1,
) -> dict[str, Any]:
    """Return current layout metadata for one combined detector frame."""
    frame = int(frame_size_samples)
    streams = int(num_input_streams)
    window = int(detector_window_samples)
    channels = int(num_selected_channels)
    if frame <= 0 or streams <= 0 or window <= 0 or channels <= 0:
        raise ValueError("frame size, streams, window, and channels must be positive.")
    if frame % window != 0:
        raise ValueError("frame_size_samples must be a multiple of detector window.")
    windows_per_stream = frame // window
    rows = streams * channels * windows_per_stream
    return {
        "frame_size_samples": frame,
        "num_input_streams": streams,
        "num_selected_channels": channels,
        "detector_window_samples": window,
        "windows_per_stream": int(windows_per_stream),
        "detector_rows_per_frame": int(rows),
        "combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        "row_index_rule": (
            "row = ((input_stream_index * num_selected_channels + "
            "selected_channel_index) * windows_per_stream) + window_index"
        ),
    }


def calibration_contract(
    *,
    dtv_bandwidth_hz: float,
    bin_enbw_hz: float,
    pilot_below_data_db: float,
    pilot_capture_efficiency: float,
) -> dict[str, float]:
    """Return data-shelf conversion constants used by a run."""
    return {
        "dtv_bandwidth_hz": float(dtv_bandwidth_hz),
        "bin_enbw_hz": float(bin_enbw_hz),
        "pilot_below_data_db": float(
            pilot_below_data_db
        ),
        "pilot_capture_efficiency": float(pilot_capture_efficiency),
    }


def threshold_contract(threshold: dict[str, Any] | None) -> dict[str, Any]:
    """Return current threshold fields, using null values when unset."""
    names = (
        "threshold_data_shelf_snr_db",
        "threshold_pilot_excess_db",
        "threshold_normalized_pilot_excess",
        "threshold_normalized_power_ratio",
        "threshold_coarse_power_ratio",
        "threshold_half_num",
        "threshold_half_den",
        "target_norm_sq",
        "reference_norm_sum_sq",
        "null_power_ratio",
    )
    if threshold is None:
        return {name: None for name in names}
    return {name: threshold.get(name) for name in names}


def mask_convention() -> dict[str, Any]:
    """Return the generic mask convention for before/after averages."""
    return {
        "schema_version": MASK_CONVENTION_SCHEMA_TOKEN,
        "mask_value_excluded": int(MASK_VALUE_EXCLUDED),
        "mask_value_included": int(MASK_VALUE_INCLUDED),
        "definition": MASKED_AVERAGE_DEFINITION,
        "masked_average_formula": MASKED_AVERAGE_FORMULA,
    }


def result_schema_object(
    *,
    frame_size_samples: int,
    num_input_streams: int,
    detector_window_samples: int,
    dtv_bandwidth_hz: float,
    bin_enbw_hz: float,
    pilot_below_data_db: float,
    pilot_capture_efficiency: float,
    threshold: dict[str, Any] | None,
    reference_offset_bins: int = LOCKED_REFERENCE_OFFSET_BINS,
    num_selected_channels: int = 1,
) -> dict[str, Any]:
    """Build the current public result-schema object for a run."""
    return {
        "schema_version": RESULT_SCHEMA_TOKEN,
        "statistic": statistic_contract(),
        "layout": result_layout(
            frame_size_samples=int(frame_size_samples),
            num_input_streams=int(num_input_streams),
            detector_window_samples=int(detector_window_samples),
            num_selected_channels=int(num_selected_channels),
        ),
        "calibration": calibration_contract(
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
            bin_enbw_hz=float(bin_enbw_hz),
            pilot_below_data_db=float(
                pilot_below_data_db
            ),
            pilot_capture_efficiency=float(pilot_capture_efficiency),
        ),
        "threshold": threshold_contract(threshold),
        "fixed_point_contract": fixed_point_contract(
            detector_window_samples=int(detector_window_samples),
            reference_offset_bins=int(reference_offset_bins),
        ),
        "mask_convention": mask_convention(),
    }
