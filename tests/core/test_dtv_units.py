# coding=utf-8
from __future__ import annotations

import math

import numpy as np
import pytest

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    EFFECTIVE_BIN_BW_HZ,
    SPREADING_LOSS_DB,
    power_terms_to_normalized_pilot_excess,
    power_terms_to_normalized_coarse_power_ratio,
    power_terms_to_coarse_power_ratio,
    normalize_coarse_power_ratio,
    normalized_coarse_power_ratio_to_pilot_excess,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
    pilot_excess_to_data_shelf_metadata,
    data_shelf_snr_db_to_normalized_power_ratio_threshold,
    data_shelf_snr_db_to_pilot_excess_db,
    data_shelf_snr_threshold_fields,
    normalized_power_ratio_threshold_to_half_rational,
)

REFERENCE_EFFECTIVE_BIN_BW_HZ = 3051.7578125
REFERENCE_SPREADING_LOSS_DB = 32.936012
RAW_PILOTPROXY_LEVEL_EXAMPLE_DB = 1.0
EXPECTED_PNR_BIN_DB_FROM_1DB_RAW_F = -5.868253
EXPECTED_SHELF_SNR_DB_FROM_1DB_RAW_F = -27.504265
REFERENCE_CHANNEL_WIDTH_HZ = 390625.0
REFERENCE_DETECTOR_WINDOW_SAMPLES = 128

THRESHOLD_SNR_SHELF_DB = -26.0
EXPECTED_THRESHOLD_PNR_BIN_DB = -4.363988
EXPECTED_THRESHOLD_RAW_F = 1.366101
EXPECTED_THRESHOLD_HALF = 0.683050614
ABS_TOL_DB = 1e-6
ABS_TOL_HALF_THRESHOLD = 1e-10
NUMDEN_ZERO_REFERENCE_NUM = 1
NUMDEN_ZERO_REFERENCE_DEN = 0
THRESHOLD_TARGET_NORM_SQ = 100
THRESHOLD_REFERENCE_NORM_SUM_SQ = 200


def test_reference_geometry_pnr_to_shelf_example_is_explicit() -> None:
    assert math.isclose(EFFECTIVE_BIN_BW_HZ, REFERENCE_EFFECTIVE_BIN_BW_HZ)
    assert math.isclose(
        SPREADING_LOSS_DB,
        REFERENCE_SPREADING_LOSS_DB,
        abs_tol=ABS_TOL_DB,
    )

    f_raw = DB_LINEAR_BASE ** (RAW_PILOTPROXY_LEVEL_EXAMPLE_DB / DB_POWER_FACTOR)
    pilot_excess_db = normalized_pilot_excess_to_db(
        normalized_coarse_power_ratio_to_pilot_excess(normalize_coarse_power_ratio(f_raw, 1.0))
    )
    shelf_snr_db = pilot_excess_db_to_data_shelf_snr_db(pilot_excess_db)

    assert math.isclose(
        pilot_excess_db,
        EXPECTED_PNR_BIN_DB_FROM_1DB_RAW_F,
        abs_tol=ABS_TOL_DB,
    )
    assert math.isclose(
        float(shelf_snr_db),
        EXPECTED_SHELF_SNR_DB_FROM_1DB_RAW_F,
        abs_tol=ABS_TOL_DB,
    )

    metadata = pilot_excess_to_data_shelf_metadata()
    assert metadata["channel_width_hz"] == REFERENCE_CHANNEL_WIDTH_HZ
    assert metadata["detector_window_samples"] == REFERENCE_DETECTOR_WINDOW_SAMPLES
    assert metadata["bin_enbw_hz"] == EFFECTIVE_BIN_BW_HZ


def test_snr_shelf_threshold_converts_to_kernel_half_threshold() -> None:
    pilot_excess_db = float(data_shelf_snr_db_to_pilot_excess_db(THRESHOLD_SNR_SHELF_DB))
    raw = data_shelf_snr_db_to_normalized_power_ratio_threshold(THRESHOLD_SNR_SHELF_DB)
    half_num, half_den = normalized_power_ratio_threshold_to_half_rational(
        raw,
        target_norm_sq=THRESHOLD_TARGET_NORM_SQ,
        reference_norm_sum_sq=THRESHOLD_REFERENCE_NORM_SUM_SQ,
    )
    fields = data_shelf_snr_threshold_fields(
        THRESHOLD_SNR_SHELF_DB,
        target_norm_sq=THRESHOLD_TARGET_NORM_SQ,
        reference_norm_sum_sq=THRESHOLD_REFERENCE_NORM_SUM_SQ,
    )

    assert math.isclose(pilot_excess_db, EXPECTED_THRESHOLD_PNR_BIN_DB, abs_tol=ABS_TOL_DB)
    assert math.isclose(raw, EXPECTED_THRESHOLD_RAW_F, abs_tol=ABS_TOL_DB)
    assert math.isclose(
        half_num / half_den,
        EXPECTED_THRESHOLD_HALF,
        abs_tol=ABS_TOL_HALF_THRESHOLD,
    )
    assert fields["threshold_half_num"] == half_num
    assert fields["threshold_half_den"] == half_den
    assert fields["threshold_data_shelf_snr_db"] == THRESHOLD_SNR_SHELF_DB


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("target_norm_sq", 0),
        ("target_norm_sq", 1.9),
        ("target_norm_sq", True),
        ("target_norm_sq", np.bool_(True)),
        ("reference_norm_sum_sq", 0),
        ("reference_norm_sum_sq", 2.9),
        ("reference_norm_sum_sq", True),
        ("max_denominator", 0),
        ("max_denominator", 3.9),
        ("max_denominator", True),
    ],
)
def test_threshold_rational_requires_exact_positive_uint64_fields(
    field: str,
    invalid: object,
) -> None:
    arguments = {
        "target_norm_sq": 1,
        "reference_norm_sum_sq": 2,
        "max_denominator": 3,
    }
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        normalized_power_ratio_threshold_to_half_rational(
            1.5,
            **arguments,
        )


def test_threshold_rational_rejects_uint64_output_overflow() -> None:
    with pytest.raises(ValueError, match="threshold numerator"):
        normalized_power_ratio_threshold_to_half_rational(
            float(1 << 64),
            target_norm_sq=1,
            reference_norm_sum_sq=1,
            max_denominator=1,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("target_norm_sq", 0),
        ("target_norm_sq", 1.9),
        ("target_norm_sq", True),
        ("reference_norm_sum_sq", 0),
        ("reference_norm_sum_sq", 2.9),
        ("reference_norm_sum_sq", True),
    ],
)
def test_normalized_power_terms_require_exact_positive_norms(
    field: str,
    invalid: object,
) -> None:
    arguments = {"target_norm_sq": 1, "reference_norm_sum_sq": 2}
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        power_terms_to_normalized_coarse_power_ratio(
            1,
            2,
            **arguments,
        )


def test_numden_zero_denominator_is_invalid_measurement() -> None:
    assert math.isnan(power_terms_to_coarse_power_ratio(
        NUMDEN_ZERO_REFERENCE_NUM,
        NUMDEN_ZERO_REFERENCE_DEN,
    ))
    assert math.isnan(power_terms_to_normalized_pilot_excess(
        NUMDEN_ZERO_REFERENCE_NUM,
        NUMDEN_ZERO_REFERENCE_DEN,
        target_norm_sq=THRESHOLD_TARGET_NORM_SQ,
        reference_norm_sum_sq=THRESHOLD_REFERENCE_NORM_SUM_SQ,
    ))
    assert math.isnan(
        normalized_pilot_excess_to_db(power_terms_to_normalized_pilot_excess(
            NUMDEN_ZERO_REFERENCE_NUM,
            NUMDEN_ZERO_REFERENCE_DEN,
            target_norm_sq=THRESHOLD_TARGET_NORM_SQ,
            reference_norm_sum_sq=THRESHOLD_REFERENCE_NORM_SUM_SQ,
        ))
    )
