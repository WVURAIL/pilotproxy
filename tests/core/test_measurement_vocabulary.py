# coding=utf-8
from __future__ import annotations

import math

import numpy as np
import pytest

from pilot_proxy.dtv_units import (
    data_shelf_snr_threshold_fields,
    normalize_coarse_power_ratio,
    normalized_coarse_power_ratio_to_pilot_excess,
    normalized_pilot_excess_to_db,
    power_terms_to_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio_db,
    power_terms_to_normalized_pilot_excess,
)
from pilot_proxy.product_contract import current_decision_contract


def test_coarse_power_ratio_is_the_raw_target_reference_ratio() -> None:
    assert power_terms_to_coarse_power_ratio(10, 20) == pytest.approx(1.0)


def test_normalized_coordinates_use_the_weight_norm_null_ratio() -> None:
    # R = 2*10/20 = 1, R_null = 2*4/10 = 0.8, Q = R/R_null = 1.25 and rho = Q - 1 = 0.25
    excess = power_terms_to_normalized_pilot_excess(
        10,
        20,
        target_norm_sq=4,
        reference_norm_sum_sq=10,
    )
    assert excess == pytest.approx(0.25)
    assert power_terms_to_normalized_coarse_power_ratio_db(
        10,
        20,
        target_norm_sq=4,
        reference_norm_sum_sq=10,
    ) == pytest.approx(10.0 * math.log10(1.25))
    assert normalized_pilot_excess_to_db(excess) == pytest.approx(10.0 * math.log10(0.25))


def test_normalized_pilot_excess_rejects_invalid_norms() -> None:
    with pytest.raises(ValueError, match="target_norm_sq"):
        power_terms_to_normalized_pilot_excess(
            1, 1, target_norm_sq=0, reference_norm_sum_sq=2
        )
    with pytest.raises(ValueError, match="reference_norm_sum_sq"):
        power_terms_to_normalized_pilot_excess(
            1, 1, target_norm_sq=1, reference_norm_sum_sq=0
        )


def test_invalid_reference_power_maps_to_nan() -> None:
    assert np.isnan(
        power_terms_to_normalized_pilot_excess(
            1, 0, target_norm_sq=1, reference_norm_sum_sq=2
        )
    )


def test_ratio_helpers_are_consistent() -> None:
    assert normalized_coarse_power_ratio_to_pilot_excess(
        normalize_coarse_power_ratio(1.2, 0.8)
    ) == pytest.approx(0.5)
    assert normalized_pilot_excess_to_db(0.5) == pytest.approx(
        10.0 * math.log10(0.5)
    )
    assert np.isnan(normalized_pilot_excess_to_db(0.0))


def test_threshold_uses_the_same_packed_weight_null_as_the_mask() -> None:
    fields = data_shelf_snr_threshold_fields(
        -26.0,
        target_norm_sq=4,
        reference_norm_sum_sq=10,
    )
    q_threshold = float(fields["threshold_normalized_power_ratio"])
    expected_half = q_threshold * 4.0 / 10.0
    assert fields["threshold_coarse_power_ratio"] == pytest.approx(0.8 * q_threshold)
    assert fields["threshold_half_float"] == pytest.approx(expected_half)
    assert fields["null_power_ratio"] == pytest.approx(0.8)


def test_current_decision_contract_uses_power_ratio_fields() -> None:
    contract = current_decision_contract()
    assert contract["measurements"] == [
        "coarse_local_reference_power_ratio",
        "fine_local_reference_power_ratio",
    ]
    # The scan stores the exact fine terms and makes no decision on them. No
    # float fine diagnostic is published, since the float ratio was never the
    # deployed statistic.
    assert contract["fine_measurement"]["terms_field"] == "fine_power_u64"
    assert "fine_diagnostic" not in contract
