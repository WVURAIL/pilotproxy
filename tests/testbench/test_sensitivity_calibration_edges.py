"""Calibration must preserve the null population and conservative Q16 boundary."""

from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    exact_q16_decision,
    order_statistic_threshold,
    q16_ceil_multiplier,
)


@pytest.mark.parametrize("count,probability", [
    (100, np.nextafter(0.1, 0)),
    (100, 0.1),
    (3, 1 / 3),
    (3, np.nextafter(1 / 3, 1)),
    (80, 0.1),
    (2, np.nextafter(0.5, 0)),
])
def test_null_order_statistic_never_exceeds_requested_probability(count, probability):
    values = np.arange(1, count + 1, dtype=float)
    original = values.copy()
    result = order_statistic_threshold(values, p_fa=probability)
    # Enumerate the candidate thresholds with rational arithmetic, independently
    # of the implementation's order-index calculation.
    accepted = [threshold for threshold in values
                if Fraction(int(np.count_nonzero(values > threshold)), count)
                <= Fraction(float(probability))]
    assert result["threshold_multiplier"] == min(accepted)
    assert result["requested_p_fa_resolved"] == (
        Fraction(1, count) <= Fraction(float(probability)))
    np.testing.assert_array_equal(values, original)


def test_calibration_preserves_unsorted_input_and_counts_threshold_ties():
    values = np.array([4, 1, 4, 2, 3, 4, 2, 1], dtype=float)
    original = values.copy()
    result = order_statistic_threshold(values, p_fa=0.25)
    assert result["threshold_multiplier"] == 4
    assert result["null_trials"] == 8
    assert result["null_exceedances"] == 0
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("values", [
    [1, np.nan], [1, np.inf], [1, -np.inf], [1, -1],
    [[1, 2], [3, 4]], [1 + 1j, 2], [],
])
def test_null_calibration_refuses_contaminated_or_malformed_population(values):
    with pytest.raises(ValueError, match="null"):
        order_statistic_threshold(values, p_fa=0.1)


@pytest.mark.parametrize("threshold", [
    np.nextafter(0.0, 1.0),
    np.nextafter(1.0, 0), 1.0, np.nextafter(1.0, 2),
    np.nextafter(float(1 << 48), 0),
])
def test_q16_rounding_is_the_smallest_conservative_representable_integer(threshold):
    result = q16_ceil_multiplier(threshold)
    exact = Fraction(float(threshold)) * 65536
    assert result - 1 < exact <= result
    assert 0 < result < 1 << 64


@pytest.mark.parametrize("threshold", [float(1 << 48), np.finfo(float).max])
def test_q16_rounding_refuses_unrepresentable_thresholds(threshold):
    with pytest.raises(OverflowError, match="uint64"):
        q16_ceil_multiplier(threshold)


@pytest.mark.parametrize("multiplier", [True, np.bool_(True), 1.0, 1.5, "1"])
def test_q16_decision_requires_exact_integer_multiplier(multiplier):
    with pytest.raises(TypeError, match="multiplier_q16"):
        exact_q16_decision(ExactResponseComponents(1, 1, 1, 1, 0, 2),
                           multiplier_q16=multiplier)


@pytest.mark.parametrize("multiplier", [-1, 0, 1 << 64])
@pytest.mark.parametrize("valid", [False, True])
def test_q16_decision_rejects_out_of_range_multiplier_even_for_invalid_trial(
        multiplier, valid):
    components = ExactResponseComponents(1, int(valid), 1, 1, 0, 2)
    with pytest.raises(ValueError, match="multiplier_q16"):
        exact_q16_decision(components, multiplier_q16=multiplier)


@pytest.mark.parametrize("multiplier", [1, np.uint64((1 << 64) - 1)])
def test_exact_q16_decision_preserves_uint64_endpoints_and_strict_ties(multiplier):
    q = int(multiplier)
    equal = ExactResponseComponents(q, 65536, 1, 1, 0, 2)
    above = ExactResponseComponents(q + 1, 65536, 1, 1, 0, 2)
    assert exact_q16_decision(equal, multiplier_q16=multiplier) == 0
    assert exact_q16_decision(above, multiplier_q16=multiplier) == 1


def test_zero_null_threshold_remains_explicitly_unrepresentable_in_deployed_q16():
    result = order_statistic_threshold(np.zeros(10), p_fa=0.1)
    assert result["threshold_multiplier"] == 0
    assert result["null_trials"] == 10
    with pytest.raises(ValueError, match="positive"):
        q16_ceil_multiplier(result["threshold_multiplier"])
