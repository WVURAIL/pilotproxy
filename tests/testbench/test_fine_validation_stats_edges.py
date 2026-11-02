"""Numerical domains and raw-crossing evidence for frozen report workflows."""

from fractions import Fraction
import math

import numpy as np
import pytest

from pilot_proxy.testbench.fine_validation_stats import (
    _higher_index,
    accepted_count_limit,
    binomial_interval,
    false_alarm_validation,
    higher_index,
    higher_rank,
    raw_crossing_brackets,
)


@pytest.mark.parametrize("snrs", [
    [-np.finfo(float).max, np.finfo(float).max],
    [np.finfo(float).max / 2, np.finfo(float).max],
    [-np.finfo(float).max, -np.finfo(float).max / 2],
])
@pytest.mark.parametrize("rates", [[0.0, 1.0], [1.0, 0.0]])
@pytest.mark.parametrize("target", [0.25, 0.5, 0.75])
def test_raw_crossings_interpolate_finite_extremes_without_overflow(snrs, rates, target):
    fraction = (Fraction(target) - Fraction(rates[0])) / (
        Fraction(rates[1]) - Fraction(rates[0]))
    expected = float((1 - fraction) * Fraction(snrs[0]) + fraction * Fraction(snrs[1]))
    with np.errstate(over="raise", invalid="raise"):
        result = raw_crossing_brackets(snrs, rates, target=target)
    key = "upward_brackets" if rates[0] < rates[1] else "downward_brackets"
    assert len(result[key]) == 1
    assert result[key][0]["estimate_db"] == pytest.approx(expected, rel=3e-15)
    assert math.isfinite(result[key][0]["estimate_db"])
    assert result["status"] == ("unique" if key == "upward_brackets" else "ambiguous")
    assert result["raw_rates"] == rates


@pytest.mark.parametrize("field", ["snr_db", "rates"])
def test_raw_crossings_refuse_complex_observations(field):
    arguments = {"snr_db": np.array([0.0, 1.0]), "rates": np.array([0.0, 1.0]), "target": 0.5}
    arguments[field] = arguments[field].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        raw_crossing_brackets(**arguments)


@pytest.mark.parametrize("n", [True, np.bool_(True), 2.0, 2.5, "2", 0, -1])
def test_higher_order_helpers_require_positive_integer_population(n):
    for helper in (higher_index, _higher_index, higher_rank):
        with pytest.raises((TypeError, ValueError), match="integer|at least"):
            helper(n, 0.1)


def test_order_helpers_share_exact_decimal_contract_and_numpy_integer_support():
    for n in [np.int64(1), np.int64(1001)]:
        index = higher_index(n, 0.001)
        assert type(index) is int
        assert higher_rank(n, 0.001) == index + 1
        assert _higher_index(n, Fraction(1, 1000)) == index
    assert higher_rank(1001, 0.001) == 1000


@pytest.mark.parametrize("field", ["trials", "family_tests"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 12.0, 12.5, "12", 0, -1])
def test_acceptance_count_requires_exact_positive_counts(field, value):
    arguments = {"trials": 12, "cap": 0.25, "family_tests": 1}
    arguments[field] = value
    with pytest.raises((TypeError, ValueError), match="integer|at least"):
        accepted_count_limit(**arguments)


@pytest.mark.parametrize("trials,cap,family,confidence", [
    (12, Fraction(1, 4), 1, Fraction(4, 5)),
    (20, Fraction(1, 2), 2, Fraction(9, 10)),
    (5, Fraction(1, 10), 1, Fraction(19, 20)),
])
def test_count_limit_matches_direct_rational_binomial_tail(trials, cap, family, confidence):
    budget = (1 - confidence) / family
    accepted = []
    for k in range(trials + 1):
        cdf = sum(Fraction(math.comb(trials, j)) * cap**j * (1-cap)**(trials-j)
                  for j in range(k+1))
        if cdf <= budget:
            accepted.append(k)
    expected = max(accepted, default=-1)
    result = accepted_count_limit(np.int64(trials), float(cap),
                                  family_tests=np.int64(family), confidence=float(confidence))
    assert type(result) is int
    assert result == expected


@pytest.mark.parametrize("probability", [Fraction(1, 10**400), 1-Fraction(1, 10**400)])
def test_binomial_refuses_confidence_that_rounds_outside_float_probability_domain(probability):
    with pytest.raises(ValueError, match="representable"):
        binomial_interval(1, 10, confidence=probability)


@pytest.mark.parametrize("field", ["nominal_pfa", "cap_pfa", "confidence"])
@pytest.mark.parametrize("probability", [Fraction(1, 10**400), 1-Fraction(1, 10**400)])
def test_false_alarm_report_refuses_unrepresentable_probability(field, probability):
    arguments = {"nominal_pfa": 0.1, "cap_pfa": 0.2, "confidence": 0.95}
    arguments[field] = probability
    with pytest.raises(ValueError, match="representable"):
        false_alarm_validation(1, 100, **arguments)


def test_count_limit_refuses_underflowed_family_error_budget():
    with pytest.raises(ValueError, match="representable"):
        accepted_count_limit(1000, 0.5, family_tests=10**400)


@pytest.mark.parametrize("target", [Fraction(1, 10**400), 1-Fraction(1, 10**400)])
def test_crossing_refuses_target_that_would_round_to_endpoint(target):
    with pytest.raises(ValueError, match="representable"):
        raw_crossing_brackets([0, 1], [0, 1], target=target)
