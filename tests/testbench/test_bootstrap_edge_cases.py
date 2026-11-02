"""Crossing and paired-bootstrap contracts for sparse or degenerate sweeps."""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.testbench import sensitivity_study
from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    _percentile_interval,
    crossing_bracket,
    monotone_detection_rates,
    paired_crossing_bootstrap,
)


def _component(value: int) -> ExactResponseComponents:
    return ExactResponseComponents(value, 1, 1, 1, 0, 4)


def _bootstrap_inputs() -> dict:
    # Both stages cross halfway between two samples for every resample.
    return {
        "snr_db": [-2.0, 0.0],
        "float_null_ratios": [1.0, 1.0],
        "fixed_null_ratios": [1.0, 1.0],
        "null_trial_keys": ["null-a", "null-b"],
        "float_h1_ratios_by_snr": {-2.0: [0.0, 0.0], 0.0: [2.0, 2.0]},
        "fixed_h1_components_by_snr": {
            -2.0: [_component(0), _component(0)],
            0.0: [_component(2), _component(2)],
        },
        "trial_keys_by_snr": {-2.0: ["a", "b"], 0.0: ["a", "b"]},
        "p_fa": 0.25,
        "target_pd": 0.5,
        "replicates": 8,
        "seed": 7,
    }


@pytest.mark.parametrize("rates", [[np.nan, 0.8], [0.0, np.inf], [-0.1, 1.0], [0.0, 1.1], [0.0j, 1.0j]])
def test_monotone_rates_reject_invalid_detection_probabilities(rates) -> None:
    with pytest.raises(ValueError, match="rates"):
        monotone_detection_rates(rates)


def test_monotone_rates_preserve_input_and_only_pool_downward_fluctuations() -> None:
    raw = np.array([0.2, 0.1, 0.7, 0.6, 1.0])
    adjusted = monotone_detection_rates(raw)
    np.testing.assert_array_equal(adjusted, [0.2, 0.2, 0.7, 0.7, 1.0])
    np.testing.assert_array_equal(raw, [0.2, 0.1, 0.7, 0.6, 1.0])


@pytest.mark.parametrize("snrs", [[0.0, 0.0], [np.nan, 1.0], [-1.0, np.inf], [-np.inf, 1.0], [0.0j, 1.0j]])
def test_crossing_rejects_nonfinite_duplicate_or_complex_snr(snrs) -> None:
    with pytest.raises(ValueError, match="snr_db"):
        crossing_bracket(snrs, [0.0, 1.0], target=0.5)


@pytest.mark.parametrize("rates", [[0.5, 0.8], [0.8, 1.0], [0.0, 0.4], [0.5, 0.5]])
def test_crossing_requires_sample_strictly_below_target(rates) -> None:
    result = crossing_bracket([-2.0, 0.0], rates, target=0.5)
    assert result["bracketed"] is False
    assert result["estimate_db"] is None


def test_crossing_sorts_pairs_and_retains_observed_upper_endpoint() -> None:
    result = crossing_bracket([0.0, -4.0, -2.0], [0.3, 0.0, 0.5], target=0.5)
    assert result["estimate_db"] == -2.0
    assert result["snr_lo_db"] == -4.0
    assert result["snr_hi_db"] == -2.0
    assert result["raw_monotone_adjustment_max"] == pytest.approx(0.2)


def test_crossing_interpolation_does_not_overflow_finite_bracket() -> None:
    limit = np.finfo(np.float64).max
    result = crossing_bracket([-limit, limit], [0.0, 1.0], target=0.5)
    assert result["estimate_db"] == 0.0


@pytest.mark.parametrize("fraction", [0.0, 0.025, 0.5, 0.975, 1.0])
@pytest.mark.parametrize(
    "lo,hi",
    [
        (-np.finfo(float).max, np.finfo(float).max),
        (0.5 * np.finfo(float).max, np.finfo(float).max),
        (-np.finfo(float).max, -0.5 * np.finfo(float).max),
        (math.ulp(0.0), math.ulp(0.0)),
        (math.ulp(0.0), 2 * math.ulp(0.0)),
        (-math.ulp(0.0), math.ulp(0.0)),
    ],
)
def test_linear_interpolation_agrees_with_exact_rational_oracle(lo, hi, fraction) -> None:
    weight = Fraction(fraction)
    expected = float((1 - weight) * Fraction(float(lo)) + weight * Fraction(float(hi)))
    observed = sensitivity_study._linear_interpolation(float(lo), float(hi), fraction)
    assert math.isfinite(observed)
    assert abs(observed - expected) <= math.ulp(expected)
    if lo == hi:
        assert observed == expected


def test_percentile_interval_uses_linear_order_statistics_without_overflow() -> None:
    limit = np.finfo(np.float64).max
    low, high = _percentile_interval([-limit, limit])
    assert low == pytest.approx(-0.95 * limit)
    assert high == pytest.approx(0.95 * limit)
    assert _percentile_interval([limit, limit]) == (limit, limit)
    assert _percentile_interval([]) == (None, None)


@pytest.mark.parametrize("values", [[0.0, np.nan], [-np.inf, 1.0], [[1.0, 2.0]], [1.0j, 2.0j]])
def test_percentile_interval_rejects_invalid_derived_samples(values) -> None:
    with pytest.raises(ValueError):
        _percentile_interval(values)


@pytest.mark.parametrize("replicates", [0, -1, 0.5, 1.5, 2.0, True, "2", np.nan, np.inf])
def test_bootstrap_requires_positive_integer_replicates(replicates) -> None:
    kwargs = _bootstrap_inputs()
    kwargs["replicates"] = replicates
    with pytest.raises(ValueError, match="replicates"):
        paired_crossing_bootstrap(**kwargs)


@pytest.mark.parametrize("seed", [-1, 0.5, 1.5, 2.0, True, "2", np.nan, np.inf])
def test_bootstrap_requires_nonnegative_integer_seed(seed) -> None:
    kwargs = _bootstrap_inputs()
    kwargs["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        paired_crossing_bootstrap(**kwargs)


@pytest.mark.parametrize("snrs", [[], [-2.0, -2.0], [-2.0, np.nan], [-2.0, np.inf], [-2.0j, 0.0j], [[-2.0, 0.0]]])
def test_bootstrap_validates_snr_grid_before_mapping_lookup(snrs) -> None:
    kwargs = _bootstrap_inputs()
    kwargs["snr_db"] = snrs
    with pytest.raises(ValueError, match="snr_db"):
        paired_crossing_bootstrap(**kwargs)


@pytest.mark.parametrize("stage", ["float_null_ratios", "fixed_null_ratios"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_bootstrap_refuses_null_rows_that_cannot_be_calibrated_together(stage, invalid) -> None:
    kwargs = _bootstrap_inputs()
    kwargs[stage] = [1.0, invalid]
    kwargs["replicates"] = 1
    kwargs["seed"] = 1
    with pytest.raises(ValueError, match="null"):
        paired_crossing_bootstrap(**kwargs)


def test_bootstrap_identical_degenerate_crossings_have_zero_width_intervals() -> None:
    kwargs = _bootstrap_inputs()
    kwargs["replicates"] = np.int64(8)
    kwargs["seed"] = np.uint64(7)
    result = paired_crossing_bootstrap(**kwargs)
    assert result["valid_bracketed_replicates"] == 8
    assert result["valid_fraction"] == 1.0
    assert result["paired_null_trials"] == 2
    assert result["paired_h1_trials_per_snr"] == 2
    for stage in ("float", "fixed_q16"):
        for summary in ("median", "bootstrap95_lo", "bootstrap95_hi"):
            assert result[f"{stage}_crossing_{summary}_db"] == -1.0
    for summary in ("median", "bootstrap95_lo", "bootstrap95_hi"):
        assert result[f"fixed_minus_float_sensitivity_loss_{summary}_db"] == 0.0


def test_bootstrap_reports_no_interval_when_only_one_stage_brackets() -> None:
    kwargs = _bootstrap_inputs()
    kwargs["float_h1_ratios_by_snr"][0.0] = [0.0, 0.0]
    result = paired_crossing_bootstrap(**kwargs)
    assert result["valid_bracketed_replicates"] == 0
    assert result["valid_fraction"] == 0.0
    for name, value in result.items():
        if "median_db" in name or "bootstrap95_" in name:
            assert value is None, name


def test_bootstrap_intersects_h1_identities_before_resampling() -> None:
    kwargs = _bootstrap_inputs()
    kwargs["trial_keys_by_snr"] = {-2.0: ["a", "b"], 0.0: ["c", "a"]}
    # Only a is common. Extra rows would change the crossing if sampled.
    kwargs["float_h1_ratios_by_snr"] = {-2.0: [0.0, 2.0], 0.0: [0.0, 2.0]}
    kwargs["fixed_h1_components_by_snr"] = {
        -2.0: [_component(0), _component(2)],
        0.0: [_component(0), _component(2)],
    }
    result = paired_crossing_bootstrap(**kwargs)
    assert result["paired_h1_trials_per_snr"] == 1
    assert result["valid_bracketed_replicates"] == 8
    assert result["float_crossing_median_db"] == -1.0
    assert result["fixed_minus_float_sensitivity_loss_median_db"] == 0.0


def test_bootstrap_preserves_zero_rank_detection_and_invalid_trial_nondetection() -> None:
    kwargs = _bootstrap_inputs()
    kwargs["float_h1_ratios_by_snr"] = {-2.0: [np.nan, np.nan], 0.0: [np.inf, np.inf]}
    kwargs["fixed_h1_components_by_snr"] = {
        -2.0: [ExactResponseComponents(0, 0, 0, 0, -1, -1)] * 2,
        0.0: [ExactResponseComponents(1, 1, 0, 1, 0, 4)] * 2,
    }
    original = copy.deepcopy(kwargs)
    result = paired_crossing_bootstrap(**kwargs)
    assert result["paired_h1_trials_per_snr"] == 2
    assert result["valid_bracketed_replicates"] == 8
    assert result["float_crossing_median_db"] == -1.0
    assert result["fixed_minus_float_sensitivity_loss_median_db"] == 0.0
    assert kwargs["fixed_h1_components_by_snr"] == original["fixed_h1_components_by_snr"]
    assert math.isnan(kwargs["float_h1_ratios_by_snr"][-2.0][0])


def test_bootstrap_median_stays_finite_near_float_limit() -> None:
    kwargs = _bootstrap_inputs()
    limit = np.finfo(np.float64).max
    old_snrs = kwargs["snr_db"]
    new_snrs = [0.75 * limit, limit]
    kwargs["snr_db"] = new_snrs
    for key in ("float_h1_ratios_by_snr", "fixed_h1_components_by_snr", "trial_keys_by_snr"):
        kwargs[key] = {new: kwargs[key][old] for old, new in zip(old_snrs, new_snrs)}
    result = paired_crossing_bootstrap(**kwargs)
    assert result["float_crossing_median_db"] == pytest.approx(0.875 * limit)
    assert result["fixed_q16_crossing_median_db"] == pytest.approx(0.875 * limit)
    assert result["fixed_minus_float_sensitivity_loss_median_db"] == 0.0


def test_bootstrap_refuses_unrepresentable_loss_instead_of_dropping_it() -> None:
    kwargs = _bootstrap_inputs()
    limit = np.finfo(np.float64).max
    grid = [-limit, -0.5 * limit, 0.5 * limit, limit]
    kwargs["snr_db"] = grid
    kwargs["float_h1_ratios_by_snr"] = dict(zip(grid, [[0.0] * 2] + [[2.0] * 2] * 3))
    kwargs["fixed_h1_components_by_snr"] = dict(
        zip(grid, [[_component(0)] * 2] * 3 + [[_component(2)] * 2])
    )
    kwargs["trial_keys_by_snr"] = {snr: ["a", "b"] for snr in grid}
    with pytest.raises(OverflowError, match="loss"):
        paired_crossing_bootstrap(**kwargs)


@pytest.mark.parametrize(
    "field,changes,match",
    [
        ("null_trial_keys", ["null-a"], "identities"),
        ("null_trial_keys", ["null-a", "null-a"], "duplicate"),
        ("float_null_ratios", [1.0], "shape"),
        ("trial_keys_by_snr", {-2.0: ["a"], 0.0: ["a", "b"]}, "identities"),
        ("trial_keys_by_snr", {-2.0: ["a", "a"], 0.0: ["a", "b"]}, "duplicate"),
        ("trial_keys_by_snr", {-2.0: ["a", "b"], 0.0: ["c", "d"]}, "no common"),
        ("float_h1_ratios_by_snr", {-2.0: [0.0], 0.0: [2.0, 2.0]}, "disagree"),
        ("float_h1_ratios_by_snr", {-2.0: np.array([0.0j, 0.0j]), 0.0: [2.0, 2.0]}, "real"),
    ],
)
def test_bootstrap_rejects_malformed_trial_pairing(field, changes, match) -> None:
    kwargs = _bootstrap_inputs()
    kwargs[field] = changes
    with pytest.raises(ValueError, match=match):
        paired_crossing_bootstrap(**kwargs)


@pytest.mark.parametrize("parameter", ["p_fa", "target_pd"])
@pytest.mark.parametrize("value", [0.0, 1.0, np.nan, np.inf])
def test_bootstrap_validates_probabilities_before_resampling(parameter, value) -> None:
    kwargs = _bootstrap_inputs()
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        paired_crossing_bootstrap(**kwargs)
