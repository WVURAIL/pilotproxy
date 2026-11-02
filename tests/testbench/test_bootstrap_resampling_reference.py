"""Check paired sensitivity resampling against a scalar rational reference."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    paired_crossing_bootstrap,
)


def _components(value: float) -> ExactResponseComponents:
    if value == float("inf"):
        return ExactResponseComponents(1, 1, 0, 1, 0, 4)
    rational = Fraction(value)
    return ExactResponseComponents(
        rational.numerator, rational.denominator, 1, 1, 0, 4
    )


def _fixture() -> dict[str, Any]:
    snrs = [-6.0, -4.0, -2.0, 0.0]
    float_rows = [
        [0.25] * 7 + [2.0],
        [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        [0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    ]
    fixed_rows = [
        [0.25] * 7 + [2.25],
        [0.4, 0.8, 1.0 + 2**-17, 1.0 + 2**-16, 1.2, 1.4, 1.7, 2.0],
        [0.6, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        [0.8, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
    ]
    return {
        "snr_db": snrs,
        "float_null_ratios": [0.75, 0.9, 1.0 + 2**-17, 1.0 + 2**-16, 1.1, 1.2, 1.4, 1.6],
        "fixed_null_ratios": [0.8, 1.0, 1.0 + 2**-17, 1.0 + 2**-17, 1.25, 1.5, 1.75, 2.0],
        "null_trial_keys": [("null", i) for i in range(8)],
        "float_h1_ratios_by_snr": dict(zip(snrs, float_rows)),
        "fixed_h1_components_by_snr": {
            snr: [_components(value) for value in row]
            for snr, row in zip(snrs, fixed_rows)
        },
        "trial_keys_by_snr": {snr: list("abcdefgh") for snr in snrs},
        "p_fa": 0.25,
        "target_pd": 0.5,
        "replicates": 257,
        "seed": 3,
    }


def _quantile(values: list[Fraction], probability: Fraction) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    weight = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    return float((1 - weight) * ordered[lower] + weight * ordered[upper])


def _reference(data: dict[str, Any]) -> dict[str, Any]:
    # Join observations by identity before resampling. Multiplicities are
    # shared across the joined columns, including every sampled SNR.
    snrs = sorted(data["snr_db"])
    nulls = list(zip(
        data["null_trial_keys"], data["float_null_ratios"], data["fixed_null_ratios"]
    ))
    records: dict[Any, dict[float, tuple[float, ExactResponseComponents]]] = {}
    for snr in snrs:
        for identity, floating, fixed in zip(
            data["trial_keys_by_snr"][snr],
            data["float_h1_ratios_by_snr"][snr],
            data["fixed_h1_components_by_snr"][snr],
        ):
            records.setdefault(identity, {})[snr] = (floating, fixed)
    identities = sorted(
        (identity for identity, rows in records.items() if len(rows) == len(snrs)),
        key=repr,
    )
    rng = np.random.default_rng(data["seed"])
    requested_cdf = 1 - Fraction(data["p_fa"])
    target = Fraction(data["target_pd"])
    crossings: list[tuple[Fraction, Fraction]] = []
    for _ in range(data["replicates"]):
        sampled_null = [nulls[int(i)] for i in rng.integers(len(nulls), size=len(nulls))]
        thresholds = []
        for column in (1, 2):
            ordered = sorted(Fraction(row[column]) for row in sampled_null)
            thresholds.append(next(
                value for rank, value in enumerate(ordered, 1)
                if Fraction(rank, len(ordered)) >= requested_cdf
            ))
        q16_rational = thresholds[1] * 65536
        q16 = -(-q16_rational.numerator // q16_rational.denominator)
        sampled_h1 = Counter(
            identities[int(i)] for i in rng.integers(len(identities), size=len(identities))
        )
        curves = [[], []]
        for snr in snrs:
            counts = [0, 0]
            for identity, multiplicity in sampled_h1.items():
                floating, fixed = records[identity][snr]
                counts[0] += multiplicity * (floating > thresholds[0])
                left = fixed.designated_num * fixed.rank_den * 65536
                right = fixed.rank_num * fixed.designated_den * q16
                counts[1] += multiplicity * (left > right)
            for column, count in enumerate(counts):
                curves[column].append(Fraction(count, len(identities)))
        estimates = []
        for rates in curves:
            # The first sampled point cannot supply a lower bracket.
            upper = next((i for i, rate in enumerate(rates) if rate >= target), None)
            if upper is None or upper == 0:
                estimates.append(None)
                continue
            lower_rate = max(rates[:upper])
            weight = (target - lower_rate) / (rates[upper] - lower_rate)
            estimates.append((1 - weight) * Fraction(snrs[upper - 1]) + weight * Fraction(snrs[upper]))
        if all(estimate is not None for estimate in estimates):
            crossings.append((estimates[0], estimates[1]))
    result: dict[str, Any] = {
        "valid_bracketed_replicates": len(crossings),
        "valid_fraction": len(crossings) / data["replicates"],
        "paired_null_trials": len(nulls),
        "paired_h1_trials_per_snr": len(identities),
    }
    samples = [
        ("float_crossing", [f for f, _ in crossings]),
        ("fixed_q16_crossing", [q for _, q in crossings]),
        ("fixed_minus_float_sensitivity_loss", [q - f for f, q in crossings]),
    ]
    for prefix, values in samples:
        result[f"{prefix}_median_db"] = _quantile(values, Fraction(1, 2))
        result[f"{prefix}_bootstrap95_lo_db"] = _quantile(values, Fraction(1, 40))
        result[f"{prefix}_bootstrap95_hi_db"] = _quantile(values, Fraction(39, 40))
    return result


def _assert_reference(data: dict[str, Any]) -> dict[str, Any]:
    expected = _reference(data)
    result = paired_crossing_bootstrap(**data)
    for field, value in expected.items():
        if value is None or isinstance(value, int):
            assert result[field] == value, field
        else:
            assert result[field] == pytest.approx(value, abs=2e-12, rel=2e-12), field
    return result


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_full_resampling_matches_independent_rational_reference(seed: int) -> None:
    data = _fixture()
    data["seed"] = seed
    result = _assert_reference(data)
    assert 1 < result["valid_bracketed_replicates"] < data["replicates"]
    assert result["fixed_minus_float_sensitivity_loss_median_db"] > 0
    assert result["fixed_minus_float_sensitivity_loss_bootstrap95_lo_db"] < result["fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"]


@pytest.mark.parametrize("probability", [np.nextafter(0.25, 0.0), np.nextafter(0.25, 1.0)])
def test_resampling_respects_the_supplied_probability_at_an_order_boundary(
    probability: float,
) -> None:
    data = _fixture()
    data["p_fa"] = probability
    assert _assert_reference(data)["valid_bracketed_replicates"] > 0


def test_only_common_h1_identities_enter_the_bootstrap_population() -> None:
    data = _fixture()
    expected = paired_crossing_bootstrap(**data)
    for i, snr in enumerate(data["snr_db"]):
        data["trial_keys_by_snr"][snr].append(f"only_at_{snr}")
        data["float_h1_ratios_by_snr"][snr].append(1e6 if i % 2 else 0.0)
        data["fixed_h1_components_by_snr"][snr].append(_components(0.0 if i % 2 else 1e6))
    assert _assert_reference(data) == expected


def test_snr_and_independent_h1_storage_permutations_preserve_seeded_result() -> None:
    data = _fixture()
    expected = paired_crossing_bootstrap(**data)
    rng = np.random.default_rng(101)
    for snr in data["snr_db"]:
        order = rng.permutation(8)
        for field in ("trial_keys_by_snr", "float_h1_ratios_by_snr", "fixed_h1_components_by_snr"):
            data[field][snr] = [data[field][snr][i] for i in order]
    data["snr_db"].reverse()
    assert _assert_reference(data) == expected


def test_paired_null_storage_permutation_uses_the_supplied_sampling_order() -> None:
    data = _fixture()
    _assert_reference(data)
    order = [6, 0, 4, 1, 7, 3, 5, 2]
    for field in ("null_trial_keys", "float_null_ratios", "fixed_null_ratios"):
        data[field] = [data[field][i] for i in order]
    result = _assert_reference(data)
    # Null rows retain their supplied order. Repeating the same representation
    # repeats its draws; reordering it may change a finite Monte Carlo estimate.
    assert paired_crossing_bootstrap(**data) == result


def test_positive_infinite_h1_response_remains_a_detection() -> None:
    data = _fixture()
    for snr in data["snr_db"][1:]:
        data["float_h1_ratios_by_snr"][snr][0] = float("inf")
        data["fixed_h1_components_by_snr"][snr][0] = _components(float("inf"))
    assert _assert_reference(data)["valid_bracketed_replicates"] > 0


def test_equal_stages_retain_zero_loss_under_shared_resampling() -> None:
    data = _fixture()
    data["fixed_null_ratios"] = [1.0] * 8
    data["float_null_ratios"] = [1.0] * 8
    data["fixed_h1_components_by_snr"] = {
        snr: [_components(value) for value in row]
        for snr, row in data["float_h1_ratios_by_snr"].items()
    }
    result = _assert_reference(data)
    for suffix in ("median_db", "bootstrap95_lo_db", "bootstrap95_hi_db"):
        assert result[f"fixed_minus_float_sensitivity_loss_{suffix}"] == 0.0


@pytest.mark.parametrize("response", [0.0, 100.0])
def test_no_sampled_bracket_produces_no_crossing_or_interval(response: float) -> None:
    data = _fixture()
    for snr in data["snr_db"]:
        data["float_h1_ratios_by_snr"][snr] = [response] * 8
        data["fixed_h1_components_by_snr"][snr] = [_components(response)] * 8
    result = _assert_reference(data)
    assert result["valid_bracketed_replicates"] == 0
    assert all(value is None for field, value in result.items() if field.endswith("_db"))


def test_one_retained_replicate_reports_its_conditional_degenerate_interval() -> None:
    data = _fixture()
    data["replicates"] = 1
    data["seed"] = 0
    result = _assert_reference(data)
    assert result["valid_bracketed_replicates"] == 1
    for prefix in ("float_crossing", "fixed_q16_crossing", "fixed_minus_float_sensitivity_loss"):
        assert result[f"{prefix}_median_db"] == result[f"{prefix}_bootstrap95_lo_db"]
        assert result[f"{prefix}_median_db"] == result[f"{prefix}_bootstrap95_hi_db"]


def test_q16_rounding_and_strict_ties_change_the_crossing() -> None:
    data = _fixture()
    data["float_null_ratios"] = [1.0 + 2**-17] * 8
    data["fixed_null_ratios"] = data["float_null_ratios"].copy()
    for snr, value in zip(data["snr_db"], [0.5, 1.0 + 2**-16, 2.0, 3.0]):
        data["float_h1_ratios_by_snr"][snr] = [value] * 8
        data["fixed_h1_components_by_snr"][snr] = [_components(value)] * 8
    result = _assert_reference(data)
    assert result["float_crossing_median_db"] == -5.0
    assert result["fixed_q16_crossing_median_db"] == -3.0
    assert result["fixed_minus_float_sensitivity_loss_median_db"] == 2.0


def test_breaking_stage_identity_pairing_changes_reference_loss() -> None:
    data = _fixture()
    broken = deepcopy(data)
    for snr in broken["snr_db"]:
        broken["fixed_h1_components_by_snr"][snr].reverse()
    correct_result = _assert_reference(data)
    broken_result = _assert_reference(broken)
    assert correct_result["fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"] != broken_result["fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"]


def test_downward_sampled_rates_use_the_prior_maximum_at_the_lower_bracket() -> None:
    data = _fixture()
    data["target_pd"] = 0.75
    data["float_null_ratios"] = [1.0] * 8
    data["fixed_null_ratios"] = [1.0] * 8
    float_rows = [[0.5] * 8, [2.0] * 4 + [0.5] * 4, [0.5] * 8, [2.0] * 8]
    fixed_rows = [[0.5] * 8, [2.0] * 3 + [0.5] * 5, [2.0] + [0.5] * 7, [2.0] * 8]
    for snr, floating, fixed in zip(data["snr_db"], float_rows, fixed_rows):
        data["float_h1_ratios_by_snr"][snr] = floating
        data["fixed_h1_components_by_snr"][snr] = [_components(value) for value in fixed]
    result = _assert_reference(data)
    assert result["valid_bracketed_replicates"] == data["replicates"]
    assert result["float_crossing_median_db"] < -0.5
