"""Recovery fits and empirical rates retain their meaning at numeric limits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from pilot_proxy.characterization import injection


def _decimal_fit(x, y, sem):
    # Independent normal-equation sums at enough precision to retain the
    # small determinant that binary64 subtraction loses for large offsets.
    with localcontext() as context:
        context.prec = 2000
        dx, dy, ds = ([Decimal.from_float(float(v)) for v in values]
                      for values in (x, y, sem))
        weights = [1 / value**2 for value in ds]
        total = sum(weights)
        sx = sum(w * a for w, a in zip(weights, dx, strict=True))
        sy = sum(w * b for w, b in zip(weights, dy, strict=True))
        sxx = sum(w * a**2 for w, a in zip(weights, dx, strict=True))
        sxy = sum(w * a * b for w, a, b in zip(weights, dx, dy, strict=True))
        determinant = total * sxx - sx**2
        return {
            "floor": float((sxx * sy - sx * sxy) / determinant),
            "gain_per_lsb2": float((total * sxy - sx * sy) / determinant),
            "floor_err": float((sxx / determinant).sqrt()),
            "gain_err": float((total / determinant).sqrt()),
        }


@pytest.mark.parametrize("x, y, sem", [
    (1e12 + np.arange(5), 0.75 + 3 * (1e12 + np.arange(5)), np.array([1, 2, 3, 4, 5.])),
    (1e200 * np.array([0, 1, 2, 4.]), np.array([0.75, 3.75, 6.75, 12.75]), np.ones(4)),
    (1e-200 * np.array([0, 1, 2, 4.]), np.array([0.75, 3.75, 6.75, 12.75]), np.ones(4)),
    (np.array([0, 1, 2, 4.]), np.array([0.75, 3.75, 6.75, 12.75]), 1e200 * np.arange(1, 5)),
    (np.array([0, 1, 2, 4.]), np.array([0.75, 3.75, 6.75, 12.75]), 1e-200 * np.arange(1, 5)),
    (np.arange(4.), np.array([0.25, 1e100, 4.25, -1e100]), np.array([1e-200, 1e200, 1e-200, 1e200])),
    (np.array([-np.finfo(float).max, np.finfo(float).max]), np.array([-1., 1.]), np.ones(2)),
    (1e-200 * np.arange(4.), 1e-200 * (0.75 + 3 * np.arange(4.)), 1e-200 * np.ones(4)),
])
def test_centered_weighted_fit_matches_high_precision_sums(x, y, sem):
    expected = _decimal_fit(x, y, sem)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = injection.weighted_linear_fit(x, y, sem)
    assert all(np.isfinite(list(actual.values())))
    # The intercept at x=0 extrapolates far outside the first fixture. Keep a
    # small absolute tolerance there; gain and formal errors are relative.
    floor_atol = 2e-6 if np.all(x > 1e10) else 0
    assert actual["floor"] == pytest.approx(expected["floor"], rel=2e-12, abs=floor_atol)
    for key in ("gain_per_lsb2", "floor_err", "gain_err"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-12, abs=0)


@pytest.mark.parametrize("x,y,sem,message", [
    ([], [], [], "at least two"),
    ([0], [1], [1], "at least two"),
    ([0, 1], [1], [1, 1], "same shape"),
    ([[0, 1]], [[1, 2]], [[1, 1]], "one-dimensional"),
    ([0, 1], [1, 2], [1, 0], "positive"),
    ([0, 1], [1, 2], [1, -1], "positive"),
    ([0, np.nan], [1, 2], [1, 1], "finite"),
    ([0, 1], [1, np.inf], [1, 1], "finite"),
    ([0, 1], [1, 2], [1, np.inf], "finite"),
    ([1, 1], [1, 2], [1, 1], "distinct"),
])
def test_undefined_weighted_fit_is_refused(x, y, sem, message):
    with pytest.raises(ValueError, match=message):
        injection.weighted_linear_fit(np.asarray(x), np.asarray(y), np.asarray(sem))


@pytest.mark.parametrize("x,y,sem", [
    (np.array([0, 1e-308]), np.array([0, 1e308]), np.ones(2)),
    (np.arange(4.), np.arange(4.), np.full(4, np.nextafter(0., 1.))),
])
def test_fit_refuses_unrepresentable_coefficients_or_errors(x, y, sem):
    with pytest.raises(ValueError, match="representable"):
        injection.weighted_linear_fit(x, y, sem)


def test_complex_fit_is_refused_without_discarding_imaginary_components():
    with pytest.raises(ValueError, match="real"):
        injection.weighted_linear_fit(np.array([0, 1]), np.array([1j, 2j]), np.ones(2))


def test_nearby_false_alarm_rates_keep_distinct_thresholds_and_counts():
    values = np.linspace(0, 1, 100)
    requested = [0.25000001, 0.25000002, 0.25, 0.25000001]
    thresholds, usable = injection.matched_thresholds(
        {"n_valid": 100, "f": values}, requested, statistics={"ratio": "f"},
    )
    assert usable == [0.25000002, 0.25000001, 0.25]
    assert set(thresholds["ratio"]) == {"0.25000002", "0.25000001", "0.25"}
    expected = {key: float(np.quantile(values, 1 - float(key))) for key in thresholds["ratio"]}
    assert thresholds["ratio"] == pytest.approx(expected, rel=0, abs=1e-15)
    # Between the two close thresholds, this value must change its decision.
    point = {"n_valid": 2, "f": np.array([0., np.mean(list(expected.values())[:2])])}
    rates = injection.detection_rates(point, thresholds, usable, statistics={"ratio": "f"})
    assert rates["pd_ratio_pfa0.25000002"] == 0.5
    assert rates["pd_ratio_pfa0.25000001"] == 0
    assert rates["pd_ratio_pfa0.25"] == 0


@pytest.mark.parametrize("pfa", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_false_alarm_rate_is_refused_by_statistics_api(pfa):
    with pytest.raises(ValueError, match="false-alarm"):
        injection.matched_thresholds(
            {"n_valid": 100, "f": np.arange(100.)}, [pfa], statistics={"ratio": "f"},
        )


@pytest.mark.parametrize("n,values", [
    (100, np.arange(3.)), (True, np.array([1.])), (3.0, np.arange(3.)),
    (3, np.array([[1., 2., 3.]])), (3, np.array([1., np.nan, 3.])),
    (3, np.array([1., np.inf, 3.])),
    (-1, np.array([])), (2, np.array([1j, 2j])),
])
def test_population_must_match_its_finite_statistic_samples(n, values):
    point = {"n_valid": n, "f": values}
    with pytest.raises(ValueError, match="n_valid|statistic"):
        injection.matched_thresholds(point, [0.5], statistics={"ratio": "f"})
    with pytest.raises(ValueError, match="n_valid|statistic"):
        injection.detection_rates(point, {"ratio": {"0.5": 1.5}}, [0.5], statistics={"ratio": "f"})


def test_empty_control_is_unsupported_and_empty_detection_population_is_refused():
    empty = {"n_valid": 0, "f": np.array([])}
    assert injection.matched_thresholds(empty, [0.5], statistics={"ratio": "f"}) == ({"ratio": {}}, [])
    with pytest.raises(ValueError, match="positive|empty"):
        injection.detection_rates(empty, {"ratio": {"0.5": 1}}, [0.5], statistics={"ratio": "f"})


def test_finite_empirical_quantile_does_not_overflow_between_extreme_endpoints():
    maximum = np.finfo(float).max
    values = np.array([-maximum] * 10 + [maximum] * 10)
    with np.errstate(over="raise", invalid="raise"):
        thresholds, usable = injection.matched_thresholds(
            {"n_valid": 20, "f": values}, [0.5, 1e-320], statistics={"ratio": "f"},
        )
    assert usable == [0.5]
    assert thresholds == {"ratio": {"0.5": 0.0}}


def test_support_boundary_keeps_ten_expected_false_alarms_requirement():
    below_boundary = float(np.nextafter(0.1, 0.0))
    thresholds, usable = injection.matched_thresholds(
        {"n_valid": 100, "f": np.arange(100.)}, [0.1, below_boundary],
        statistics={"ratio": "f"},
    )
    assert usable == [0.1]
    assert set(thresholds["ratio"]) == {"0.1"}


def test_finite_quantiles_preserve_numpy_endpoint_arithmetic():
    values = np.random.default_rng(2018).normal(size=100)
    requested = [0.1, 0.3, 0.5, 0.6, 0.75, 0.9]
    thresholds, usable = injection.matched_thresholds(
        {"n_valid": len(values), "f": values}, requested, statistics={"ratio": "f"},
    )
    for pfa in usable:
        assert thresholds["ratio"][f"{pfa:g}"] == float(np.quantile(values, 1 - pfa))


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_nonfinite_detection_threshold_is_refused(threshold):
    with pytest.raises(ValueError, match="threshold"):
        injection.detection_rates(
            {"n_valid": 2, "f": np.array([1., 2.])},
            {"ratio": {"0.5": threshold}}, [0.5], statistics={"ratio": "f"},
        )
