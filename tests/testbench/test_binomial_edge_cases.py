"""Wilson intervals retain valid binomial inputs and representable tails."""

from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from pilot_proxy.testbench import evaluate_snr, sensitivity_study


@pytest.fixture(params=[evaluate_snr.wilson_interval, sensitivity_study.wilson_interval],
                ids=["snr", "sensitivity"])
def interval(request):
    return request.param


def _decimal_interval(successes, trials, z):
    with localcontext() as context:
        context.prec = 500
        n, k, critical = Decimal(int(trials)), Decimal(int(successes)), Decimal.from_float(float(z))
        p = k / n
        denominator = 1 + critical**2 / n
        center = (p + critical**2 / (2*n)) / denominator
        radius = critical * (p*(1-p)/n + critical**2/(4*n*n)).sqrt() / denominator
        return float(center-radius), float(center+radius)


@pytest.mark.parametrize("successes,trials,z", [
    (1, 10**12, 2), (1000, 10**14, 1.959963984540054),
    (1, 10**200, 2), (0, 10, 1e200), (10, 10, 1e100),
    (3, 10, 0), (np.int64(7), np.int64(20), np.float64(2)),
])
def test_interval_matches_high_precision_score_roots(interval, successes, trials, z):
    result = interval(successes, trials, z=z)
    expected = _decimal_interval(successes, trials, z)
    assert all(math.isfinite(value) for value in result)
    assert 0 <= result[0] <= successes / trials <= result[1] <= 1
    for observed, reference in zip(result, expected, strict=True):
        assert observed == pytest.approx(reference, rel=2e-14, abs=0)


@pytest.mark.parametrize("successes,trials", [
    (-1, 10), (11, 10), (0, -1), (1, 0),
    (1.5, 10), (1, 2.5), (True, 10), (1, True),
    (np.bool_(True), 10), (1, np.bool_(True)),
])
def test_invalid_counts_are_not_clipped_or_truncated(interval, successes, trials):
    with pytest.raises(ValueError, match="successes|trials"):
        interval(successes, trials)


@pytest.mark.parametrize("z", [-1, np.nan, np.inf, -np.inf])
def test_invalid_score_width_is_refused(interval, z):
    with pytest.raises(ValueError, match="z"):
        interval(2, 10, z=z)


def test_empty_binomial_sample_remains_undefined(interval):
    assert all(math.isnan(value) for value in interval(0, 0))
