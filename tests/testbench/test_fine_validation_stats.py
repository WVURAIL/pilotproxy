from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.stats import binom

from pilot_proxy.testbench.fine_validation_stats import (
    ALWAYS_MASKED_Q16,
    MAX_Q16,
    binomial_interval,
    empirical_null_threshold,
    exact_q16_null_threshold,
    false_alarm_validation,
    raw_crossing_brackets,
)


@pytest.mark.parametrize('n', [1, 5, 100, 5000])
def test_exact_binomial_zero_all_closed_forms(n):
    empty = binomial_interval(0, n, side='upper')
    full = binomial_interval(n, n, side='lower')
    assert empty['lower'] == 0
    assert empty['upper'] == pytest.approx(-math.expm1(math.log(.05)/n))
    assert full['upper'] == 1
    assert full['lower'] == pytest.approx(.05**(1/n))
    assert binomial_interval(0, n)['upper'] == pytest.approx(1-.025**(1/n))


@pytest.mark.parametrize('k,n', [(1, 8), (4, 17), (97, 100), (50, 5000)])
def test_binomial_endpoints_invert_binomial_tails(k, n):
    ci = binomial_interval(k, n)
    assert binom.sf(k-1, n, ci['lower']) == pytest.approx(.025, rel=1e-9)
    assert binom.cdf(k, n, ci['upper']) == pytest.approx(.025, rel=1e-9)
    assert ci['lower'] < k/n < ci['upper']


@pytest.mark.parametrize('args', [(True, 2), (1, 0), (-1, 8), (9, 8), (1.0, 2)])
def test_binomial_rejects_invalid_counts(args):
    with pytest.raises((ValueError, TypeError)):
        binomial_interval(*args)


def test_family_cap_is_separate_from_nominal_and_pointwise_precision():
    single = false_alarm_validation(50, 5000, nominal_pfa=.01, cap_pfa=.02)
    family = false_alarm_validation(50, 5000, nominal_pfa=.01, cap_pfa=.02, family_tests=184)
    assert family['simultaneous_upper']['upper'] > single['simultaneous_upper']['upper']
    assert family['cap_demonstrated']
    assert family['nominal_pfa'] == .01 and family['cap_pfa'] == .02
    assert family['pointwise'] == single['pointwise']
    assert not family['physical_acceptance']
    assert not false_alarm_validation(200, 5000, nominal_pfa=.01, cap_pfa=.02)['cap_demonstrated']


@pytest.mark.parametrize('n,p', [(10, Fraction(1, 10)), (2000, Fraction(1, 100)), (2000, Fraction(1, 20)), (5, Fraction(1, 3))])
def test_float_higher_threshold_matches_observed_order_and_keeps_ties(n, p):
    values = np.arange(n, dtype=float) / 10
    before = values.copy()
    result = empirical_null_threshold(values, pfa=p)
    exact_index = ((1-p).numerator*(n-1)+(1-p).denominator-1)//(1-p).denominator
    assert result['index'] == exact_index
    assert result['threshold'] == values[exact_index]
    assert result['exceedances'] == n-exact_index-1
    np.testing.assert_array_equal(values, before)
    repeated = empirical_null_threshold([1., 1., 1., 2., 2.], pfa=.4)
    assert repeated['threshold'] == 2 and repeated['exceedances'] == 0


@pytest.mark.parametrize('bad', [[1, np.inf], [1, np.nan], [], [-1, 1], [True, 1], [1+2j]])
def test_null_calibration_never_drops_or_coerces_invalids(bad):
    with pytest.raises((ValueError, TypeError)):
        empirical_null_threshold(bad, pfa=.01)


def test_q16_threshold_preserves_adjacent_uint64_maxima_and_sentinel():
    values = np.asarray([MAX_Q16-2, MAX_Q16-1, MAX_Q16], dtype=np.uint64)
    selected = exact_q16_null_threshold(values, pfa=Fraction(1, 2))
    assert selected['threshold'] == MAX_Q16-1
    assert selected['exceedances'] == 1
    unavailable = exact_q16_null_threshold([1, 2, ALWAYS_MASKED_Q16], pfa=.1)
    assert unavailable['status'] == 'unavailable'
    assert unavailable['threshold'] is None
    assert unavailable['selected_requirement'] == ALWAYS_MASKED_Q16
    assert unavailable['trials'] == 3 and unavailable['sentinel_trials'] == 1
    kept_sentinel = exact_q16_null_threshold([1, 2, 3, ALWAYS_MASKED_Q16], pfa=.5)
    assert kept_sentinel['threshold'] == 3 and kept_sentinel['exceedances'] == 1


@pytest.mark.parametrize('bad', [[0], [False], [2.0], [ALWAYS_MASKED_Q16+1], []])
def test_q16_refuses_undecoded_zero_or_noninteger_requirement(bad):
    with pytest.raises((TypeError, ValueError)):
        exact_q16_null_threshold(bad, pfa=.01)


def test_raw_crossing_preserves_downward_fluctuation_without_cummax():
    result = raw_crossing_brackets([0, 1, 2, 3], [.1, .4, .2, .8], target=.5)
    assert result['status'] == 'unique'
    assert result['upward_brackets'][0]['indices'] == [2, 3]
    assert result['estimate_db'] == pytest.approx(2.5)
    assert result['raw_rates'] == [.1, .4, .2, .8]


@pytest.mark.parametrize('rates', [[.1, .9, .1, .9], [.9, .1, .9, 1], [.1, .5, .5, .9], [.1, .5, .1, .9]])
def test_multiple_crossings_plateau_or_touch_are_ambiguous(rates):
    result = raw_crossing_brackets(range(len(rates)), rates, target=.5)
    assert result['status'] == 'ambiguous'
    assert result['estimate_db'] is None


@pytest.mark.parametrize('rates', [[.1, .2], [.8, .9], [.5, .9]])
def test_unbracketed_crossing_has_no_extrapolated_estimate(rates):
    result = raw_crossing_brackets([0, 1], rates, target=.5)
    assert result['status'] == 'unbracketed'
    assert result['estimate_db'] is None


def test_exact_target_single_hit_is_one_bracket():
    result = raw_crossing_brackets([0, 1, 2], [.1, .5, .9], target=.5)
    assert result['status'] == 'unique' and result['estimate_db'] == 1
    assert result['exact_target_indices'] == [1]


@pytest.mark.parametrize('x,y', [([1, 0], [.1, .9]), ([0, 0], [.1, .9]), ([0, 1], [.1, np.nan]), ([0, 1], [-.1, .9]), ([0], [.5])])
def test_crossing_invalid_grid_refuses(x, y):
    with pytest.raises(ValueError):
        raw_crossing_brackets(x, y, target=.5)
