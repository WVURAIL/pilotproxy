"""The score cache: exact weight, threshold, conditional-summary and missingness regressions."""
from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.characterization.score_cache import MAX_WORD, candidate_cache, event_weights


def fraction(value):
    return None if value is None else Fraction(int(value['numerator']), int(value['denominator']))


def candidate(rows, word):
    return next(row for row in rows if row['threshold'] == {'kind': 'finite', 'eta_q16': str(word)})


def test_equal_event_weights_and_conditional_mean_use_original_population():
    words = np.array([10, 20, 40, 100], np.uint64)
    inverse = np.array([0, 1, 1, 1])
    reference, rows, arrays, _ = candidate_cache(words, np.ones(4, bool), np.zeros(4, bool), inverse, 40, 20)
    assert fraction(reference['mean_word']) == Fraction(95, 3)
    row = candidate(rows, 40)
    assert fraction(row['retained_probability']) == Fraction(5, 6)
    assert fraction(row['conditional_mean_word']) == 18
    assert fraction(row['g']) == Fraction(54, 95)
    np.testing.assert_array_equal(arrays['weight_numerator'], [3, 1, 1, 1])
    assert arrays['weight_denominator_decimal'].item() == '6'
    # Re-equalizing surviving events would give 20; conditioning the original
    # event/frame measure gives 18 instead.
    assert fraction(row['conditional_mean_word']) != 20


def test_invalid_event_retains_mass_and_supported_reference_is_explicit():
    words = np.array([10, MAX_WORD, 30], np.uint64)
    rank = np.array([True, False, True])
    reference, rows, _, _ = candidate_cache(words, rank, np.zeros(3, bool), np.array([0, 1, 2]), MAX_WORD, 10)
    assert fraction(reference['supported_probability']) == Fraction(2, 3)
    assert fraction(reference['mean_word']) == 20
    endpoint = candidate(rows, MAX_WORD)
    assert fraction(endpoint['retained_probability']) == Fraction(2, 3)
    assert fraction(endpoint['unsupported_probability']) == Fraction(1, 3)
    assert fraction(endpoint['threshold_rejected_supported_probability']) == 0
    assert fraction(endpoint['g']) == 1


def test_exact_uint64_boundary_and_weighted_sum_do_not_overflow():
    words = np.array([2**63, 2**63+1, MAX_WORD], np.uint64)
    ref, rows, arrays, _ = candidate_cache(words, np.ones(3, bool), np.zeros(3, bool), np.arange(3), 2**63, 2**63+1)
    first = candidate(rows, 2**63)
    np.testing.assert_array_equal(arrays['candidate_keep'][first['mask_npz_row']], [True, False, False])
    assert fraction(first['conditional_mean_word']) == 2**63
    assert fraction(ref['mean_word']) == Fraction(2**65, 3)


def test_numeric_zero_and_permanent_mask_are_distinct_with_zero_scores():
    ref, rows, arrays, _ = candidate_cache(np.zeros(2, np.uint64), np.ones(2, bool), np.zeros(2, bool), np.arange(2), 0, 0)
    zero = candidate(rows, 0)
    permanent = next(row for row in rows if row['permanent_mask'])
    assert fraction(ref['mean_word']) == 0
    assert fraction(zero['retained_probability']) == 1
    assert zero['g'] is None and zero['transfer_status'] == 'unavailable_zero_reference_mean'
    assert fraction(permanent['retained_probability']) == 0
    assert permanent['conditional_mean_word'] is None and permanent['g'] is None
    assert not arrays['candidate_keep'][permanent['mask_npz_row']].any()


def test_no_supported_reference_never_fabricates_mean_or_g():
    ref, rows, _, _ = candidate_cache(np.array([1, 2], np.uint64), np.zeros(2, bool), np.zeros(2, bool), np.arange(2), 2, 1)
    assert ref['mean_word'] is None
    assert fraction(ref['supported_probability']) == 0
    assert all(row['g'] is None and row['conditional_mean_word'] is None for row in rows)
    assert all(row['transfer_status'] == 'unavailable_no_supported_reference' for row in rows)


def test_quantiles_use_event_weights_and_equality_not_float_word_rounding():
    ref, rows, arrays, _ = candidate_cache(np.array([10, 20, 30, 40], np.uint64), np.ones(4, bool), np.zeros(4, bool), np.array([0, 1, 1, 1]), 40, 30)
    median = next(row for row in rows if 'quantile_1_over_2' in row['roles'])
    assert median['threshold']['eta_q16'] == '10'
    assert fraction(median['retained_probability']) == Fraction(1, 2)
    assert len(rows) <= 26
    assert np.array_equal(arrays['candidate_keep'][-1], np.zeros(4, bool))
    assert fraction(ref['supported_probability']) == 1


def test_always_masked_word_is_not_admitted_at_maximum():
    _, rows, arrays, _ = candidate_cache(np.array([1, MAX_WORD], np.uint64), np.ones(2, bool), np.array([False, True]), np.arange(2), MAX_WORD, 1)
    endpoint = candidate(rows, MAX_WORD)
    np.testing.assert_array_equal(arrays['candidate_keep'][endpoint['mask_npz_row']], [True, False])


def test_large_lcm_weights_remain_exact_and_have_safe_string_storage():
    lengths = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    inverse = np.repeat(np.arange(len(lengths)), lengths)
    weights, denominator, counts = event_weights(inverse)
    assert sum(weights) == denominator and denominator > MAX_WORD
    for event in range(len(counts)):
        assert Fraction(int(sum(weights[inverse == event])), denominator) == Fraction(1, len(counts))
    _, rows, arrays, encoding = candidate_cache(np.ones(len(inverse), np.uint64), np.ones(len(inverse), bool), np.zeros(len(inverse), bool), inverse, 1, 1)
    assert encoding == 'decimal_strings'
    assert arrays['weight_numerator_decimal'].dtype.kind == 'U'
    assert fraction(candidate(rows, 1)['retained_probability']) == 1


@pytest.mark.parametrize('inverse', [np.array([]), np.array([-1]), np.array([0, 2]), np.array([0.0, 1.0])])
def test_invalid_event_accounting_refused(inverse):
    with pytest.raises(ValueError):
        event_weights(inverse)


@pytest.mark.parametrize('word', [True, 1.0, -1, 2**64])
def test_inexact_or_invalid_current_word_refused(word):
    with pytest.raises(ValueError, match='Exact finite'):
        candidate_cache(np.array([1], np.uint64), np.array([True]), np.array([False]), np.array([0]), word, 1)
