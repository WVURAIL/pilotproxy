"""Fixed-rank tolerance limits and the separate null-sampling qualification."""

import numpy as np
import pytest
from scipy.stats import beta, binom

from pilot_proxy.testbench import fine_validation_stats as fa


IID = {"independent_null": True, "independence_source": "fixed iid test generator"}


def test_allowed_count_and_rank_at_two_hundred_thousand():
    result = fa.confidence_threshold(np.arange(200000), pfa=0.001, **IID)
    assert result["status"] == "available"
    assert result["allowed_exceedances"] == result["exceedances"] == 176
    assert result["rank"] == 199824 and result["index"] == 199823
    assert result["threshold"] == 199823
    assert result["pfa_upper"] == pytest.approx(0.000997126322601402)
    assert beta.ppf(0.95, 178, 200000 - 177) > 0.001
    assert binom.sf(176, 200000, 0.001) >= 0.95
    assert not result["independent_validation"]


def test_zero_count_feasibility_does_not_relax_project_support():
    assert fa.confidence_count_limit(2994, pfa=0.001) == -1
    assert fa.confidence_count_limit(2995, pfa=0.001) == 0
    result = fa.confidence_threshold(np.arange(2995), pfa=0.001, **IID)
    assert result["status"] == "unsupported: independent trials 2995 < 10000"
    assert result["threshold"] is None
    assert result["pfa_upper"] <= 0.001
    assert fa.confidence_threshold([1.0], pfa=0.001, **IID)["allowed_exceedances"] == -1


def test_q16_ties_keep_fixed_design_bound_and_smallest_integer():
    values = [65536] * 200000
    values[-177:] = [65538] * 2 + [65540] * 175
    result = fa.exact_q16_confidence_threshold(values, pfa=0.001, **IID)
    assert result["threshold"] == 65538
    assert result["exceedances"] == 175
    assert sum(v > result["threshold"] - 1 for v in values) == 177
    assert result["allowed_exceedances"] == 176
    assert result["pfa_upper"] == beta.ppf(0.95, 177, 200000 - 176)
    assert result["pfa_upper"] > beta.ppf(0.95, 176, 200000 - 175)


def test_integer_precision_and_sentinel_trial_accounting():
    values = [2**63 + 7] * 10000
    result = fa.exact_q16_confidence_threshold(values, pfa=0.001, **IID)
    assert result["threshold"] == 2**63 + 7 and isinstance(result["threshold"], int)
    k = result["allowed_exceedances"]
    values[-k:] = [fa.ALWAYS_MASKED_Q16] * k
    result = fa.exact_q16_confidence_threshold(values, pfa=0.001, **IID)
    assert result["status"] == "available" and result["trials"] == 10000
    assert result["sentinel_trials"] == result["exceedances"] == k
    values[-k - 1] = fa.ALWAYS_MASKED_Q16
    refused = fa.exact_q16_confidence_threshold(values, pfa=0.001, **IID)
    assert "always-masked sentinel" in refused["status"] and refused["threshold"] is None


def test_stronger_targets_and_confidence_only_raise_the_rank():
    loose = fa.confidence_count_limit(10000, pfa=0.01, confidence=0.9)
    assert fa.confidence_count_limit(10000, pfa=0.001, confidence=0.9) < loose
    assert fa.confidence_count_limit(10000, pfa=0.01, confidence=0.99) < loose


@pytest.mark.parametrize("n", [3, 5, 8])
def test_discrete_iid_coverage_by_exhaustive_binomial_counts(n):
    alpha, confidence, mass_at_one = 0.3, 0.5, 0.4
    k = fa.confidence_count_limit(n, pfa=alpha, confidence=confidence)
    if k < 0:
        return
    coverage = 0.0
    for ones in range(n + 1):
        result = fa.confidence_threshold([0.0] * (n - ones) + [1.0] * ones,
                                         pfa=alpha, confidence=confidence, **IID)
        selected = result["selected_score"]
        true_tail = mass_at_one if selected == 0 else 0.0
        coverage += binom.pmf(ones, n, mass_at_one) * (true_tail <= alpha)
    assert coverage >= confidence


def test_unknown_dependence_is_not_certified_by_raw_size():
    values = np.repeat(np.linspace(0, 1, 200), 1000)
    result = fa.confidence_threshold(values, pfa=0.001)
    assert result["status"] == "unavailable: independent null sampling not established"
    assert result["threshold"] is None
    assert 1 - 0.999**200 < 0.19  # actual coverage of this replicated-sample maximum


@pytest.mark.parametrize("values", [[1.0, np.nan], [1.0, np.inf], [True, 1.0], [[1.0]]])
def test_confidence_scores_are_not_silently_trimmed(values):
    with pytest.raises((TypeError, ValueError)):
        fa.confidence_threshold(values, pfa=0.001, **IID)


def test_qualification_requires_a_source_and_valid_design_parameters():
    with pytest.raises(ValueError, match="sampling source"):
        fa.confidence_threshold([1.0], pfa=0.001, independent_null=True)
    for n in (False, 20.5, 0):
        with pytest.raises((TypeError, ValueError)):
            fa.confidence_count_limit(n, pfa=0.001)
    for confidence in (True, 0, 1, np.nan):
        with pytest.raises((TypeError, ValueError)):
            fa.confidence_threshold([1.0], pfa=0.001, confidence=confidence)
