from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    MAX_MULTIPLIER_Q16,
    fine_mask_decision,
    fine_required_multiplier_q16,
)
from pilot_proxy.testbench.fine_validation_scores import (
    FineScoreGeometry,
    construct_geometry,
    decode_requirements,
    fixed_scores,
    float_scores,
)


@pytest.mark.parametrize("anchor", [0, 1, 14, 29, 198, 202, 254, 255])
def test_anchor_only_guard_and_quarter_ranks(anchor):
    geometry = construct_geometry(anchor)
    designated = {(anchor + k) % 256 for k in range(-2, 3)}
    expected = set(range(0, 256, 2)) - designated
    assert set(geometry.designated_bins) == designated
    assert set(np.flatnonzero(geometry.bulk_mask)) == expected
    assert geometry.n_bulk == (125 if anchor % 2 == 0 else 126)
    assert geometry.ranks.tolist() == ([31, 62, 93, 124] if anchor % 2 == 0 else [31, 62, 94, 125])
    assert not geometry.bulk_mask.flags.writeable


@pytest.mark.parametrize("anchor", [-1, 256, True, 1.5])
def test_invalid_anchor_refused(anchor):
    with pytest.raises((ValueError, TypeError)):
        construct_geometry(anchor)


def reference_kwargs(geometry, rank):
    return dict(anchor_bin=geometry.anchor_bin, designated_half_width=2,
                bulk_mask=geometry.bulk_mask, cfar_rank=int(rank))


@pytest.mark.parametrize("anchor", [0, 29, 202, 255])
def test_exact_scores_match_frozen_independent_rank_reference(anchor):
    rng = np.random.default_rng(7328 + anchor)
    # Wide uint64 inputs force sums/cross products beyond machine integers.
    powers = rng.integers(0, 1 << 63, (5, 3, 256), dtype=np.uint64)
    geometry = construct_geometry(anchor)
    result = fixed_scores(powers, geometry)
    logical = decode_requirements(result["required_q16"], result["always_masked"], result["valid"])
    for i, row in enumerate(powers):
        for j, rank in enumerate(geometry.ranks):
            kwargs = reference_kwargs(geometry, rank)
            reference = fine_required_multiplier_q16(row, **kwargs)
            assert result["valid"][i, j] == reference.valid
            assert logical[i, j] == reference.multiplier_q16
            boundary = int(logical[i, j])
            for threshold in {max(1, boundary - 1), boundary, min(MAX_MULTIPLIER_Q16, boundary + 1)}:
                decision = fine_mask_decision(row, multiplier_q16=threshold, **kwargs)
                assert decision.mask == int(threshold < boundary)


@pytest.mark.parametrize("boundary", [(1 << 53) + 1, (1 << 53) + 2, (1 << 64) - 2, (1 << 64) - 1])
def test_adjacent_large_exact_q16_boundaries_survive_float_aliasing(boundary):
    geometry = construct_geometry(29)
    row = np.ones((1, 3, 256), dtype=np.uint64)
    row[:, 0, :] = 65536
    row[0, 0, geometry.designated_bins] = boundary
    result = fixed_scores(row, geometry)
    assert result["required_q16"].tolist() == [[boundary] * 4]
    assert not result["always_masked"].any()
    for rank in geometry.ranks:
        kwargs = reference_kwargs(geometry, rank)
        assert fine_mask_decision(row[0], multiplier_q16=boundary - 1, **kwargs).mask == 1
        assert fine_mask_decision(row[0], multiplier_q16=boundary, **kwargs).mask == 0


def test_sentinel_and_invalid_are_distinct_and_never_dropped():
    geometry = construct_geometry(202)
    powers = np.ones((3, 3, 256), dtype=np.uint64)
    powers[0, 0, geometry.bulk_mask] = 0  # positive designated / zero bulk
    powers[1, 1:, geometry.bulk_mask] = 0  # no usable bulk bins
    powers[2, 0, geometry.bulk_mask] = 1
    powers[2, 0, geometry.designated_bins] = MAX_MULTIPLIER_Q16
    result = fixed_scores(powers, geometry)
    assert result["required_q16"].shape == (3, 4)
    assert result["always_masked"].tolist() == [[True] * 4, [False] * 4, [True] * 4]
    assert result["valid"].tolist() == [[True] * 4, [False] * 4, [True] * 4]
    assert np.all(result["required_q16"] == 0)
    assert np.isinf(result["fixed_float_response"][0]).all()
    assert np.isfinite(result["fixed_float_response"][2]).all()  # finite float can exceed the deployable Q16 range
    assert np.isnan(result["fixed_float_response"][1]).all()
    with pytest.raises(ValueError, match="invalid fine ranks"):
        decode_requirements(result["required_q16"], result["always_masked"], result["valid"])
    decoded = decode_requirements(result["required_q16"][[0, 2]], result["always_masked"][[0, 2]], result["valid"][[0, 2]])
    assert decoded.tolist() == [[ALWAYS_MASKED_Q16] * 4] * 2


def test_zero_designated_and_degenerate_references_obey_frozen_forced_keep():
    geometry = construct_geometry(0)
    powers = np.zeros((2, 3, 256), dtype=np.uint64)
    powers[:, 1:, geometry.bulk_mask] = 1
    powers[0, 0, geometry.designated_bins] = 10  # ignored: designated den=0
    powers[1, 1:, geometry.designated_bins] = 1  # valid 0/0 rank comparison
    result = fixed_scores(powers, geometry)
    assert result["valid"].all()
    assert np.all(result["required_q16"] == 1)
    assert np.all(result["fixed_float_response"] == 0)
    for row in powers:
        for rank in geometry.ranks:
            decision = fine_mask_decision(row, multiplier_q16=1, **reference_kwargs(geometry, rank))
            assert decision.valid and decision.mask == 0


def test_rank_is_not_recomputed_after_missing_reference_bins():
    geometry = construct_geometry(202)
    powers = np.ones((1, 3, 256), dtype=np.uint64)
    removed = np.flatnonzero(geometry.bulk_mask)[-2:]
    powers[0, 1:, removed] = 0
    result = fixed_scores(powers, geometry)
    assert result["n_bulk"].tolist() == [123]
    assert result["valid"].tolist() == [[True, True, True, False]]
    assert result["required_q16"].tolist() == [[65536, 65536, 65536, 0]]


def test_float_ratios_have_expected_order_statistics_and_scale_invariance():
    geometry = construct_geometry(255)
    powers = np.ones((2, 3, 256), dtype=float)
    bulk = np.flatnonzero(geometry.bulk_mask)
    powers[:, 0, bulk] = np.arange(1, len(bulk) + 1)
    powers[:, 0, geometry.designated_bins] = [2, 50, 200, 10, 1]
    powers[1] *= 13.0
    result = float_scores(powers, geometry)
    expected = 200 / (geometry.ranks + 1)
    np.testing.assert_allclose(result["response"], np.broadcast_to(expected, (2, 4)), rtol=2e-15)
    np.testing.assert_array_equal(result["rank_bin"][0], bulk[geometry.ranks])
    assert result["valid"].all()


def test_coarse_normalization_is_once_and_reference_sum_cannot_wrap():
    geometry = construct_geometry(202)
    fine = np.ones((2, 3, 256), dtype=np.uint64)
    maximum = np.iinfo(np.uint64).max
    coarse = np.array([[maximum, maximum, maximum], [1, 0, 0]], dtype=np.uint64)
    result = fixed_scores(fine, geometry, coarse_powers=coarse, mu0=2.0)
    np.testing.assert_array_equal(result["coarse_ratio"], [0.5, 0.0])
    np.testing.assert_array_equal(result["coarse_valid"], [True, False])
    assert np.all(result["fixed_float_response"] == 1)  # fine norm cancels


def test_extreme_finite_float_powers_do_not_overflow_reference_sum():
    geometry = construct_geometry(0)
    powers = np.full((1, 3, 256), 1e308)
    result = float_scores(powers, geometry, coarse_powers=powers[:, :, 0])
    assert np.all(result["response"] == 1)
    assert result["coarse_ratio"][0] == 1


def test_invalid_reference_cannot_sort_before_valid_infinite_float_ratio():
    geometry = construct_geometry(0)
    powers = np.ones((1, 3, 256))
    bulk = np.flatnonzero(geometry.bulk_mask)
    powers[0, 0, bulk] = 1e308
    powers[0, 1:, bulk] = 1e-308
    powers[0, 1:, bulk[0]] = 0
    result = float_scores(powers, geometry)
    assert result["valid"].tolist() == [[True, True, True, False]]
    assert all(b != bulk[0] for b in result["rank_bin"][0, :3])


def test_fraction_oracle_demonstrates_exact_ratio_not_float_rank_sort():
    geometry = construct_geometry(29)
    powers = np.ones((1, 3, 256), dtype=np.uint64)
    bulk = np.flatnonzero(geometry.bulk_mask)
    # All these adjacent values alias to the same float64; input bin order
    # intentionally runs opposite to exact rational order.
    powers[0, 0, bulk] = np.arange(len(bulk), dtype=np.uint64)[::-1] + (1 << 60)
    powers[0, 0, geometry.designated_bins] = (1 << 60) + 60
    result = fixed_scores(powers, geometry)
    for column, rank in enumerate(geometry.ranks):
        value = Fraction(((1 << 60) + 60) * 65536, (1 << 60) + int(rank))
        expected = -(-value.numerator // value.denominator)
        assert int(result["required_q16"][0, column]) == expected
        assert result["rank_bin"][0, column] == bulk[-1 - int(rank)]
    assert result["required_q16"].tolist() == [[65537, 65536, 65536, 65536]]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_bad_float_power_refused_without_changing_denominator(bad):
    powers = np.ones((2, 3, 256))
    powers[1, 0, 42] = bad
    with pytest.raises(ValueError, match="no rows are dropped"):
        float_scores(powers, construct_geometry(0))


def test_wrong_integer_representation_and_geometry_refused():
    with pytest.raises(TypeError, match="uint64"):
        fixed_scores(np.ones((1, 3, 256), dtype=np.int64), construct_geometry(0))
    original = construct_geometry(0)
    wrong = FineScoreGeometry(0, original.designated_bins, original.bulk_mask,
                              np.array([0, 1, 2, 3]), original.n_bulk)
    with pytest.raises(ValueError, match="geometry differs"):
        fixed_scores(np.ones((1, 3, 256), dtype=np.uint64), wrong)


@pytest.mark.parametrize("mu0", [0, -1, np.inf, np.nan, True])
def test_invalid_norm_refused_even_without_coarse_output(mu0):
    with pytest.raises((ValueError, TypeError)):
        float_scores(np.ones((1, 3, 256)), construct_geometry(0), mu0=mu0)


def test_decode_refuses_ambiguous_encoding_and_bad_flags():
    with pytest.raises(ValueError, match="sentinel"):
        decode_requirements(np.array([0], dtype=np.uint64), np.array([False]), np.array([True]))
    with pytest.raises(TypeError, match="Boolean"):
        decode_requirements(np.array([1], dtype=np.uint64), np.array([0]), np.array([True]))


def test_empty_batches_preserve_shapes():
    geometry = construct_geometry(0)
    result = fixed_scores(np.empty((0, 3, 256), dtype=np.uint64), geometry)
    assert result["required_q16"].shape == (0, 4)
    assert result["fixed_float_response"].shape == (0, 4)
