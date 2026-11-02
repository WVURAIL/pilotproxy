"""Exact sensitivity shards and binomial detection intervals."""

import math

import numpy as np
import pytest

from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    components_from_columns,
    exact_columns,
    exact_q16_decision,
    wilson_interval,
)


def test_exact_columns_preserve_integer_decisions_through_npz(tmp_path):
    maximum = (1 << 64) - 1
    components = [
        ExactResponseComponents((1 << 53) + 1, 1, 1 << 53, 1, 255, 2),
        ExactResponseComponents((1 << 63) + 3, maximum,
                                (1 << 63) + 1, maximum, 0, 254),
        ExactResponseComponents(maximum - 2, maximum,
                                maximum - 2, maximum, 128, 130),
        ExactResponseComponents(0, 0, 0, 0, -1, -1),
    ]
    expected = {
        "designated_num": [(1 << 53) + 1, (1 << 63) + 3, maximum - 2, 0],
        "designated_den": [1, maximum, maximum, 0],
        "rank_num": [1 << 53, (1 << 63) + 1, maximum - 2, 0],
        "rank_den": [1, maximum, maximum, 0],
        "designated_bin": [255, 0, 128, -1],
        "rank_bin": [2, 254, 130, -1],
    }
    columns = exact_columns(item for item in components)
    assert set(columns) == set(expected)
    for name, values in expected.items():
        dtype = np.int16 if name.endswith("_bin") else np.uint64
        assert columns[name].dtype == dtype
        np.testing.assert_array_equal(columns[name], np.asarray(values, dtype=dtype))

    path = tmp_path / "exact-shard.npz"
    np.savez(path, **columns)
    with np.load(path, allow_pickle=False) as archive:
        restored = components_from_columns(archive)
    assert restored == components
    assert all(type(value) is int for item in restored for value in vars(item).values())
    # The first two ratios exceed one by less than float64 can resolve.
    # Equality and the invalid sentinel must still produce no detection.
    assert [exact_q16_decision(item, multiplier_q16=65536)
            for item in restored] == [1, 1, 0, 0]


def test_empty_exact_columns_keep_the_shard_schema():
    columns = exact_columns(iter(()))
    assert set(columns) == {
        "designated_num", "designated_den", "rank_num", "rank_den",
        "designated_bin", "rank_bin",
    }
    for name, column in columns.items():
        assert column.shape == (0,)
        assert column.dtype == (np.int16 if name.endswith("_bin") else np.uint64)
    assert components_from_columns(columns) == []


@pytest.mark.parametrize("malformation", ["missing", "misaligned"])
def test_exact_shard_columns_require_complete_aligned_rows(malformation):
    columns = {
        "designated_num": [3, 7], "designated_den": [2, 4],
        "rank_num": [1, 5], "rank_den": [1, 3],
        "designated_bin": [0, 2], "rank_bin": [4, 6],
    }
    if malformation == "missing":
        del columns["rank_den"]
        with pytest.raises(KeyError, match="rank_den"):
            components_from_columns(columns)
    else:
        columns["rank_den"].pop()
        with pytest.raises(ValueError, match="equal lengths"):
            components_from_columns(columns)


def test_exact_columns_refuse_values_outside_the_shard_integer_range():
    with pytest.raises(OverflowError):
        exact_columns([ExactResponseComponents(1 << 64, 1, 1, 1, 0, 2)])


@pytest.mark.parametrize("trials", [1, 20, 100])
def test_wilson_endpoints_match_closed_forms(trials):
    zero = wilson_interval(0, trials, z=2)
    complete = wilson_interval(trials, trials, z=2)
    assert zero == pytest.approx((0, 4 / (trials + 4)), abs=1e-15)
    assert complete == pytest.approx((trials / (trials + 4), 1), abs=1e-15)


@pytest.mark.parametrize("successes", [1, 5, 10, 19])
def test_wilson_interior_bounds_invert_the_binomial_score(successes):
    trials, z = 20, 2
    lo, hi = wilson_interval(successes, trials, z=z)
    assert 0 < lo < successes / trials < hi < 1
    for bound in (lo, hi):
        score_squared = (successes - trials * bound) ** 2 / (
            trials * bound * (1 - bound))
        assert score_squared == pytest.approx(z * z, rel=2e-14)


def test_wilson_interval_is_undefined_without_trials():
    assert all(math.isnan(bound) for bound in wilson_interval(0, 0))
