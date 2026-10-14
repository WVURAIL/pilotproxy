# coding=utf-8
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.testbench.mask_frontier import (
    ALWAYS_MASKED_Q16,
    MIN_RETAINED_FRAMES,
    Q16_SCALE,
    bootstrap_frontier,
    candidate_multipliers_q16,
    coarse_normalized_excess,
    frontier_points,
    kept_at,
    kept_frame_means,
    measured_floor_db,
    minimum_mask_envelope,
    required_multipliers_by_rank,
    residual_score_histogram,
    shelf_db_from_excess,
    systematic_residuals,
)
from pilot_proxy.testbench.sensitivity_study import designated_bins


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    designated = designated_bins(64, 2)
    bulk = independent_bin_mask(
        256, pad_factor=2, designated_bins=designated, guard_fine_bins=1
    )
    return designated, bulk


def _powers(target: dict[int, int], reference: int) -> np.ndarray:
    powers = np.zeros((3, 256), dtype=np.uint64)
    powers[1, :] = reference
    powers[2, :] = reference
    for index, value in target.items():
        powers[0, index] = value
    return powers


def test_required_boundary_is_the_exact_upward_rounded_ratio() -> None:
    designated, bulk = _geometry()
    bulk_bins = np.flatnonzero(bulk)
    powers = _powers({int(designated[2]): 700}, reference=100)
    for offset, value in enumerate(bulk_bins):
        powers[0, int(value)] = 100 + offset
    boundaries = required_multipliers_by_rank(
        powers, designated=designated, bulk_mask=bulk
    )
    assert len(boundaries) == int(bulk.sum())
    # F2 = 2 p_target / (p_lower + p_upper); the designated maximum is 700.
    for rank_index, offset in enumerate(range(len(bulk_bins))):
        rank_target = 100 + offset
        exact = Fraction(700, rank_target) * Q16_SCALE
        assert boundaries[rank_index] == math.ceil(exact)


def test_a_dead_rank_is_always_masked_and_a_dead_designated_set_requires_nothing() -> None:
    designated, bulk = _geometry()
    bulk_bins = np.flatnonzero(bulk)
    powers = _powers({int(designated[0]): 500}, reference=100)
    powers[0, int(bulk_bins[0])] = 0
    for offset, value in enumerate(bulk_bins[1:], start=1):
        powers[0, int(value)] = 10 * offset
    boundaries = required_multipliers_by_rank(
        powers, designated=designated, bulk_mask=bulk
    )
    assert boundaries[0] == ALWAYS_MASKED_Q16

    quiet = _powers({}, reference=100)
    for offset, value in enumerate(bulk_bins):
        quiet[0, int(value)] = 10 + offset
    assert set(
        required_multipliers_by_rank(quiet, designated=designated, bulk_mask=bulk)
    ) == {1}


def test_the_bulk_mask_may_not_contain_a_designated_bin() -> None:
    designated, bulk = _geometry()
    contaminated = np.array(bulk, dtype=bool)
    contaminated[int(designated[0])] = True
    with pytest.raises(ValueError):
        required_multipliers_by_rank(
            _powers({int(designated[0]): 1}, reference=1),
            designated=designated,
            bulk_mask=contaminated,
        )


def test_keep_rule_is_inclusive_and_excludes_the_always_masked_sentinel() -> None:
    required = np.asarray([1, 100, 101, ALWAYS_MASKED_Q16], dtype=object)
    keep = kept_at(required, 101)
    assert list(keep) == [True, True, True, False]
    assert list(kept_at(required, ALWAYS_MASKED_Q16 - 1)) == [True, True, True, False]


def test_candidate_staircase_is_the_distinct_deployable_values_plus_the_floor() -> None:
    assert candidate_multipliers_q16([5, 5, 2, ALWAYS_MASKED_Q16]) == (1, 2, 5)
    with pytest.raises(ValueError):
        candidate_multipliers_q16([])


def test_normalized_excess_uses_the_exact_integer_marginals() -> None:
    excess = coarse_normalized_excess(
        2200, 1000, 1000, target_norm_sq=11, reference_norm_sum_sq=20
    )
    assert excess == pytest.approx(2200 * 20 / (2000 * 11) - 1.0)
    assert math.isnan(
        coarse_normalized_excess(1, 0, 0, target_norm_sq=1, reference_norm_sum_sq=1)
    )


def test_shelf_conversion_refuses_a_non_positive_excess() -> None:
    assert shelf_db_from_excess(1.0, offset_db=-3.5) == pytest.approx(-3.5)
    assert math.isnan(shelf_db_from_excess(-1e-6, offset_db=0.0))
    assert math.isnan(shelf_db_from_excess(float("nan"), offset_db=0.0))


def test_floor_is_a_percentile_of_the_off_population_and_refuses_a_small_one() -> None:
    shelf = np.linspace(-60.0, -40.0, 200)
    floor = measured_floor_db(shelf)
    assert floor["evidence"] == "measured"
    assert floor["floor_db"] == pytest.approx(float(np.percentile(shelf, 90.0)))
    assert measured_floor_db(shelf[:10])["evidence"] == "refused"
    assert math.isnan(measured_floor_db(shelf[:10])["floor_db"])


def test_no_frame_is_booked_below_the_floor() -> None:
    shelf = np.asarray([-30.0, -80.0, math.nan])
    residual = systematic_residuals(shelf, floor_linear=1e-5)
    assert residual[0] == pytest.approx(1e-3)
    assert residual[1] == pytest.approx(1e-5)
    assert residual[2] == pytest.approx(1e-5)


def _synthetic_frames(count: int = 200) -> tuple[list[int], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260907)
    required = [int(value) for value in rng.integers(Q16_SCALE, 8 * Q16_SCALE, count)]
    claim = rng.uniform(1e-5, 1e-2, count)
    truth = rng.uniform(0.0, 1e-2, count)
    return required, claim, truth


def test_prefix_sums_reproduce_the_direct_kept_frame_means() -> None:
    required, claim, truth = _synthetic_frames()
    candidates = candidate_multipliers_q16(required)
    histogram = residual_score_histogram(
        required, claim, truth, candidates=candidates, rho=7, bulk_size=124
    )
    assert histogram.frame_count == len(required)
    points = frontier_points(histogram)
    assert points
    for point in points:
        direct_claim = kept_frame_means(required, claim, eta_q16=point["eta_q16"])
        direct_truth = kept_frame_means(required, truth, eta_q16=point["eta_q16"])
        assert direct_claim["kept"] == point["kept"]
        assert point["r_sys"] == pytest.approx(direct_claim["mean"], rel=1e-12)
        assert point["r_injected"] == pytest.approx(direct_truth["mean"], rel=1e-12)
        assert point["masked_fraction"] == pytest.approx(
            1.0 - point["kept"] / point["frames"]
        )


def test_candidates_below_the_retained_floor_are_not_evaluated() -> None:
    required, claim, truth = _synthetic_frames()
    candidates = candidate_multipliers_q16(required)
    histogram = residual_score_histogram(
        required, claim, truth, candidates=candidates, rho=1, bulk_size=124
    )
    points = frontier_points(histogram)
    assert all(point["kept"] >= MIN_RETAINED_FRAMES for point in points)
    assert min(point["kept"] for point in points) >= MIN_RETAINED_FRAMES


def test_the_envelope_is_the_least_mask_inside_the_allowance() -> None:
    points = [
        {"rho": 1, "eta_q16": 10, "masked_fraction": 0.6, "r_sys": 1e-4},
        {"rho": 1, "eta_q16": 20, "masked_fraction": 0.3, "r_sys": 5e-4},
        {"rho": 1, "eta_q16": 30, "masked_fraction": 0.1, "r_sys": 5e-3},
    ]
    chosen = minimum_mask_envelope(points, 1e-3)
    assert chosen is not None and chosen["eta_q16"] == 20
    assert minimum_mask_envelope(points, 1e-9) is None


def test_bootstrap_intervals_bracket_the_full_sample_point() -> None:
    required, claim, truth = _synthetic_frames(400)
    candidates = candidate_multipliers_q16(required)
    histogram = residual_score_histogram(
        required, claim, truth, candidates=candidates, rho=3, bulk_size=124
    )
    points = {point["eta_q16"]: point for point in frontier_points(histogram)}
    intervals = bootstrap_frontier(
        required,
        claim,
        truth,
        candidates=candidates,
        rho=3,
        bulk_size=124,
        replicates=200,
        seed=11,
    )
    inside = 0
    for eta, point in points.items():
        low, high = intervals[eta]["r_sys"]
        inside += int(low <= point["r_sys"] <= high)
    assert inside >= 0.8 * len(points)


def test_the_histogram_refuses_a_negative_or_ragged_residual_column() -> None:
    required, claim, truth = _synthetic_frames(40)
    candidates = candidate_multipliers_q16(required)
    with pytest.raises(ValueError):
        residual_score_histogram(
            required, -claim, truth, candidates=candidates, rho=1, bulk_size=124
        )
    with pytest.raises(ValueError):
        residual_score_histogram(
            required, claim[:-1], truth, candidates=candidates, rho=1, bulk_size=124
        )
