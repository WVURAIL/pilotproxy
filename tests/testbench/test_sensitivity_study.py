# coding=utf-8
from __future__ import annotations

import math

import numpy as np

from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
)
from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.fxfft import fine_power_fx, fine_power_ratio_fx
from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    canonical_seed,
    crossing_bracket,
    designated_bins,
    exact_q16_decision,
    exact_response_components,
    float_fine_power_ratio,
    float_fine_powers_by_stream,
    float_response_ratio,
    order_statistic_threshold,
    paired_crossing_bootstrap,
    q16_ceil_multiplier,
    stage_seed,
)


def test_common_random_seed_is_order_independent_and_omits_snr() -> None:
    a = stage_seed(
        20260820,
        purpose="h1",
        physical_channel=14,
        offset_fine_bins=0.5,
        trial_index=7,
    )
    b = stage_seed(
        20260820,
        purpose="h1",
        physical_channel=14,
        offset_fine_bins=0.5,
        trial_index=7,
    )
    assert a == b
    assert a != stage_seed(
        20260820,
        purpose="null",
        physical_channel=14,
        offset_fine_bins=0.5,
        trial_index=7,
    )
    assert canonical_seed(1, "x", 2) != canonical_seed(1, "x", 3)


def test_designated_bins_wrap_circularly() -> None:
    np.testing.assert_array_equal(
        designated_bins(0, 2), np.asarray([254, 255, 0, 1, 2])
    )


def test_exact_response_ratio_matches_float_view_of_same_powers() -> None:
    rng = np.random.default_rng(4)
    streams = 3
    rows = streams * 128
    packed = rng.integers(0, 256, size=(rows, 128), dtype=np.uint8).astype(np.int8)
    weights = rng.integers(0, 256, size=(3, 128), dtype=np.uint8).astype(np.int8)
    projections = matched_filter_row_projections_cpu_reference_packed(
        packed, weights, 4
    )
    powers = fine_power_fx(projections, num_streams=streams)
    f2 = fine_power_ratio_fx(powers)
    designated = designated_bins(0, 2)
    bulk = independent_bin_mask(256, designated_bins=designated)
    rank = int(np.count_nonzero(bulk) // 2)

    exact = exact_response_components(
        powers, designated=designated, bulk_mask=bulk, cfar_rank=rank
    )
    viewed = float_response_ratio(
        f2, designated=designated, bulk_mask=bulk, cfar_rank=rank
    )

    assert exact.valid
    assert exact.response_ratio() == viewed


def test_float_per_stream_powers_sum_to_existing_statistic() -> None:
    rng = np.random.default_rng(19)
    rows = rng.normal(size=(3 * 128, 8)) + 1j * rng.normal(size=(3 * 128, 8))
    weights = rng.normal(size=(3, 8)) + 1j * rng.normal(size=(3, 8))
    per_stream = float_fine_powers_by_stream(rows, weights, num_streams=3)
    summed = per_stream.sum(axis=1)
    den = summed[1] + summed[2]
    expected = np.divide(
        2.0 * summed[0], den, out=np.zeros(256), where=den > 0
    )

    assert per_stream.shape == (3, 3, 256)
    np.testing.assert_allclose(
        float_fine_power_ratio(rows, weights, num_streams=3), expected
    )


def test_empirical_threshold_and_q16_are_conservative() -> None:
    ratios = np.arange(1.0, 101.0)
    calibration = order_statistic_threshold(ratios, p_fa=0.1)
    assert calibration["threshold_multiplier"] == 90.0
    assert calibration["null_exceedances"] == 10
    q16 = q16_ceil_multiplier(1.000001)
    assert q16 / 65536.0 >= 1.000001

    components = ExactResponseComponents(
        designated_num=3,
        designated_den=2,
        rank_num=1,
        rank_den=1,
        designated_bin=0,
        rank_bin=4,
    )
    assert exact_q16_decision(components, multiplier_q16=65536) == 1
    assert exact_q16_decision(components, multiplier_q16=2 * 65536) == 0

    zero_rank = ExactResponseComponents(1, 1, 0, 1, 0, 4)
    assert zero_rank.valid
    assert math.isinf(zero_rank.response_ratio())
    assert exact_q16_decision(zero_rank, multiplier_q16=2 * 65536) == 1


def test_crossing_requires_an_observed_adjacent_bracket() -> None:
    result = crossing_bracket([-3, -2, -1], [0.1, 0.4, 0.8], target=0.5)
    assert result["bracketed"] is True
    assert -2.0 < result["estimate_db"] < -1.0
    missing = crossing_bracket([-3, -2], [0.1, 0.2], target=0.5)
    assert missing["bracketed"] is False
    assert missing["estimate_db"] is None


def test_paired_bootstrap_reports_fixed_minus_float_loss() -> None:
    # Deterministic toy curves with clear brackets.  Fixed responses are shifted
    # one grid point toward higher SNR relative to float responses.
    snrs = [-3.0, -2.0, -1.0, 0.0]
    n = 80
    float_null = np.linspace(0.8, 1.2, n)
    fixed_null = np.linspace(0.8, 1.2, n)
    float_h1 = {
        -3.0: np.full(n, 0.9),
        -2.0: np.r_[np.full(20, 0.9), np.full(60, 1.3)],
        -1.0: np.r_[np.full(5, 0.9), np.full(75, 1.3)],
        0.0: np.full(n, 1.3),
    }

    def component(value: float) -> ExactResponseComponents:
        # ratio = designated_num/rank_num with both denominators one
        scale = 1_000_000
        return ExactResponseComponents(
            int(round(value * scale)), 1, scale, 1, 0, 4
        )

    fixed_values = {
        -3.0: np.full(n, 0.9),
        -2.0: np.full(n, 0.9),
        -1.0: np.r_[np.full(20, 0.9), np.full(60, 1.3)],
        0.0: np.r_[np.full(5, 0.9), np.full(75, 1.3)],
    }
    fixed_h1 = {key: [component(v) for v in values] for key, values in fixed_values.items()}
    result = paired_crossing_bootstrap(
        snr_db=snrs,
        float_null_ratios=float_null,
        fixed_null_ratios=fixed_null,
        null_trial_keys=[("null", index) for index in range(n)],
        float_h1_ratios_by_snr=float_h1,
        fixed_h1_components_by_snr=fixed_h1,
        trial_keys_by_snr={
            snr: [("h1", index) for index in range(n)] for snr in snrs
        },
        p_fa=0.1,
        target_pd=0.5,
        replicates=50,
        seed=99,
    )
    assert result["valid_bracketed_replicates"] > 0
    assert result["fixed_minus_float_sensitivity_loss_median_db"] > 0.0
    assert math.isfinite(result["fixed_minus_float_sensitivity_loss_median_db"])
    assert result["null_pairing_mode"] == "explicit_trial_identities"
    assert result["h1_pairing_mode"].startswith("intersection_of_explicit")


def test_paired_bootstrap_aligns_shuffled_explicit_trial_identities() -> None:
    snrs = [-2.0, -1.0, 0.0]
    keys = [("h1", index) for index in range(12)]
    permutations = {
        -2.0: np.arange(12),
        -1.0: np.asarray([7, 2, 11, 0, 8, 5, 1, 10, 3, 9, 6, 4]),
        0.0: np.asarray([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
    }

    def component(value: float) -> ExactResponseComponents:
        return ExactResponseComponents(int(value * 100), 1, 100, 1, 0, 4)

    canonical = {
        -2.0: np.asarray([0.9] * 9 + [1.3] * 3),
        -1.0: np.asarray([0.9] * 5 + [1.3] * 7),
        0.0: np.asarray([0.9] * 1 + [1.3] * 11),
    }
    float_h1 = {
        snr: canonical[snr][permutations[snr]] for snr in snrs
    }
    fixed_h1 = {
        snr: [component(value) for value in canonical[snr][permutations[snr]]]
        for snr in snrs
    }
    trial_keys = {
        snr: [keys[int(index)] for index in permutations[snr]] for snr in snrs
    }
    null = np.linspace(0.8, 1.2, 24)
    result = paired_crossing_bootstrap(
        snr_db=snrs,
        float_null_ratios=null,
        fixed_null_ratios=null,
        null_trial_keys=[("null", index) for index in range(null.size)],
        float_h1_ratios_by_snr=float_h1,
        fixed_h1_components_by_snr=fixed_h1,
        trial_keys_by_snr=trial_keys,
        p_fa=0.1,
        target_pd=0.5,
        replicates=20,
        seed=3,
    )

    assert result["paired_h1_trials_per_snr"] == 12
    assert result["valid_bracketed_replicates"] > 0
    assert abs(result["fixed_minus_float_sensitivity_loss_median_db"]) < 1e-12
