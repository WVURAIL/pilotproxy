"""Independent event-copy oracle and failure cases for joint upper bands."""
from dataclasses import replace

import numpy as np
import pytest

from pilot_proxy.characterization.confidence_band import simultaneous_upper_band
from pilot_proxy.characterization.threshold_rule import ChannelFrames


def frames_fixture(events=12):
    size = np.arange(events) % 4 + 2
    event = np.repeat(np.arange(events), size)
    within = np.concatenate([np.arange(n) for n in size])
    required = np.asarray(1 + 2 * within, dtype=np.uint64)
    # Asymmetry is intentional: using the wrong sign need not give the same band.
    excess = .2 * event + .1 * within + .004 * event**3
    weight = 1. / (events * size[event])
    return ChannelFrames(99, required, weight, np.ones(len(event), bool), event, excess, 4, 4)


def direct_moments(frames, words, masks, labels, multiplicity):
    ids, cluster = np.unique(labels, return_inverse=True)
    result = {}
    for name, mask in masks.items():
        means, errors = [], []
        for word in words:
            keep = frames.supported & mask & (frames.required <= word)
            a = np.bincount(cluster[keep], weights=frames.weight[keep] * frames.excess[keep], minlength=len(ids))
            b = np.bincount(cluster[keep], weights=frames.weight[keep], minlength=len(ids))
            denominator = np.dot(multiplicity, b)
            if denominator == 0:
                means.append(np.nan); errors.append(np.nan)
            else:
                mean = np.dot(multiplicity, a) / denominator
                # Each sampled copy is one cluster: multiplicity, not its square.
                error = np.sqrt(np.dot(multiplicity, (a - mean * b)**2)) / denominator
                means.append(mean); errors.append(error)
        result[name] = (np.asarray(means), np.asarray(errors))
    return result


def direct_bootstrap(frames, words, masks, labels, seed, replicates, eligible):
    clusters = len(np.unique(labels))
    original = direct_moments(frames, words, masks, labels, np.ones(clusters))
    draws = np.random.default_rng(seed).multinomial(clusters, np.full(clusters, 1/clusters), size=replicates)
    pivots = []
    for draw in draws:
        boot = direct_moments(frames, words, masks, labels, draw)
        parts = []
        for name in masks:
            mean, se = boot[name]
            use = eligible[name]
            if np.any(~np.isfinite(mean[use]) | ~np.isfinite(se[use]) | (se[use] <= 1e-14)):
                parts.append(np.inf)
            elif use.any():
                parts.append(np.max((original[name][0][use] - mean[use]) / se[use]))
        pivots.append(max(parts))
    return original, np.asarray(pivots)


def run(frames, words=None, **kwargs):
    if words is None:
        words = np.array([2, 4, 8], dtype=np.uint64)
    settings = dict(bootstrap_replicates=79, seed=631, min_kept_events=2,
                    min_effective_events=1.01, confidence=.8, shelf=1.)
    settings.update(kwargs)
    return simultaneous_upper_band(frames, words, **settings)


def test_sparse_moments_and_every_pivot_match_independent_event_copy_oracle():
    f = frames_fixture()
    words = np.array([2, 4, 8], dtype=np.uint64)
    masks = {"full": np.ones(len(f.event), bool), "last": f.event >= 4}
    result = run(f, words, windows=masks)
    eligible = {name: row["eligible"] for name, row in result["windows"].items()}
    original, pivots = direct_bootstrap(f, words, masks, f.event, 631, 79, eligible)
    for name in masks:
        row = result["windows"][name]
        np.testing.assert_allclose(row["signed_mean"], original[name][0], rtol=2e-14)
        np.testing.assert_allclose(row["se"], original[name][1], rtol=2e-13)
    np.testing.assert_allclose(result["bootstrap_max_pivots"], pivots, rtol=2e-12, atol=2e-13)
    expected_critical = np.sort(pivots)[int(np.ceil(.8 * 78))]
    assert result["critical"] == pytest.approx(expected_critical, rel=2e-12)
    for name, row in result["windows"].items():
        np.testing.assert_allclose(row["signed_upper"], original[name][0] + expected_critical * original[name][1])


def test_upper_pivot_sign_is_not_reversed_for_skewed_event_population():
    f = frames_fixture(20)
    result = run(f, np.array([8], np.uint64), bootstrap_replicates=299, confidence=.95)
    masks = {"full": np.ones(len(f.event), bool)}
    _, upper_pivots = direct_bootstrap(f, result["words"], masks, f.event, 631, 299,
                                     {"full": np.array([True])})
    wrong = np.quantile(-upper_pivots, .95, method="higher")
    assert abs(result["critical"] - wrong) > .1
    assert result["critical"] == pytest.approx(np.quantile(upper_pivots, .95, method="higher"))


def test_overlapping_and_identical_windows_share_the_same_draw():
    f = frames_fixture(30)
    full = np.ones(len(f.event), bool)
    one = run(f)
    both = run(f, windows={"a": full, "b": full.copy()})
    np.testing.assert_array_equal(one["bootstrap_max_pivots"], both["bootstrap_max_pivots"])
    np.testing.assert_array_equal(both["windows"]["a"]["upper"], both["windows"]["b"]["upper"])
    assert both["eligible_family_size"] == 2 * one["eligible_family_size"]


def test_zero_kept_events_remain_resampling_units_and_failed_draws_are_not_discarded():
    f = frames_fixture(10)
    # Only two of ten events can survive the sole threshold. Some draws retain
    # no events or only one distinct contribution and must yield infinite pivots.
    required = np.where(f.event < 2, 1, 9).astype(np.uint64)
    f = replace(f, required=required)
    result = run(f, np.array([1], np.uint64), bootstrap_replicates=399, confidence=.95)
    assert result["resampling_cluster_count"] == 10
    assert result["windows"]["full"]["kept_events"][0] == 2
    assert result["diagnostics"]["completed_replicates"] == 399
    assert result["diagnostics"]["failed_replicates"] > 20
    assert np.isposinf(result["critical"])
    assert np.isposinf(result["windows"]["full"]["upper"][0])
    assert result["status"].startswith("unavailable:")


def test_support_is_frozen_and_ineligible_points_never_receive_a_finite_bound():
    f = frames_fixture(12)
    f = replace(f, required=np.where(f.event < 3, 1, 5).astype(np.uint64))
    result = run(f, np.array([1, 5], np.uint64), min_kept_events=5)
    row = result["windows"]["full"]
    np.testing.assert_array_equal(row["eligible"], [False, True])
    assert row["status"][0] == "insufficient_kept_events"
    assert np.isposinf(row["upper"][0]) and np.isfinite(row["upper"][1])
    assert result["eligible_family_size"] == 1


def test_effective_support_refuses_one_dominant_cluster():
    f = frames_fixture(40)
    weight = f.weight * np.where(f.event == 0, 10000., 1.)
    result = run(replace(f, weight=weight), min_effective_events=20.)
    assert not result["windows"]["full"]["eligible"].any()
    assert result["diagnostics"]["completed_replicates"] == 0
    assert np.isposinf(result["windows"]["full"]["upper"]).all()


def test_signed_values_survive_until_final_reporting_transform():
    f = frames_fixture(30)
    f = replace(f, excess=f.excess - 1000.)
    result = run(f)
    row = result["windows"]["full"]
    assert (row["signed_mean"] < 0).all() and (row["signed_upper"] < 0).all()
    np.testing.assert_array_equal(row["upper"], np.zeros(3))
    assert (row["se"] > 0).all()


def test_temporal_clusters_preserve_unequal_original_weight_and_match_oracle():
    f = frames_fixture(18)
    # Unequal numbers of original events in resampling blocks, with original
    # equal-event weights untouched.
    labels = np.where(f.event < 9, f.event // 3, 3 + (f.event - 9) // 2)
    result = run(f, cluster_labels=labels, bootstrap_replicates=99, confidence=.7)
    masks = {"full": np.ones(len(f.event), bool)}
    original, pivots = direct_bootstrap(f, result["words"], masks, labels, 631, 99,
                                     {"full": result["windows"]["full"]["eligible"]})
    assert result["original_event_count"] == 18
    assert result["resampling_cluster_count"] == len(np.unique(labels))
    np.testing.assert_allclose(result["windows"]["full"]["signed_mean"], original["full"][0])
    np.testing.assert_allclose(result["windows"]["full"]["se"], original["full"][1])
    np.testing.assert_allclose(result["bootstrap_max_pivots"], pivots, rtol=2e-12, atol=2e-13)


def test_seed_reproduces_results_independently_of_batch_size_and_weight_scale():
    f = frames_fixture(30)
    a = run(f, batch_size=1)
    b = run(f, batch_size=17)
    c = run(replace(f, weight=7 * f.weight), batch_size=3)
    np.testing.assert_array_equal(a["bootstrap_max_pivots"], b["bootstrap_max_pivots"])
    np.testing.assert_allclose(a["bootstrap_max_pivots"], c["bootstrap_max_pivots"], rtol=1e-12, atol=1e-12)
    assert not np.array_equal(a["bootstrap_max_pivots"], run(f, seed=632)["bootstrap_max_pivots"])


def test_large_common_shift_preserves_standard_error():
    f = frames_fixture(30)
    a = run(f)
    b = run(replace(f, excess=f.excess + 1e8))
    np.testing.assert_allclose(a["windows"]["full"]["se"], b["windows"]["full"]["se"], rtol=1e-8)
    np.testing.assert_allclose(a["windows"]["full"]["signed_mean"] + 1e8,
                               b["windows"]["full"]["signed_mean"], rtol=1e-15)


def test_cancellation_dominated_small_subset_is_refused_not_certified():
    f = frames_fixture(40)
    excess = np.where(f.required <= 2, 1e8 + 1e-5 * f.event, 2e8 + f.event)
    result = run(replace(f, excess=excess))
    row = result["windows"]["full"]
    assert not row["eligible"][0]
    assert np.isposinf(row["upper"][0])


def test_empty_and_constant_windows_are_explicitly_unsupported():
    f = frames_fixture(40)
    result = run(replace(f, excess=np.ones(len(f.event))),
                 windows={"empty": np.zeros(len(f.event), bool), "constant": np.ones(len(f.event), bool)})
    assert set(result["windows"]["empty"]["status"]) == {"no_kept_weight"}
    assert set(result["windows"]["constant"]["status"]) == {"zero_standard_error"}
    assert result["status"] == "unavailable: no eligible points"


@pytest.mark.parametrize("words", [np.array([1, 2]), np.array([0, 1], np.uint64),
                                  np.array([2, 1], np.uint64), np.array([1, 1], np.uint64),
                                  np.array([], np.uint64)])
def test_bad_grids_are_rejected(words):
    with pytest.raises((ValueError, TypeError)):
        run(frames_fixture(), words)


@pytest.mark.parametrize("option", [{"confidence": 1}, {"confidence": True}, {"bootstrap_replicates": 1},
                                   {"seed": -1}, {"batch_size": 0}, {"min_kept_events": 1},
                                   {"min_effective_events": 1}, {"shelf": -1}, {"windows": {}}])
def test_bad_options_are_rejected(option):
    with pytest.raises((ValueError, TypeError)):
        run(frames_fixture(), **option)


def test_invalid_arrays_and_split_events_are_not_silently_repaired():
    f = frames_fixture()
    for bad in [replace(f, weight=np.zeros(len(f.event))),
                replace(f, excess=np.full(len(f.event), np.nan)),
                replace(f, supported=np.ones(len(f.event), dtype=int)),
                replace(f, required=f.required.astype(float))]:
        with pytest.raises((ValueError, TypeError)):
            run(bad)
    with pytest.raises(ValueError, match="split an original event"):
        run(f, cluster_labels=np.arange(len(f.event)))
    with pytest.raises(TypeError):
        run(f, windows={"full": np.ones(len(f.event), dtype=int)})


def test_unsupported_frames_remain_in_original_event_population():
    f = frames_fixture(30)
    supported = f.event >= 5
    excess = np.where(supported, f.excess, np.nan)
    result = run(replace(f, supported=supported, excess=excess))
    assert result["resampling_cluster_count"] == 30
    np.testing.assert_array_equal(result["windows"]["full"]["kept_events"], [25, 25, 25])


def test_model_widening_preserves_raw_statistics_and_does_not_mutate_input():
    from pilot_proxy.characterization.confidence_band import widen_upper_band

    original = run(frames_fixture())
    widened = widen_upper_band(original, 2.)
    assert widened["critical"] == 2 * max(original["critical"], 1.)
    assert "model_widening" not in original
    np.testing.assert_array_equal(widened["bootstrap_max_pivots"], original["bootstrap_max_pivots"])
    for key, row in original["windows"].items():
        got = widened["windows"][key]
        for field in ("signed_mean", "se", "eligible", "kept_weight", "kept_events"):
            np.testing.assert_array_equal(got[field], row[field])
        use = row["eligible"]
        np.testing.assert_allclose(got["signed_upper"][use],
            row["signed_mean"][use] + widened["critical"] * row["se"][use])
        assert np.all(got["signed_upper"][use] >= row["signed_upper"][use])


def test_model_widening_refuses_overflow_and_narrowing():
    from pilot_proxy.characterization.confidence_band import widen_upper_band

    original = run(frames_fixture())
    original["critical"] = 1e308
    with pytest.raises(ValueError, match="not finite"):
        widen_upper_band(original, 10.)
    with pytest.raises(ValueError, match=">= 1"):
        widen_upper_band(original, .5)


def test_model_widening_floors_negative_critical_and_preserves_unavailability():
    from pilot_proxy.characterization.confidence_band import widen_upper_band

    original = run(frames_fixture())
    original["critical"] = -.5
    assert widen_upper_band(original, 2.)["critical"] == 2.
    original["status"] = "unavailable"
    original["critical"] = float("inf")
    widened = widen_upper_band(original, 2.)
    assert widened["status"] == "unavailable"
    assert np.isinf(widened["critical"])
