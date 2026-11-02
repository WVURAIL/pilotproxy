"""The reduced threshold rule, checked against independent event calculations."""
import json
from fractions import Fraction
import numpy as np
import pytest
from pilot_proxy.characterization import threshold_rule as t
from pilot_proxy.characterization.score_cache import sha
def synthetic(seed=1, n=600, events=80):
    rng = np.random.default_rng(seed)
    event = rng.integers(0, events, n)
    counts = np.bincount(event, minlength=events)
    weight = 1.0 / (events * counts[event])
    return t.ChannelFrames(channel=99, required=rng.integers(40000, 400000, n).astype(np.uint64), weight=weight,
                           supported=rng.random(n) > 0.05, event=event, excess=rng.normal(0.05, 0.4, n),
                           current_word=200000)


def direct(f, word):
    m = t.keep(f, word)
    w, q, e = f.weight[m], f.excess[m], f.event[m]
    mean = (w * q).sum() / w.sum()
    cluster = np.bincount(e, weights=w * (q - mean), minlength=f.event.max() + 1)
    return w.sum() / f.weight.sum(), mean, np.sqrt((cluster ** 2).sum()) / w.sum(), int(m.sum())


def fraction_summary(frames, word):
    """Evaluate the stated weighted mean and cluster equation in exact rationals."""
    total = sum(Fraction(float(w)) for w in frames.weight)
    clusters = {}
    count = 0
    for required, weight, supported, event, excess in zip(
            frames.required, frames.weight, frames.supported, frames.event, frames.excess):
        if not supported or int(required) > word:
            continue
        weight, excess = Fraction(float(weight)), Fraction(float(excess))
        a, b = clusters.get(int(event), (Fraction(0), Fraction(0)))
        clusters[int(event)] = (a + weight * excess, b + weight)
        count += 1
    retained = sum(b for a, b in clusters.values())
    mean = sum(a for a, b in clusters.values()) / retained
    variance = sum((a - mean * b)**2 for a, b in clusters.values()) / retained**2
    return float(retained / total), float(mean), np.sqrt(float(variance)), count, len(clusters)


def assert_fraction_sweep(frames):
    sweep = t.sweep(frames, shelf=1)
    for i, word in enumerate(sweep['word']):
        kappa, mean, error, count, events = fraction_summary(frames, int(word))
        assert sweep['kappa'][i] == pytest.approx(kappa, rel=3e-14)
        assert sweep['mean_excess'][i] == pytest.approx(mean, rel=3e-14, abs=3e-16)
        assert sweep['x'][i] == pytest.approx(max(mean, 0), rel=3e-14, abs=3e-16)
        assert sweep['se'][i] == pytest.approx(error, rel=2e-8, abs=1e-18)
        assert sweep['kept_frames'][i] == count
        assert sweep['kept_events'][i] == events
    return sweep


@pytest.mark.parametrize('excess', [
    [1, 1 + 2**-26],
    [-.1, -.1 + 1e-10],
    [1e8, 1e8 + 2**-20],
    [-.2, .1],
])
def test_clustered_error_retains_small_spread_and_clips_only_the_signed_mean(excess):
    frames = t.ChannelFrames(32, np.array([1000, 1000], dtype=np.uint64),
                             np.array([.5, .5]), np.ones(2, dtype=bool),
                             np.array([0, 1]), np.array(excess, dtype=float))
    assert_fraction_sweep(frames)


def test_sampling_margin_prevents_false_acceptance_of_nearly_equal_events():
    frames = t.ChannelFrames(32, np.array([1000, 1000], dtype=np.uint64),
                             np.array([.5, .5]), np.ones(2, dtype=bool),
                             np.array([0, 1]), np.array([1, 1 + 2**-26]))
    _, mean, error, _, _ = fraction_summary(frames, 1000)
    power_limit = (mean**2 + (mean + 2 * error)**2) / 2
    result = t.decide(t.sweep(frames, shelf=1), .1, power_scale=1, power_limit=power_limit)
    assert result['deployed_word'] is None


def test_sweep_preserves_original_weights_and_all_supported_change_points():
    frames = t.ChannelFrames(
        32, np.array([1, 1, 2**63, 2**63, 2**64 - 1, 7], dtype=np.uint64),
        np.array([1/3, 1/6, 1/6, 1/9, 1/9, 1/9]),
        np.array([True, True, True, True, True, False]),
        np.array([4, 100, 100, 2**40, 2**40, 2**40]),
        np.array([-.2, .1, -.1, .2, .3, np.nan]), current_word=1)
    sweep = assert_fraction_sweep(frames)
    np.testing.assert_array_equal(sweep['word'], np.array([1, 2**63, 2**64 - 1], dtype=np.uint64))
    assert sweep['kappa'][-1] == pytest.approx(8/9)


@pytest.mark.parametrize('seed', range(5))
def test_sweep_matches_fraction_equation_for_small_random_populations(seed):
    rng = np.random.default_rng(seed)
    events = np.tile(np.arange(5), 6)
    rng.shuffle(events)
    frames = t.ChannelFrames(32, rng.integers(1, 9, 30).astype(np.uint64),
                             np.full(30, 1/30), rng.random(30) > .2,
                             events, rng.normal(.01, .2, 30))
    assert_fraction_sweep(frames)


def test_clustered_error_is_invariant_to_a_common_excess_offset():
    values = np.array([0, 1, -1, 2, -2, 3], dtype=float) * 2**-25
    frames = t.ChannelFrames(32, np.array([1, 1, 2, 3, 4, 4], dtype=np.uint64),
                             np.full(6, 1/6), np.ones(6, dtype=bool),
                             np.array([0, 0, 1, 1, 2, 2]), values.copy())
    baseline = assert_fraction_sweep(frames)
    frames.excess = values + 1
    translated = assert_fraction_sweep(frames)
    np.testing.assert_allclose(translated['se'], baseline['se'], rtol=2e-8, atol=1e-18)


def test_sweep_preserves_small_cluster_spread_after_a_small_weight_outlier():
    count = 48
    frames = t.ChannelFrames(32, np.arange(1, count + 1, dtype=np.uint64),
                             np.r_[1e-12, np.full(count - 1, (1 - 1e-12)/(count - 1))],
                             np.ones(count, dtype=bool), np.arange(count),
                             np.r_[0, 1 + np.arange(count - 1) * 2**-35])
    assert_fraction_sweep(frames)


@pytest.mark.parametrize('single_event', [False, True])
def test_zero_cluster_variance_for_constant_excess_or_one_event(single_event):
    count = 20
    excess = np.linspace(-.2, .3, count) if single_event else np.full(count, .1)
    events = np.zeros(count, dtype=int) if single_event else np.arange(count)
    frames = t.ChannelFrames(32, np.arange(1, count + 1, dtype=np.uint64),
                             np.full(count, 1/count), np.ones(count, dtype=bool),
                             events, excess)
    sweep = assert_fraction_sweep(frames)
    np.testing.assert_array_equal(sweep['se'], np.zeros(count))


@pytest.mark.parametrize('values, relative_weights', [
    ([.1, .2], [1, 1]),
    ([-.2, .3], [1, 2]),
    ([.01, .07], [1, 11]),
    ([-.2, .3, .3], [1, 1, 1]),
])
def test_identical_cluster_means_have_zero_error_despite_within_event_variation(values, relative_weights):
    count = 12
    size = len(values)
    frames = t.ChannelFrames(
        32, np.repeat(np.arange(1, count + 1, dtype=np.uint64), size),
        np.tile(np.asarray(relative_weights) / (count * sum(relative_weights)), count),
        np.ones(count * size, dtype=bool), np.repeat(np.arange(count), size),
        np.tile(values, count))
    result = assert_fraction_sweep(frames)
    np.testing.assert_array_equal(result['se'], np.zeros(count))
    # A real difference must survive even when most event means are identical.
    frames.excess[-1] += 1e-13
    changed = assert_fraction_sweep(frames)
    assert changed['se'][-1] > 0


def test_clustered_error_preserves_a_representable_subnormal_result():
    smallest = np.nextafter(0.0, 1.0)
    frames = t.ChannelFrames(32, np.ones(2, dtype=np.uint64), np.full(2, .5),
                             np.ones(2, dtype=bool), np.arange(2),
                             np.array([0, 2 * smallest]))
    result = t.sweep(frames, shelf=1)
    # The exact error is smallest / sqrt(2), which rounds to smallest.
    assert result['mean_excess'][0] == smallest
    assert result['se'][0] == smallest


def test_clustered_error_avoids_overflow_of_the_squared_error():
    frames = t.ChannelFrames(32, np.ones(2, dtype=np.uint64), np.full(2, .5),
                             np.ones(2, dtype=bool), np.arange(2),
                             np.array([-1e300, 1e300]))
    result = t.sweep(frames, shelf=1)
    assert result['mean_excess'][0] == 0
    assert result['se'][0] == pytest.approx(1e300 / np.sqrt(2), rel=1e-15)


def test_clustered_error_preserves_finite_results_across_large_exponent_ranges():
    frames = t.ChannelFrames(32, np.ones(2, dtype=np.uint64), np.array([1e-300, 1e300]),
                             np.ones(2, dtype=bool), np.arange(2),
                             np.array([-1e300, 1e300]))
    result = t.sweep(frames, shelf=1)
    assert result['kappa'][0] == 1
    assert result['mean_excess'][0] == 1e300
    assert result['se'][0] == pytest.approx(2 * np.sqrt(2) * 1e-300, rel=1e-15, abs=0)


def test_the_sweep_matches_direct_computation_at_every_word():
    f = synthetic()
    s = t.sweep(f)
    assert np.all(np.diff(s["word"].astype(np.int64)) > 0)
    for i, word in enumerate(s["word"]):
        kappa, mean, se, frames = direct(f, int(word))
        assert s["kappa"][i] == pytest.approx(kappa, rel=1e-12)
        assert s["mean_excess"][i] == pytest.approx(mean, rel=1e-9, abs=1e-12)
        assert s["se"][i] == pytest.approx(t.SHELF * se, rel=1e-6, abs=1e-9)
        assert s["kept_frames"][i] == frames
    assert s["x"].min() >= 0


REFERENCE_HALF = 1 << 40


def fine_for_excess(excess):
    """Fine powers whose all-bin sums give ``excess`` (norms 1 : 2, so mu_0 = 1).

    With ``P_0 = round((e + 1) 2**40)`` and ``P_1 = P_2 = 2**40``, the excess formed
    from the sums is ``P_0 / 2**40 - 1``, which differs from ``e`` by at most ``2**-41``.
    """
    excess = np.asarray(excess, dtype=np.float64)
    finite = np.isfinite(excess)
    fine = np.zeros((len(excess), 3, 4), dtype=np.uint64)
    # A frame power is nonnegative, so an excess below -1 has no fine form.
    fine[finite, 0, 0] = np.rint(np.maximum(excess[finite] + 1.0, 0.0) * REFERENCE_HALF).astype(np.uint64)
    fine[finite, 1, 1] = REFERENCE_HALF  # a NaN excess is a zero reference power
    fine[finite, 2, 2] = REFERENCE_HALF
    return fine



def sweep_of(upper, kappa):
    n = len(kappa)
    return dict(word=np.arange(1, n + 1, dtype=np.uint64) * 1000,
                kappa=np.asarray(kappa), x=np.asarray(upper) / 2,
                se=np.asarray(upper) / 4, kept_frames=np.arange(2, n + 2),
                kept_events=np.full(n, 2))

def test_selects_maximum_retention_across_all_passing_words():
    s = sweep_of([.5, 2, .8, 3, .9], [.1, .2, .4, .8, .95])
    d = t.decide(s, 1, power_scale=1, power_limit=1)
    assert d["deployed_word"] == 5000
    assert d["deployed"]["kappa"] == .95

def test_no_lifetime_cutoff():
    d = t.decide(sweep_of([.5], [.001]), 1, power_scale=1, power_limit=1)
    assert d["deployed"]["time_years"] == 1000

def test_temperature_amplitude_is_squared():
    s = sweep_of([.8], [.5])
    assert t.decide(s, 1, power_scale=1, power_limit=1)["deployed_word"] == 1000
    assert t.decide(s, 1, power_scale=2, power_limit=1)["deployed_word"] is None

def test_exclusion_and_insufficient_clusters():
    s = sweep_of([.2], [.5])
    assert t.decide(s, 1, power_scale=1, power_limit=1, exclude="no matching pilot")["deployed_word"] is None
    s["kept_events"][:] = 1
    assert t.decide(s, 1, power_scale=1, power_limit=1)["deployed_word"] is None

@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_power_normalization(scale):
    with pytest.raises(ValueError, match="power_scale"):
        t.decide(sweep_of([.5], [.2]), 1, power_scale=scale, power_limit=1)

def test_no_supported_frames():
    f = synthetic()
    f.supported[:] = False
    assert t.decide(t.sweep(f), 1, power_scale=1, power_limit=1)["deployed_word"] is None

def test_product_power_and_exact_mask_readback(tmp_path):
    f = synthetic()
    product = tmp_path / "product.npz"
    # The excess is read from the frame powers, which are the all-bin sums of the
    # fine powers (kernel core 3.0.0). The stored v5 column holds decoy row-domain data.
    np.savez(product, normalized_pilot_excess=np.c_[f.excess, f.excess] + .5,
             fine_power_u64=fine_for_excess(f.excess), target_norm_sq=np.array([1]),
             reference_norm_sum_sq=np.array([2]))
    cache = tmp_path / "ch99.npz"
    np.savez(cache, source_row_index=np.arange(len(f.required)), required_q16=f.required,
             probability_weight=f.weight, supported=f.supported, frame_event_index=f.event,
             current_keep=t.keep(f, f.current_word))
    meta = dict(original_product_identity=dict(path=str(product), bytes=product.stat().st_size,
                                               sha256=sha(product)),
                npz_sha256=sha(cache), current_threshold=dict(eta_q16=str(f.current_word)))
    (tmp_path / "ch99.json").write_text(json.dumps(meta))
    g = t.load_frames(99, tmp_path)
    np.testing.assert_allclose(g.excess, np.maximum(f.excess, -1), rtol=0, atol=1e-12)
    meta["npz_sha256"] = "0" * 64
    (tmp_path / "ch99.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="cache"):
        t.load_frames(99, tmp_path)

def test_legacy_table_is_refused(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"schema": "rfisher-channel-table-v2"}))
    with pytest.raises(ValueError, match="schema v3"):
        t.load_table(p)


def test_packaged_table_preserves_channel_response_and_temperature_normalization(tmp_path):
    table = t.load_table()
    assert [row["channel"] for row in table["channels"]] == list(range(14, 38))
    for row in table["channels"]:
        normalization = row["normalization"]
        expected_scale = normalization["temperature_mK"] ** 2 / normalization["mode_volume"]
        expected_response = abs(row["bias_dv_per_unit_a"] / row["sigma_dv"])
        assert t.power_scale(table, row["channel"]) == pytest.approx(expected_scale, rel=1e-13)
        assert t.response(table, row["channel"]) == pytest.approx(expected_response, rel=1e-13)
    exported = tmp_path / "channel-table.json"
    exported.write_text(json.dumps(table))
    assert t.load_table(exported) == table


def test_finite_power_limit_overrides_a_passing_linear_bias():
    s = sweep_of([.3, .8], [.2, .9])
    result = t.decide(s, 1, power_scale=1, power_limit=.5)
    assert result["deployed_word"] == 1000
    assert result["deployed"]["a_hi"] == pytest.approx(.3**2)
    assert result["power_limit"] == .5


def test_power_limit_equality_is_allowed():
    s = sweep_of([.5], [.9])
    assert t.decide(s, 1, power_scale=1, power_limit=.25)["deployed_word"] == 1000


@pytest.mark.parametrize("limit", [0, -1, np.nan, np.inf])
def test_invalid_finite_power_limit(limit):
    with pytest.raises(ValueError, match="power_limit"):
        t.decide(sweep_of([.3], [.9]), 1, power_scale=1, power_limit=limit)


def test_finite_limit_is_required():
    with pytest.raises(TypeError, match="power_limit"):
        t.decide(sweep_of([.3], [.9]), 1, power_scale=1)


def test_doubling_the_proxy_quadruples_model_power():
    s = sweep_of([.25], [.5])
    before = t.bias_over_sigma(s, 1, power_scale=3)
    s["x"] *= 2
    s["se"] *= 2
    after = t.bias_over_sigma(s, 1, power_scale=3)
    np.testing.assert_allclose(after, np.asarray(before) * 4)


def test_channel_normalization_must_be_explicit():
    with pytest.raises(ValueError, match="normalization"):
        t.power_scale({"channels": [{"channel": 14}]}, 14)
