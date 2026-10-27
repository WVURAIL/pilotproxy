"""The threshold rule: the sweep against direct computation, each branch of the
rule, the guard, and the inputs it reads."""
import json

import numpy as np
import pytest

from pilot_proxy.characterization import knee
from pilot_proxy.characterization import threshold_rule as t

needs_measurements = pytest.mark.skipif(t._STACK_RATIO_RECORD is None or t._GAIN is None, reason="withheld from the code-only cut (CHIME_DERIVED.md)")


def synthetic(seed=1, n=600, events=80):
    rng = np.random.default_rng(seed)
    event = rng.integers(0, events, n)
    counts = np.bincount(event, minlength=events)
    weight = 1.0 / (events * counts[event])
    return t.ChannelFrames(channel=99, required=rng.integers(40000, 400000, n).astype(np.uint64), weight=weight,
                           supported=rng.random(n) > 0.05, event=event, excess=rng.normal(0.05, 0.4, n),
                           knee_word=200000, current_word=200000)


def direct(f, word):
    m = t.keep(f, word)
    w, q, e = f.weight[m], f.excess[m], f.event[m]
    mean = (w * q).sum() / w.sum()
    cluster = np.bincount(e, weights=w * (q - mean), minlength=f.event.max() + 1)
    return w.sum() / f.weight.sum(), mean, np.sqrt((cluster ** 2).sum()) / w.sum(), int(m.sum())


@needs_measurements
def test_constants():
    assert t.SHELF == pytest.approx(3051.7578125 / 6e6 * 10 ** 1.13, rel=1e-15)
    assert t.FRAME_SECONDS == pytest.approx(0.04194304)
    assert t.STACK_AMPLITUDE_RATIO == pytest.approx(np.median(list(t.STACK_AMPLITUDE_RATIO_BY_BAND.values())))
    assert t.STACK_AMPLITUDE_RATIO_SCALE == "calibrated" and 0.05 < t.STACK_AMPLITUDE_RATIO < 0.12
    assert t.CALENDAR_YEARS_PER_RETAINED_YEAR == 1.0
    assert t.LEGACY_CALENDAR_FACTOR == pytest.approx(6.574, abs=1e-3)


def test_the_packaged_table():
    table = t.load_table()
    assert [r["channel"] for r in table["channels"]] == list(range(14, 38))
    row = next(r for r in table["channels"] if r["channel"] == 19)
    assert t.response(table, 19) == abs(row["response_dv"])
    assert row["a_tol"] == pytest.approx(1 / t.response(table, 19))


def test_the_sweep_matches_direct_computation_at_every_word():
    f = synthetic()
    s = t.sweep(f)
    assert np.all(np.diff(s["word"].astype(np.int64)) > 0)
    for i, word in enumerate(s["word"]):
        kappa, mean, se, frames = direct(f, int(word))
        assert s["kappa"][i] == pytest.approx(kappa, rel=1e-12)
        assert s["mean_excess"][i] == pytest.approx(mean, rel=1e-9, abs=1e-12)
        # the running sums leave about 1e-11 of rounding where the true error is zero (one frame)
        assert s["se"][i] == pytest.approx(t.SHELF * se, rel=1e-6, abs=1e-9)
        assert s["kept_frames"][i] == frames
    assert s["x"].min() >= 0


def sweep_of(r_meas, r_high, kappa):
    """A sweep whose measured and upper-end bias / sigma are given, with R = 1."""
    words = np.arange(1, len(kappa) + 1, dtype=np.uint64) * 1000
    to_x = lambda r: np.sqrt(np.asarray(r, float) / t.FRAME_SAMPLES) / t.STACK_AMPLITUDE_RATIO  # noqa: E731
    x = to_x(r_meas)
    se = (to_x(r_high) - x) / t.GUARD_STANDARD_ERRORS
    n = len(kappa)
    return dict(word=words, kappa=np.asarray(kappa, float), x=x, se=se, mean_excess=x / t.SHELF,
                kept_frames=np.arange(1, n + 1), kept_events=np.arange(1, n + 1))


@needs_measurements
def test_a_band_whose_knee_passes_is_deployed_at_its_knee():
    s = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 0.7, 0.9], [0.1, 0.4, 0.8, 0.9])
    d = t.decide(s, knee_word=4000, R=1.0, gain=1.0)
    assert d["outcome"] == "useful" and d["deployed_word"] == 4000 and d["below_knee"] is None and d["reason"] is None
    assert d["deployed_at"] == "knee"
    assert d["knee"]["time_years"] == pytest.approx(t.CALENDAR_YEARS_PER_RETAINED_YEAR / 0.9)


@needs_measurements
def test_the_guard_can_fail_a_knee_the_measured_power_passes():
    s = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 0.7, 1.4], [0.1, 0.4, 0.8, 0.9])
    d = t.decide(s, knee_word=4000, R=1.0, gain=1.0)
    assert d["knee"]["r"] == pytest.approx(0.8) and d["knee"]["r_hi"] == pytest.approx(1.4)
    assert d["outcome"] == "useful" and d["deployed_at"] == "below the knee" and d["deployed_word"] == 3000


@needs_measurements
def test_a_failing_knee_tightens_to_the_passing_threshold_closest_to_it():
    # passes at words 1000 and 3000, fails at 2000 and at the knee, and no word above the knee is read
    s = sweep_of([0.1, 2.0, 0.3, 2.0, 0.1], [0.2, 3.0, 0.9, 3.0, 0.2], [0.1, 0.2, 0.4, 0.9, 0.95])
    d = t.decide(s, knee_word=4000, R=1.0, gain=1.0)
    assert d["outcome"] == "useful" and d["deployed_word"] == 3000 and d["reason"] is None
    assert d["deployed"]["time_years"] == pytest.approx(t.CALENDAR_YEARS_PER_RETAINED_YEAR / 0.4)


@needs_measurements
def test_nothing_passes_leaves_no_evidence_below_the_knee():
    s = sweep_of([2.0, 3.0], [3.0, 4.0], [0.5, 0.9])
    d = t.decide(s, knee_word=2000, R=1.0, gain=1.0)
    assert d["outcome"] == "not useful" and d["below_knee"] is None and d["deployed_word"] is None
    assert d["reason"] == "above tolerance"


@needs_measurements
def test_an_excluded_band_is_not_deployed_even_when_its_knee_passes():
    s = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 0.7, 0.9], [0.1, 0.4, 0.8, 0.9])
    d = t.decide(s, knee_word=4000, R=1.0, gain=1.0, exclude="marker off the detector's frequency")
    assert d["outcome"] == "not useful" and d["deployed_word"] is None
    assert d["reason"] == "marker off the detector's frequency" and d["knee"]["r_hi"] == pytest.approx(0.9)


@needs_measurements
def test_the_coherence_limit():
    s = sweep_of([0.1, 0.25], [0.2, 0.5], [0.5, 0.9])
    d = t.decide(s, knee_word=2000, R=1.0, gain=1.0)
    assert d["knee"]["coherence_limit_s"] == pytest.approx(t.FRAME_SECONDS / 0.5)


@needs_measurements
def test_a_knee_beyond_the_time_limit_is_not_useful():
    s = sweep_of([0.1, 0.2], [0.2, 0.3], [0.01, 0.02])
    d = t.decide(s, knee_word=2000, R=1.0, gain=1.0)
    assert d["outcome"] == "not useful"
    assert t.decide(s, knee_word=2000, R=1.0, gain=1.0, max_years=1000)["outcome"] == "useful"


@needs_measurements
def test_evidence_below_the_knee_beyond_the_time_limit_is_still_reported():
    # below the knee the tolerance is met only at 2% of frames, which takes 50 years
    s = sweep_of([0.1, 2.0], [0.5, 3.0], [0.02, 0.9])
    d = t.decide(s, knee_word=2000, R=1.0, gain=1.0)
    assert d["outcome"] == "not useful" and d["below_knee"]["time_years"] == pytest.approx(t.CALENDAR_YEARS_PER_RETAINED_YEAR / 0.02)
    assert d["reason"] == "out of time" and d["deployed_word"] is None


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


def _knee(f, rule):
    return int(knee.propose([int(w) for w in f.required], [str(int(e)) for e in f.event],
                            valid=[bool(v) for v in f.supported], rule=rule)["knee_word_q16"])


def v1_knee(f):
    """The knee a cache records (rule v1). The rule reads the frames again with rule v2."""
    return _knee(f, knee.RULE_ID_V1)


def write_cache(tmp_path, f, current_keep, product_rows=None):
    n = len(f.required)
    rows = np.arange(n) if product_rows is None else product_rows
    product = tmp_path / "product.npz"
    # The excess is read from the frame powers, which are the all-bin sums of the
    # fine powers (kernel core 3.0.0). The stored v5 column holds decoy row-domain data.
    np.savez(product, normalized_pilot_excess=np.c_[f.excess, f.excess] + 0.5,
             fine_power_u64=fine_for_excess(f.excess),
             target_norm_sq=np.asarray([1], dtype=np.int64),
             reference_norm_sum_sq=np.asarray([2], dtype=np.int64))
    meta = {"original_product_identity": {"path": str(product), "bytes": product.stat().st_size},
            "knee_threshold": {"eta_q16": str(v1_knee(f))}, "current_threshold": {"eta_q16": str(f.current_word)}}
    (tmp_path / "ch99.json").write_text(json.dumps(meta))
    np.savez(tmp_path / "ch99.npz", source_row_index=rows, required_q16=f.required, probability_weight=f.weight,
             supported=f.supported, frame_event_index=f.event, current_keep=current_keep)


def test_loading_a_score_cache(tmp_path):
    f = synthetic()
    write_cache(tmp_path, f, t.keep(f, f.current_word))
    g = t.load_frames(99, tmp_path)
    fine = fine_for_excess(f.excess)
    from_fine = np.where(fine[:, 1, 1] > 0, fine[:, 0, 0].astype(np.float64) / REFERENCE_HALF - 1.0, np.nan)
    assert np.array_equal(g.excess, from_fine, equal_nan=True) and g.knee_word == _knee(f, knee.RULE_ID)
    np.testing.assert_allclose(g.excess, np.maximum(f.excess, -1.0), rtol=0, atol=1e-12)
    assert t.sweep(g)["kappa"][-1] == pytest.approx(f.weight[f.supported].sum() / f.weight.sum())


def test_a_cache_whose_mask_the_rule_does_not_reproduce_is_refused(tmp_path):
    f = synthetic()
    wrong = t.keep(f, f.current_word).copy()
    wrong[np.flatnonzero(wrong)[0]] = False
    write_cache(tmp_path, f, wrong)
    with pytest.raises(ValueError, match="accepted mask"):
        t.load_frames(99, tmp_path)



def test_the_admissible_minimum_gain_matches_the_dump_lengths():
    assert round(t.minimum_family_gain(4), 2) == 6.75
    assert round(t.minimum_family_gain(11), 2) == 15.23
    assert round(t.minimum_family_gain(33), 2) == 40.00


def test_the_gain_is_the_sum_over_every_lag():
    d = np.arange(1, 200001)
    for tau in (0.3, 1.7, 5.0, 12.4):
        assert t.gain_of(tau, "exp") == pytest.approx(1 + 2 * np.exp(-d / tau).sum(), rel=1e-12)
        assert t.gain_of(tau, "tri") == pytest.approx(1 + 2 * np.clip(1 - d / tau, 0, None).sum(), rel=1e-12)


@needs_measurements
def test_every_band_is_priced_by_the_dumps_that_measured_the_coherent_fraction():
    assert t.GAIN_FRAMES == 33
    assert t.temporal_gain(18) == t.temporal_gain(20) == t.minimum_family_gain(33)


@needs_measurements
def test_the_bias_scales_with_the_gain():
    s = sweep_of([0.5, 0.9], [0.1, 0.2], [0.2, 0.3])
    r1, _ = t.bias_over_sigma(s, 1.0, gain=1.0)
    r8, _ = t.bias_over_sigma(s, 1.0, gain=8.0)
    assert np.allclose(r8, 8 * r1)



# the last-year guard
PASS = ([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 0.7, 0.9], [0.1, 0.4, 0.8, 0.9])   # deployed at its knee, 4000


@needs_measurements
def test_a_word_that_passes_on_the_last_year_is_unchanged():
    full = sweep_of(*PASS)
    last = sweep_of([0.3, 0.4, 0.6, 0.9], [0.4, 0.5, 0.8, 0.99], [0.1, 0.3, 0.7, 0.9])   # a different knee would not matter
    d = t.decide_guarded(full, 4000, last, R=1.0, gain=1.0)
    assert not d["guard_acted"] and d["last_year_r_hi"] == pytest.approx(0.99)
    alone = t.decide(full, 4000, 1.0, gain=1.0)
    assert {k: d[k] for k in alone} == alone and d["full_era_word"] == 4000 and d["deployed_at"] == "knee"


@needs_measurements
def test_a_last_year_failure_moves_to_the_loosest_word_passing_on_both():
    full = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 1.2, 0.7, 0.9], [0.4, 0.5, 0.8, 0.9])   # 2000 fails on the full era
    last = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 1.5, 1.5], [0.2, 0.5, 0.8, 0.9])   # 3000 and 4000 fail last year
    d = t.decide_guarded(full, 4000, last, R=1.0, gain=1.0)
    assert d["guard_acted"] and d["last_year_r_hi"] == pytest.approx(1.5) and d["full_era_word"] == 4000
    assert d["deployed_word"] == 1000 and d["deployed_at"] == "below the knee" and d["outcome"] == "useful"
    assert d["deployed"]["kappa"] == 0.4 and d["deployed"]["r_hi"] == pytest.approx(0.2)   # read on the full era


@needs_measurements
def test_a_word_last_year_rejects_everywhere_is_not_deployed():
    last = sweep_of([2.0, 3.0, 3.0, 3.0], [3.0, 4.0, 4.0, 4.0], [0.2, 0.5, 0.8, 0.9])
    d = t.decide_guarded(sweep_of(*PASS), 4000, last, R=1.0, gain=1.0)
    assert d["guard_acted"] and d["outcome"] == "not useful" and d["deployed_word"] is None
    assert d["reason"] == t.GUARD_FAILED == "fails the tolerance on its last year" and d["full_era_word"] == 4000


@needs_measurements
def test_the_guard_s_word_must_meet_the_time_limit_on_the_full_era():
    full = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 0.7, 0.9], [0.01, 0.4, 0.8, 0.9])   # 1000 needs 100 ideal years
    last = sweep_of([0.1, 0.2, 0.5, 0.8], [0.2, 1.3, 1.5, 1.5], [0.2, 0.5, 0.8, 0.9])
    d = t.decide_guarded(full, 4000, last, R=1.0, gain=1.0)
    assert d["guard_acted"] and d["reason"] == t.GUARD_FAILED and d["deployed"] is None


@needs_measurements
@pytest.mark.parametrize("full,exclude,reason", [
    (sweep_of([2.0, 3.0], [3.0, 4.0], [0.5, 0.9]), None, "above tolerance"),
    (sweep_of([0.1, 2.0], [0.5, 3.0], [0.02, 0.9]), None, "out of time"),
    (sweep_of(*PASS), "marker off the detector's frequency", "marker off the detector's frequency")])
def test_a_band_the_full_era_rejects_keeps_its_decision(full, exclude, reason):
    knee_word = int(full["word"][-1])
    alone = t.decide(full, knee_word, 1.0, gain=1.0, exclude=exclude)
    last = sweep_of([2.0, 3.0], [3.0, 4.0], [0.5, 0.9])
    d = t.decide_guarded(full, knee_word, last, R=1.0, gain=1.0, exclude=exclude)
    assert d["reason"] == reason and d["deployed"] is None and not d["guard_acted"] and d["last_year_r_hi"] is None
    assert {k: d[k] for k in alone} == alone


def test_the_last_year_is_its_frames_with_the_era_s_knee():
    f = synthetic()
    f.time = np.linspace(0, 3 * 365.25 * 86400, len(f.required))
    y = t.last_year(f)
    sel = f.time >= f.time[-1] - 365.25 * 86400
    assert np.array_equal(y.required, f.required[sel]) and np.array_equal(y.weight, f.weight[sel])
    assert np.array_equal(y.excess, f.excess[sel]) and y.knee_word == f.knee_word
    assert t.sweep(y)["kappa"][-1] == pytest.approx(f.weight[sel & f.supported].sum() / f.weight[sel].sum())


def test_the_last_year_needs_frame_times():
    with pytest.raises(ValueError, match="frame_time"):
        t.last_year(synthetic())


@needs_measurements
def test_a_finite_budget_tightens_and_one_changes_nothing():
    f = synthetic()
    s = t.sweep(f)
    R = 1.0 / float(np.nanmax(t.bias_over_sigma(s, 1.0, 1.0)[1]))     # the largest r_hi of the sweep is 1
    full = t.decide_guarded(s, f.knee_word, s, R, gain=1.0)
    same = t.decide_guarded(s, f.knee_word, s, R, gain=1.0, budget=1.0)
    assert full == same
    tight = t.decide_guarded(s, f.knee_word, s, R, gain=1.0, budget=0.5)
    if tight["deployed"] is not None:
        assert tight["deployed"]["r_hi"] <= 0.5 and tight["deployed_word"] <= (full["deployed_word"] or 2**63)


@needs_measurements
def test_calendar_cost_and_historical_duty_are_explicit():
    full = sweep_of([0.1, 0.2], [0.2, 0.3], [0.1, 0.5])
    ideal = t.decide_guarded(full, 2000, full, 1.0, gain=1.0)
    historical = t.decide_guarded(full, 2000, full, 1.0, gain=1.0,
                                  calendar_factor=t.LEGACY_CALENDAR_FACTOR)
    assert ideal["deployed"]["time_years"] == 2.0
    assert historical["deployed_word"] == ideal["deployed_word"]
    assert historical["deployed"]["time_years"] == pytest.approx(2 * t.LEGACY_CALENDAR_FACTOR)
    # With only 10% of frames kept, the calendar time under the historical duty exceeds the 20-year limit.
    assert t.decide(full, 1000, 1.0, gain=1.0)["deployed_word"] == 1000
    assert t.decide(full, 1000, 1.0, gain=1.0,
                    calendar_factor=t.LEGACY_CALENDAR_FACTOR)["reason"] == "out of time"


@needs_measurements
def test_no_recent_supported_frames_refuses_the_candidate():
    frames = synthetic()
    frames.supported[:] = False
    empty = t.sweep(frames)
    assert empty["word"].dtype == np.uint64 and empty["word"].size == 0
    result = t.decide_guarded(sweep_of(*PASS), 4000, empty, 1.0, gain=1.0)
    assert result["reason"] == t.GUARD_FAILED and result["deployed_word"] is None
    assert result["guard_acted"] and result["last_year_r_hi"] is None
    assert t.decide(empty, 4000, 1.0, gain=1.0)["deployed_word"] is None


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_invalid_calendar_factor_is_refused(factor):
    with pytest.raises(ValueError, match="calendar_factor"):
        t.decide({}, 1, 1.0, gain=1.0, calendar_factor=factor)


@pytest.mark.parametrize("budget", [0, -1, 1.1, np.nan, np.inf])
def test_invalid_policy_budget_is_refused(budget):
    with pytest.raises(ValueError, match="budget"):
        t.decide({}, 1, 1.0, gain=1.0, budget=budget)
