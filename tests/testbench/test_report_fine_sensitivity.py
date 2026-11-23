import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import report_fine_sensitivity as R
from pilot_proxy.testbench.fine_validation_stats import empirical_null_threshold, exact_q16_null_threshold


def test_multiplicities_reproduce_the_same_explicit_bootstrap_draws():
    counts = R.resampling(np.arange(7), np.arange(100, 111), replicates=20, seed=214)
    rng = np.random.default_rng(214)
    for i in range(20):
        np.testing.assert_array_equal(counts["null_counts"][i], np.bincount(rng.integers(7, size=7), minlength=7))
        np.testing.assert_array_equal(counts["h1_counts"][i], np.bincount(rng.integers(11, size=11), minlength=11))
    same = R.resampling(np.arange(7), np.arange(100, 111), replicates=20, seed=214)
    np.testing.assert_array_equal(counts["h1_counts"], same["h1_counts"])


@pytest.mark.parametrize("exact", [False, True])
def test_weighted_thresholds_and_rates_equal_expanded_multisets(exact):
    weights = R.resampling(np.arange(8), np.arange(100, 112), replicates=40, seed=9)
    null = np.array([7, 2, 2, 3, 8, 5, 1, 4], dtype=object if exact else float)
    sample = np.array([np.arange(1, 13), np.arange(3, 15)], dtype=object if exact else float)
    if exact:
        null[-1] = R.SENTINEL
        sample[0, 0] = R.SENTINEL
        null[0] = (1 << 63) + 1
    thresholds, available = R.weighted_thresholds(null, weights["null_counts"], pfa=.25, exact=exact)
    rates = R.bootstrap_rates(sample, thresholds, available, weights["h1_counts"], exact=exact, batch=7)
    calibrator = exact_q16_null_threshold if exact else empirical_null_threshold
    for b in range(40):
        expanded_null = np.repeat(null, weights["null_counts"][b])
        record = calibrator(expanded_null, pfa=.25)
        assert available[b] == (record["status"] == "available")
        if available[b]:
            assert thresholds[b] == record["threshold"]
            expected = np.mean(np.repeat(sample, weights["h1_counts"][b], axis=1) > record["threshold"], axis=1)
            np.testing.assert_array_equal(rates[b], expected)
        else:
            assert np.isnan(rates[b]).all()


def test_uint64_adjacent_threshold_ties_and_always_masked_decisions():
    samples = np.array([[2**63, 0], [2**63 + 1, 0]], dtype=np.uint64)
    valid = np.array([[True, False], [True, True]])
    always = np.array([[False, False], [False, True]])
    result = R.curve_summary(samples, valid, 2**63, [-50, -49], exact=True, always=always)
    assert result["counts"] == [1, 1]
    assert result["invalid_trials"] == [0, 1]
    assert result["trials"] == 2 and not result["response_qualified"]


def test_pairing_retains_identical_stage_zero_loss_and_censoring():
    snrs = np.arange(-3, 4, dtype=float)
    null = np.linspace(1, 2, 20)
    h1 = np.tile((snrs + 4)[:, None], (1, 32))
    weights = R.resampling(np.arange(20), np.arange(100, 132), replicates=30, seed=5)
    samples0, samples1, qualification, points = {}, {}, {}, {"0.05": {}}
    for stage in R.STAGES:
        exact = stage == R.Q16_STAGE
        samples0[stage] = np.array(np.ceil(null * 65536), dtype=object) if exact else null
        if exact:
            samples0[stage] = np.array([int(v) for v in samples0[stage]], dtype=object)
            samples1[stage] = np.array([[int(v) for v in row] for row in np.ceil(h1 * 65536)], dtype=object)
        else:
            samples1[stage] = h1
        qualification[stage] = True
        rates = np.mean(h1 > 2, axis=1)
        points["0.05"][stage] = {str(t): R.raw_crossing_brackets(snrs, rates, target=t) for t in (.5, .9)}
    arrays, summaries = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    assert np.all(arrays["loss_valid"])
    assert np.nanmax(np.abs(arrays["loss_db"])) == 0
    assert all(v["pointwise_percentile95_db"] == [0, 0] for v in summaries)
    qualification[R.STAGES[1]] = False
    arrays2, summaries2 = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    assert not arrays2["loss_valid"][:, :, 7].any()  # total ATSC-to-Q16
    total = [v for v in summaries2 if v["pair"] == "total_atsc_to_q16"]
    assert all(v["pointwise_percentile95_db"] is None and v["valid_replicates"] == 0 for v in total)


# Each stage's H1 curve is the stage-0 curve delayed by SHIFTS[stage] dB, so
# each pair's loss (right minus left) is the difference of its two shifts.
SHIFTS = (0, 1, 3, 7, 12, 20, 30)
PAIR_LOSS_DB = {"tone_to_atsc_model": 1, "input_quantization": 2, "weight_quantization": 6,
                "weights_after_input": 9, "input_after_weights": 5, "fixed_transform": 8,
                "q16_decision": 10, "total_atsc_to_q16": 29}


def test_pairing_reports_each_named_stage_contrast_with_its_sign():
    snrs = np.arange(-3, 34, dtype=float)
    null = np.linspace(1, 2, 20)
    weights = R.resampling(np.arange(20), np.arange(100, 132), replicates=30, seed=5)
    samples0, samples1, qualification, points = {}, {}, {}, {"0.05": {}}
    for stage, shift in zip(R.STAGES, SHIFTS):
        h1 = np.tile((snrs + 4 - shift)[:, None], (1, 32))
        if stage == R.Q16_STAGE:
            samples0[stage] = np.array([int(v) for v in np.ceil(null * 65536)], dtype=object)
            samples1[stage] = np.array([[int(v) for v in row] for row in np.ceil(np.maximum(h1, 1) * 65536)], dtype=object)
        else:
            samples0[stage], samples1[stage] = null, np.maximum(h1, 0)
        qualification[stage] = True
        rates = np.mean(h1 > 2, axis=1)
        points["0.05"][stage] = {str(t): R.raw_crossing_brackets(snrs, rates, target=t) for t in (.5, .9)}
    arrays, summaries = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    assert np.all(arrays["loss_valid"])
    names = [label for _, _, label in R.PAIRS]
    assert sorted(names) == sorted(PAIR_LOSS_DB)
    for pair, name in enumerate(names):
        np.testing.assert_allclose(arrays["loss_db"][:, :, pair], PAIR_LOSS_DB[name], rtol=0, atol=1e-9)
    for summary in summaries:
        assert summary["pointwise_percentile95_db"] == pytest.approx([PAIR_LOSS_DB[summary["pair"]]] * 2)
    qualification[R.STAGES[1]] = False
    arrays, _ = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    with_stage1 = [name in ("tone_to_atsc_model", "input_quantization", "weight_quantization",
                            "total_atsc_to_q16") for name in names]
    assert [not arrays["loss_valid"][:, :, pair].any() for pair in range(len(names))] == with_stage1
    assert [arrays["loss_valid"][:, :, pair].all() for pair in range(len(names))] == [not v for v in with_stage1]


def test_ambiguous_and_unbracketed_bootstrap_rows_remain_censored():
    values = np.array([[0, .8, .2, 1], [0, .1, .2, .3], [0, .2, .8, 1]])
    estimates, status = R.crossing_arrays([0, 1, 2, 3], values, .5)
    assert status.tolist() == [2, 1, 0]
    assert np.isnan(estimates[:2]).all() and estimates[2] == pytest.approx(1.5)


def test_raw_grid_discretization_is_outer_difference_not_point_difference():
    a = R.raw_crossing_brackets([-3, 0, 3], [0, .7, 1], target=.5)
    b = R.raw_crossing_brackets([-3, 0, 3], [0, .3, 1], target=.5)
    bound = R.loss_bounds(a, b)
    assert bound["lower_db"] == 0 and bound["upper_db"] == 6
    assert 0 < bound["point_loss_db"] < 6


# Even (bulk) target powers ramp 80..120, so the four ranks have distinct
# reference values and thresholds. Tones and bin 1 are odd and untouched.
BULK_RAMP = 80 + (40 * np.arange(128)) // 127


def engineering_cohort(tmp_path, *, coarse_members=None):
    spec = importlib.util.spec_from_file_location("sensitivity_fixture_builder", ROOT / "tests/testbench/test_reduce_fine_validation.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    out = helper.fixture(tmp_path)
    plan = R.read_json(out / "plan.json")
    ids = np.load(out / "trial-identities.npy", allow_pickle=False)
    extra = []
    for i in range(2):
        identity = f"literal-fine-validation-campaign-v1:fixture-choice:evaluation:{i}".encode()
        payload = 2 + (R.hashlib.sha256(identity).digest()[0] & 1)
        extra.append(("evaluation", 2048, i, 300 + i, 300 + i, payload))
    ids = np.concatenate([ids, np.array(extra, dtype=ids.dtype)])
    np.save(out / "trial-identities.npy", ids, allow_pickle=False)
    plan["identities_sha256"] = R.sha(out / "trial-identities.npy")
    plan["counts"]["evaluation"] = 2
    plan["loss_rule"] = {"bootstrap_replicates": 12}
    helper.write(out / "plan.json", plan)
    # These test fixtures are unsealed, so each shard is rebound to the edited plan.
    for path in (out / "raw").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
            metadata = json.loads(str(data["meta_json"].item()))
        for name in ("fixed_fine", "float_fine"):
            arrays[name][..., 0, ::2] = BULK_RAMP
        metadata["plan_sha256"] = R.sha(out / "plan.json")
        np.savez(path, **arrays, meta_json=np.asarray(json.dumps(metadata)))
        receipt = R.read_json(path.with_suffix(".json"))
        receipt.update(metadata=metadata, sha256=R.sha(path))
        helper.write(path.with_suffix(".json"), receipt)
    D = helper.REDUCER
    D.score_stage(out, "null_calibration")
    D.calibrate(out)
    D.score_stage(out, "discovery")
    grid = D.freeze_grid(out)
    for case in range(7):
        snrs = grid["grids"][f"ch14/case{case}"]
        fine = np.full((2, len(snrs), 3, 256), 100, dtype=np.uint64)
        fine[..., 0, ::2] = BULK_RAMP
        coarse = np.full((2, len(snrs), 3), 100, dtype=np.uint64)
        # Above -49.5 dB the designated tone is the anchor's null tone plus 2, and
        # bin 1, in no bulk or designated set, lifts the frame power.
        tone, high = helper.TONE_BINS[case], np.asarray(snrs) >= -49.5
        fine[:, high, 0, tone] = helper.NULL_TONES[case] + 2
        fine[:, high, 0, 1] = 700
        coarse[:, high, 0] = 150
        floating = np.repeat(fine[:, :, None], 5, axis=2).astype(float)
        for stage in range(5):
            # The null stage scales reversed, so stage s carries 1 + 0.05 (4 - s).
            floating[:, high, stage, 0, tone] *= helper.stage_scale(4 - stage)
        if coarse_members is not None:
            coarse = np.full_like(coarse, coarse_members)
        metadata = {"plan_sha256": R.sha(out / "plan.json"), "phase": "evaluation", "channel": 14,
            "case_index": case, "streams": 2048, "start": 0, "stop": 2, "snr_db": snrs,
            "float_stages": plan["float_stages"], "raw_seeds": [300, 301], "payloads": [v[-1] for v in extra],
            "axes": ["trial", "snr", "stage_if_float", "term", "bin_if_fine"],
            "evaluation_grid_sha256": R.sha(out / "evaluation-grid.json")}
        path = out / "raw/evaluation" / f"ch14_case{case}_M2048_000000.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fixed_fine=fine, fixed_coarse=coarse,
                 float_fine=floating,
                 float_coarse=np.repeat(coarse[:, :, None], 5, axis=2).astype(float),
                 clip_count=np.zeros((2, len(snrs)), dtype=np.uint32), meta_json=np.asarray(json.dumps(metadata)))
        helper.write(path.with_suffix(".json"), {"metadata": metadata, "sha256": R.sha(path), "completed_utc": R.utc()})
    return out


def test_complete_small_native_cohort_end_to_end(tmp_path):
    out = engineering_cohort(tmp_path)
    report = R.run(out, "evaluation")
    assert report["coverage_complete"] and len(report["cells"]) == 7
    assert report["all_profile_equivalence"] == "not established"
    assert report["trials_per_cell_snr"] == 2
    assert not report["thresholds_changed"] and not report["physical_certification"]
    # The H1 tone clears the thresholds of float stages 0-2 and the fixed
    # stages, not those of float stages 3 and 4.
    detected = {*R.FLOAT_STAGES[:3], R.FLOAT_STAGES[5], R.Q16_STAGE}
    seeds = set()
    for cell in report["cells"]:
        saved = R.read_json(out / cell["report"])
        assert len(saved["fine_curves"]) == 168 and len(saved["coarse_curves"]) == 18
        for stage in R.STAGES:
            ranked = {c["rank_index"]: c["threshold"] for c in saved["fine_curves"]
                      if c["stage"] == stage and c["geometry"] == "calibrated" and c["pfa"] == .01}
            assert len(set(ranked.values())) == 4
        snrs = saved["snr_db"]
        below = sum(v < -49.5 for v in snrs)
        assert below == 4
        step = [0] * below + [2] * (len(snrs) - below)
        for curve in saved["fine_curves"] + saved["coarse_curves"]:
            hit = curve["kind"] == "coarse" or (curve["geometry"] == "calibrated" and curve["stage"] in detected)
            assert curve["counts"] == (step if hit else [0] * len(snrs))
            assert curve["invalid_trials"] == [0] * len(snrs)
            crossing = curve["crossings"]["0.5"]
            assert crossing["status"] == ("unique" if hit else "unbracketed")
            if hit:
                bracket = crossing["upward_brackets"][0]
                assert bracket["snr_lo_db"] < -49.5 <= bracket["snr_hi_db"]
        assert len(saved["paired_contrasts"]) == 32
        seeds.add(saved["bootstrap_seed"])
        assert all(not item["equivalence_demonstrated"] for item in saved["paired_contrasts"])
    assert len(seeds) == 1
    R.rehash(out, report["outputs_sha256"])


def test_frame_level_curves_use_the_fine_sums_not_the_raw_coarse_members(tmp_path):
    (tmp_path / "consistent").mkdir()
    (tmp_path / "altered").mkdir()
    consistent = engineering_cohort(tmp_path / "consistent")
    altered = engineering_cohort(tmp_path / "altered", coarse_members=0)
    reports = [R.run(out, "evaluation") for out in (consistent, altered)]
    for a, b in zip(reports[0]["cells"], reports[1]["cells"]):
        curves_a = R.read_json(consistent / a["report"])["coarse_curves"]
        curves_b = R.read_json(altered / b["report"])["coarse_curves"]
        assert len(curves_a) == len(curves_b) == 18
        for x, y in zip(curves_a, curves_b):
            assert (x["stage"], x["pfa"]) == (y["stage"], y["pfa"])
            assert x["counts"] == y["counts"]
            assert x["invalid_trials"] == y["invalid_trials"]
            assert not any(y["invalid_trials"])


def test_missing_heldout_cell_refused_before_report_creation(tmp_path):
    out = engineering_cohort(tmp_path)
    (out / "raw/evaluation/ch14_case6_M2048_000000.json").unlink()
    with pytest.raises(FileNotFoundError):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()


def test_shared_or_duplicated_noise_ids_cannot_be_independent_bootstrap():
    with pytest.raises(ValueError, match="Unique disjoint"):
        R.resampling(np.array([1, 2]), np.array([2, 3]), replicates=3, seed=1)
    with pytest.raises(ValueError, match="Unique disjoint"):
        R.resampling(np.array([1, 1]), np.array([2, 3]), replicates=3, seed=1)


def test_authenticated_wrong_raw_dtype_refused_before_any_report(tmp_path):
    out = engineering_cohort(tmp_path)
    path = out / "raw/evaluation/ch14_case6_M2048_000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data["fixed_fine"] = data["fixed_fine"].astype(np.float64)
    np.savez(path, **data)
    receipt = R.read_json(path.with_suffix(".json"))
    receipt["sha256"] = R.sha(path)
    path.with_suffix(".json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="header shape/dtype"):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()


def _edit_json(path, change):
    value = R.read_json(path)
    change(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("damage,message", [
    ("stray_shard", "missing or extra shards"),
    ("stale_receipt_hash", "identity/hash differs"),
    ("receipt_before_grid", "does not follow the frozen grid"),
    ("grid_before_calibration", "Calibration/grid chronology differs"),
    ("grid_omits_calibration", "Grid does not bind this calibration"),
    ("grid_omits_dependency", "Grid omits or changes a frozen calibration dependency"),
])
def test_heldout_authentication_refusals(tmp_path, damage, message):
    out = engineering_cohort(tmp_path)
    raw = out / "raw/evaluation/ch14_case6_M2048_000000.npz"
    grid = out / "evaluation-grid.json"
    if damage == "stray_shard":
        for suffix in (".npz", ".json"):
            shutil.copy(raw.with_suffix(suffix), raw.with_name("stray" + suffix))
    elif damage == "stale_receipt_hash":
        with np.load(raw, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        data["fixed_fine"] = data["fixed_fine"] + np.uint64(1)
        np.savez(raw, **data)
    elif damage == "receipt_before_grid":
        _edit_json(raw.with_suffix(".json"), lambda r: r.update(completed_utc="2021-01-01T00:00:00+00:00"))
    else:
        change = {"grid_before_calibration": lambda g: g.update(frozen_utc="2020-06-01T00:00:00+00:00"),
                  "grid_omits_calibration": lambda g: g["calibration_inputs_sha256"].pop("calibration.json"),
                  "grid_omits_dependency": lambda g: g["calibration_inputs_sha256"].pop(
                      "scores/null_calibration/ch14.npz")}[damage]
        _edit_json(grid, change)
        grid.with_suffix(".sha256").write_text(R.sha(grid) + "\n")  # resealed
    with pytest.raises(ValueError, match=message):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()
