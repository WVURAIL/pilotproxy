from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest


DRIVER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools/current_geometry_sensitivity.py")
)


def _shard(directory, purpose, snr, *, indices, ratio=1.0, gpu=False,
           mismatch=0, replacements=None, metadata=None, name="sample",
           backend="full-frame"):
    indices = np.asarray(indices)
    n = indices.size
    meta = {
        "config_sha256": "synthetic-report-fixture",
        "physical_channel": 14,
        "offset_fine_bins": 0.0,
        "snr_db": snr,
        "base_seed": 7,
        "trials": n,
        "simulation_backend": backend,
        "gpu_executed": gpu,
        "gpu_full_frame_executed": gpu,
        "gpu_execution_form": "synthetic-parity-fixture" if gpu else None,
    }
    meta.update(metadata or {})
    payload = {
        DRIVER["SHARD_META_KEY"]: np.asarray(json.dumps(meta)),
        "trial_index": indices,
        "clip_fraction": np.zeros(n),
        "cpu_q16_mask": np.full(n, int(ratio > 1), dtype=np.int8),
        "gpu_q16_mask": np.full(n, int(ratio > 1) if gpu else -1, dtype=np.int8),
        "gpu_fine_mismatch": np.full(n, mismatch if gpu else -1, dtype=np.int8),
        "gpu_mask_mismatch": np.full(n, 0 if gpu else -1, dtype=np.int8),
    }
    for stage in DRIVER["FLOAT_RESPONSE_STAGES"]:
        payload[f"ratio__{stage}"] = np.full(n, ratio, dtype=np.float64)
    for key, value in {
        "designated_num": round(ratio * 2), "designated_den": 2,
        "rank_num": 1, "rank_den": 1, "designated_bin": 0, "rank_bin": 2,
    }.items():
        payload[f"fixed__{key}"] = np.full(n, value, dtype=np.uint64)
    payload.update(replacements or {})
    path = directory / "shards" / purpose / f"{name}_{snr}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def _load(directory, *, purpose="h1", snr=-2.0):
    return DRIVER["_load_shards"](
        directory, purpose=purpose, config_sha256="synthetic-report-fixture",
        physical_channel=14, offset_fine_bins=0.0, snr_db=snr,
    )


def _campaign(directory, *, upper_indices=None, high_ratio=2.0,
              backend="full-frame", audit=False, gpu=False, mismatch=0):
    args = SimpleNamespace(
        physical_channel=[14], offset_fine_bins=[0.0], output_dir=directory,
        snr_db=[-2.0, 0.0], num_streams=2048, sufficient_pool_streams=256,
        p_fa=0.01, bootstrap_replicates=4, seed=12,
        nyquist_zone=DRIVER["MIRRORED_NYQUIST_ZONE"],
        designated_half_width=2, guard_fine_bins=1, null_quantile=0.5,
    )
    config = {
        "config_sha256": "synthetic-report-fixture",
        "scientific_scope": "current_chime_profile_geometry", "geometry": {},
        "simulation": {"primary_backend": backend},
        "full_frame_audit_design": {
            "physical_channels": [14] if audit else [],
            "offsets_fine_bins": [0.0] if audit else [],
            "snr_db": [-2.0, 0.0],
            "minimum_trials_per_null_and_h1_point": 16,
            "acceptance": {"maximum_sufficient_negative_power_clip_fraction": 1e-6},
        },
    }
    bank = DRIVER["DetectorWeightBank"](explicit_path=DRIVER["DEFAULT_WEIGHTS_PATH"])
    _shard(directory, "null", None, indices=np.arange(1000), gpu=gpu)
    _shard(directory, "h1", -2.0, indices=np.arange(1000), ratio=0.5,
           gpu=gpu, mismatch=mismatch)
    _shard(directory, "h1", 0.0,
           indices=np.arange(1000) if upper_indices is None else upper_indices,
           ratio=high_ratio, gpu=gpu)
    calibration = DRIVER["_build_calibration"](args, config, bank)
    return args, config, bank, calibration


def _report(campaign):
    report = DRIVER["_build_report"](*campaign)
    return report, report["profiles"]["ch14_off_p0p000000"]


def test_shard_rejects_duplicate_identities_within_one_file(tmp_path):
    _shard(tmp_path, "h1", -2.0, indices=[3, 3])
    with pytest.raises(RuntimeError, match="overlapping trial coordinates"):
        _load(tmp_path)


@pytest.mark.parametrize("indices", [[0.5, 1.5], [[0, 1]], [-1, 0]])
def test_shard_rejects_invalid_trial_coordinates(tmp_path, indices):
    _shard(tmp_path, "h1", -2.0, indices=indices)
    with pytest.raises(RuntimeError, match="invalid trial coordinates"):
        _load(tmp_path)


def test_shard_sorting_keeps_response_columns_paired(tmp_path):
    stage = DRIVER["STAGE_ATSC_FLOAT"]
    _shard(tmp_path, "h1", -2.0, indices=[8, 2], replacements={
        f"ratio__{stage}": [80.0, 20.0], "fixed__designated_num": [8, 2],
    })
    loaded = _load(tmp_path)
    assert loaded["trial_keys"] == [(7, 2), (7, 8)]
    np.testing.assert_array_equal(loaded["ratios"][stage], [20.0, 80.0])
    assert [item.designated_num for item in loaded["exact"]] == [2, 8]


def test_shard_rejects_overlap_across_files(tmp_path):
    _shard(tmp_path, "h1", -2.0, indices=[1, 2], name="a")
    _shard(tmp_path, "h1", -2.0, indices=[2, 3], name="b")
    with pytest.raises(RuntimeError, match="overlapping trial coordinates"):
        _load(tmp_path)


@pytest.mark.parametrize("field", [
    "ratio__" + DRIVER["STAGE_ATSC_FLOAT"], "fixed__rank_num", "clip_fraction",
    "cpu_q16_mask", "gpu_q16_mask", "gpu_fine_mismatch", "gpu_mask_mismatch",
])
@pytest.mark.parametrize("shape", [(3,), (2, 1)])
def test_shard_rejects_misaligned_trial_columns(tmp_path, field, shape):
    _shard(tmp_path, "h1", -2.0, indices=[1, 0],
           replacements={field: np.ones(shape)})
    with pytest.raises(RuntimeError, match="trial column"):
        _load(tmp_path)


def test_report_round_trips_literal_crossings_and_paired_intervals(tmp_path):
    campaign = _campaign(tmp_path)
    report, profile = _report(campaign)
    for target, crossing in [("0.5", -1.0), ("0.9", -0.2)]:
        assert profile["stages"][DRIVER["STAGE_ATSC_FLOAT"]]["crossings"][target][
            "estimate_db"] == pytest.approx(crossing)
        interval = profile["paired_bootstrap"][target]
        assert interval["valid_bracketed_replicates"] == 4
        assert interval["paired_h1_trials_per_snr"] == 1000
        assert interval["fixed_minus_float_sensitivity_loss_bootstrap95_lo_db"] == 0
        assert interval["fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"] == 0
    assert profile["claim_status"] == "eligible_for_precision_review"
    saved = json.loads((tmp_path / "sensitivity_report.json").read_text(),
                       parse_constant=lambda value: pytest.fail(value))
    assert saved == report
    with (tmp_path / "sensitivity_curves.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * (len(DRIVER["REPORT_STAGES"]) - 1)
    assert {row["trials"] for row in rows} == {"1000"}


def test_report_withholds_unsupported_crossings(tmp_path):
    _, profile = _report(_campaign(tmp_path, high_ratio=0.5))
    assert profile["claim_status"] == "preliminary_or_code_validation"
    for target in ("0.5", "0.9"):
        assert profile["fixed_minus_float_point_estimates"][target][
            "fixed_minus_float_sensitivity_loss_db"] is None
        interval = profile["paired_bootstrap"][target]
        assert interval["valid_bracketed_replicates"] == 0
        assert interval["fixed_minus_float_sensitivity_loss_median_db"] is None
        assert interval["fixed_minus_float_sensitivity_loss_bootstrap95_lo_db"] is None


def test_report_retains_curves_when_h1_populations_are_disjoint(tmp_path):
    _, profile = _report(_campaign(tmp_path, upper_indices=np.arange(1000, 2000)))
    assert profile["claim_status"] == "preliminary_or_code_validation"
    assert profile["paired_h1_trials_per_snr"] == 0
    for target in ("0.5", "0.9"):
        assert profile["paired_bootstrap"][target]["status"] == "unavailable"
        assert "no common H1 trial identities" in profile["paired_bootstrap"][target]["reason"]
    assert (tmp_path / "sensitivity_curves.csv").is_file()


def test_precision_review_requires_1000_shared_h1_trials(tmp_path):
    _, profile = _report(_campaign(tmp_path, upper_indices=np.arange(999, 1999)))
    assert profile["paired_bootstrap"]["0.5"]["paired_h1_trials_per_snr"] == 1
    assert profile["claim_status"] == "preliminary_or_code_validation"
    assert not profile["precision_review_prerequisites"][
        "at_least_1000_paired_h1_trials_per_snr"]


def test_audit_requires_gpu_execution_at_every_declared_point(tmp_path):
    campaign = _campaign(tmp_path, backend="sufficient-statistic", audit=True)
    _shard(tmp_path, "audit_null", None, indices=np.arange(16), gpu=True)
    _shard(tmp_path, "audit_h1", -2.0, indices=np.arange(16), ratio=0.5)
    _shard(tmp_path, "audit_h1", 0.0, indices=np.arange(16), ratio=2.0)
    report, profile = _report(campaign)
    assert profile["full_frame_audit"]["comparison_acceptance"]["passes"]
    assert not profile["full_frame_audit"]["complete_for_precision_review"]
    assert not report["stratified_full_frame_audit"]["complete_for_precision_review"]
    assert profile["claim_status"] == "preliminary_or_code_validation"


def test_precision_review_rejects_primary_gpu_mismatches(tmp_path):
    _, profile = _report(_campaign(tmp_path, gpu=True, mismatch=1))
    assert profile["gpu_parity"]["fine_power_mismatches"] == 1000
    assert profile["claim_status"] == "preliminary_or_code_validation"


def test_complete_audit_retains_precision_review_eligibility(tmp_path):
    campaign = _campaign(tmp_path, backend="sufficient-statistic", audit=True)
    for purpose, snr, ratio in [("audit_null", None, 1.0),
                                ("audit_h1", -2.0, 0.5), ("audit_h1", 0.0, 2.0)]:
        _shard(tmp_path, purpose, snr, indices=np.arange(16), ratio=ratio, gpu=True)
    report, profile = _report(campaign)
    assert profile["full_frame_audit"]["full_frame_gpu_coverage_complete"]
    assert report["stratified_full_frame_audit"]["complete_for_precision_review"]
    assert profile["claim_status"] == "eligible_for_precision_review"


def test_report_requires_source_null_shards_even_with_saved_calibration(tmp_path):
    campaign = _campaign(tmp_path)
    for path in (tmp_path / "shards" / "null").glob("*.npz"):
        path.unlink()
    with pytest.raises(RuntimeError, match="no null shards for channel 14"):
        _report(campaign)
    assert not (tmp_path / "sensitivity_report.json").exists()


def test_report_without_h1_shards_stays_preliminary(tmp_path):
    campaign = _campaign(tmp_path)
    for path in (tmp_path / "shards" / "h1").glob("*.npz"):
        path.unlink()
    _, profile = _report(campaign)
    assert profile["h1_snrs_present_db"] == []
    assert profile["paired_bootstrap"] == {}
    assert profile["claim_status"] == "preliminary_or_code_validation"


def test_negative_power_clipping_still_withholds_loss(tmp_path):
    campaign = _campaign(tmp_path)
    _shard(tmp_path, "h1", 0.0, indices=np.arange(1000), ratio=2.0, metadata={
        "negative_power_clip_fraction_by_stage": {DRIVER["STAGE_ATSC_FLOAT"]: 1e-3},
    })
    _, profile = _report(campaign)
    assert profile["claim_status"] == "preliminary_or_code_validation"
    for target in ("0.5", "0.9"):
        assert profile["fixed_minus_float_point_estimates"][target][
            "fixed_minus_float_sensitivity_loss_db"] is None
        interval = profile["paired_bootstrap"][target]
        assert interval["valid_bracketed_replicates"] == 4
        assert interval["fixed_minus_float_sensitivity_loss_median_db"] is None
        assert interval["fixed_minus_float_sensitivity_loss_bootstrap95_lo_db"] is None
        assert interval["fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"] is None
        assert "negative-power clipping" in interval["sensitivity_loss_withheld_reason"]


@pytest.mark.parametrize("invalid", [np.nan, -np.inf, -1.0])
def test_ks_audit_refuses_invalid_response_population(invalid):
    comparison = DRIVER["_ks_comparison"]([1.0] * 15 + [invalid], [1.0] * 16)
    assert not comparison["passes_declared_rule"]
    assert comparison["distance"] is None


def test_ks_audit_keeps_positive_infinity_in_the_response_population():
    comparison = DRIVER["_ks_comparison"]([1.0] + [np.inf] * 15, [1.0] * 16)
    assert comparison["distance"] == 15 / 16
    assert not comparison["passes_declared_rule"]


def test_ks_audit_accepts_matching_zero_rank_populations():
    comparison = DRIVER["_ks_comparison"]([1.0] + [np.inf] * 15,
                                         [1.0] + [np.inf] * 15)
    assert comparison["distance"] == 0
    assert comparison["passes_declared_rule"]


@pytest.mark.parametrize("field,value", [
    ("ratio__" + DRIVER["STAGE_ATSC_FLOAT"], 1 + 2j),
    ("gpu_q16_mask", 256), ("gpu_mask_mismatch", 0.5),
    ("gpu_fine_mismatch", 128), ("cpu_q16_mask", -2),
])
def test_shard_refuses_lossy_column_conversions(tmp_path, field, value):
    _shard(tmp_path, "h1", -2.0, indices=[0, 1],
           replacements={field: np.full(2, value)})
    with pytest.raises(RuntimeError, match="trial column"):
        _load(tmp_path)


def test_ks_audit_refuses_complex_responses():
    assert not DRIVER["_ks_comparison"](np.ones(16) + 2j, np.ones(16))[
        "passes_declared_rule"]


def test_report_refuses_null_population_added_after_calibration(tmp_path):
    campaign = _campaign(tmp_path)
    _shard(tmp_path, "null", None, indices=np.arange(1000, 1010), name="later")
    with pytest.raises(RuntimeError, match="null population differs from calibration"):
        _report(campaign)


def test_report_does_not_round_unresolved_null_probability_up_to_one_trial(tmp_path):
    args, config, bank, _ = _campaign(tmp_path)
    args.p_fa = 1 / 3
    _shard(tmp_path, "null", None, indices=np.arange(3))
    calibration = DRIVER["_build_calibration"](args, config, bank)
    _, profile = _report((args, config, bank, calibration))
    assert not profile["precision_review_prerequisites"][
        "requested_pfa_resolved_by_null_count"]
