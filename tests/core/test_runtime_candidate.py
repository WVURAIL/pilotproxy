"""Tests of archive candidate binding and refusal. All products here are synthetic."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from pilot_proxy import operating_policy as op, runtime_calibration as rc
from pilot_proxy.runtime_bundle import (
    REQUIRED_RUNTIME_BUNDLE_FILES,
    _validate_fine_calibration,
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)
from pilot_proxy.runtime_candidate import PLAN_SCHEMA, build_archive_candidate

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ref(path)


def ref(path):
    return {"path": str(path), "sha256": rc.sha256(path)}


def refresh(case):
    meta, plan = case["meta"], case["plan"]
    meta["original_product_identity"] = {**ref(case["product"]), "bytes": case["product"].stat().st_size}
    plan["score_metadata"]["29"] = write(case["meta_path"], meta)
    write(case["plan_path"], plan)


@pytest.fixture
def case(tmp_path):
    bundle = tmp_path / "input_bundle"
    receiver = ROOT / "configs/receiver_profiles/chime_dtv_fengine.json"
    core = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"
    export_runtime_weight_bundle(receiver_profile_path=receiver, detector_core_profile_path=core,
        weight_coordinate_system="post_nyquist_zone_normalization", physical_channels=[14, 29], output_dir=bundle)
    profiles = json.loads((bundle / "pilot_profiles.json").read_text())
    profile = next(p for p in profiles["profiles"] if p["physical_channel"] == 29)
    packed = (bundle / "weights.bin").read_bytes()[profile["weight_bank_offset_bytes"]:]
    product = tmp_path / "product.npz"
    product_fields = {
        "physical_channel": np.asarray(29), "freq_id": np.asarray(profile["receiver_channel_id"]),
        "weights_hash": np.asarray(hashlib.sha256(packed).hexdigest()),
        "target_norm_sq": np.asarray([profile["target_norm_sq"]]),
        "reference_norm_sum_sq": np.asarray([profile["reference_norm_sum_sq"]]),
        "detector_contract_json": np.asarray((bundle / "detector_contract.json").read_text()),
        "decision_contract_json": np.asarray(json.dumps({"fine_measurement": {
            "method": "exact_fine_power_terms", "terms_field": "fine_power_u64"}})),
        "fine_power_u64": np.ones((1, 3, 256), dtype=np.uint64),
    }
    np.savez(product, **product_fields)
    score = tmp_path / "ch29.npz"
    np.savez(score, required_q16=np.asarray([65536], dtype=np.uint64), supported=np.asarray([True]),
             frame_event_index=np.asarray([0]), probability_weight=np.asarray([1.0]))
    bins = [b for b in range(0, 256, 2) if b not in range(60, 65)]
    meta = {"channel": 29, "physical_science_qualified": False, "threshold_activated": False,
            "current_months": ["synthetic-era"], "npz_sha256": rc.sha256(score), "frames": 1, "events": 1,
            "score_geometry": {"anchor_bin": 62, "designated_half_width": 2,
                "cfar_rank_zero_based": 62, "bulk_bins": bins, "pad_factor": 2,
                "intended_bulk_size": len(bins), "designated_bins": [60, 61, 62, 63, 64]}}
    decisions_path = tmp_path / "decisions.json"
    decisions = write(decisions_path, {"inputs": {"scores": str(tmp_path), "frame_samples": 16384}, "channels": [
        {"channel": 14, "outcome": "not useful", "reason": "synthetic permanent-mask example"},
        {"channel": 29, "outcome": "useful", "deployed_word": 65536, "deployed_at": "knee",
         "kept_frames": 1, "kept_events": 1, "kappa": 1.0},
        {"channel": 37, "outcome": "useful", "deployed_word": 70000},
    ]})
    plan = {"schema_version": PLAN_SCHEMA, "decisions": decisions,
            "bundle_files": {name: ref(bundle / name) for name in REQUIRED_RUNTIME_BUNDLE_FILES},
            "score_metadata": {}}
    case = {"plan": plan, "plan_path": tmp_path / "plan.json", "meta": meta,
            "meta_path": tmp_path / "ch29.json", "product": product,
            "product_fields": product_fields, "receiver": receiver, "output": tmp_path / "candidate"}
    refresh(case)
    return case


def test_candidate_is_exactly_bound_but_not_a_deployment_configuration(case):
    report = build_archive_candidate(case["plan_path"], case["output"])
    assert report["candidate_configuration_complete"]
    assert not report["deployment_configuration_complete"]
    assert not report["activation_authorized"] and not report["science_certification"]
    assert report["kotekan_parameters"] is None
    assert report["selection_granularity"] == "detector_frame"
    assert report["permanent_mask_channels"] == [14]
    assert not report["control_channel_37_bound"]
    bundle = case["output"] / "bundle"
    assert validate_runtime_weight_bundle(bundle_dir=bundle)["valid"]
    p = json.loads((bundle / "pilot_profiles.json").read_text())
    fine = p["profiles"][1]["fine_calibration"]
    assert fine["cfar_multiplier_q16"] == 65536
    assert fine["provenance"]["null_quantile"] is None
    assert fine["provenance"]["threshold_selection"]["qualification"] == "archive_review_only"
    evidence = case["plan"]["decisions"]
    plan = {"schema_version": op.PLAN_SCHEMA, "receiver_profile": ref(case["receiver"]),
            "receiver_channel_id_namespace": "chime_freq_id", "decisions": evidence,
            "bundle_files": {name: ref(bundle / name) for name in REQUIRED_RUNTIME_BUNDLE_FILES},
            "era": {"id": "synthetic-era", "scope": "synthetic test", "evidence": [evidence]},
            "consumer": {"name": op.CONSUMER_NAME, "version": "synthetic", "build_sha256": evidence["sha256"],
                         "capabilities": sorted(op.REQUIRED_CAPABILITIES), "evidence": [evidence]}}
    policy_path = case["output"] / "policy_plan.json"
    write(policy_path, plan)
    policy = op.analyze_policy(policy_path)
    assert policy["candidate_configuration_complete"] and policy["pending_freq_ids"] == []
    assert policy["review_only_candidate_freq_ids"] == [614]
    assert policy["required_threshold_bindings"][0]["eta_q16"] == "65536"
    assert not policy["deployment_configuration_complete"] and policy["kotekan_parameters"] is None
    with pytest.raises(ValueError, match="archive review candidate"):
        op.write_policy(policy_path, case["output"] / "policy", require_complete=True)


@pytest.mark.parametrize("field,value,match", [
    ("weights_hash", "0" * 64, "packed weights differ"),
    ("physical_channel", 28, "physical channel differs"),
    ("freq_id", 613, "receiver frequency differs"),
    ("target_norm_sq", [1], "target_norm_sq differ"),
])
def test_product_binding_refuses_mismatch(case, field, value, match):
    case["product_fields"][field] = np.asarray(value)
    np.savez(case["product"], **case["product_fields"])
    refresh(case)
    with pytest.raises(ValueError, match=match):
        build_archive_candidate(case["plan_path"], case["output"])
    assert not case["output"].exists()


def test_coordinate_mismatch_is_refused(case):
    contract = json.loads(case["product_fields"]["detector_contract_json"].item())
    contract["input_coordinate_system"] = "different-coordinate"
    case["product_fields"]["detector_contract_json"] = np.asarray(json.dumps(contract))
    np.savez(case["product"], **case["product_fields"])
    refresh(case)
    with pytest.raises(ValueError, match="input_coordinate_system differ"):
        build_archive_candidate(case["plan_path"], case["output"])


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(channel=28), "metadata channel differs"),
    (lambda m: m.update(current_months=[]), "Missing archive epochs"),
    (lambda m: m["score_geometry"].update(designated_bins=[62]), "Designated bins"),
    (lambda m: m["score_geometry"].update(cfar_rank_zero_based=255), "rank or guard"),
    (lambda m: m.update(npz_sha256="0" * 64), "File identity mismatch"),
])
def test_metadata_refusal(case, change, match):
    change(case["meta"])
    refresh(case)
    with pytest.raises(ValueError, match=match):
        build_archive_candidate(case["plan_path"], case["output"])
    assert not case["output"].exists()


def test_product_digest_is_checked(case):
    refresh(case)
    with case["product"].open("ab") as stream:
        stream.write(b"changed")
    case["meta"]["original_product_identity"]["bytes"] = case["product"].stat().st_size
    case["plan"]["score_metadata"]["29"] = write(case["meta_path"], case["meta"])
    write(case["plan_path"], case["plan"])
    with pytest.raises(ValueError, match="Original product SHA256 differs"):
        build_archive_candidate(case["plan_path"], case["output"])


@pytest.mark.parametrize("key,value", [
    ("method", "null_quantile"), ("decision_word_q16", 1), ("decision_channel", 14),
    ("qualification", "science_qualified"), ("selection_granularity", "correlator_pair"),
    ("score_metadata_sha256", "0" * 64),
])
def test_candidate_provenance_cannot_claim_other_selection_or_qualification(case, key, value):
    build_archive_candidate(case["plan_path"], case["output"])
    p = json.loads((case["output"] / "bundle/pilot_profiles.json").read_text())
    p["profiles"][1]["fine_calibration"]["provenance"]["threshold_selection"][key] = value
    errors = []
    _validate_fine_calibration(profiles=p["profiles"], errors=errors)
    assert errors and any(key in error["check"] for error in errors)


def test_candidate_cannot_claim_a_null_quantile(case):
    build_archive_candidate(case["plan_path"], case["output"])
    p = json.loads((case["output"] / "bundle/pilot_profiles.json").read_text())
    p["profiles"][1]["fine_calibration"]["provenance"]["null_quantile"] = 0.95
    errors = []
    _validate_fine_calibration(profiles=p["profiles"], errors=errors)
    assert any(error["check"].endswith("null_quantile") for error in errors)


@pytest.mark.parametrize("key,value", [("kept_frames", 2), ("kept_events", 2), ("kappa", 0.9)])
def test_same_channel_population_must_reproduce_decision(case, key, value):
    path = Path(case["plan"]["decisions"]["path"])
    decision = json.loads(path.read_text())
    decision["channels"][1][key] = value
    case["plan"]["decisions"] = write(path, decision)
    write(case["plan_path"], case["plan"])
    with pytest.raises(ValueError, match=f"Decision {key} does not reproduce"):
        build_archive_candidate(case["plan_path"], case["output"])


def test_same_channel_scores_with_other_keep_set_are_refused(case):
    score = case["meta_path"].with_suffix(".npz")
    np.savez(score, required_q16=np.asarray([65537], dtype=np.uint64), supported=np.asarray([True]),
             frame_event_index=np.asarray([0]), probability_weight=np.asarray([1.0]))
    case["meta"]["npz_sha256"] = rc.sha256(score)
    refresh(case)
    with pytest.raises(ValueError, match="Decision kept_frames does not reproduce"):
        build_archive_candidate(case["plan_path"], case["output"])


def test_decision_frame_geometry_must_match(case):
    path = Path(case["plan"]["decisions"]["path"])
    decision = json.loads(path.read_text())
    decision["inputs"]["frame_samples"] = 8192
    case["plan"]["decisions"] = write(path, decision)
    write(case["plan_path"], case["plan"])
    with pytest.raises(ValueError, match="frame_samples differs"):
        build_archive_candidate(case["plan_path"], case["output"])


def test_unequal_frame_weights_within_one_event_are_refused(case):
    score = case["meta_path"].with_suffix(".npz")
    np.savez(score, required_q16=np.asarray([65536, 65536], dtype=np.uint64),
             supported=np.asarray([True, True]), frame_event_index=np.asarray([0, 0]),
             probability_weight=np.asarray([0.25, 0.75]))
    case["meta"]["npz_sha256"] = rc.sha256(score)
    case["meta"]["frames"] = 2
    refresh(case)
    with pytest.raises(ValueError, match="equal original frame weights"):
        build_archive_candidate(case["plan_path"], case["output"])


def add_control(case, tmp_path, *, word=70000):
    """Re-export the input bundle with a channel 37 profile and give the control its scores."""
    bundle = tmp_path / "input_bundle37"
    core = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"
    export_runtime_weight_bundle(receiver_profile_path=case["receiver"], detector_core_profile_path=core,
        weight_coordinate_system="post_nyquist_zone_normalization", physical_channels=[14, 29, 37],
        output_dir=bundle)
    profiles = {p["physical_channel"]: p for p in json.loads((bundle / "pilot_profiles.json").read_text())["profiles"]}
    weights = (bundle / "weights.bin").read_bytes()
    for channel in (29, 37):
        profile = profiles[channel]
        packed = weights[profile["weight_bank_offset_bytes"]:profile["weight_bank_offset_bytes"]
                         + profile["weight_bank_nbytes"]]
        fields = {**case["product_fields"], "physical_channel": np.asarray(channel),
                  "freq_id": np.asarray(profile["receiver_channel_id"]),
                  "weights_hash": np.asarray(hashlib.sha256(packed).hexdigest()),
                  "target_norm_sq": np.asarray([profile["target_norm_sq"]]),
                  "reference_norm_sum_sq": np.asarray([profile["reference_norm_sum_sq"]])}
        product = tmp_path / f"product{channel}.npz"
        np.savez(product, **fields)
        meta = json.loads(json.dumps(case["meta"]))
        meta["channel"] = channel
        meta["original_product_identity"] = {**ref(product), "bytes": product.stat().st_size}
        if channel == 37:
            score = tmp_path / "ch37.npz"
            np.savez(score, required_q16=np.asarray([word], dtype=np.uint64), supported=np.asarray([True]),
                     frame_event_index=np.asarray([0]), probability_weight=np.asarray([1.0]))
            meta["npz_sha256"] = rc.sha256(score)
        case["plan"]["score_metadata"][str(channel)] = write(tmp_path / f"ch{channel}.json", meta)
    path = Path(case["plan"]["decisions"]["path"])
    decisions = json.loads(path.read_text())
    decisions["channels"][2].update(deployed_word=word, kept_frames=1, kept_events=1, kappa=1.0)
    case["plan"]["decisions"] = write(path, decisions)
    case["plan"]["bundle_files"] = {name: ref(bundle / name) for name in REQUIRED_RUNTIME_BUNDLE_FILES}
    write(case["plan_path"], case["plan"])


def test_a_channel_37_profile_is_bound_as_the_control(case, tmp_path):
    add_control(case, tmp_path)
    report = build_archive_candidate(case["plan_path"], case["output"])
    assert report["control_channel_37_bound"] and report["permanent_mask_channels"] == [14]
    profiles = {p["physical_channel"]: p for p in
                json.loads((case["output"] / "bundle/pilot_profiles.json").read_text())["profiles"]}
    assert profiles[29]["fine_calibration"]["status"] == "calibrated"
    control = profiles[37]["fine_calibration"]
    assert control["status"] == "control" and control["cfar_multiplier_q16"] == 70000
    assert control["provenance"]["threshold_selection"]["decision_channel"] == 37
    assert validate_runtime_weight_bundle(bundle_dir=case["output"] / "bundle")["valid"]


def test_a_channel_37_profile_without_a_control_word_is_refused(case, tmp_path):
    add_control(case, tmp_path)
    path = Path(case["plan"]["decisions"]["path"])
    decisions = json.loads(path.read_text())
    decisions["channels"][2] = {"channel": 37, "outcome": "not useful", "reason": "synthetic"}
    case["plan"]["decisions"] = write(path, decisions)
    write(case["plan_path"], case["plan"])
    with pytest.raises(ValueError, match="give the control no word"):
        build_archive_candidate(case["plan_path"], case["output"])


@pytest.mark.parametrize("line_in_bulk", [False, True])
def test_score_geometry_guard_band_is_carried_and_enforced(case, line_in_bulk):
    geometry = case["meta"]["score_geometry"]
    bins = [b for b in geometry["bulk_bins"] if not 62 - 5 <= b <= 62 + 5]
    if line_in_bulk:
        bins.append(66)                            # 4 bins from the anchor, inside h + g = 5
    geometry.update(marker_guard_bins=3, bulk_bins=sorted(bins), intended_bulk_size=len(bins))
    refresh(case)
    if line_in_bulk:
        with pytest.raises(ValueError, match="rank or guard"):
            build_archive_candidate(case["plan_path"], case["output"])
        return
    build_archive_candidate(case["plan_path"], case["output"])
    p = json.loads((case["output"] / "bundle/pilot_profiles.json").read_text())
    assert p["profiles"][1]["fine_calibration"]["marker_guard_bins"] == 3
