"""Bind frozen archive operating words to numerical profiles for review only.

This builder does not select thresholds, calibrate physical power, qualify a
consumer's masking granularity, or authorize activation.

When the input bundle has a channel 37 profile, the channel 37 decision's word
is bound to it as the noise control (status ``control``), the same way, and the
plan must name its score metadata. Without one the candidate is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from pilot_proxy import operating_policy, runtime_calibration
from pilot_proxy.fine_decision import FINE_DECISION_VERSION, pack_bulk_mask
from pilot_proxy.runtime_bundle import (
    CONTROL_PHYSICAL_CHANNEL,
    FINE_CALIBRATION_STATUS_CONTROL,
    REQUIRED_RUNTIME_BUNDLE_FILES,
    validate_runtime_weight_bundle,
)

PLAN_SCHEMA = "pilotproxy_archive_candidate_plan_v1"
REVIEW_SCHEMA = "pilotproxy_archive_candidate_review_v1"
METHOD = "operating_curve_candidate_v1"
_REQUIRE = runtime_calibration._require


def _fields(value, expected, label):
    _REQUIRE(isinstance(value, dict) and set(value) == set(expected), f"{label}: fields differ")


def _bound(reference, base, bindings):
    _fields(reference, {"path", "sha256"}, "file reference")
    path = (base / runtime_calibration._text(reference["path"], "file path")).resolve()
    if str(path) in bindings:
        _REQUIRE(bindings[str(path)] == reference["sha256"], "Conflicting source identities")
    return runtime_calibration._bound_file(reference, base, bindings)


def _metadata_geometry(meta, channel):
    _REQUIRE(type(meta.get("channel")) is int and meta["channel"] == channel,
             "Score metadata channel differs")
    _REQUIRE(meta.get("physical_science_qualified") is False
             and meta.get("threshold_activated") is False,
             "Archive input must explicitly remain unqualified and unactivated")
    epochs = meta.get("current_months")
    _REQUIRE(isinstance(epochs, list) and epochs and len(set(epochs)) == len(epochs)
             and all(isinstance(e, str) and e.strip() for e in epochs), "Missing archive epochs")
    g = meta.get("score_geometry")
    _REQUIRE(isinstance(g, dict), "Missing frozen score geometry")
    anchor = runtime_calibration._integer(g.get("anchor_bin"), "anchor_bin", 0, 255)
    width = runtime_calibration._integer(g.get("designated_half_width"), "designated_half_width", 0, 127)
    rank = runtime_calibration._integer(g.get("cfar_rank_zero_based"), "cfar_rank", 0, 255)
    bins = g.get("bulk_bins")
    _REQUIRE(isinstance(bins, list) and bins
             and all(type(b) is int and 0 <= b < 256 for b in bins)
             and len(set(bins)) == len(bins), "Invalid frozen bulk bins")
    _REQUIRE(g.get("pad_factor") == 2 and g.get("intended_bulk_size") == len(bins),
             "Frozen bulk size or pad factor differs")
    designated = g.get("designated_bins")
    _REQUIRE(isinstance(designated, list)
             and all(type(b) is int for b in designated)
             and sorted(designated) == sorted((anchor + i) % 256 for i in range(-width, width + 1)),
             "Designated bins are not the runtime's contiguous anchor-centered set")
    # optional bulk guard band beyond the designated set (anchors.bulk_mask marker_guard_bins)
    guard = runtime_calibration._integer(g.get("marker_guard_bins", 0), "marker_guard_bins", 0, 127 - width)
    exclusion = max(width + guard, 2)
    _REQUIRE(rank < len(bins) and not any((anchor + i) % 256 in bins
             for i in range(-exclusion, exclusion + 1)), "Frozen rank or guard is not runtime compatible")
    mask = np.zeros(256, dtype=bool)
    mask[bins] = True
    result = {"status": "calibrated", "decision_version": FINE_DECISION_VERSION,
              "anchor_bin": anchor, "designated_half_width": width, "cfar_rank": rank,
              "bulk_mask_words_hex": [f"0x{int(w):016x}" for w in pack_bulk_mask(mask)]}
    if "marker_guard_bins" in g:
        result["marker_guard_bins"] = guard
    return result


def _decision_population_replay(score_path, meta, decision):
    """Reproduce the saved decision's selection without recomputing its forecast."""
    with np.load(score_path, allow_pickle=False) as z:
        required, supported = z["required_q16"], z["supported"]
        event, weight = z["frame_event_index"], z["probability_weight"]
    _REQUIRE(required.dtype == np.dtype("uint64") and required.ndim == 1 and required.size > 0,
             "Scores must contain uint64 required words")
    _REQUIRE(supported.dtype == np.dtype("bool") and supported.shape == required.shape,
             "Scores must contain matching boolean support")
    _REQUIRE(event.dtype.kind in "iu" and event.shape == required.shape and np.all(event >= 0),
             "Scores must contain nonnegative event IDs")
    _REQUIRE(weight.dtype.kind == "f" and weight.shape == required.shape
             and np.all(np.isfinite(weight)) and np.all(weight > 0), "Invalid score probability weights")
    unique, inverse = np.unique(event, return_inverse=True)
    counts = np.bincount(inverse)
    frame_event_mass = weight * counts[inverse]
    _REQUIRE(np.allclose(frame_event_mass, frame_event_mass[0], rtol=1e-10, atol=1e-14),
             "Score population does not give equal original frame weights within equal-mass events")
    masses = np.bincount(inverse, weights=weight)
    _REQUIRE(np.allclose(masses, masses[0], rtol=1e-10, atol=1e-14),
             "Score population does not give equal original event mass")
    _REQUIRE(meta.get("frames") == required.size and meta.get("events") == unique.size,
             "Score metadata population counts differ")
    word = operating_policy._q16(decision["deployed_word"])
    keep = supported & (required <= np.uint64(word))
    kept_frames, kept_events = int(keep.sum()), int(np.unique(event[keep]).size)
    kappa = float(weight[keep].sum() / weight.sum())
    for key, value in (("kept_frames", kept_frames), ("kept_events", kept_events)):
        _REQUIRE(type(decision.get(key)) is int and decision[key] == value,
                 f"Decision {key} does not reproduce on the selected score population")
    expected_kappa = decision.get("kappa")
    _REQUIRE(type(expected_kappa) in (int, float) and math.isfinite(expected_kappa)
             and math.isclose(kappa, expected_kappa, rel_tol=1e-10, abs_tol=1e-12),
             "Decision kappa does not reproduce on the selected score population")
    return {"kept_frames": kept_frames, "kept_events": kept_events, "kappa": kappa,
            "word_q16": word, "keep_rule": "supported and required_q16 <= word_q16",
            "kappa_absolute_tolerance": 1e-12, "kappa_relative_tolerance": 1e-10}


def _product_proof(meta, profile, contract, weight_bytes, base, bindings):
    identity = meta.get("original_product_identity")
    _REQUIRE(isinstance(identity, dict), "Missing original product identity")
    product = (base / runtime_calibration._text(identity.get("path"), "product path")).resolve()
    expected = runtime_calibration._digest(identity.get("sha256"), "product SHA256")
    size = runtime_calibration._integer(identity.get("bytes"), "product bytes", 1)
    _REQUIRE(product.is_file(), f"Missing archive product: {product}")
    stat = product.stat()
    _REQUIRE(stat.st_size == size, "Original product size differs")
    start = profile["weight_bank_offset_bytes"]
    packed = weight_bytes[start:start + profile["weight_bank_nbytes"]]
    expected_weights = hashlib.sha256(packed).hexdigest()
    with np.load(product, allow_pickle=False) as z:
        _REQUIRE(profile["receiver_channel_id_namespace"] == "chime_freq_id",
                 "This archive candidate builder requires CHIME receiver IDs")
        for field, expected_value, label in (("physical_channel", profile["physical_channel"], "physical channel"),
                                       ("freq_id", profile["receiver_channel_id"], "receiver frequency")):
            value = z[field]
            _REQUIRE(value.size == 1 and value.dtype.kind in "iu"
                     and int(value.reshape(-1)[0]) == expected_value, f"Archive {label} differs")
        _REQUIRE(str(z["weights_hash"].item()) == expected_weights,
                 "Archive and bundle packed weights differ")
        actual_contract = json.loads(str(z["detector_contract_json"].item()))
        for key in ("detector_window_samples", "num_weight_terms", "sample_bits_per_component",
                    "input_format", "input_coordinate_system", "input_preprocessing",
                    "weight_coordinate_system"):
            _REQUIRE(key in contract and actual_contract.get(key) == contract[key],
                     f"Archive and bundle {key} differ")
        for key in ("target_norm_sq", "reference_norm_sum_sq"):
            value = z[key]
            _REQUIRE(value.size == 1 and value.dtype.kind in "iu"
                     and int(value.reshape(-1)[0]) == profile[key], f"Archive and bundle {key} differ")
        measurement = json.loads(str(z["decision_contract_json"].item())).get("fine_measurement", {})
        _REQUIRE(measurement.get("method") == "exact_fine_power_terms"
                 and measurement.get("terms_field") == "fine_power_u64",
                 "Archive does not declare the exact fine-power product")
        _REQUIRE("fine_power_u64" in z.files, "Archive lacks declared fine powers")
    _REQUIRE(runtime_calibration.sha256(product) == expected, "Original product SHA256 differs")
    after = product.stat()
    _REQUIRE((after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns),
             "Original product changed during validation")
    bindings[str(product)] = expected
    return {"path": str(product), "sha256": expected, "packed_profile_sha256": expected_weights,
            "coordinate_contract_matches": True}


def build_archive_candidate(plan_path, output_dir):
    """Verify every input before writing a new, explicitly unqualified bundle."""
    plan_path = Path(plan_path).resolve()
    output = Path(output_dir).resolve()
    _REQUIRE(not output.exists(), "Output directory already exists")
    plan = runtime_calibration._read(plan_path)
    _fields(plan, {"schema_version", "decisions", "bundle_files", "score_metadata"}, "candidate plan")
    _REQUIRE(plan["schema_version"] == PLAN_SCHEMA, "Unknown candidate plan schema")
    bindings = {str(plan_path): runtime_calibration.sha256(plan_path)}
    decisions_path = _bound(plan["decisions"], plan_path.parent, bindings)
    decisions = operating_policy._decisions(decisions_path)
    decision_inputs = json.loads(decisions_path.read_text(encoding="utf-8")).get("inputs", {})
    _REQUIRE(isinstance(decision_inputs, dict), "Missing decision input provenance")
    score_directory = (decisions_path.parent / runtime_calibration._text(
        decision_inputs.get("scores"), "decision score directory")).resolve()
    control_decision = decisions.get(CONTROL_PHYSICAL_CHANNEL)
    decisions = {ch: d for ch, d in decisions.items() if 14 <= ch <= 36}
    _REQUIRE(decisions, "No television-channel decisions")
    finite = {ch: d for ch, d in decisions.items() if d["outcome"] == "useful"}
    _REQUIRE(finite, "No candidate operating words")
    _fields(plan["bundle_files"], REQUIRED_RUNTIME_BUNDLE_FILES, "bundle files")
    files = {name: _bound(ref, plan_path.parent, bindings) for name, ref in plan["bundle_files"].items()}
    _REQUIRE(len({p.parent for p in files.values()}) == 1
             and all(p.name == name for name, p in files.items()), "Bundle files must retain names in one directory")
    validation = validate_runtime_weight_bundle(bundle_dir=next(iter(files.values())).parent)
    _REQUIRE(validation["valid"], f"Input runtime bundle invalid: {validation['errors']}")
    profiles = runtime_calibration._read(files["pilot_profiles.json"])
    by_channel = {p["physical_channel"]: p for p in profiles["profiles"]}
    control = CONTROL_PHYSICAL_CHANNEL in by_channel
    _REQUIRE(len(by_channel) == len(profiles["profiles"])
             and set(by_channel) - {CONTROL_PHYSICAL_CHANNEL} == set(decisions),
             "Bundle must contain exactly one profile for every declared television channel")
    if control:
        _REQUIRE(control_decision is not None and control_decision["outcome"] == "useful",
                 "The bundle has a channel 37 profile but the decisions give the control no word")
        finite[CONTROL_PHYSICAL_CHANNEL] = control_decision
    _fields(plan["score_metadata"], {str(ch) for ch in finite}, "score metadata channels")
    _REQUIRE(all(p["fine_calibration"]["status"] == "pending_campaign" for p in by_channel.values()),
             "Start with an unpopulated exported bundle; do not replace existing calibration")
    contract = runtime_calibration._read(files["detector_contract.json"])
    _REQUIRE(type(decision_inputs.get("frame_samples")) is int
             and decision_inputs["frame_samples"] == 128 * contract["detector_window_samples"],
             "Decision frame_samples differs from the bundle transform geometry")
    weights = files["weights.bin"].read_bytes()
    proofs, small_inputs = [], set(bindings)
    for channel, decision in sorted(finite.items()):
        metadata_path = _bound(plan["score_metadata"][str(channel)], plan_path.parent, bindings)
        _REQUIRE(metadata_path == score_directory / f"ch{channel}.json",
                 "Selected score metadata is not the population path named by the decision record")
        meta = runtime_calibration._read(metadata_path)
        geometry = _metadata_geometry(meta, channel)
        score_path = _bound({"path": str(metadata_path.with_suffix(".npz")),
                             "sha256": meta.get("npz_sha256")}, plan_path.parent, bindings)
        replay = _decision_population_replay(score_path, meta, decision)
        proof = _product_proof(meta, by_channel[channel], contract, weights, metadata_path.parent, bindings)
        small_inputs.update((str(metadata_path), str(score_path)))
        word = operating_policy._q16(decision["deployed_word"])
        selection = {"method": METHOD, "decision_channel": channel, "decision_word_q16": word,
                     "decision_record_sha256": bindings[str(decisions_path)],
                     "score_metadata_sha256": bindings[str(metadata_path)],
                     "score_population_sha256": bindings[str(score_path)],
                     "selection_granularity": "detector_frame", "qualification": "archive_review_only"}
        geometry["cfar_multiplier_q16"] = word
        if channel == CONTROL_PHYSICAL_CHANNEL:
            geometry["status"] = FINE_CALIBRATION_STATUS_CONTROL
        geometry["provenance"] = {
            "epochs": meta["current_months"], "null_quantile": None,
            "source_product_sha256": list(dict.fromkeys([selection["decision_record_sha256"],
                selection["score_metadata_sha256"], selection["score_population_sha256"], proof["sha256"]])),
            "threshold_selection": selection,
            "note": "Frozen archive detector-frame operating-curve candidate. Numerical profile populated; physical transfer, consumer masking granularity, and activation remain unqualified.",
        }
        by_channel[channel]["fine_calibration"] = geometry
        proofs.append({"physical_channel": channel, "receiver_channel_id": by_channel[channel]["receiver_channel_id"],
                       "word_q16": word, "score_metadata": str(metadata_path), "product_proof": proof,
                       "decision_population_replay": replay})
    for path in sorted(small_inputs):
        _REQUIRE(runtime_calibration.sha256(path) == bindings[path], f"Input changed during review: {path}")
    output.mkdir(parents=True, exist_ok=False)
    bundle_dir = output / "bundle"
    bundle_dir.mkdir()
    for name, source in files.items():
        if name not in {"pilot_profiles.json", "sha256sums.txt"}:
            shutil.copyfile(source, bundle_dir / name)
    runtime_calibration._write_json(bundle_dir / "pilot_profiles.json", profiles)
    (bundle_dir / "sha256sums.txt").write_text("".join(
        f"{runtime_calibration.sha256(bundle_dir / name)}  {name}\n"
        for name in REQUIRED_RUNTIME_BUNDLE_FILES if name != "sha256sums.txt"), encoding="utf-8")
    validation = validate_runtime_weight_bundle(bundle_dir=bundle_dir)
    _REQUIRE(validation["valid"], f"Candidate bundle invalid: {validation['errors']}")
    report = {
        "schema_version": REVIEW_SCHEMA, "status": "archive_review_only",
        "candidate_configuration_complete": True, "deployment_configuration_complete": False,
        "activation_authorized": False, "science_certification": False,
        "consumer_capability_implementation_verified": False, "kotekan_parameters": None,
        "selection_granularity": "detector_frame", "null_quantile_threshold_claim": False,
        "profiles": proofs, "permanent_mask_channels": sorted(set(decisions) - set(finite)),
        "control_channel_37_bound": control, "runtime_bundle_validation": validation,
        "input_files_sha256": bindings,
        "source_plan": str(plan_path), "builder_source_sha256": runtime_calibration.sha256(__file__),
        "dependency_sources_sha256": {**operating_policy._identity(),
                                      "operating_policy.py": runtime_calibration.sha256(operating_policy.__file__)},
        "bundle_files_sha256": {name: runtime_calibration.sha256(bundle_dir / name)
                                for name in REQUIRED_RUNTIME_BUNDLE_FILES},
        "remaining_acceptance": [
            "Re-evaluate retention, retained proxy, sampling guard, and the decision rule at the actual consumer masking granularity, including CHIMEv1 pair expansion and frame alignment.",
            "Qualify current receiver/input state, physical transfer, and temporal covariance for the claimed science use.",
            "Verify the consumer build, full-load behavior, and release approval before activation.",
        ],
    }
    runtime_calibration._write_json(output / "candidate_review.json", report)
    shutil.copyfile(plan_path, output / "source_plan.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_archive_candidate(args.plan, args.output_dir)
    print(json.dumps({"status": report["status"], "candidate_profiles": len(report["profiles"]),
                      "deployment_configuration_complete": False, "activation_authorized": False}))


if __name__ == "__main__":
    main()
