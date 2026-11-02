"""Runtime command workflows with synthetic, hash-bound archive evidence."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pilot_proxy import runtime_bundle, runtime_candidate


ROOT = Path(__file__).resolve().parents[2]
RECEIVER = ROOT / "configs/receiver_profiles/chime_dtv_fengine.json"
CORE = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"


def _read(path):
    def refuse_constant(value):
        raise AssertionError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=refuse_constant)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, data):
    path.write_text(json.dumps(data, allow_nan=False) + "\n")


def _bundle_args(output, coordinate="post_nyquist_zone_normalization"):
    return ["--receiver-profile", str(RECEIVER), "--detector-core-profile", str(CORE),
            "--weight-coordinate-system", coordinate, "--output-dir", str(output)]


@pytest.mark.parametrize("coordinate,reverse", [
    ("post_nyquist_zone_normalization", True),
    ("raw_input_frequency_coordinate", False),
])
def test_bundle_cli_preserves_channel_selection_coordinates_and_binary_norms(
    tmp_path, capsys, coordinate, reverse,
):
    output = tmp_path / "runtime bundle"
    assert runtime_bundle.main(_bundle_args(output, coordinate) + [
        "--physical-channel", "29", "--physical-channel", "14",
        "--physical-channel-range", "14:15",
    ]) == 0
    assert set(p.name for p in output.iterdir()) == set(runtime_bundle.REQUIRED_RUNTIME_BUNDLE_FILES)
    printed = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    assert printed == {
        "detector_contract": str(output / "detector_contract.json"),
        "pilot_profiles": str(output / "pilot_profiles.json"),
        "weights": str(output / "weights.bin"),
        "weights_manifest": str(output / "weights.manifest.json"),
        "sha256sums": str(output / "sha256sums.txt"),
    }
    manifest = _read(output / "weights.manifest.json")
    profiles = _read(output / "pilot_profiles.json")
    contract = _read(output / "detector_contract.json")
    assert manifest["physical_channels"] == [14, 15, 29]
    assert manifest["weight_profile_shape"] == [3, 128]
    assert manifest["weight_profile_nbytes"] == 384
    assert contract["weight_coordinate_system"] == coordinate
    assert contract["input_preprocessing"]["time_reverse_detector_windows_before_kernel"] is reverse
    payload = (output / "weights.bin").read_bytes()
    assert len(payload) == 1152
    assert [p["physical_channel"] for p in profiles["profiles"]] == [14, 15, 29]
    for index, row in enumerate(profiles["profiles"]):
        assert row["weight_bank_offset_bytes"] == 384 * index
        assert row["weight_bank_nbytes"] == 384
        chunk = payload[384 * index:384 * (index + 1)]
        norms = []
        for term in range(3):
            norm = 0
            for byte in chunk[128 * term:128 * (term + 1)]:
                real, imag = byte & 15, byte >> 4
                real -= 16 if real >= 8 else 0
                imag -= 16 if imag >= 8 else 0
                norm += real * real + imag * imag
            norms.append(norm)
        assert row["target_norm_sq"] == norms[0]
        assert row["reference_norm_sum_sq"] == norms[1] + norms[2]
        # F uses the mean reference power, so its null ratio has a factor of two.
        assert row["null_power_ratio"] == 2 * norms[0] / (norms[1] + norms[2])
        assert row["fine_calibration"]["status"] == "pending_campaign"
    sums = dict(line.split("  ")[::-1] for line in (output / "sha256sums.txt").read_text().splitlines())
    assert set(sums) == set(runtime_bundle.REQUIRED_RUNTIME_BUNDLE_FILES) - {"sha256sums.txt"}
    assert all(_digest(output / name) == digest for name, digest in sums.items())
    assert runtime_bundle.validate_runtime_weight_bundle(bundle_dir=output)["valid"]


@pytest.mark.parametrize("selection,message", [
    ([], "at least one physical channel"),
    (["--physical-channel", "13"], "physical channel"),
    (["--physical-channel-range", "14:70"], "physical channel"),
])
def test_bundle_cli_refuses_invalid_channel_selection_before_output(tmp_path, selection, message):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match=message):
        runtime_bundle.main(_bundle_args(output) + selection)
    assert not output.exists()


def test_export_refuses_duplicate_profiles_before_output(tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="multiple physical channels map to the same coarse channel"):
        runtime_bundle.export_runtime_weight_bundle(
            receiver_profile_path=RECEIVER, detector_core_profile_path=CORE,
            physical_channels=[29, 29],
            weight_coordinate_system="post_nyquist_zone_normalization", output_dir=output,
        )
    assert not output.exists()


@pytest.fixture
def archive_plan(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    runtime_bundle.main(_bundle_args(bundle) + ["--physical-channel", "29"])
    profile = _read(bundle / "pilot_profiles.json")["profiles"][0]
    product = inputs / "product.npz"
    np.savez(product, physical_channel=np.asarray(29), freq_id=np.asarray(614),
             weights_hash=np.asarray(_digest(bundle / "weights.bin")),
             target_norm_sq=np.asarray([profile["target_norm_sq"]]),
             reference_norm_sum_sq=np.asarray([profile["reference_norm_sum_sq"]]),
             detector_contract_json=np.asarray((bundle / "detector_contract.json").read_text()),
             decision_contract_json=np.asarray(json.dumps({"fine_measurement": {
                 "method": "exact_fine_power_terms", "terms_field": "fine_power_u64"}})),
             fine_power_u64=np.ones((6, 3, 256), dtype=np.uint64))
    scores = inputs / "scores"
    scores.mkdir()
    score = scores / "ch29.npz"
    np.savez(score, required_q16=np.asarray([65535, 65536, 65537, 65536, 1, 2], dtype=np.uint64),
             supported=np.asarray([True, True, True, False, True, False]),
             frame_event_index=np.asarray([10, 10, 10, 99, 99, 99]),
             probability_weight=np.full(6, 1 / 6))
    bulk = [i for i in range(0, 256, 2) if i not in (60, 62, 64)]
    metadata = {"channel": 29, "physical_science_qualified": False, "threshold_activated": False,
                "current_months": ["synthetic-era"], "npz_sha256": _digest(score),
                "frames": 6, "events": 2,
                "original_product_identity": {"path": "../product.npz", "sha256": _digest(product),
                                              "bytes": product.stat().st_size},
                "score_geometry": {"anchor_bin": 62, "designated_half_width": 2,
                    "cfar_rank_zero_based": 62, "bulk_bins": bulk, "pad_factor": 2,
                    "intended_bulk_size": 125, "designated_bins": [60, 61, 62, 63, 64]}}
    _write(scores / "ch29.json", metadata)
    _write(inputs / "decisions.json", {"inputs": {"scores": "scores", "frame_samples": 16384},
           "channels": [{"channel": 29, "outcome": "useful", "deployed_word": 65536,
                         "deployed_at": "knee", "kept_frames": 3, "kept_events": 2, "kappa": 0.5}]})

    def reference(path):
        return {"path": str(path.relative_to(inputs)), "sha256": _digest(path)}

    plan = {"schema_version": "pilotproxy_archive_candidate_plan_v1",
            "decisions": reference(inputs / "decisions.json"),
            "bundle_files": {name: reference(bundle / name) for name in runtime_bundle.REQUIRED_RUNTIME_BUNDLE_FILES},
            "score_metadata": {"29": reference(scores / "ch29.json")}}
    plan_path = inputs / "plan.json"
    _write(plan_path, plan)
    return plan_path, tmp_path / "candidate"


def _candidate_main(monkeypatch, plan, output):
    monkeypatch.setattr(sys, "argv", ["runtime_candidate", "--plan", str(plan), "--output-dir", str(output)])
    return runtime_candidate.main()


def test_candidate_cli_binds_relative_inputs_and_keeps_review_only_status(
    archive_plan, tmp_path, monkeypatch, capsys,
):
    plan, output = archive_plan
    before = {str(p): _digest(p) for p in plan.parent.rglob("*") if p.is_file()}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    capsys.readouterr()
    _candidate_main(monkeypatch, plan, output)
    assert json.loads(capsys.readouterr().out) == {
        "status": "archive_review_only", "candidate_profiles": 1,
        "deployment_configuration_complete": False, "activation_authorized": False,
    }
    report = _read(output / "candidate_review.json")
    assert report["status"] == "archive_review_only"
    assert report["candidate_configuration_complete"] is True
    for field in ("deployment_configuration_complete", "activation_authorized", "science_certification",
                  "consumer_capability_implementation_verified", "null_quantile_threshold_claim"):
        assert report[field] is False
    assert report["kotekan_parameters"] is None
    assert report["permanent_mask_channels"] == report["unmasked_channels"] == []
    assert report["control_channel_37_bound"] is False
    proof = report["profiles"][0]
    assert proof["physical_channel"] == 29 and proof["receiver_channel_id"] == 614
    assert proof["word_q16"] == 65536
    assert proof["decision_population_replay"]["kept_frames"] == 3
    assert proof["decision_population_replay"]["kept_events"] == 2
    assert proof["decision_population_replay"]["kappa"] == pytest.approx(0.5, rel=0, abs=1e-15)
    assert report["input_files_sha256"] == before
    assert (output / "source_plan.json").read_bytes() == plan.read_bytes()
    bundle = output / "bundle"
    assert report["bundle_files_sha256"] == {p.name: _digest(p) for p in bundle.iterdir()}
    assert report["runtime_bundle_validation"]["valid"] is True
    fine = _read(bundle / "pilot_profiles.json")["profiles"][0]["fine_calibration"]
    assert fine["anchor_bin"] == 62 and fine["cfar_rank"] == 62
    assert fine["cfar_multiplier_q16"] == 65536
    assert fine["provenance"]["null_quantile"] is None
    selection = fine["provenance"]["threshold_selection"]
    assert selection["qualification"] == "archive_review_only"
    assert selection["selection_granularity"] == "detector_frame"
    assert {str(p): _digest(p) for p in plan.parent.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("failure", ["changed_decisions", "existing_output", "unknown_argument"])
def test_candidate_cli_refuses_before_publishing_success(archive_plan, monkeypatch, capsys, failure):
    plan, output = archive_plan
    capsys.readouterr()
    if failure == "changed_decisions":
        with (plan.parent / "decisions.json").open("a") as stream:
            stream.write(" ")
        with pytest.raises(ValueError, match="File identity mismatch"):
            _candidate_main(monkeypatch, plan, output)
    elif failure == "existing_output":
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("existing review\n")
        with pytest.raises(ValueError, match="Output directory already exists"):
            _candidate_main(monkeypatch, plan, output)
        assert sentinel.read_text() == "existing review\n"
        assert list(output.iterdir()) == [sentinel]
    else:
        monkeypatch.setattr(sys, "argv", ["runtime_candidate", "--plan", str(plan),
                                        "--output-dir", str(output), "--activate"])
        with pytest.raises(SystemExit) as exc:
            runtime_candidate.main()
        assert exc.value.code == 2
    assert capsys.readouterr().out == ""
    if failure != "existing_output":
        assert not output.exists()
