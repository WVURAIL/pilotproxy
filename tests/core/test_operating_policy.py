"""The threshold rule's decisions compile to the permanent mask and the deployed words; refusal paths."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy import operating_policy as op, runtime_calibration as rc
from pilot_proxy.config.project import default_project
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_POST_NYQUIST_ZONE
from pilot_proxy.runtime_bundle import REQUIRED_RUNTIME_BUNDLE_FILES, export_runtime_weight_bundle

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "tests/data/threshold_decisions_2026-10-12.json"   # tools/threshold_rule.py of 2026-10-12
_spec = importlib.util.spec_from_file_location("policy_example", ROOT / "tools/make_operating_policy_example.py")
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def ref(path):
    return {"path": str(path), "sha256": rc.sha256(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return ref(path)


def read(path):
    return json.loads(Path(path).read_text())


def refresh_bundle(directory):
    names = [n for n in REQUIRED_RUNTIME_BUNDLE_FILES if n != "sha256sums.txt"]
    (directory / "sha256sums.txt").write_text("".join(f"{rc.sha256(directory / name)}  {name}\n" for name in names))
    return {name: ref(directory / name) for name in REQUIRED_RUNTIME_BUNDLE_FILES}


def calibrated(word, source):
    mask = np.zeros(256, dtype=bool)
    mask[::2] = True
    mask[(62 + np.arange(-4, 5)) % 256] = False
    words = [hex(sum(int(mask[i * 64 + j]) << j for j in range(64))) for i in range(4)]
    return {"status": "calibrated", "decision_version": "fine_decision_v1", "anchor_bin": 62,
            "designated_half_width": 2, "bulk_mask_words_hex": words, "cfar_rank": 61,
            "cfar_multiplier_q16": int(word),
            "provenance": {"epochs": ["test-era"], "null_quantile": .95, "source_product_sha256": [source]}}


@pytest.fixture
def case(tmp_path):
    path = example.make_example(tmp_path / "example")
    return path, read(path), path.parent


def analyze(case):
    path, plan, _ = case
    write(path, plan)
    return op.analyze_policy(path)


def change_decisions(case, change):
    path, plan, directory = case
    decisions = read(directory / "decisions.json")
    change({row["channel"]: row for row in decisions["channels"]})
    plan["decisions"] = write(directory / "decisions.json", decisions)


def change_bundle(case, change):
    _, plan, directory = case
    profiles = read(directory / "bundle/pilot_profiles.json")
    change({row["physical_channel"]: row for row in profiles["profiles"]})
    write(directory / "bundle/pilot_profiles.json", profiles)
    plan["bundle_files"] = refresh_bundle(directory / "bundle")


def test_decisions_compile_to_the_permanent_mask_and_the_deployed_word(case):
    path, _, tmp = case
    report = op.write_policy(path, tmp / "review", require_complete=True)
    assert report["deployment_configuration_complete"] is True
    assert report["receiver_channel_id_namespace"] == "chime_freq_id"
    assert report["permanent_mask_freq_ids"] == [844] and report["require_fine_freq_ids"] == [614]
    assert report["kotekan_parameters"] == {"permanent_mask_freq_ids": [844], "require_fine_freq_ids": [614]}
    (rejected,) = [row for row in report["channels"] if row["physical_channel"] == 14]
    assert rejected["threshold"] == {"kind": "reject_all"} and rejected["reason"] == "above tolerance"
    assert report["required_threshold_bindings"][0]["eta_q16"] == "65536"
    assert not report["activation_authorized"] and not report["science_certification"]
    assert not report["consumer_capability_implementation_verified"]
    assert op.validate_policy(tmp / "review") == report
    with pytest.raises(FileExistsError):
        op.write_policy(path, tmp / "review")


def test_the_control_band_is_reported_not_assigned(case):
    report = analyze(case)
    assert [row["physical_channel"] for row in report["not_in_policy"]] == [37]
    assert 37 not in [row["physical_channel"] for row in report["channels"]]


def test_a_deployed_channel_is_pending_until_the_bundle_carries_its_word(case):
    change_bundle(case, lambda rows: rows[29].update(fine_calibration=read(case[2] / "bundle/pilot_profiles.json")
                                                      ["profiles"][0]["fine_calibration"]))
    report = analyze(case)
    assert report["deployment_configuration_complete"] is False and report["kotekan_parameters"] is None
    assert report["pending_freq_ids"] == [614] and report["require_fine_freq_ids"] == []
    assert report["permanent_mask_freq_ids"] == [844]
    with pytest.raises(ValueError, match="incomplete"):
        op.write_policy(case[0], case[2] / "refused", require_complete=True)


def test_the_bundle_word_must_equal_the_decided_word(case):
    change_bundle(case, lambda rows: rows[29]["fine_calibration"].update(cfar_multiplier_q16=65537))
    with pytest.raises(ValueError, match="differs from the decided word"):
        analyze(case)


@pytest.mark.parametrize("change, match", [
    (lambda rows: rows[29].update(deployed_word=None), "deployed word"),
    (lambda rows: rows[29].update(deployed_word=0), "deployed word"),
    (lambda rows: rows[29].update(deployed_word="65536.0"), "deployed word"),
    (lambda rows: rows[14].update(reason=None), "reason"),
    (lambda rows: rows[14].update(outcome="excise"), "outcome"),
])
def test_each_decision_states_its_word_or_its_reason(case, change, match):
    change_decisions(case, change)
    with pytest.raises((TypeError, ValueError), match=match):
        analyze(case)


def test_the_namespace_must_match_the_receiver_profile(case):
    case[1]["receiver_channel_id_namespace"] = "chord_freq_id"
    with pytest.raises(ValueError, match="namespace"):
        analyze(case)
    case[1]["receiver_channel_id_namespace"] = "frequency_index"
    with pytest.raises(ValueError, match="namespace"):
        analyze(case)


def test_the_chime_freq_id_is_the_projects(case, monkeypatch):
    project = default_project()

    class Shifted:
        def target_freq_id(self, band):
            return project.target_freq_id(band) + 1

    monkeypatch.setattr(op, "default_project", lambda: Shifted())
    with pytest.raises(ValueError, match="differs from the project's"):
        analyze(case)


def test_a_channel_needs_its_bundle_profile(case):
    decisions = read(case[2] / "decisions.json")
    decisions["channels"].append({**decisions["channels"][0], "channel": 30})   # no bundle profile for 30
    case[1]["decisions"] = write(case[2] / "decisions.json", decisions)
    with pytest.raises(ValueError, match="exactly one profile"):
        analyze(case)


def test_declared_consumer_must_explicitly_admit_support_and_mask_capabilities(case):
    case[1]["consumer"]["capabilities"].remove("dtv_permanent_mask_freq_ids_v1")
    with pytest.raises(ValueError, match="capabilities"):
        analyze(case)


def test_era_and_source_identity_are_required(case):
    case[1]["era"]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="File identity mismatch"):
        analyze(case)
    del case[1]["era"]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        analyze(case)


def test_source_change_during_compilation_cannot_be_sealed(case, monkeypatch):
    real = op._binding

    def change_source(*args, **kwargs):
        result = real(*args, **kwargs)
        evidence = case[2] / "synthetic_evidence.json"
        evidence.write_text(evidence.read_text() + "\n")
        return result

    monkeypatch.setattr(op, "_binding", change_source)
    with pytest.raises(ValueError, match="Source changed during policy compilation"):
        analyze(case)


def test_conflicting_path_identities_are_refused():
    with pytest.raises(ValueError, match="Conflicting source identities"):
        op._remember({"/fixture": "a" * 64}, {"/fixture": "b" * 64})


def test_validation_detects_changed_output_and_source(case):
    path, _, tmp = case
    op.write_policy(path, tmp / "review")
    output = tmp / "review/policy.json"
    output.write_text(output.read_text() + "\n")
    with pytest.raises(ValueError, match="Compiled policy changed"):
        op.validate_policy(tmp / "review")


def test_cli_compiles_and_validates_without_activation(case, capsys):
    from pilot_proxy.cli import main

    path, _, tmp = case
    main(["compile-operating-policy", "--plan", str(path), "--output-dir", str(tmp / "cli-review"), "--require-complete"])
    main(["validate-operating-policy", "--policy-dir", str(tmp / "cli-review")])
    assert "activation is not authorized" in capsys.readouterr().out


@pytest.mark.skipif(not FROZEN.is_file(), reason="withheld from the code-only cut (CHIME_DERIVED.md)")
def test_the_frozen_decisions_compile_for_the_chime_upgrade(tmp_path):
    receiver = ROOT / "configs/receiver_profiles/chime_dtv_fengine.json"
    bundle = tmp_path / "bundle"
    export_runtime_weight_bundle(
        receiver_profile_path=receiver,
        detector_core_profile_path=ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json",
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        physical_channels=list(range(14, 37)), output_dir=bundle,
    )
    evidence = write(tmp_path / "evidence.json", {"scope": "test of the frozen decisions"})
    plan_path = tmp_path / "plan.json"
    plan = {"schema_version": op.PLAN_SCHEMA, "receiver_profile": ref(receiver),
            "receiver_channel_id_namespace": "chime_freq_id", "decisions": ref(FROZEN),
            "bundle_files": refresh_bundle(bundle),
            "era": {"id": "current-2026-10", "scope": "the decisions' current eras", "evidence": [ref(FROZEN)]},
            "consumer": {"name": op.CONSUMER_NAME, "version": "test", "build_sha256": evidence["sha256"],
                         "capabilities": sorted(op.REQUIRED_CAPABILITIES), "evidence": [evidence]}}
    decisions = {row["channel"]: row for row in read(FROZEN)["channels"]}
    project = default_project()
    deployed = {project.target_freq_id(str(ch)): row["deployed_word"]
                for ch, row in decisions.items() if row["outcome"] == "useful" and ch <= 36}
    # not deployed: above tolerance (17, 24), out of time (22, 30, 31, 35, 36), excluded (19, 33)
    assert sorted(ch for ch, row in decisions.items() if row["outcome"] == "not useful") == [17, 19, 22, 24, 30, 31, 33, 35, 36]
    permanent = [506, 521, 552, 583, 598, 690, 721, 798]        # channel 19 (767) is left unmasked

    write(plan_path, plan)
    report = op.analyze_policy(plan_path)
    assert report["permanent_mask_freq_ids"] == permanent and report["unmasked_freq_ids"] == [767]
    assert report["pending_freq_ids"] == sorted(deployed) and report["kotekan_parameters"] is None
    assert [row["physical_channel"] for row in report["not_in_policy"]] == [37]
    for row in report["channels"]:
        decision = decisions[row["physical_channel"]]
        assert row["frequency_ids"] == [project.target_freq_id(str(row["physical_channel"]))]
        if decision["outcome"] == "useful":
            assert row["threshold"] == {"kind": "finite", "eta_q16": str(decision["deployed_word"])}
        elif decision["reason"] in op.UNMASKED_REASONS:
            assert row["threshold"] == {"kind": "pass_all"} and row["reason"] == decision["reason"]
        else:
            assert row["threshold"] == {"kind": "reject_all"} and row["reason"] == decision["reason"]

    profiles = read(bundle / "pilot_profiles.json")
    for profile in profiles["profiles"]:
        if profile["receiver_channel_id"] in deployed:
            profile["fine_calibration"] = calibrated(deployed[profile["receiver_channel_id"]], rc.sha256(FROZEN))
    write(bundle / "pilot_profiles.json", profiles)
    plan["bundle_files"] = refresh_bundle(bundle)
    write(plan_path, plan)
    report = op.analyze_policy(plan_path)
    assert report["deployment_configuration_complete"] is True
    assert report["kotekan_parameters"] == {"permanent_mask_freq_ids": permanent,
                                            "require_fine_freq_ids": sorted(deployed)}
    assert {int(b["receiver_channel_id"]): int(b["eta_q16"]) for b in report["required_threshold_bindings"]} == deployed
    assert not set(report["permanent_mask_freq_ids"]) & set(report["require_fine_freq_ids"])
    assert not set(report["unmasked_freq_ids"]) & (set(permanent) | set(deployed))


def with_control(case, *, word=72850, capability=True):
    """Re-export the example bundle with a channel 37 profile at the control status."""
    path, plan, directory = case
    bundle = directory / "bundle37"
    export_runtime_weight_bundle(
        receiver_profile_path=directory / "receiver_profile.json",
        detector_core_profile_path=directory / "detector_core_profile.json",
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        physical_channels=[14, 29, 37], output_dir=bundle,
    )
    profiles = read(bundle / "pilot_profiles.json")
    rows = {row["physical_channel"]: row for row in profiles["profiles"]}
    source = plan["decisions"]["sha256"]
    rows[29]["fine_calibration"] = calibrated(65536, source)
    rows[37]["fine_calibration"] = {**calibrated(word, source), "status": "control"}
    write(bundle / "pilot_profiles.json", profiles)
    plan["bundle_files"] = refresh_bundle(bundle)
    if capability:
        plan["consumer"]["capabilities"] = sorted(op.REQUIRED_CAPABILITIES | {op.CONTROL_CAPABILITY})


def test_the_channel_37_control_compiles_to_control_freq_ids(case):
    with_control(case)
    path, _, tmp = case
    write(path, case[1])
    report = op.write_policy(path, tmp / "review", require_complete=True)
    assert report["not_in_policy"] == []
    assert report["control_freq_ids"] == [491]
    assert report["kotekan_parameters"] == {"permanent_mask_freq_ids": [844], "require_fine_freq_ids": [614],
                                            "control_freq_ids": [491]}
    (control,) = [row for row in report["channels"] if row["physical_channel"] == 37]
    assert control["threshold"] == {"kind": "control", "eta_q16": "72850"}
    assert report["control_bindings"][0]["eta_q16"] == "72850"
    assert 491 not in report["require_fine_freq_ids"] + report["permanent_mask_freq_ids"]
    assert op.validate_policy(tmp / "review") == report


@pytest.mark.parametrize("change, match", [
    (dict(word=72851), "differs from the decided word"),
    (dict(capability=False), "dtv_control_freq_ids_v1"),
])
def test_the_control_is_refused_off_its_word_or_without_the_consumer_capability(case, change, match):
    with_control(case, **change)
    with pytest.raises(ValueError, match=match):
        analyze(case)


def test_a_control_profile_needs_the_control_decision_word(case):
    with_control(case)
    change_decisions(case, lambda rows: rows[37].update(outcome="not useful", reason="no control word",
                                                         deployed_word=None))
    with pytest.raises(ValueError, match="control profile but the decision has no word"):
        analyze(case)
