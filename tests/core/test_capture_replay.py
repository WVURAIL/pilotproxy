"""Native fixture replay, authenticated refusals, and calibration handoff."""

import copy
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy import capture_replay as replay
from pilot_proxy import runtime_calibration as calibration
from pilot_proxy.capture_reference import (
    CHORD_METADATA_SIZE,
    CaptureGeometry,
    expected_products,
)
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_RAW_INPUT

SPEC = importlib.util.spec_from_file_location(
    "capture_example",
    Path(__file__).resolve().parents[2] / "tools/make_capture_replay_example.py",
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def read(path):
    return json.loads(path.read_text())


def make_plan(tmp_path, **kwargs):
    return example.make_example(tmp_path / "inputs", **kwargs)


def update_contract(plan_path, field, mutate):
    plan = read(plan_path)
    path = plan_path.parent / plan[field]["path"]
    data = read(path)
    mutate(data)
    example.write_json(path, data)
    plan[field] = example.file_ref(path, plan_path.parent)
    producer_path = plan_path.parent / plan["producer"]["path"]
    producer = read(producer_path)
    producer["binding"][f"{field}_sha256"] = plan[field]["sha256"]
    example.write_json(producer_path, producer)
    plan["producer"] = example.file_ref(producer_path, plan_path.parent)
    example.write_json(plan_path, plan)


def update_raw(plan_path, kind, change):
    plan = read(plan_path)
    path = plan_path.parent / plan["raw_files"][kind][0]["path"]
    data = bytearray(path.read_bytes())
    change(data)
    path.write_bytes(data)
    plan["raw_files"][kind][0] = example.file_ref(path, plan_path.parent)
    example.write_json(plan_path, plan)


def test_example_preserves_exact_valid_and_invalid_scores(tmp_path):
    plan = make_plan(tmp_path)
    output = tmp_path / "replay"
    report = replay.write_replay(plan, output)
    assert report["status"] == "invalid_support"
    assert report["product_agreement"] is True
    assert (
        report["num_blocks"],
        report["num_frequencies"],
        report["score_rows"],
        report["invalid_rows"],
    ) == (2, 2, 2, 1)
    assert report["telescope_qualification"] is False
    rows = read(output / "scores_CH14.json")["scores"]
    assert rows[0]["expected_coarse_powers"] == ["35040840", "29834371", "35519122"]
    assert {
        key: rows[0][key]
        for key in (
            "valid",
            "n_bulk",
            "rank_bin",
            "limiting_bin",
            "required_multiplier_q16",
        )
    } == {
        "valid": True,
        "n_bulk": 123,
        "rank_bin": 68,
        "limiting_bin": 64,
        "required_multiplier_q16": "82009",
    }
    assert rows[1]["valid"] is False
    assert rows[1]["n_bulk"] == 0
    assert rows[1]["required_multiplier_q16"] is None
    assert rows[1]["expected_mask"] == 0
    assert (
        rows[1]["voltage_frames"][0]["payload_offset"]
        > rows[0]["voltage_frames"][0]["payload_offset"]
    )
    assert len(read(output / "population_CH14.json")["scores"]) == 2
    profile = read(output / "profile_CH14.json")
    with pytest.raises(ValueError, match="Invalid score/support"):
        calibration._load_population(
            output / "population_CH14.json",
            profile,
            calibration.sha256(output / "profile_CH14.json"),
            "calibration",
        )
    assert replay.validate_replay(output) == report


@pytest.mark.parametrize("kind", ("masks", "powers"))
def test_product_mismatch_is_retained_and_suppresses_population(tmp_path, kind):
    plan = make_plan(tmp_path)
    update_raw(
        plan,
        kind,
        lambda data: data.__setitem__(
            4 + CHORD_METADATA_SIZE, data[4 + CHORD_METADATA_SIZE] ^ 1
        ),
    )
    output = tmp_path / "replay"
    report = replay.write_replay(plan, output)
    assert report["status"] == "product_mismatch"
    assert report["product_agreement"] is False
    assert report["invalid_rows"] == 1
    assert len(read(output / "scores_CH14.json")["scores"]) == 2
    assert not (output / "population_CH14.json").exists()
    assert replay.validate_replay(output) == report


def test_unbound_frequency_must_also_match(tmp_path):
    plan = make_plan(tmp_path)
    update_raw(plan, "masks", lambda data: data.__setitem__(5 + CHORD_METADATA_SIZE, 1))
    artifacts = replay.replay_capture(plan)
    assert artifacts["replay.json"]["status"] == "product_mismatch"
    assert "population_CH14.json" not in artifacts


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    (
        (
            "packing_contract",
            lambda p: p.__setitem__("num_dishes", 2),
            "receiver input population",
        ),
        (
            "packing_contract",
            lambda p: p.__setitem__("sample_rate_hz", "1"),
            "sample-rate",
        ),
        (
            "packing_contract",
            lambda p: p.__setitem__("spectral_sense", "inverted"),
            "spectral sense",
        ),
        (
            "packing_contract",
            lambda p: p.__setitem__("sample_encoding", "complex64"),
            "native packing",
        ),
        (
            "packing_contract",
            lambda p: p.__setitem__(
                "time_reverse_detector_windows_before_kernel", False
            ),
            "preprocessing",
        ),
        (
            "packing_contract",
            lambda p: p["frequency_map"][0].__setitem__("rf_center_hz", "1"),
            "RF-center",
        ),
        (
            "packing_contract",
            lambda p: p["timing"]["utc_reference"].__setitem__(
                "tick_period_seconds", "1"
            ),
            "FPGA/sample-rate",
        ),
        (
            "input_state_contract",
            lambda p: p["streams"][1].__setitem__(
                "input_id", p["streams"][0]["input_id"]
            ),
            "Repeated input identity",
        ),
        (
            "input_state_contract",
            lambda p: p["streams"][1].__setitem__("polarization_index", 0),
            "polarization order",
        ),
        (
            "input_state_contract",
            lambda p: p.__setitem__("valid_until_fpga_seq_num", 1000001),
            "validity interval",
        ),
        (
            "receiver_profile",
            lambda p: p.__setitem__("receiver_profile_id", "different-receiver"),
            "bundle receiver",
        ),
    ),
)
def test_contract_mismatch_refused_after_hash_refresh(tmp_path, field, mutate, message):
    plan = make_plan(tmp_path)
    update_contract(plan, field, mutate)
    with pytest.raises(ValueError, match=message):
        replay.write_replay(plan, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


@pytest.mark.parametrize(
    ("offset", "format_string", "value", "message"),
    (
        (32, "<q", 999999, "timing"),
        (40, "<i", 2, "timing"),
        (68, "<i", 9, "layout"),
        (76, "<i", 4096, "layout"),
        (360, "<q", 2, "layout"),
        (440, "<q", 1, "layout"),
        (520, "<q", 1, "layout"),
        (532, "<i", 2409, "coarse_freq"),
        (49684, "<i", 2, "upchannelized"),
    ),
)
def test_native_metadata_mismatch_refused_after_hash_refresh(
    tmp_path, offset, format_string, value, message
):
    plan = make_plan(tmp_path)
    update_raw(
        plan,
        "voltage",
        lambda data: struct.pack_into(format_string, data, 4 + offset, value),
    )
    with pytest.raises(ValueError, match=message):
        replay.replay_capture(plan)


@pytest.mark.parametrize("field", ("enabled", "samples_per_detector_frame"))
def test_detector_configuration_mismatch_refused(tmp_path, field):
    plan_path = make_plan(tmp_path)
    plan = read(plan_path)
    producer_path = plan_path.parent / plan["producer"]["path"]
    producer = read(producer_path)
    producer["detector"][field] = (
        False if field != "samples_per_detector_frame" else 16384
    )
    example.write_json(producer_path, producer)
    plan["producer"] = example.file_ref(producer_path, plan_path.parent)
    example.write_json(plan_path, plan)
    with pytest.raises(ValueError, match="detector configuration"):
        replay.replay_capture(plan_path)


def test_native_hash_change_refused(tmp_path):
    plan = make_plan(tmp_path)
    raw = plan.parent / "voltage.raw"
    raw.write_bytes(raw.read_bytes() + b"x")
    with pytest.raises(ValueError, match="File identity mismatch"):
        replay.replay_capture(plan)


@pytest.mark.parametrize(
    "change", ("omit", "duplicate", "drop_invalid", "duplicate_unit")
)
def test_profile_and_population_membership_refused(tmp_path, change):
    path = make_plan(tmp_path)
    plan = read(path)
    if change == "omit":
        plan["profiles"] = []
    elif change == "duplicate":
        plan["profiles"].append(copy.deepcopy(plan["profiles"][0]))
    elif change == "drop_invalid":
        plan["profiles"][0]["population"]["units"].pop()
    else:
        plan["profiles"][0]["population"]["units"][1]["independent_unit_id"] = plan[
            "profiles"
        ][0]["population"]["units"][0]["independent_unit_id"]
    example.write_json(path, plan)
    with pytest.raises(
        ValueError, match="profiles|profile|every captured block|Repeated population"
    ):
        replay.replay_capture(path)


def test_raw_coordinate_weights_replay_without_reversal(tmp_path):
    path = make_plan(tmp_path, weight_coordinate_system=WEIGHT_COORDINATE_RAW_INPUT)
    report = replay.replay_capture(path)["replay.json"]
    assert report["product_agreement"] is True
    assert report["invalid_rows"] == 1


def test_no_population_export_without_supplied_declaration(tmp_path):
    path = make_plan(tmp_path)
    plan = read(path)
    del plan["profiles"][0]["population"]
    example.write_json(path, plan)
    artifacts = replay.replay_capture(path)
    assert "population_CH14.json" not in artifacts
    assert len(artifacts["scores_CH14.json"]["scores"]) == 2


def test_relabeling_identical_capture_does_not_change_trial_identity(tmp_path):
    path = make_plan(tmp_path, valid_only=True)
    first = replay.replay_capture(path)
    plan = read(path)
    plan["capture_id"] = "relabel"
    for unit in plan["profiles"][0]["population"]["units"]:
        unit["trial_id"] += "-relabel"
    example.write_json(path, plan)
    second = replay.replay_capture(path)
    assert [row["trial_id"] for row in first["population_CH14.json"]["scores"]] == [
        row["trial_id"] for row in second["population_CH14.json"]["scores"]
    ]
    assert first["profile_CH14.json"] == second["profile_CH14.json"]


def test_separate_capture_populations_share_profile(tmp_path):
    path = make_plan(tmp_path, valid_only=True)
    calibration_output = tmp_path / "calibration-replay"
    replay.write_replay(path, calibration_output)
    plan = read(path)
    plan["capture_id"] = "synthetic-evaluation-example"
    plan["first_fpga_seq_num"] += 16384
    population = plan["profiles"][0]["population"]
    population["role"] = "evaluation"
    population["population_id"] = "synthetic-evaluation-fixture"
    for unit in population["units"]:
        unit["trial_id"] += "-evaluation"
        unit["independent_unit_id"] += "-evaluation"
    geometry = CaptureGeometry(
        num_times=8192, k=64, frequency_ids=(2408, 1600), num_dishes=1
    )
    bundle = read(path.parent / "bundle/pilot_profiles.json")
    weights = (path.parent / "bundle/weights.bin").read_bytes()
    recordings = {kind: bytearray() for kind in ("voltage", "masks", "powers")}
    rng = np.random.default_rng(20261007)
    for block in range(2):
        voltage = rng.integers(0, 256, geometry.voltage_frame_size, dtype=np.uint8)
        products = expected_products(voltage, bundle, weights, geometry=geometry)
        payloads = {
            "voltage": voltage.tobytes(),
            "masks": products.masks.tobytes(),
            "powers": products.powers.astype("<u8").tobytes(),
        }
        for kind, payload in payloads.items():
            recordings[kind].extend(
                example.native_frame(
                    payload,
                    kind=kind,
                    sequence=plan["first_fpga_seq_num"] + block * 8192,
                    counter=block,
                    geometry=geometry,
                )
            )
    for kind, data in recordings.items():
        target = path.parent / f"evaluation-{kind}.raw"
        target.write_bytes(data)
        plan["raw_files"][kind] = [example.file_ref(target, path.parent)]
    evaluation_plan = path.parent / "evaluation-plan.json"
    example.write_json(evaluation_plan, plan)
    evaluation_output = tmp_path / "evaluation-replay"
    replay.write_replay(evaluation_plan, evaluation_output)
    assert (calibration_output / "profile_CH14.json").read_bytes() == (
        evaluation_output / "profile_CH14.json"
    ).read_bytes()


def test_output_and_input_tampering_refused_by_revalidation(tmp_path):
    path = make_plan(tmp_path)
    output = tmp_path / "replay"
    replay.write_replay(path, output)
    with pytest.raises(ValueError, match="must not exist"):
        replay.write_replay(path, output)
    scores = output / "scores_CH14.json"
    original = scores.read_bytes()
    scores.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="output identity mismatch"):
        replay.validate_replay(output)
    scores.write_bytes(original)
    (path.parent / "synthetic_gains.json").write_text("changed")
    with pytest.raises(ValueError, match="File identity mismatch"):
        replay.validate_replay(output)
