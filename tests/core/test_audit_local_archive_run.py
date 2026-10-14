from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "tools" / "audit_local_archive_run.py"
SPEC = importlib.util.spec_from_file_location("audit_local_archive_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_local_archive_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_local_archive_run)

CONTRACT_FIXTURE_SCRIPT = Path(__file__).with_name("test_current_product_contract.py")
CONTRACT_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "current_product_contract_fixture", CONTRACT_FIXTURE_SCRIPT
)
assert CONTRACT_FIXTURE_SPEC is not None and CONTRACT_FIXTURE_SPEC.loader is not None
current_product_contract_fixture = importlib.util.module_from_spec(
    CONTRACT_FIXTURE_SPEC
)
CONTRACT_FIXTURE_SPEC.loader.exec_module(current_product_contract_fixture)


def _fixture(tmp_path: Path) -> dict[str, object]:
    inventory = tmp_path / "inventory.jsonl"
    row = {"scope": "scope.raw", "event": "100", "name": "file.h5", "freq_id": 506}
    inventory.write_text(json.dumps(row) + "\n", encoding="utf-8")
    inventory_sha256 = hashlib.sha256(inventory.read_bytes()).hexdigest()
    unit_key = audit_local_archive_run.logical_unit_key(
        row["scope"], row["event"], row["name"]
    )
    run_dir = tmp_path / "run"
    product_dir = run_dir / "_per_pilot"
    staging_dir = tmp_path / "staging"
    product_dir.mkdir(parents=True)
    staging_dir.mkdir()
    package_sha256 = "1" * 64
    kernel_sha256 = "2" * 64
    np.savez(
        product_dir / "506.npz",
        schema_version=np.asarray("pilotproxy_per_pilot_product_v5"),
        freq_id=np.asarray([506], dtype=np.int64),
        physical_channel=np.asarray([36], dtype=np.int32),
        chime_frequency_hz=np.asarray([800e6 - 506 * 390625.0]),
        nfft=np.asarray(16384, dtype=np.int64),
        detector_window_samples=np.asarray(128, dtype=np.int64),
        num_input_streams=np.asarray(2048, dtype=np.int64),
        max_chunks_per_file=np.asarray(-1, dtype=np.int64),
        fine_status=np.asarray("enabled"),
        fine_power_u64=np.zeros((1, 3, 256), dtype=np.uint64),
        detector_version=np.asarray(
            f"source={package_sha256} kernel_sha256={kernel_sha256}"
        ),
        weight_bank_sha256=np.asarray("4" * 64),
        weight_manifest_sha256=np.asarray("5" * 64),
        detector_contract_json=np.asarray(
            json.dumps(
                {
                    "schema_version": "pilotproxy_detector_contract_v1",
                    "fine_reduction": {
                        "pad_factor": 2,
                        "designated_bins": [10],
                    },
                }
            )
        ),
        decision_contract_json=np.asarray(json.dumps({"active": "coarse"})),
        mask_rule=np.asarray("positive excess"),
        sample_rate_hz=np.asarray(390625.0),
        sense=np.asarray(-1),
        target_norm_sq=np.asarray(1),
        reference_norm_sum_sq=np.asarray(2),
        fine_pad_factor=np.asarray(2),
        fine_num_bins=np.asarray(256),
        fine_p_fa=np.asarray(0.001),
        fine_guard_fine_bins=np.asarray(1),
        pilot_below_data_db=np.asarray(11.3),
        bin_enbw_hz=np.asarray(1.0),
        dtv_bandwidth_hz=np.asarray(6.0e6),
        pilot_capture_efficiency=np.asarray(1.0),
        unit_order=np.asarray([unit_key]),
        unit_keys=np.asarray([unit_key]),
        source_event_keys=np.asarray(
            [audit_local_archive_run.source_event_key(unit_key, 506)]
        ),
        unit_scope=np.asarray(["scope.raw"]),
        archive_version=np.asarray(["1"]),
        unit_git_version_tag=np.asarray(["receiver"]),
        unit_input_map_sha256=np.asarray(["3" * 64]),
        unit_collection_server=np.asarray(["host-a"]),
        frame_index=np.asarray([0], dtype=np.int64),
    )
    execution = {
        "preserve_source_order": True,
        "download_workers": 4,
        "max_staged_files": 8,
        "checkpoint_every": 250,
        "staging_dir": str(staging_dir.resolve()),
    }
    scope = {
        "schema_version": "pilotproxy_chime_scan_scope_v1",
        "complete": True,
        "source": "cadc-datatrail",
        "input": {
            "inventory_path": str(inventory.resolve()),
            "inventory_sha256": inventory_sha256,
        },
        "requested_selections": [[506]],
        "allow_partial": False,
        "max_files": None,
        "max_chunks_per_file": None,
        "fine_retention": {"requested": "on", "resolved": "enabled"},
        "execution": execution,
        "execution_attempts": [execution],
        "totals": {
            "pilots_requested": 1,
            "requested": 1,
            "enumerated": 1,
            "completed": 1,
            "capped": 0,
            "failed": 0,
            "quarantined": 0,
            "unprocessed": 0,
            "extra_completed": 0,
        },
        "pilots": [
            {
                "selection": [506],
                "status": "complete",
                "enumerated": 1,
                "completed": 1,
                "capped": 0,
                "failed": 0,
                "quarantined": 0,
                "unprocessed": 0,
                "extra_completed": 0,
            }
        ],
        "terminal_combine": {"status": "combined"},
    }
    (run_dir / "scan_scope.json").write_text(json.dumps(scope), encoding="utf-8")
    return {
        "inventory_path": inventory,
        "run_dir": run_dir,
        "staging_dir": staging_dir,
        "package_sha256": package_sha256,
        "kernel_sha256": kernel_sha256,
        "weight_bank_sha256": "4" * 64,
        "weight_manifest_sha256": "5" * 64,
        "expected_inventory_sha256": inventory_sha256,
        "expected_units": 1,
        "expected_freq_ids": (506,),
    }


def test_closeout_accounts_for_inventory_and_product(tmp_path: Path, monkeypatch) -> None:
    arguments = _fixture(tmp_path)
    monkeypatch.setattr(
        audit_local_archive_run, "validate_current_product_identity", lambda product: None
    )
    report = audit_local_archive_run.audit_run(**arguments)
    assert report["inventory_units"] == 1
    assert report["product_units"] == 1
    assert report["per_freq_units"] == {"506": 1}


def test_closeout_rejects_source_token_difference(tmp_path: Path, monkeypatch) -> None:
    arguments = _fixture(tmp_path)
    arguments["package_sha256"] = "4" * 64
    monkeypatch.setattr(
        audit_local_archive_run, "validate_current_product_identity", lambda product: None
    )
    with pytest.raises(audit_local_archive_run.CloseoutError, match="package source"):
        audit_local_archive_run.audit_run(**arguments)


def test_closeout_rejects_weight_artifact_difference(tmp_path: Path, monkeypatch) -> None:
    arguments = _fixture(tmp_path)
    arguments["weight_bank_sha256"] = "6" * 64
    monkeypatch.setattr(
        audit_local_archive_run, "validate_current_product_identity", lambda product: None
    )
    with pytest.raises(audit_local_archive_run.CloseoutError, match="weight-bank"):
        audit_local_archive_run.audit_run(**arguments)


def test_common_identity_rejects_cross_product_mismatch(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    product_path = arguments["run_dir"] / "_per_pilot" / "506.npz"
    with np.load(product_path, allow_pickle=False) as product:
        reference = audit_local_archive_run.common_product_identity(product)
        second_product = {field: product[field] for field in product.files}
    second_product["detector_contract_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": "pilotproxy_detector_contract_v1",
                "fine_reduction": {"pad_factor": 2, "designated_bins": [20]},
            }
        )
    )
    assert audit_local_archive_run.common_product_identity(second_product) == reference
    candidate = dict(reference)
    candidate["weight_bank_sha256"] = "6" * 64
    with pytest.raises(audit_local_archive_run.CloseoutError, match="weight_bank"):
        audit_local_archive_run.require_common_identity(
            reference, candidate, "521.npz"
        )


def test_common_identity_allows_channel_specific_norms_and_decisions() -> None:
    first = current_product_contract_fixture.current_product()
    first["weight_bank_sha256"] = np.asarray("4" * 64)
    first["weight_manifest_sha256"] = np.asarray("5" * 64)
    first["detector_contract_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": "pilotproxy_detector_contract_v1",
                "fine_reduction": {"pad_factor": 4, "designated_bins": [10]},
            }
        )
    )
    first["weights_hash"] = np.asarray("channel-14-row")
    first["reference_placement_json"] = np.asarray("{\"channel\":14}")

    second = {
        field: value.copy() if isinstance(value, np.ndarray) else value
        for field, value in first.items()
    }
    second["physical_channel"] = np.asarray([15], dtype=np.int32)
    second["freq_id"] = np.asarray([829], dtype=np.int64)
    second["target_norm_sq"] = np.asarray([2], dtype=np.int64)
    second["reference_norm_sum_sq"] = np.asarray([3], dtype=np.int64)
    second["p_target_u64"] = np.asarray([[2]], dtype=np.uint64)
    second["p_ref_lower_u64"] = np.asarray([[1]], dtype=np.uint64)
    second["p_ref_upper_u64"] = np.asarray([[1]], dtype=np.uint64)
    second["p_ref_sum_u64"] = np.asarray([[2]], dtype=np.uint64)
    second["reject_mask"] = np.asarray([[1]], dtype=np.uint8)
    second["weights_hash"] = np.asarray("channel-15-row")
    second["reference_placement_json"] = np.asarray("{\"channel\":15}")
    second["detector_contract_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": "pilotproxy_detector_contract_v1",
                "fine_reduction": {"pad_factor": 4, "designated_bins": [20]},
            }
        )
    )

    audit_local_archive_run.validate_current_product_identity(first)
    audit_local_archive_run.validate_current_product_identity(second)
    first_identity = audit_local_archive_run.common_product_identity(first)
    second_identity = audit_local_archive_run.common_product_identity(second)
    audit_local_archive_run.require_common_identity(
        first_identity, second_identity, "829.npz"
    )


def _replace_product_field(path: Path, field: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as product:
        fields = {name: product[name] for name in product.files}
    fields[field] = value
    np.savez(path, **fields)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physical_channel", np.asarray([35], dtype=np.int32), "physical channel"),
        (
            "chime_frequency_hz",
            np.asarray([1.0], dtype=np.float64),
            "CHIME frequency",
        ),
    ],
)
def test_closeout_rejects_channel_frequency_geometry(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    arguments = _fixture(tmp_path)
    product_path = arguments["run_dir"] / "_per_pilot" / "506.npz"
    _replace_product_field(product_path, field, value)
    monkeypatch.setattr(
        audit_local_archive_run, "validate_current_product_identity", lambda product: None
    )
    with pytest.raises(audit_local_archive_run.CloseoutError, match=message):
        audit_local_archive_run.audit_run(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_event_keys", np.asarray(["wrong-event"]), "source-event"),
        ("archive_version", np.asarray([""]), "receiver identity"),
        ("unit_collection_server", np.asarray([""]), "receiver identity"),
    ],
)
def test_closeout_rejects_incomplete_source_or_receiver_identity(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    arguments = _fixture(tmp_path)
    product_path = arguments["run_dir"] / "_per_pilot" / "506.npz"
    _replace_product_field(product_path, field, value)
    monkeypatch.setattr(
        audit_local_archive_run, "validate_current_product_identity", lambda product: None
    )
    with pytest.raises(audit_local_archive_run.CloseoutError, match=message):
        audit_local_archive_run.audit_run(**arguments)
