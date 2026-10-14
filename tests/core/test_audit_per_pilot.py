from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.product_contract import PER_PILOT_PRODUCT_SCHEMA_TOKEN


SCRIPT = Path(__file__).parents[2] / "tools" / "audit_per_pilot.py"
SPEC = importlib.util.spec_from_file_location("audit_per_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_per_pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_per_pilot)


def _product() -> dict[str, np.ndarray]:
    nfft = 16384
    total_samples = 2 * nfft * 2048
    fine = np.ones((1, 3, 256), dtype=np.uint64)
    return {
        "schema_version": np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        "decision_contract_json": np.asarray(
            json.dumps(audit_per_pilot.current_decision_contract())
        ),
        "detector_contract_json": np.asarray(
            json.dumps(
                {
                    "schema_version": "pilotproxy_detector_contract_v1",
                    "fine_reduction": {"pad_factor": 2},
                }
            )
        ),
        "detector_version": np.asarray(
            "pilot-proxy/0 kernel=2.3.0 K=128 "
            f"{PER_PILOT_PRODUCT_SCHEMA_TOKEN} source={'1' * 64}"
        ),
        "mask_rule": np.asarray(NORMALIZED_POSITIVE_EXCESS_MASK_RULE),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "pilot_frequency_hz": np.asarray([470_309_441.0]),
        "chime_frequency_hz": np.asarray([470_312_500.0]),
        "pilot_in_band": np.asarray([1], dtype=np.uint8),
        "nfft": np.asarray(nfft, dtype=np.int64),
        "detector_window_samples": np.asarray(128, dtype=np.int64),
        "num_input_streams": np.asarray(2048, dtype=np.int64),
        "frame_index": np.asarray([0], dtype=np.int64),
        "p_target_u64": np.asarray([[3]], dtype=np.uint64),
        "p_ref_lower_u64": np.asarray([[2]], dtype=np.uint64),
        "p_ref_upper_u64": np.asarray([[2]], dtype=np.uint64),
        "p_ref_sum_u64": np.asarray([[4]], dtype=np.uint64),
        "coarse_power_ratio": np.asarray([[1.5]], dtype=np.float64),
        "normalized_coarse_power_ratio_db": np.asarray([[10.0 * np.log10(1.5)]]),
        "pilot_excess_db": np.asarray([[10.0 * np.log10(0.5)]]),
        "estimated_data_shelf_snr_db": np.asarray([[0.0]], dtype=np.float64),
        "normalized_pilot_excess": np.asarray([[0.5]], dtype=np.float64),
        "valid": np.asarray([[1]], dtype=np.uint8),
        "reject_mask": np.asarray([[1]], dtype=np.uint8),
        "target_norm_sq": np.asarray(1, dtype=np.int64),
        "reference_norm_sum_sq": np.asarray(2, dtype=np.int64),
        "rational_overflow_count": np.asarray(0, dtype=np.uint64),
        "baseband_power_linear": np.asarray([[1.0]], dtype=np.float64),
        "railed_sample_count": np.asarray([[0]], dtype=np.uint64),
        "fill_sample_count": np.asarray([[0]], dtype=np.uint64),
        "railed_sample_total": np.asarray([[total_samples]], dtype=np.uint64),
        "fine_status": np.asarray("enabled"),
        "fine_num_bins": np.asarray(256, dtype=np.int64),
        "fine_power_u64": fine,
        "integrated_spectrum_before_mask": np.ones(nfft, dtype=np.float64),
        "integrated_spectrum_after_mask": np.zeros(nfft, dtype=np.float64),
        "unit_order": np.asarray(["unit"], dtype=str),
        "unit_time0_ctime": np.asarray([1_700_000_000.0]),
        "unit_time0_fpga": np.asarray([1], dtype=np.uint64),
        "unit_event_id": np.asarray([1], dtype=np.int64),
        "unit_delta_time": np.asarray([nfft / 390_625.0]),
        "archive_version": np.asarray(["1"], dtype=str),
        "unit_git_version_tag": np.asarray(["receiver-build"], dtype=str),
        "unit_input_map_sha256": np.asarray(["a" * 64], dtype=str),
        "unit_collection_server": np.asarray(["host-a"], dtype=str),
        "unit_scope": np.asarray(["triggered"], dtype=str),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
    }


def test_audit_rejects_corrupt_reference_split(tmp_path: Path) -> None:
    product = _product()
    valid_path = tmp_path / "valid.npz"
    np.savez(valid_path, **product)
    valid_audit, _stats = audit_per_pilot.audit_file(valid_path, None, None)
    assert valid_audit.failures == []

    product["p_ref_lower_u64"] = np.asarray([[3]], dtype=np.uint64)
    path = tmp_path / "844.npz"
    np.savez(path, **product)

    audit, _stats = audit_per_pilot.audit_file(path, None, None)

    assert (
        "lower + upper reference powers match p_ref_sum_u64 without overflow"
        in audit.failures
    )


def test_reference_split_rejects_uint64_wrap() -> None:
    maximum = np.iinfo(np.uint64).max

    assert not audit_per_pilot.reference_split_matches(
        np.asarray([[maximum]], dtype=np.uint64),
        np.asarray([[1]], dtype=np.uint64),
        np.asarray([[0]], dtype=np.uint64),
    )


def test_audit_rejects_wrapped_sample_counts(tmp_path: Path) -> None:
    product = _product()
    product["railed_sample_count"] = np.asarray(
        [[np.iinfo(np.uint64).max]], dtype=np.uint64
    )
    product["fill_sample_count"] = np.asarray([[1]], dtype=np.uint64)
    path = tmp_path / "844.npz"
    np.savez(path, **product)

    audit, _stats = audit_per_pilot.audit_file(path, None, None)

    assert (
        "railed_sample_count + fill_sample_count <= railed_sample_total"
        in audit.failures
    )


def test_audit_rejects_malformed_input_map_hash(tmp_path: Path) -> None:
    product = _product()
    product["unit_input_map_sha256"] = np.asarray(["A" * 64], dtype=str)
    path = tmp_path / "844.npz"
    np.savez(path, **product)

    audit, _stats = audit_per_pilot.audit_file(path, None, None)

    assert "input-map hashes are lowercase SHA-256" in audit.failures


def test_audit_rejects_empty_unit_scope(tmp_path: Path) -> None:
    product = _product()
    product["unit_scope"] = np.asarray([""], dtype=str)
    path = tmp_path / "844.npz"
    np.savez(path, **product)

    audit, _stats = audit_per_pilot.audit_file(path, None, None)

    assert "unit scopes are nonempty" in audit.failures


def test_audit_rejects_archive_unit_without_receiver_state(tmp_path: Path) -> None:
    product = _product()
    product["unit_git_version_tag"] = np.asarray([""], dtype=str)
    path = tmp_path / "844.npz"
    np.savez(path, **product)

    audit, _stats = audit_per_pilot.audit_file(path, None, None)

    assert "archive units carry receiver-state identity" in audit.failures
