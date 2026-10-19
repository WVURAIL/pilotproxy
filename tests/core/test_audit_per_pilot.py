from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.product_contract import (
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    PER_PILOT_PRODUCT_V5_SCHEMA_TOKEN,
    coarse_columns_from_fine,
)


SCRIPT = Path(__file__).parents[2] / "tools" / "audit_per_pilot.py"
SPEC = importlib.util.spec_from_file_location("audit_per_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_per_pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_per_pilot)


def _product() -> dict[str, np.ndarray]:
    nfft = 16384
    total_samples = 2 * nfft * 2048
    # Frame powers (3, 2, 2) are the all-bin sums of these fine powers.
    fine = np.zeros((1, 3, 256), dtype=np.uint64)
    fine[0, :, 0] = (3, 2, 2)
    return {
        "schema_version": np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        "decision_contract_json": np.asarray(
            json.dumps(audit_per_pilot.current_decision_contract())
        ),
        "detector_contract_json": np.asarray(
            json.dumps(
                {
                    "schema_version": "pilotproxy_detector_contract_v2",
                    "fine_reduction": {
                        "pad_factor": 2,
                        "coarse_power_definition": "sum_of_fxfft256_fine_powers",
                    },
                }
            )
        ),
        "detector_version": np.asarray(
            "pilot-proxy/0 kernel=3.0.0 K=128 "
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
        "pilot_below_data_db": np.asarray(11.3),
        "bin_enbw_hz": np.asarray(3051.7578125),
        "dtv_bandwidth_hz": np.asarray(6.0e6),
        "pilot_capture_efficiency": np.asarray(1.0),
    }


def _consistent(product: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    product.update(coarse_columns_from_fine(product))
    return product


def test_audit_accepts_a_v6_product_and_enforces_the_fine_sum_invariant(tmp_path: Path) -> None:
    product = _consistent(_product())
    path = tmp_path / "844.npz"
    np.savez(path, **product)
    audit, _stats = audit_per_pilot.audit_file(path, None, None)
    assert audit.failures == []

    # One count added to one fine bin breaks the exact sum.
    product["fine_power_u64"] = product["fine_power_u64"].copy()
    product["fine_power_u64"][0, 0, 1] += np.uint64(1)
    np.savez(path, **product)
    audit, _stats = audit_per_pilot.audit_file(path, None, None)
    assert "frame powers == all-bin sums of fine_power_u64 (v6 invariant)" in audit.failures


def test_audit_reads_a_v5_product_of_record(tmp_path: Path) -> None:
    product = _consistent(_product())
    product["schema_version"] = np.asarray(PER_PILOT_PRODUCT_V5_SCHEMA_TOKEN)
    product["detector_version"] = np.asarray(
        f"pilot-proxy/0 kernel=2.3.0 K=128 {PER_PILOT_PRODUCT_V5_SCHEMA_TOKEN} source={'1' * 64}"
    )
    product["detector_contract_json"] = np.asarray(json.dumps(
        {"schema_version": "pilotproxy_detector_contract_v1", "fine_reduction": {"pad_factor": 2}}
    ))
    product["decision_contract_json"] = np.asarray(
        json.dumps(audit_per_pilot.v5_decision_contract())
    )
    # v5 stored row-domain powers that are not the fine sums: reported, not failed.
    product["fine_power_u64"] = product["fine_power_u64"] * np.uint64(256)
    path = tmp_path / "844.npz"
    np.savez(path, **product)
    audit, _stats = audit_per_pilot.audit_file(path, None, None)
    assert audit.failures == []
    assert any("v5 coarse ratio from fine sums" in note for note in audit.notes)

    # Any other change to the v5 decision contract fails, not only a method.
    decision = audit_per_pilot.v5_decision_contract()
    decision["active_decision"]["output_field"] = "mask"
    product["decision_contract_json"] = np.asarray(json.dumps(decision))
    np.savez(path, **product)
    audit, _stats = audit_per_pilot.audit_file(path, None, None)
    assert "decision_contract_json" in audit.failures


def test_v5_decision_contract_is_the_stored_record() -> None:
    # The exact decision_contract_json a v5 product of record stores.
    stored = (
        '{"active_decision":{"implementation":"host_exact_integer_comparison",'
        '"method":"coarse_normalized_positive_excess","output_field":"reject_mask"},'
        '"fine_candidate_decision":{"active":false,"calibration_status":"pending_campaign",'
        '"method":"fine_order_statistic_cfar"},"fine_measurement":{"method":'
        '"exact_fine_power_terms","no_scan_time_decision":"The scan applies no fine '
        'decision. A float ratio formed with numpy.fft was previously stored and was '
        'never bit-equal to the deployed statistic; it is no longer written.",'
        '"role":"measurement_only_no_scan_time_decision","terms_field":"fine_power_u64",'
        '"terms_note":"exact uint64 [3, num_bins] from the frozen fxfft256: the deployed '
        "fine statistic's numerators and denominators. The scan stores these and nothing "
        "derived from them, so the ratio, the CFAR baseline, the designated-set decision, "
        'and any threshold are recomputed in post-processing at whatever operating point '
        'is calibrated."},"measurements":["coarse_local_reference_power_ratio",'
        '"fine_local_reference_power_ratio"]}'
    )
    assert json.loads(stored) == audit_per_pilot.v5_decision_contract()


def test_audit_rejects_corrupt_reference_split(tmp_path: Path) -> None:
    product = _consistent(_product())
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
