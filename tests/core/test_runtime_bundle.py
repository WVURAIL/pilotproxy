# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_proxy.detector_contract import (
    INPUT_COORDINATE_POST_NYQUIST_ZONE_NORMALIZED,
    INPUT_COORDINATE_RAW_INPUT,
    WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
    WEIGHT_COORDINATE_RAW_INPUT,
)
from pilot_proxy.provenance import file_sha256
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CHIME_FINE_OFFSET_HZ = -3_059.0


def _layout_by_channel(manifest: dict, channel: int) -> dict:
    for row in manifest["target_reference_layout"]:
        if int(row["physical_channel"]) == int(channel):
            return row
    raise AssertionError(f"missing physical channel {channel}")


def test_export_runtime_weight_bundle_writes_compact_profiles(tmp_path) -> None:
    output_dir = tmp_path / "bundle"

    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )

    for path in outputs.values():
        assert path.exists()

    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))

    assert contract["schema_version"] == "pilotproxy_detector_contract_v2"
    assert contract["per_frequency_threshold"] is False
    assert contract["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    assert (
        contract["input_coordinate_system"]
        == INPUT_COORDINATE_POST_NYQUIST_ZONE_NORMALIZED
    )
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )
    assert profiles["schema_version"] == "pilotproxy_runtime_pilot_profiles_v1"
    assert manifest["schema_version"] == "pilotproxy_runtime_weights_manifest_v1"
    assert profiles["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    assert manifest["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    assert profiles["detector_contract_sha256"] == manifest["detector_contract_sha256"]
    assert profiles["weights_sha256"] == file_sha256(outputs["weights"])
    assert [row["physical_channel"] for row in profiles["profiles"]] == [14, 21]
    assert all("chime_frequency_hz" not in row for row in profiles["profiles"])
    assert profiles["profiles"][0]["weight_bank_index"] == 0
    assert profiles["profiles"][0]["weight_bank_offset_bytes"] == 0
    assert profiles["profiles"][1]["weight_bank_index"] == 1
    assert profiles["profiles"][1]["weight_bank_offset_bytes"] == (
        manifest["weight_profile_nbytes"]
    )
    assert outputs["weights"].stat().st_size == (
        2 * manifest["weight_profile_nbytes"]
    )
    sha_text = outputs["sha256sums"].read_text("utf-8")
    assert "detector_contract.json" in sha_text
    assert "pilot_profiles.json" in sha_text
    assert "weights.bin" in sha_text
    assert "weights.manifest.json" in sha_text

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is True
    assert report["num_errors"] == 0


@pytest.mark.parametrize("invalid_channel", [14.9, True, "14"])
def test_export_runtime_bundle_requires_exact_physical_channels(
    tmp_path,
    invalid_channel: object,
) -> None:
    with pytest.raises(TypeError, match="physical channel.*integer"):
        export_runtime_weight_bundle(
            receiver_profile_path=(
                CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
            ),
            detector_core_profile_path=(
                CONFIGS_DIR
                / "detector_core"
                / "pilotproxy_cuda_local_reference_power_ratio.json"
            ),
            physical_channels=[invalid_channel],
            weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
            output_dir=tmp_path / "bundle",
        )


def test_chime_runtime_bundle_post_coordinate_uses_detector_coordinate(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    channel_14 = _layout_by_channel(manifest, 14)
    channel_21 = _layout_by_channel(manifest, 21)

    assert contract["input_coordinate_system"] == (
        INPUT_COORDINATE_POST_NYQUIST_ZONE_NORMALIZED
    )
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is True
    )
    assert channel_14["target_offset_hz"] == pytest.approx(-CHIME_FINE_OFFSET_HZ)
    assert channel_14["lower_reference_edge_wrapped"] is True
    assert channel_14["upper_reference_edge_wrapped"] is False
    assert channel_21["lower_reference_edge_wrapped"] is False
    assert channel_21["upper_reference_edge_wrapped"] is False


def test_chime_runtime_bundle_raw_coordinate_uses_native_inverted_coordinate(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_RAW_INPUT,
        output_dir=output_dir,
    )
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    channel_14 = _layout_by_channel(manifest, 14)
    channel_21 = _layout_by_channel(manifest, 21)

    assert contract["input_coordinate_system"] == INPUT_COORDINATE_RAW_INPUT
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )
    assert channel_14["target_offset_hz"] == pytest.approx(CHIME_FINE_OFFSET_HZ)
    assert channel_14["lower_reference_edge_wrapped"] is False
    assert channel_14["upper_reference_edge_wrapped"] is True
    assert channel_21["lower_reference_edge_wrapped"] is False
    assert channel_21["upper_reference_edge_wrapped"] is False


def test_validate_runtime_weight_bundle_reports_bad_offset(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["profiles"][1]["weight_bank_offset_bytes"] = 1
    outputs["pilot_profiles"].write_text(json.dumps(profiles), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "sha256sums.pilot_profiles.json" in checks
    assert "pilot_profiles.detector_contract_sha256" not in checks
    assert "pilot_profiles.profile_offset_alignment" in checks


def test_validate_runtime_weight_bundle_reports_coordinate_mismatch(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    manifest["weight_coordinate_system"] = "raw_input_frequency_coordinate"
    outputs["weights_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "weight_coordinate_system.consistency" in checks


def test_validate_runtime_weight_bundle_reports_input_coordinate_mismatch(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["input_coordinate_system"] = "raw_input_frequency_coordinate"
    outputs["pilot_profiles"].write_text(json.dumps(profiles), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "input_coordinate_system.consistency" in checks
    assert "pilot_profiles.input_coordinate_system" in checks


def _export_reference_bundle(tmp_path):
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    return output_dir, outputs


def _rewrite_profiles(outputs, profiles) -> None:
    outputs["pilot_profiles"].write_text(
        json.dumps(profiles), encoding="utf-8"
    )


def _refresh_sha256sum(outputs, key: str) -> None:
    path = outputs[key]
    sums_path = outputs["sha256sums"]
    lines = sums_path.read_text(encoding="utf-8").splitlines()
    replacement = f"{file_sha256(path)}  {path.name}"
    sums_path.write_text(
        "\n".join(
            replacement if line.endswith(f"  {path.name}") else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )


def _calibrated_fine_block(anchor: int = 62) -> tuple[dict, object]:
    from pilot_proxy.fine_decision import pack_bulk_mask
    from pilot_proxy.fine_reduction import independent_bin_mask

    half_width = 2
    designated_window = [
        (anchor + offset) % 256 for offset in range(-half_width, half_width + 1)
    ]
    bulk = independent_bin_mask(256, designated_bins=designated_window)
    words = pack_bulk_mask(bulk)
    return (
        {
            "status": "calibrated",
            "decision_version": "fine_decision_v1",
            "anchor_bin": anchor,
            "designated_half_width": half_width,
            "bulk_mask_words_hex": [f"0x{word:016x}" for word in words],
            "cfar_rank": int(sum(bulk) // 2),
            "cfar_multiplier_q16": int(1.5 * 65536),
            "provenance": {
                "epochs": ["2026Q1"],
                "null_quantile": 0.999,
                "source_product_sha256": ["d" * 64],
            },
        },
        bulk,
    )


def test_exported_bundle_carries_pending_fine_calibration(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    for row in profiles["profiles"]:
        block = row["fine_calibration"]
        assert block["status"] == "pending_campaign"
        assert block["decision_version"] == "fine_decision_v1"
        assert block["anchor_bin"] is None
        assert block["designated_half_width"] == 2
        assert block["cfar_rank"] is None
        assert block["cfar_multiplier_q16"] is None
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is True


def test_current_bundle_rejects_missing_norm_and_fine_fields(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    del profiles["profiles"][0]["target_norm_sq"]
    del profiles["profiles"][0]["fine_calibration"]
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.profile_required_fields" in checks
    assert "pilot_profiles.fine_calibration[0]" in checks


def test_current_bundle_rejects_wrong_detector_contract_schema(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    contract["schema_version"] = "pilotproxy_detector_contract_v0"
    outputs["detector_contract"].write_text(json.dumps(contract), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "detector_contract.current_schema" in checks


def test_malformed_detector_contract_geometry_returns_invalid_report(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    contract["num_weight_terms"] = []
    outputs["detector_contract"].write_text(json.dumps(contract), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "detector_contract.current_schema" in checks
    assert "detector_contract.num_weight_terms" in checks


def test_bundle_binds_manifest_channel_to_profile_after_checksum_refresh(
    tmp_path,
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["profiles"][0]["physical_channel"] = 99
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "sha256sums.pilot_profiles.json" not in checks
    assert "runtime_bundle.channel_binding" in checks


def test_bundle_binds_target_layout_to_profile_after_checksum_refresh(
    tmp_path,
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    manifest["target_reference_layout"][0]["physical_channel"] = 99
    outputs["weights_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_sha256sum(outputs, "weights_manifest")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "sha256sums.weights.manifest.json" not in checks
    assert "runtime_bundle.channel_binding" in checks


def test_bundle_binds_weight_offsets_to_profile_order_after_checksum_refresh(
    tmp_path,
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    rows = profiles["profiles"]
    rows[0]["weight_bank_offset_bytes"], rows[1]["weight_bank_offset_bytes"] = (
        rows[1]["weight_bank_offset_bytes"],
        rows[0]["weight_bank_offset_bytes"],
    )
    norm_fields = (
        "target_norm_sq",
        "reference_norm_sum_sq",
        "null_power_ratio",
        "positive_excess_half_threshold_num",
        "positive_excess_half_threshold_den",
    )
    for field in norm_fields:
        rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "sha256sums.pilot_profiles.json" not in checks
    assert "pilot_profiles.weight_bank_offset_order" in checks


@pytest.mark.parametrize(
    "invalid_offset",
    (
        True,
        0.0,
        "0",
        "not-an-integer",
    ),
)
def test_bundle_reports_non_integer_profile_offsets(
    tmp_path, invalid_offset
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["profiles"][0]["weight_bank_offset_bytes"] = invalid_offset
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.profile_offsets" in checks
    assert "pilot_profiles.weight_bank_offset_order" in checks


def test_bundle_reports_huge_json_numeric_offset_without_crashing(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    serialized = json.dumps(profiles)
    original = '"weight_bank_offset_bytes": 0'
    assert original in serialized
    outputs["pilot_profiles"].write_text(
        serialized.replace(
            original,
            '"weight_bank_offset_bytes": 1e999',
            1,
        ),
        encoding="utf-8",
    )

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.profile_offsets" in checks
    assert "pilot_profiles.weight_bank_offset_order" in checks


def test_bundle_reports_nonfinite_manifest_profile_size_without_crashing(
    tmp_path,
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    serialized = json.dumps(manifest)
    original = (
        f'"weight_profile_nbytes": {manifest["weight_profile_nbytes"]}'
    )
    assert original in serialized
    outputs["weights_manifest"].write_text(
        serialized.replace(
            original,
            '"weight_profile_nbytes": 1e999',
            1,
        ),
        encoding="utf-8",
    )

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "weights_manifest.weight_profile_nbytes" in checks


@pytest.mark.parametrize("anchor", [62, 63])
def test_calibrated_fine_block_validates_when_consistent(tmp_path, anchor) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block(anchor)
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    # the profiles were rewritten in place, so only the sha256sums entry fails
    # and the fine_calibration block adds no errors
    checks = {e["check"] for e in report["errors"]}
    assert not any("fine_calibration" in c for c in checks), checks


@pytest.mark.parametrize("anchor,population", [(62, 125), (63, 126)])
def test_calibrated_fine_block_accepts_characterization_bulk(
    tmp_path, anchor, population
) -> None:
    from pilot_proxy.detectors.narrowband_marker.anchors import bulk_mask
    from pilot_proxy.fine_decision import pack_bulk_mask

    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block(anchor)
    bulk = bulk_mask(anchor, pad_factor=2, guard_fine_bins=1)
    assert int(bulk.sum()) == population
    block["bulk_mask_words_hex"] = [f"0x{w:016x}" for w in pack_bulk_mask(bulk)]
    block["cfar_rank"] = 63
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    checks = {e["check"] for e in report["errors"]}
    assert not any("fine_calibration" in c for c in checks), checks


@pytest.mark.parametrize("rank", [0, 61, 62, 63, 122])
def test_calibrated_fine_block_accepts_configured_supported_rank(tmp_path, rank) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, bulk = _calibrated_fine_block()
    assert int(sum(bulk)) == 123
    assert block["cfar_rank"] == 61
    block["cfar_rank"] = rank
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"], report["errors"]


@pytest.mark.parametrize("rank", [123, 255])
def test_calibrated_fine_block_rejects_insufficient_reference_population(tmp_path, rank) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, bulk = _calibrated_fine_block()
    assert int(sum(bulk)) == 123
    block["cfar_rank"] = rank
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert not report["valid"]
    assert any("cfar_rank" in error["check"] for error in report["errors"])


def test_calibrated_fine_block_rejects_inconsistencies(tmp_path) -> None:
    from pilot_proxy.fine_decision import pack_bulk_mask
    from pilot_proxy.fine_reduction import independent_bin_mask

    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    bulk = independent_bin_mask(256, designated_bins=[200])
    words = pack_bulk_mask(bulk)
    base = {
        "status": "calibrated",
        "decision_version": "fine_decision_v1",
        "anchor_bin": 62,  # the anchor is in the bulk mask, since only bin 200 was designated
        "designated_half_width": 2,
        "bulk_mask_words_hex": [f"0x{w:016x}" for w in words],
        "cfar_rank": 5000,  # deeper than the bulk population
        "cfar_multiplier_q16": 0,  # not positive
        "provenance": {},
    }
    profiles["profiles"][0]["fine_calibration"] = dict(base)
    profiles["profiles"][1]["fine_calibration"] = {
        "status": "unheard_of",
        "decision_version": "fine_decision_v2",
    }
    _rewrite_profiles(outputs, profiles)
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is False
    checks = {e["check"] for e in report["errors"]}
    assert any("cfar_rank" in c for c in checks)
    assert any("cfar_multiplier_q16" in c for c in checks)
    assert any("bulk_mask_words_hex" in c for c in checks)
    assert any("provenance" in c for c in checks)
    assert any("status" in c for c in checks)
    assert any("decision_version" in c for c in checks)


@pytest.mark.parametrize("multiplier_q16", (-1, 0, 1 << 64))
def test_calibrated_fine_block_rejects_multiplier_outside_uint64(
    tmp_path, multiplier_q16
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block["cfar_multiplier_q16"] = multiplier_q16
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    assert any(
        error["check"]
        == "pilot_profiles.fine_calibration[0].cfar_multiplier_q16"
        and "uint64 range" in error["message"]
        for error in report["errors"]
    )


@pytest.mark.parametrize("invalid_word", ("-0x1", "0x10000000000000000"))
def test_calibrated_fine_block_rejects_mask_word_outside_uint64(
    tmp_path, invalid_word
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block["bulk_mask_words_hex"][0] = invalid_word
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    assert any(
        error["check"]
        == "pilot_profiles.fine_calibration[0].bulk_mask_words_hex"
        and "uint64" in error["message"]
        for error in report["errors"]
    )


def test_calibrated_fine_block_requires_complete_provenance(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block["provenance"] = {}
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.fine_calibration[0].provenance.required_fields" in checks


@pytest.mark.parametrize("anchor", [0, 62, 255])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("half_width", [0, 2, 3])
def test_calibrated_fine_block_excludes_guard_bins_from_bulk(
    tmp_path, anchor, side, half_width
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block(anchor)
    block["designated_half_width"] = half_width
    guard_bin = (anchor + side * max(half_width, 2)) % 256
    words = [int(word, 16) for word in block["bulk_mask_words_hex"]]
    words[guard_bin >> 6] |= 1 << (guard_bin & 63)
    block["bulk_mask_words_hex"] = [f"0x{word:016x}" for word in words]
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    errors = [
        error
        for error in report["errors"]
        if error["check"]
        == "pilot_profiles.fine_calibration[0].bulk_mask_words_hex"
    ]
    assert any("designated/guard bin" in error["message"] for error in errors)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("anchor_bin", True),
        ("designated_half_width", 2.0),
        ("cfar_rank", "100"),
        ("cfar_multiplier_q16", float("inf")),
    ),
)
def test_calibrated_fine_block_rejects_non_integer_fields_without_crashing(
    tmp_path, field_name, invalid_value
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block[field_name] = invalid_value
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    assert any(
        error["check"]
        == f"pilot_profiles.fine_calibration[0].{field_name}"
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    "invalid_words",
    (
        "0000",
        [0, 0, 0, 0],
    ),
)
def test_calibrated_fine_block_requires_a_list_of_hex_strings(
    tmp_path, invalid_words
) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block["bulk_mask_words_hex"] = invalid_words
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    assert any(
        error["check"]
        == "pilot_profiles.fine_calibration[0].bulk_mask_words_hex"
        for error in report["errors"]
    )


def _export_bundle_with_control(tmp_path):
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 37],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=output_dir,
    )
    return output_dir, outputs


def _fine_errors(report) -> list[dict]:
    return [e for e in report["errors"] if "fine_calibration" in e["check"]]


@pytest.mark.parametrize(
    "statuses,valid",
    [
        (("calibrated", "control"), True),
        (("calibrated", "pending_campaign"), True),
        (("control", "control"), False),       # a DTV channel cannot be the control
        (("calibrated", "calibrated"), False),  # channel 37 is never calibrated
    ],
)
def test_only_channel_37_is_the_control(tmp_path, statuses, valid) -> None:
    output_dir, outputs = _export_bundle_with_control(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    assert [row["physical_channel"] for row in profiles["profiles"]] == [14, 37]
    for row, status in zip(profiles["profiles"], statuses):
        if status != "pending_campaign":
            block, _bulk = _calibrated_fine_block()
            block["status"] = status
            row["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is valid, report["errors"]
    if not valid:
        assert any("only channel 37 is the control" in e["message"] for e in _fine_errors(report))


def test_control_block_is_validated_like_a_calibrated_one(tmp_path) -> None:
    output_dir, outputs = _export_bundle_with_control(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block()
    block.update(status="control", cfar_rank=255)
    del block["provenance"]["null_quantile"]
    profiles["profiles"][1]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    checks = {e["check"] for e in _fine_errors(validate_runtime_weight_bundle(bundle_dir=output_dir))}
    assert "pilot_profiles.fine_calibration[1].cfar_rank" in checks
    assert "pilot_profiles.fine_calibration[1].provenance.required_fields" in checks


@pytest.mark.parametrize("guard", [0, 3, 5])
@pytest.mark.parametrize("offset", [3, 4, 5, 6, 7])
def test_marker_guard_bins_keep_the_guard_band_out_of_the_bulk(tmp_path, guard, offset) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block(62)
    block["marker_guard_bins"] = guard
    line = 62 + offset
    words = [int(word, 16) for word in block["bulk_mask_words_hex"]]
    words[line >> 6] |= 1 << (line & 63)
    block["bulk_mask_words_hex"] = [f"0x{word:016x}" for word in words]
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    inside = offset <= 2 + guard
    assert report["valid"] is not inside, report["errors"]


@pytest.mark.parametrize("guard", [-1, 126, 1.5, True])
def test_marker_guard_bins_must_be_an_integer_in_range(tmp_path, guard) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    block, _bulk = _calibrated_fine_block(62)
    block["marker_guard_bins"] = guard
    profiles["profiles"][0]["fine_calibration"] = block
    _rewrite_profiles(outputs, profiles)
    _refresh_sha256sum(outputs, "pilot_profiles")
    checks = {e["check"] for e in _fine_errors(validate_runtime_weight_bundle(bundle_dir=output_dir))}
    assert "pilot_profiles.fine_calibration[0].marker_guard_bins" in checks
