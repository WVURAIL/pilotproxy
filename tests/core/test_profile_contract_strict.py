# coding=utf-8
"""Ground-zero tests for exact receiver and detector-core documents."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.integration.detector_core import (
    DetectorCoreProfile,
    load_detector_core_profile,
)
from pilot_proxy.integration.receiver_profile import (
    ReceiverProfile,
    load_receiver_profile,
    receiver_profile_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIVER_PROFILE_DIR = REPO_ROOT / "configs" / "receiver_profiles"
DETECTOR_CORE_PATH = (
    REPO_ROOT / "configs" / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
)
ACTIVE_WEIGHT_MANIFESTS = (
    REPO_ROOT / "weights" / "chime_dtv_weights_k128.bin.manifest.json",
    REPO_ROOT / "weights" / "chord_dtv_weights_k64.bin.manifest.json",
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    "path",
    sorted(RECEIVER_PROFILE_DIR.glob("*.json")),
    ids=lambda path: path.name,
)
def test_shipped_receiver_profiles_are_exact_canonical_documents(path: Path) -> None:
    raw = _read_json(path)
    profile = load_receiver_profile(path)

    assert profile.to_dict() == raw
    assert ReceiverProfile.from_dict(raw).to_dict() == raw
    assert receiver_profile_hash(profile) == receiver_profile_hash(raw)


def test_receiver_profile_rejects_flat_document() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        ReceiverProfile.from_dict(
            {
                "schema_version": "pilotproxy_receiver_profile_v1",
                "name": "flat_profile",
                "sample_rate_hz": 800_000_000.0,
            }
        )


def test_receiver_profile_rejects_unknown_top_level_field() -> None:
    raw = _read_json(RECEIVER_PROFILE_DIR / "reference_800mhz_pfb.json")
    raw["name"] = raw["receiver_profile_id"]

    with pytest.raises(ValueError, match="unknown fields"):
        ReceiverProfile.from_dict(raw)


def test_receiver_profile_rejects_unknown_nested_field() -> None:
    raw = _read_json(RECEIVER_PROFILE_DIR / "reference_800mhz_pfb.json")
    raw["channelizer"]["coarse_channel_width_hz"] = 390_625.0

    with pytest.raises(ValueError, match="unknown fields"):
        ReceiverProfile.from_dict(raw)


def test_receiver_profile_rejects_missing_baseband_frame() -> None:
    raw = _read_json(RECEIVER_PROFILE_DIR / "reference_800mhz_pfb.json")
    del raw["baseband_frame"]

    with pytest.raises(ValueError, match="missing required fields"):
        ReceiverProfile.from_dict(raw)


def test_receiver_profile_rejects_frame_not_divisible_by_window() -> None:
    raw = _read_json(RECEIVER_PROFILE_DIR / "reference_800mhz_pfb.json")
    raw["framing"]["frame_size_samples"] = 4

    with pytest.raises(ValueError, match="integer multiple"):
        ReceiverProfile.from_dict(raw)


def test_explicit_frame_and_preprocessing_semantics_are_pinned() -> None:
    reference = load_receiver_profile(
        RECEIVER_PROFILE_DIR / "reference_800mhz_pfb.json"
    )
    chime = load_receiver_profile(RECEIVER_PROFILE_DIR / "chime_dtv_fengine.json")
    chord = load_receiver_profile(RECEIVER_PROFILE_DIR / "chord_dtv_fengine.json")
    pathfinder = load_receiver_profile(
        RECEIVER_PROFILE_DIR / "chord_pathfinder_dtv_fengine.json"
    )

    assert reference.frame_center_normalized(0) == 0.5
    assert reference.forbidden_dc_normalized(0) == 0.5
    assert reference.time_reverse_detector_windows_before_kernel is False

    for profile in (chime, chord, pathfinder):
        assert profile.frame_center_normalized(0) == 0.0
        assert profile.forbidden_dc_normalized(0) == 0.0
        assert profile.time_reverse_detector_windows_before_kernel is True


@pytest.mark.parametrize("invalid_index", [843.9, True, np.bool_(True), "843"])
@pytest.mark.parametrize(
    "method",
    [
        "coarse_channel_center_hz",
        "frame_center_normalized",
        "forbidden_dc_normalized",
    ],
)
def test_receiver_grid_methods_require_exact_channel_index(
    method: str,
    invalid_index: object,
) -> None:
    profile = load_receiver_profile(
        RECEIVER_PROFILE_DIR / "chime_dtv_fengine.json"
    )

    with pytest.raises(TypeError, match="coarse channel index.*integer"):
        getattr(profile, method)(invalid_index)


def test_detector_core_document_is_exact_and_canonical() -> None:
    raw = _read_json(DETECTOR_CORE_PATH)
    profile = load_detector_core_profile(DETECTOR_CORE_PATH)

    assert profile.to_dict() == raw
    assert DetectorCoreProfile.from_dict(raw).to_dict() == raw
    assert profile.reference_offset_bins == profile.skipped_guard_bins + 1


@pytest.mark.parametrize(
    "invalid_window",
    [64.9, True, np.bool_(True), "64"],
)
def test_detector_core_programmatic_window_requires_exact_integer(
    invalid_window: object,
) -> None:
    profile = load_detector_core_profile(DETECTOR_CORE_PATH)

    with pytest.raises(TypeError, match="detector_window_samples.*integer"):
        profile.with_detector_window_samples(invalid_window)


def test_detector_core_replace_rejects_fractional_fixed_point_limit() -> None:
    from dataclasses import replace

    profile = load_detector_core_profile(DETECTOR_CORE_PATH)
    limits = dict(profile.fixed_point_limits)
    limits["power_sum_accumulator_bits"] = 64.9

    with pytest.raises(TypeError, match="fixed_point_limits.*integer"):
        replace(profile, fixed_point_limits=limits)


def test_detector_core_rejects_partial_document() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        DetectorCoreProfile.from_dict(
            {"kernel_contract": {"skipped_guard_bins": 1}}
        )


def test_detector_core_rejects_top_level_kernel_field() -> None:
    raw = _read_json(DETECTOR_CORE_PATH)
    raw["skipped_guard_bins"] = 1

    with pytest.raises(ValueError, match="unknown fields"):
        DetectorCoreProfile.from_dict(raw)


def test_detector_core_rejects_reference_offset_input() -> None:
    raw = _read_json(DETECTOR_CORE_PATH)
    raw["kernel_contract"]["reference_offset_bins"] = 2

    with pytest.raises(ValueError, match="unknown fields"):
        DetectorCoreProfile.from_dict(raw)


def test_detector_core_rejects_omitted_contract_field() -> None:
    raw = _read_json(DETECTOR_CORE_PATH)
    del raw["kernel_contract"]["coarse_power_ratio_definition"]

    with pytest.raises(ValueError, match="missing required fields"):
        DetectorCoreProfile.from_dict(raw)


@pytest.mark.parametrize(
    "manifest_path",
    ACTIVE_WEIGHT_MANIFESTS,
    ids=lambda path: path.name,
)
def test_active_weight_manifests_embed_exact_profiles(manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    embedded = manifest["receiver_profile"]
    profile = ReceiverProfile.from_dict(copy.deepcopy(embedded))

    assert embedded == profile.to_dict()
    assert manifest["receiver_profile_hash"] == receiver_profile_hash(profile)

    weights_path = REPO_ROOT / manifest["artifacts"]["weights_path"]
    actual_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert manifest["artifacts"]["weights_sha256"] == actual_sha256
