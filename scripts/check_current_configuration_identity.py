#!/usr/bin/env python3
# coding=utf-8
"""Reject chronological configuration IDs on current PilotProxy surfaces."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pilot_proxy.detector_contract import (
    ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION,
    DETECTOR_POWER_RATIO_DEFINITION,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_detector_contract,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration.defaults import DEFAULT_DETECTOR_CORE_PROFILE
from pilot_proxy.integration.detector_core import load_detector_core_profile
from pilot_proxy.integration.receiver_profile import (
    load_receiver_profile,
    receiver_profile_hash,
)
from pilot_proxy.integration.schemas import (
    DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO,
)
from pilot_proxy.integration.stream_layout import load_stream_map

CANONICAL_CORE_ID = "pilotproxy_cuda_local_reference_power_ratio"
CANONICAL_REFERENCE_METHOD = "adaptive_circular_reference_placement"
EXPECTED_PROFILES = {
    "reference_800mhz_pfb.json": "reference_800mhz_pfb",
    "chime_dtv_fengine.json": "chime_dtv_fengine",
    "chord_dtv_fengine.json": "chord_dtv_fengine",
    "chord_pathfinder_dtv_fengine.json": "chord_pathfinder_dtv_fengine",
}
EXPECTED_STREAM_MAPS = {
    "chime_feed_pol_example.json": "chime_dtv_fengine",
    "chord_dish_pol_example.json": "chord_dtv_fengine",
    "chord_pathfinder_dish_pol_example.json": "chord_pathfinder_dtv_fengine",
}
ACTIVE_WEIGHTS = {
    "chime_dtv_weights_k128.bin": "chime_dtv_fengine",
    "chord_dtv_weights_k64.bin": "chord_dtv_fengine",
}

SCAN_ROOTS = (
    ROOT / ".github",
    ROOT / "analysis",
    ROOT / "configs",
    ROOT / "docs",
    ROOT / "examples",
    ROOT / "paper",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "tests",
    ROOT / "tools",
    ROOT / "weights",
)
EXTRA_FILES = (
    ROOT / "README.md",
    ROOT / "INTEGRATION.md",
    ROOT / "Makefile",
    ROOT / "pyproject.toml",
    ROOT / "CITATION.cff",
)
EXCLUDED_PARTS = {
    ".git", "__pycache__", "generated", "dist", "build", "out", "auxil",
    "provenance", "evidence", "migrations",
}
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
    ".py", ".pyi", ".md", ".rst", ".tex", ".bib", ".sh",
    ".json", ".toml", ".yml", ".yaml", ".txt", ".cff", ".csv",
}
BANNED = {
    "pilotproxy_cuda_" + "fstat_v1": CANONICAL_CORE_ID,
    "DETECTOR_CORE_ID_PILOT_PROXY_CUDA_" + "V1": (
        "DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO"
    ),
    "reference_800mhz_pfb_" + "v1": "reference_800mhz_pfb",
    "chime_dtv_fengine_" + "v1": "chime_dtv_fengine",
    "chime_dtv_fengine_" + "v2": "chime_dtv_fengine",
    "chord_dtv_fengine_" + "v1": "chord_dtv_fengine",
    "chord_pathfinder_dtv_fengine_" + "v1": (
        "chord_pathfinder_dtv_fengine"
    ),
    "adaptive_circular_reference_placement_" + "v1": (
        CANONICAL_REFERENCE_METHOD
    ),
    "all_rows_" + "statistic": "all_rows_coarse_power_ratio_definition",
    '"statistic": "F = 2 * sum(P_target)': (
        "coarse_power_ratio_definition"
    ),
    "Pilot-Informed F-Statistic Detection of Sub-Noise RFI": (
        "Pilot-Informed Local-Reference Power-Ratio Detection of Sub-Noise RFI"
    ),
    "F-statistic detector": "local-reference power-ratio detector",
    "F-Statistic Detector": "Local-Reference Power-Ratio Detector",
    "F-statistic detection": "local-reference power-ratio detection",
    "F-Statistic Detection": "Local-Reference Power-Ratio Detection",
}


def _iter_files():
    seen: set[Path] = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.name == "Makefile":
                seen.add(path)
    for path in EXTRA_FILES:
        if path.is_file():
            seen.add(path)
    yield from sorted(seen)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    gate = Path(__file__).resolve()

    old_core = ROOT / "configs" / "detector_core" / (
        "pilotproxy_cuda_" + "fstat_v1.json"
    )
    new_core = ROOT / "configs" / "detector_core" / (
        "pilotproxy_cuda_local_reference_power_ratio.json"
    )
    if old_core.exists():
        errors.append(f"retired detector-core config still exists: {old_core}")
    if not new_core.is_file():
        errors.append(f"canonical detector-core config is missing: {new_core}")
    if DEFAULT_DETECTOR_CORE_PROFILE.resolve() != new_core.resolve():
        errors.append(
            "DEFAULT_DETECTOR_CORE_PROFILE does not point to the canonical config"
        )

    try:
        core = load_detector_core_profile(new_core)
        if core.detector_core_id != CANONICAL_CORE_ID:
            errors.append(
                f"detector_core_id={core.detector_core_id!r}; expected "
                f"{CANONICAL_CORE_ID!r}"
            )
        if (
            DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO
            != CANONICAL_CORE_ID
        ):
            errors.append("detector-core Python identity constant is inconsistent")
        core_doc = core.to_dict()["kernel_contract"]
        if "statistic" in core_doc or "pilot_excess" in core_doc:
            errors.append("detector-core contract retains ambiguous retired keys")
        if core_doc.get("coarse_power_ratio_definition") != (
            "R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)"
        ):
            errors.append("detector-core coarse power-ratio definition is incorrect")
        if core_doc.get("raw_pilot_excess_definition") != (
            "rho_raw = R_coarse - 1 (diagnostic only)"
        ):
            errors.append("detector-core raw pilot-excess definition is incorrect")
    except Exception as exc:
        errors.append(f"canonical detector-core profile failed to load: {exc}")

    public_contract = build_detector_contract(
        detector_window_samples=128,
        skipped_guard_bins=1,
        reference_offset_bins=2,
        num_weight_terms=3,
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    )
    if "statistic" in public_contract or "all_rows_statistic" in public_contract:
        errors.append("public detector contract retains ambiguous statistic keys")
    if public_contract.get("coarse_power_ratio_definition") != (
        DETECTOR_POWER_RATIO_DEFINITION
    ):
        errors.append("public coarse power-ratio definition is incorrect")
    if public_contract.get("all_rows_coarse_power_ratio_definition") != (
        ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION
    ):
        errors.append("public all-rows power-ratio definition is incorrect")

    profiles: dict[str, object] = {}
    profile_dir = ROOT / "configs" / "receiver_profiles"
    for filename, expected_id in EXPECTED_PROFILES.items():
        path = profile_dir / filename
        try:
            profile = load_receiver_profile(path)
            profiles[expected_id] = profile
        except Exception as exc:
            errors.append(f"{path}: failed to load: {exc}")
            continue
        if profile.receiver_profile_id != expected_id:
            errors.append(
                f"{path}: receiver_profile_id={profile.receiver_profile_id!r}; "
                f"expected {expected_id!r}"
            )
        if re.search(r"_v[0-9]+$", profile.receiver_profile_id):
            errors.append(f"{path}: receiver_profile_id retains chronology")
        if profile.compatible_detector_core_id != CANONICAL_CORE_ID:
            errors.append(
                f"{path}: compatible_detector_core_id is not canonical"
            )

    stream_dir = ROOT / "configs" / "stream_maps"
    for filename, expected_profile_id in EXPECTED_STREAM_MAPS.items():
        path = stream_dir / filename
        try:
            stream_map = load_stream_map(path)
        except Exception as exc:
            errors.append(f"{path}: failed to load: {exc}")
            continue
        if stream_map.receiver_profile_id != expected_profile_id:
            errors.append(
                f"{path}: receiver_profile_id={stream_map.receiver_profile_id!r}; "
                f"expected {expected_profile_id!r}"
            )
        if expected_profile_id not in profiles:
            errors.append(f"{path}: referenced receiver profile is unavailable")

    weight_dir = ROOT / "weights"
    for filename, expected_profile_id in ACTIVE_WEIGHTS.items():
        path = weight_dir / filename
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        try:
            bank = DetectorWeightBank(explicit_path=path)
            manifest = _load_json(manifest_path)
        except Exception as exc:
            errors.append(f"{path}: failed to load active weight bank: {exc}")
            continue
        if bank.header.profile_name != expected_profile_id:
            errors.append(
                f"{path}: header profile_name={bank.header.profile_name!r}; "
                f"expected {expected_profile_id!r}"
            )
        embedded = manifest.get("receiver_profile")
        if not isinstance(embedded, dict):
            errors.append(f"{manifest_path}: receiver_profile is missing")
            continue
        if embedded.get("receiver_profile_id") != expected_profile_id:
            errors.append(f"{manifest_path}: embedded receiver profile ID is stale")
        if (
            embedded.get("detector_adapter", {}).get("compatible_detector_core_id")
            != CANONICAL_CORE_ID
        ):
            errors.append(f"{manifest_path}: embedded detector-core ID is stale")
        try:
            expected_hash = receiver_profile_hash(embedded)
        except Exception as exc:
            errors.append(f"{manifest_path}: embedded profile is invalid: {exc}")
        else:
            if manifest.get("receiver_profile_hash") != expected_hash:
                errors.append(f"{manifest_path}: receiver_profile_hash is stale")
        artifacts = manifest.get("artifacts", {})
        if artifacts.get("weights_sha256") != _sha256(path):
            errors.append(f"{manifest_path}: weights_sha256 does not bind the binary")
        layouts = manifest.get("target_reference_layout", [])
        for row in layouts:
            if not isinstance(row, dict):
                errors.append(f"{manifest_path}: layout row is not an object")
                break
            if row.get("reference_selection_method") != CANONICAL_REFERENCE_METHOD:
                errors.append(
                    f"{manifest_path}: reference_selection_method is not canonical"
                )
                break

    for path in _iter_files():
        if path.resolve() == gate:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for retired, replacement in BANNED.items():
            if retired in text:
                errors.append(
                    f"{path.relative_to(ROOT)} contains retired identity "
                    f"{retired!r}; use {replacement!r}"
                )

    if errors:
        print("current configuration identity: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("current configuration identity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
