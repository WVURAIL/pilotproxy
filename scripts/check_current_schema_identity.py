#!/usr/bin/env python3
# coding=utf-8
"""Reject pre-ground-zero schema tokens and stale generated source copies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pilot_proxy.detector_contract import (
    DETECTOR_CONTRACT_SCHEMA_NAME,
    DETECTOR_CONTRACT_SCHEMA_REVISION,
    DETECTOR_CONTRACT_SCHEMA_TOKEN,
    CHIME_RUN_CONFIG_SCHEMA_NAME,
    CHIME_RUN_CONFIG_SCHEMA_REVISION,
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_NAME,
    CHIME_STATS_SCHEMA_REVISION,
    CHIME_STATS_SCHEMA_TOKEN,
)
from pilot_proxy.detector_geometry import (
    STREAM_LAYOUT_SCHEMA_NAME,
    STREAM_LAYOUT_SCHEMA_REVISION,
    STREAM_LAYOUT_SCHEMA_TOKEN,
)
from pilot_proxy.chime.products import (
    CHIME_INPUT_MANIFEST_SCHEMA_NAME,
    CHIME_INPUT_MANIFEST_SCHEMA_REVISION,
    CHIME_INPUT_MANIFEST_SCHEMA_TOKEN,
    SCAN_INPUT_MANIFEST_SCHEMA_NAME,
    SCAN_INPUT_MANIFEST_SCHEMA_REVISION,
    SCAN_INPUT_MANIFEST_SCHEMA_TOKEN,
)
from pilot_proxy.detector_weights import (
    WEIGHT_MANIFEST_SCHEMA_NAME,
    WEIGHT_MANIFEST_SCHEMA_REVISION,
    WEIGHT_MANIFEST_SCHEMA_TOKEN,
)
from pilot_proxy.integration.defaults import DEFAULT_DETECTOR_CORE_PROFILE
from pilot_proxy.integration.detector_core import load_detector_core_profile
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.schemas import (
    DETECTOR_CORE_PROFILE_SCHEMA_NAME,
    DETECTOR_CORE_PROFILE_SCHEMA_REVISION,
    DETECTOR_CORE_PROFILE_SCHEMA_TOKEN,
    RECEIVER_PROFILE_SCHEMA_NAME,
    RECEIVER_PROFILE_SCHEMA_REVISION,
    RECEIVER_PROFILE_SCHEMA_TOKEN,
    STREAM_MAP_SCHEMA_NAME,
    STREAM_MAP_SCHEMA_REVISION,
    STREAM_MAP_SCHEMA_TOKEN,
)
from pilot_proxy.integration.stream_layout import load_stream_map
from pilot_proxy.result_schema import (
    MASK_CONVENTION_SCHEMA_NAME,
    MASK_CONVENTION_SCHEMA_REVISION,
    MASK_CONVENTION_SCHEMA_TOKEN,
    RESULT_SCHEMA_NAME,
    RESULT_SCHEMA_REVISION,
    RESULT_SCHEMA_TOKEN,
)
from pilot_proxy.runtime_bundle import (
    RUNTIME_BUNDLE_VALIDATION_SCHEMA_NAME,
    RUNTIME_BUNDLE_VALIDATION_SCHEMA_REVISION,
    RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN,
    RUNTIME_PILOT_PROFILES_SCHEMA_NAME,
    RUNTIME_PILOT_PROFILES_SCHEMA_REVISION,
    RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN,
    RUNTIME_WEIGHT_MANIFEST_SCHEMA_NAME,
    RUNTIME_WEIGHT_MANIFEST_SCHEMA_REVISION,
    RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN,
)
from pilot_proxy.schema_identity import schema_token

EXPECTED = {
    "detector_core_profile": (DETECTOR_CORE_PROFILE_SCHEMA_NAME, DETECTOR_CORE_PROFILE_SCHEMA_REVISION, DETECTOR_CORE_PROFILE_SCHEMA_TOKEN, "pilotproxy_detector_core_profile"),
    "receiver_profile": (RECEIVER_PROFILE_SCHEMA_NAME, RECEIVER_PROFILE_SCHEMA_REVISION, RECEIVER_PROFILE_SCHEMA_TOKEN, "pilotproxy_receiver_profile"),
    "stream_map": (STREAM_MAP_SCHEMA_NAME, STREAM_MAP_SCHEMA_REVISION, STREAM_MAP_SCHEMA_TOKEN, "pilotproxy_stream_map"),
    "stream_layout": (STREAM_LAYOUT_SCHEMA_NAME, STREAM_LAYOUT_SCHEMA_REVISION, STREAM_LAYOUT_SCHEMA_TOKEN, "pilotproxy_stream_layout"),
    "result_schema": (RESULT_SCHEMA_NAME, RESULT_SCHEMA_REVISION, RESULT_SCHEMA_TOKEN, "pilotproxy_result_schema"),
    "mask_convention": (MASK_CONVENTION_SCHEMA_NAME, MASK_CONVENTION_SCHEMA_REVISION, MASK_CONVENTION_SCHEMA_TOKEN, "pilotproxy_mask_convention"),
    "detector_contract": (DETECTOR_CONTRACT_SCHEMA_NAME, DETECTOR_CONTRACT_SCHEMA_REVISION, DETECTOR_CONTRACT_SCHEMA_TOKEN, "pilotproxy_detector_contract"),
    "chime_run_config": (CHIME_RUN_CONFIG_SCHEMA_NAME, CHIME_RUN_CONFIG_SCHEMA_REVISION, CHIME_RUN_CONFIG_SCHEMA_TOKEN, "pilotproxy_chime_run_config"),
    "chime_stats": (CHIME_STATS_SCHEMA_NAME, CHIME_STATS_SCHEMA_REVISION, CHIME_STATS_SCHEMA_TOKEN, "pilotproxy_chime_stats"),
    "chime_input_manifest": (CHIME_INPUT_MANIFEST_SCHEMA_NAME, CHIME_INPUT_MANIFEST_SCHEMA_REVISION, CHIME_INPUT_MANIFEST_SCHEMA_TOKEN, "pilotproxy_chime_input_manifest"),
    "scan_input_manifest": (SCAN_INPUT_MANIFEST_SCHEMA_NAME, SCAN_INPUT_MANIFEST_SCHEMA_REVISION, SCAN_INPUT_MANIFEST_SCHEMA_TOKEN, "pilotproxy_scan_input_manifest"),
    "weight_manifest": (WEIGHT_MANIFEST_SCHEMA_NAME, WEIGHT_MANIFEST_SCHEMA_REVISION, WEIGHT_MANIFEST_SCHEMA_TOKEN, "pilotproxy_weight_manifest"),
    "runtime_weights": (RUNTIME_WEIGHT_MANIFEST_SCHEMA_NAME, RUNTIME_WEIGHT_MANIFEST_SCHEMA_REVISION, RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN, "pilotproxy_runtime_weights_manifest"),
    "runtime_pilots": (RUNTIME_PILOT_PROFILES_SCHEMA_NAME, RUNTIME_PILOT_PROFILES_SCHEMA_REVISION, RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN, "pilotproxy_runtime_pilot_profiles"),
    "runtime_validation": (RUNTIME_BUNDLE_VALIDATION_SCHEMA_NAME, RUNTIME_BUNDLE_VALIDATION_SCHEMA_REVISION, RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN, "pilotproxy_runtime_bundle_validation"),
}
# Every current schema is at revision 1 except the three whose meaning kernel
# core 3.0.0 changed (frame powers are the all-bin sums of the fine powers):
# the detector contract (2), the result schema (2) and the detector-core
# profile (3; its revision 2 token is retired above). The earlier revisions
# survive only in outputs of record.
EXPECTED_REVISION = {"detector_contract": 2, "result_schema": 2, "detector_core_profile": 3}
PROFILE_FILES = ("reference_800mhz_pfb.json", "chime_dtv_fengine.json", "chord_dtv_fengine.json", "chord_pathfinder_dtv_fengine.json")
STREAM_MAP_FILES = ("chime_feed_pol_example.json", "chord_dish_pol_example.json", "chord_pathfinder_dish_pol_example.json")
ACTIVE_MANIFESTS = ("chime_dtv_weights_k128.bin.manifest.json", "chord_dtv_weights_k64.bin.manifest.json")
SCAN_ROOTS = (ROOT/".github", ROOT/"analysis", ROOT/"configs", ROOT/"docs", ROOT/"examples", ROOT/"paper", ROOT/"scripts", ROOT/"src", ROOT/"tests", ROOT/"tools", ROOT/"weights")
EXTRA_FILES = (ROOT/"README.md", ROOT/"INTEGRATION.md", ROOT/"Makefile", ROOT/"pyproject.toml", ROOT/"CITATION.cff", ROOT/".gitignore")
EXCLUDED_PARTS = {".git", "__pycache__", "generated", "dist", "build", "out", "auxil", "provenance", "evidence", "migrations"}
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".pyi", ".md", ".rst", ".tex", ".bib", ".sh", ".json", ".toml", ".yml", ".yaml", ".txt", ".cff", ".csv"}
RETIRED = {
    "pilotproxy_detector_core_profile_"+"v2", "fstat_receiver_profile_"+"v1",
    "fstat_stream_map_"+"v1", "fstat_stream_layout_"+"v1",
    "pilot_proxy_result_"+"v2", "fstat_mask_convention_"+"v1",
    "fstat_chime_run_config_"+"v2", "fstat_chime_stats_"+"v2",
    "fstat_weight_manifest_"+"v2", "fstat_runtime_weights_manifest_"+"v1",
    "fstat_runtime_pilot_profiles_"+"v1", "fstat_runtime_bundle_validation_"+"v1",
    "pilot_proxy_injection_"+"v1", "fstat_chime_input_manifest_"+"v1",
    "fstat_chime_scan_input_manifest_"+"v1",
    "pilotproxy_chime_detector_contract_"+"v1",
    "fstat_chime_product_validation_"+"v1", "fstat_chime_integrated_spectra_"+"v1",
    "pilot_proxy_detection_"+"v1", "pilot_proxy_validation_report_"+"v1",
    "fstat_atsc_detector_input_"+"v1", "fstat_atsc_waveform_quality_"+"v1",
    "fstat_atsc_waveform_audit_"+"v1", "pilot_proxy_result_summary_"+"v1",
    "pilot_proxy_cleaning_tradeoff_"+"v1", "pilot_proxy_injection_recovery_"+"v1",
}
RETIRED_CONSTANTS = {
    "DETECTOR_CORE_PROFILE_SCHEMA_"+"VERSION", "RECEIVER_PROFILE_SCHEMA_"+"VERSION",
    "STREAM_MAP_SCHEMA_"+"VERSION", "STREAM_LAYOUT_SCHEMA_"+"VERSION",
    "RESULT_SCHEMA_"+"VERSION", "MASK_CONVENTION_"+"VERSION",
    "DETECTOR_CONTRACT_SCHEMA_"+"VERSION",
    "CHIME_DETECTOR_CONTRACT_SCHEMA_"+"VERSION",
    "CHIME_RUN_CONFIG_SCHEMA_"+"VERSION",
    "CHIME_STATS_SCHEMA_"+"VERSION", "WEIGHT_MANIFEST_SCHEMA_"+"VERSION",
    "RUNTIME_WEIGHT_MANIFEST_SCHEMA_"+"VERSION", "RUNTIME_PILOT_PROFILES_SCHEMA_"+"VERSION",
    "RUNTIME_BUNDLE_VALIDATION_SCHEMA_"+"VERSION",
}


def _iter_files():
    seen: set[Path] = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() not in seen:
                seen.add(path.resolve()); yield path
    for path in EXTRA_FILES:
        if path.is_file() and path.resolve() not in seen:
            yield path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return value


def main() -> int:
    errors: list[str] = []
    for label, (name, revision, token, expected_name) in EXPECTED.items():
        if (name != expected_name or revision != EXPECTED_REVISION.get(label, 1)
                or token != schema_token(name, revision)):
            errors.append(f"{label}: identity mismatch name={name!r} revision={revision!r} token={token!r}")
    build_root = ROOT / "build"
    build_files = (
        [
            path
            for path in build_root.rglob("*")
            if path.is_file() or path.is_symlink()
        ]
        if build_root.exists()
        else []
    )
    if build_files:
        errors.append(
            "generated build files remain in the source tree: "
            + ", ".join(
                str(path.relative_to(ROOT)) for path in build_files[:10]
            )
        )
    ignore = (ROOT/".gitignore").read_text(encoding="utf-8")
    if not any(line.strip() == "build/" for line in ignore.splitlines()):
        errors.append(".gitignore does not exclude build/")
    core = load_detector_core_profile(DEFAULT_DETECTOR_CORE_PROFILE)
    if core.schema_version != DETECTOR_CORE_PROFILE_SCHEMA_TOKEN:
        errors.append("detector-core config has the wrong schema token")
    for filename in PROFILE_FILES:
        if load_receiver_profile(ROOT/"configs"/"receiver_profiles"/filename).schema_version != RECEIVER_PROFILE_SCHEMA_TOKEN:
            errors.append(f"{filename}: wrong receiver-profile schema token")
    for filename in STREAM_MAP_FILES:
        if load_stream_map(ROOT/"configs"/"stream_maps"/filename).schema_version != STREAM_MAP_SCHEMA_TOKEN:
            errors.append(f"{filename}: wrong stream-map schema token")
    for filename in ACTIVE_MANIFESTS:
        doc = _load_json(ROOT/"weights"/filename)
        if doc.get("schema_version") != WEIGHT_MANIFEST_SCHEMA_TOKEN:
            errors.append(f"{filename}: wrong weight-manifest schema token")
        profile = doc.get("receiver_profile")
        if not isinstance(profile, dict) or profile.get("schema_version") != RECEIVER_PROFILE_SCHEMA_TOKEN:
            errors.append(f"{filename}: embedded receiver profile has wrong schema token")
    for path in _iter_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT)
        for token in sorted(RETIRED):
            if token in text: errors.append(f"{rel}: retired schema token {token!r}")
        for token in sorted(RETIRED_CONSTANTS):
            if token in text: errors.append(f"{rel}: retired schema constant {token!r}")
    if errors:
        for error in errors: print(error, file=sys.stderr)
        return 1
    print("current schema identity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
