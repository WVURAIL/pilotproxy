#!/usr/bin/env python3
# coding=utf-8
"""Validate current-only receiver and detector-core profile contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pilot_proxy.integration.detector_core import load_detector_core_profile
from pilot_proxy.integration.receiver_profile import (
    ReceiverProfile,
    load_receiver_profile,
    receiver_profile_hash,
)

RECEIVER_PROFILE_DIR = ROOT / "configs" / "receiver_profiles"
DETECTOR_CORE_DIR = ROOT / "configs" / "detector_core"
ACTIVE_WEIGHT_MANIFESTS = (
    ROOT / "weights" / "chime_dtv_weights_k128.bin.manifest.json",
    ROOT / "weights" / "chord_dtv_weights_k64.bin.manifest.json",
)
RETIRED_SOURCE_TERMS = {
    "src/pilot_proxy/integration/receiver_profile.py": (
        "_is_nested_receiver_profile",
        "_from_nested_dict",
        "to_nested_dict",
        "_mapping_or_empty",
        "_mapping_or_none",
    ),
    "src/pilot_proxy/integration/detector_core.py": (
        "_contract_value",
        "DEPRECATED_DETECTOR_SPACING_FIELDS",
        "DEPRECATED_THRESHOLD_CONTRACT_FIELDS",
        "DERIVED_DETECTOR_SPACING_INPUT_FIELDS",
    ),
    "src/pilot_proxy/integration/weight_generation.py": (
        "BASEBAND_FRAME_MODE_LEGACY",
        "BASEBAND_FRAME_LEGACY_WARNING",
        "legacy center-at-Nyquist",
        "profile.to_nested_dict()",
        "profile.name",
    ),
}

GLOBAL_RETIRED_TERMS = (
    "to_" + "nested_dict",
    "receiver_profile_" + "nested",
    "profile." + "name",
    "legacy_center_" + "nyquist_default",
)
GLOBAL_SCAN_ROOTS = ("src", "tests", "scripts", "tools")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    for path in sorted(RECEIVER_PROFILE_DIR.glob("*.json")):
        try:
            raw = _load_json(path)
            profile = load_receiver_profile(path)
            canonical = profile.to_dict()
            if raw != canonical:
                errors.append(
                    f"{path}: file is not the canonical strict serialization"
                )
            reparsed = ReceiverProfile.from_dict(canonical)
            if reparsed.to_dict() != canonical:
                errors.append(f"{path}: strict round-trip changed the profile")
        except Exception as exc:  # noqa: BLE001 - aggregate all contract failures.
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    for path in sorted(DETECTOR_CORE_DIR.glob("*.json")):
        try:
            raw = _load_json(path)
            profile = load_detector_core_profile(path)
            if raw != profile.to_dict():
                errors.append(
                    f"{path}: file is not the canonical strict serialization"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    for manifest_path in ACTIVE_WEIGHT_MANIFESTS:
        try:
            manifest = _load_json(manifest_path)
            embedded = manifest.get("receiver_profile")
            if not isinstance(embedded, dict):
                raise ValueError("receiver_profile must be an object")
            profile = ReceiverProfile.from_dict(embedded)
            if embedded != profile.to_dict():
                raise ValueError("embedded receiver_profile is not canonical")
            expected_hash = receiver_profile_hash(profile)
            got_hash = manifest.get("receiver_profile_hash")
            if got_hash != expected_hash:
                raise ValueError(
                    "receiver_profile_hash mismatch: "
                    f"{got_hash!r} != {expected_hash!r}"
                )
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("artifacts must be an object")
            weights_path = ROOT / str(artifacts["weights_path"])
            expected_weights_hash = str(artifacts["weights_sha256"])
            actual_weights_hash = _file_sha256(weights_path)
            if actual_weights_hash != expected_weights_hash:
                raise ValueError(
                    "weights_sha256 mismatch: "
                    f"{actual_weights_hash} != {expected_weights_hash}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{manifest_path}: {type(exc).__name__}: {exc}")

    for relative_path, terms in RETIRED_SOURCE_TERMS.items():
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for term in terms:
            if term in text:
                errors.append(f"{relative_path}: retired term remains: {term!r}")

    this_file = Path(__file__).resolve()
    for root_name in GLOBAL_SCAN_ROOTS:
        scan_root = ROOT / root_name
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            if path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8")
            for term in GLOBAL_RETIRED_TERMS:
                if term in text:
                    errors.append(
                        f"{path.relative_to(ROOT)}: retired profile term remains: {term!r}"
                    )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("current profile contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
