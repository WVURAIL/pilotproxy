#!/usr/bin/env python3
# coding=utf-8
"""Reject removed product schemas and aliases from current project surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = {
    "pilotproxy_detector_datatrawl_" + "v2": "pilotproxy_per_pilot_product_v1",
    "pilotproxy_detector_datatrawl_" + "v3": "pilotproxy_per_pilot_product_v1",
    "pilotproxy_chime_detector_contract_" + "v1": "pilotproxy_detector_contract_v1",
    "fine_" + "nd_flag_rate": "fine_null_bulk_exceedance_fraction",
    "_fine_" + "ndrate": "_fine_null_bulk_exceedance_fraction",
    "product_schema_" + "v2.md": "PER_PILOT_PRODUCT_FIELDS.md",
    "product_schema_" + "v3.md": "FINE_REDUCTION_PRODUCTS.md",
    "product_schema_" + "v4.md": "PRODUCT_SCHEMA.md",
    "repair_fine_frame_" + "labels.py": "regenerate unsupported products",
}
CURRENT_FILES = (
    ROOT / "README.md",
    ROOT / "INTEGRATION.md",
    ROOT / "Makefile",
    ROOT / "docs" / "PRODUCT_SCHEMA.md",
    ROOT / "docs" / "PER_PILOT_PRODUCT_FIELDS.md",
    ROOT / "docs" / "FINE_REDUCTION_PRODUCTS.md",
    ROOT / "docs" / "METHOD_SPEC.md",
    ROOT / "docs" / "DATA_PRODUCTS.md",
    ROOT / "docs" / "CANFAR_RUNBOOK.md",
    ROOT / "docs" / "CHIME_RUN_WORKFLOW.md",
    ROOT / "docs" / "VALIDATION_GATES.md",
    ROOT / "docs" / "KOTEKAN_INTERFACE_PREP.md",
    ROOT / "docs" / "PilotProxy_DS001_Data_Sheet.tex",
    ROOT / "docs" / "PilotProxy_UG001_User_Guide.tex",
    ROOT / "docs" / "generated" / "specs.tex",
)
RUNTIME_TREES = (ROOT / "src", ROOT / "scripts", ROOT / "tools")
TEXT_SUFFIXES = {".py", ".md", ".tex", ".json", ".sh", ".h", ".cu", ".cpp"}
EXCLUDED_PARTS = {"provenance", "evidence", "out", "auxil", "__pycache__"}


def current_files() -> list[Path]:
    files = [path for path in CURRENT_FILES if path.is_file()]
    for tree in RUNTIME_TREES:
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.name == Path(__file__).name:
                continue
            if path == ROOT / "src" / "pilot_proxy" / "archived_product_keys.py":
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def searchable_text(text: str) -> str:
    r"""Normalize presentation escapes before checking retired identifiers.

    TeX writes identifiers as ``name\_with\_underscores``. Searching only the
    source spelling allowed a retired schema token to survive in generated
    formal-document input, so compare both the source and rendered spelling.
    """
    return text + "\n" + text.replace(r"\_", "_")


def main() -> int:
    failures: list[str] = []
    for path in current_files():
        text = searchable_text(path.read_text(encoding="utf-8"))
        for removed, replacement in BANNED.items():
            if removed in text:
                failures.append(
                    f"{path.relative_to(ROOT)} contains removed term {removed!r}; "
                    f"use {replacement!r}"
                )
    for removed_path in (
        ROOT / "docs" / "product_schema_v2.md",
        ROOT / "docs" / "product_schema_v3.md",
        ROOT / "docs" / "product_schema_v4.md",
        ROOT / "docs" / "nonpilot_mode_spec.md",
        ROOT / "tools" / "repair_fine_frame_labels.py",
        ROOT / "scripts" / "trim_pilot_amplitude.py",
    ):
        if removed_path.exists():
            failures.append(f"removed path still exists: {removed_path.relative_to(ROOT)}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("current product vocabulary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
