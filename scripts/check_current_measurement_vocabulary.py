#!/usr/bin/env python3
# coding=utf-8
"""Reject retired detector measurement fields on current product surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "src",
    ROOT / "tests",
    ROOT / "analysis",
    ROOT / "scripts",
    ROOT / "tools",
    ROOT / "docs",
    ROOT / "examples",
    ROOT / "configs",
)
EXTRA_FILES = (
    ROOT / "README.md",
    ROOT / "INTEGRATION.md",
    ROOT / "Makefile",
    ROOT / "pyproject.toml",
    ROOT / "CITATION.cff",
    ROOT / "weights" / "chime_dtv_weights_k128.bin.manifest.json",
    ROOT / "weights" / "chord_dtv_weights_k64.bin.manifest.json",
)
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "generated",
    "provenance",
    "evidence",
    "migrations",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".rst", ".tex", ".json", ".toml", ".yml", ".yaml",
    ".txt", ".csv", ".cff", ".sh",
}
BANNED = {
    "fstat_" + "raw": "coarse_power_ratio",
    "fstat_" + "level_db": "normalized_coarse_power_ratio_db",
    "fstat_" + "fine": "fine_power_ratio",
    "pnr_" + "bin_db": "pilot_excess_db",
    "snr_" + "shelf_db": "estimated_data_shelf_snr_db",
    "pilot_excess_" + "corrected": "normalized_pilot_excess",
    "pilot_excess_" + "linear": "raw_pilot_excess",
    "ref_norm_" + "sum_sq": "reference_norm_sum_sq",
    "norm_corrected_" + "mu0": "null_power_ratio_from_weight_norms",
    "v1_" + "fstat": "coarse_power_ratio_from_marginal",
    "norm_corrected_" + "positive_excess": "normalized_positive_excess",
    "emit_row_" + "sums": "emit_row_projections",
    "supports_row_" + "sums": "supports_row_projections",
    "compute_row_" + "sums_i32": "compute_row_projections_i32",
    "row_" + "sums": "matched_filter_row_projections",
    "\"positive_" + "excess\"": "normalized_positive_excess_decision",
    "coarse_local_reference_" + "f_statistic": (
        "coarse_local_reference_power_ratio"
    ),
    "fine_local_reference_" + "f_statistic": "fine_local_reference_power_ratio",
    "LEGACY_" + "NORMALIZED_POSITIVE_EXCESS_MASK_RULE": (
        "remove unsupported historical mask-rule dependencies"
    ),
    "LEGACY_" + "NORMALIZED_POSITIVE_EXCESS_EQUIVALENT_RULE": (
        "remove unsupported historical mask-rule dependencies"
    ),
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


def main() -> int:
    failures: list[str] = []
    gate_path = Path(__file__).resolve()
    # the one designated home for the archive's historical key names
    exempt = {ROOT / "src" / "pilot_proxy" / "archived_product_keys.py"}
    for path in _iter_files():
        if path.resolve() == gate_path or path in exempt:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for old, new in BANNED.items():
            if old in text:
                failures.append(
                    f"{path.relative_to(ROOT)} contains retired term {old!r}; "
                    f"use {new!r}"
                )
    if failures:
        print("current measurement vocabulary: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("current measurement vocabulary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
