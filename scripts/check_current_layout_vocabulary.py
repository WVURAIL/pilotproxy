#!/usr/bin/env python3
# coding=utf-8
"""Reject retired detector-layout classes and serialized aliases."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACES = (
    ROOT / "src/pilot_proxy/detector_geometry.py",
    ROOT / "src/pilot_proxy/integration/__init__.py",
    ROOT / "src/pilot_proxy/integration/schemas.py",
    ROOT / "src/pilot_proxy/integration/packing.py",
    ROOT / "src/pilot_proxy/integration/stream_layout.py",
    ROOT / "src/pilot_proxy/chime/frame_adapter.py",
    ROOT / "src/pilot_proxy/testbench/evaluate_snr.py",
    ROOT / "src/pilot_proxy/testbench/quantize.py",
    ROOT / "tests/core/test_frame_geometry_adapter.py",
    ROOT / "tests/core/test_chord_profiles.py",
    ROOT / "tests/core/test_integration_contract.py",
    ROOT / "tests/core/test_stream_layout_profile_window.py",
    ROOT / "tests/chime/test_chime_frame_adapter.py",
)

BANNED = {
    "Detector" + "InputLayout": "DetectorFrameLayout",
    "Input" + "StreamLayout": "DetectorFrameLayout",
    "derive_detector_" + "input_layout": "DetectorFrameLayout",
    "input_layout_" + "metadata": "DetectorFrameLayout(...).to_dict()",
    "windows_per_" + "block": "windows_per_stream",
    '"samples_per_' + 'frame"': '"frame_size_samples"',
    '"windows_per_' + 'feed"': '"windows_per_stream"',
    '"num_' + 'feeds"': '"num_input_streams"',
    '"detector_rows_per_' + 'block"': '"detector_rows_per_frame"',
    "COMBINE_MODE_INCOHERENT_POWER_SUM_OVER_" + "STREAMS": (
        "COMBINE_MODE_COMBINED_STREAMS"
    ),
}


def main() -> int:
    failures: list[str] = []
    for path in SURFACES:
        if not path.is_file():
            failures.append(f"missing enforced layout surface: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for retired, replacement in BANNED.items():
            if retired in text:
                failures.append(
                    f"{path.relative_to(ROOT)} contains retired term "
                    f"{retired!r}; use {replacement!r}"
                )
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("current detector-layout vocabulary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
