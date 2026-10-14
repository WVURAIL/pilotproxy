#!/usr/bin/env python3
"""Generate the deterministic TeX facts shared by the formal documents.

Scientific and schema values come from the same checked-in profiles, source
constants, and weight manifest used by the runtime. The generator deliberately
uses only the Python standard library so documentation can be checked without a
GPU or an installed PilotProxy environment.

Usage::

    python tools/generate_doc_specs.py
    python tools/generate_doc_specs.py --check

Check mode performs every source-consistency check and, when a generated file
is present, verifies its bytes. It never creates or modifies a file.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "generated" / "specs.tex"

CHIME_PROFILE_PATH = (
    ROOT / "configs" / "receiver_profiles" / "chime_dtv_fengine.json"
)
DETECTOR_CORE_PATH = (
    ROOT
    / "configs"
    / "detector_core"
    / "pilotproxy_cuda_local_reference_power_ratio.json"
)
WEIGHT_MANIFEST_PATH = (
    ROOT / "weights" / "chime_dtv_weights_k128.bin.manifest.json"
)
WEIGHT_BANK_PATH = ROOT / "weights" / "chime_dtv_weights_k128.bin"
VERSION_SOURCE_PATH = ROOT / "src" / "pilot_proxy" / "_version.py"
PRODUCT_CONTRACT_PATH = ROOT / "src" / "pilot_proxy" / "product_contract.py"
FINE_REDUCTION_PATH = ROOT / "src" / "pilot_proxy" / "fine_reduction.py"

SHORT_DIGEST_HEX_CHARS = 16
CAPTURE_LOSS_EXAMPLE_OFFSETS_HZ = (300.0, 500.0)


def _die(msg: str) -> None:
    print(f"generate_doc_specs: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"cannot read {path.relative_to(ROOT)}: {exc}")
    if not isinstance(value, dict):
        _die(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def _source_constant(path: Path, name: str) -> Any:
    """Read a literal module-level constant without importing the package."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        _die(f"cannot parse {path.relative_to(ROOT)}: {exc}")
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        try:
            return ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            _die(
                f"{name} in {path.relative_to(ROOT)} must remain a literal "
                f"for the stdlib-only documentation generator: {exc}"
            )
    _die(f"{name} not found in {path.relative_to(ROOT)}")


def package_version() -> str:
    value = _source_constant(VERSION_SOURCE_PATH, "__version__")
    if not isinstance(value, str) or not value:
        _die("src/pilot_proxy/_version.py must define a non-empty __version__")
    return value


def per_pilot_schema_token() -> str:
    name = _source_constant(
        PRODUCT_CONTRACT_PATH, "PER_PILOT_PRODUCT_SCHEMA_NAME"
    )
    revision = _source_constant(
        PRODUCT_CONTRACT_PATH, "PER_PILOT_PRODUCT_SCHEMA_REVISION"
    )
    if not isinstance(name, str) or not isinstance(revision, int):
        _die("per-pilot schema name/revision constants have invalid types")
    return f"{name}_v{revision}"


def kernel_core_version() -> str:
    text = (ROOT / "cuda" / "config.h").read_text(encoding="utf-8")
    parts: dict[str, str] = {}
    for field in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define\s+FSTAT_CORE_VERSION_{field}\s+(\d+)", text)
        if not match:
            _die(f"FSTAT_CORE_VERSION_{field} not found in cuda/config.h")
        parts[field] = match.group(1)
    return "{MAJOR}.{MINOR}.{PATCH}".format(**parts)


def sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:SHORT_DIGEST_HEX_CHARS]


def manifest_facts(manifest: dict[str, Any]) -> dict[str, Any]:
    layout = manifest.get("target_reference_layout")
    if not isinstance(layout, list) or not layout:
        _die("weight manifest has no target_reference_layout entries")
    mus: list[float] = []
    adaptive: list[int] = []
    for entry in layout:
        if not isinstance(entry, dict):
            _die("weight manifest target_reference_layout entries must be objects")
        target_norm_sq = int(entry["target_norm_sq"])
        reference_norm_sum_sq = int(entry["reference_norm_sum_sq"])
        if reference_norm_sum_sq <= 0:
            _die("weight manifest contains a non-positive reference norm")
        mus.append(2.0 * target_norm_sq / reference_norm_sum_sq)
        if entry.get("adaptive_reference_placement"):
            adaptive.append(int(entry["physical_channel"]))
    return {
        "null_power_ratio_min": min(mus),
        "null_power_ratio_max": max(mus),
        "n_channels": len(layout),
        "adaptive_channels": adaptive,
    }


def _require_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        _die(f"{label} is inconsistent: {actual!r} != {expected!r}")


def _geometry() -> dict[str, Any]:
    profile = _json_object(CHIME_PROFILE_PATH)
    core = _json_object(DETECTOR_CORE_PATH)
    channelizer = profile["channelizer"]
    framing = profile["framing"]
    streams = profile["input_streams"]
    adapter = profile["detector_adapter"]
    kernel_contract = core["kernel_contract"]

    input_sample_rate_hz = float(channelizer["input_sample_rate_hz"])
    pfb_fft_size = int(channelizer["pfb_fft_size"])
    coarse_rate_hz = input_sample_rate_hz / pfb_fft_size
    detector_window_samples = int(adapter["detector_window_samples"])
    if detector_window_samples != int(kernel_contract["detector_window_samples"]):
        _die("CHIME receiver profile and detector-core profile select different K")
    _require_close(
        "coarse channel spacing",
        float(channelizer["coarse_channel_center_offset_hz"]),
        coarse_rate_hz,
    )
    detector_bin_hz = coarse_rate_hz / detector_window_samples
    _require_close(
        "detector_adapter.fine_bin_enbw_hz",
        float(adapter["fine_bin_enbw_hz"]),
        detector_bin_hz,
    )

    frame_samples = int(framing["frame_size_samples"])
    if frame_samples <= 0 or frame_samples % detector_window_samples:
        _die("CHIME frame size must be a positive multiple of K")
    windows_per_stream = frame_samples // detector_window_samples
    fine_pad_factor = int(
        _source_constant(FINE_REDUCTION_PATH, "FINE_PAD_FACTOR")
    )
    fine_num_bins = fine_pad_factor * windows_per_stream
    fine_bin_hz = detector_bin_hz / fine_num_bins
    return {
        "coarse_rate_hz": coarse_rate_hz,
        "detector_window_samples": detector_window_samples,
        "detector_bin_hz": detector_bin_hz,
        "fine_pad_factor": fine_pad_factor,
        "fine_num_bins": fine_num_bins,
        "fine_bin_hz": fine_bin_hz,
        "dtv_bandwidth_hz": float(adapter["dtv_bandwidth_hz"]),
        "num_input_streams": int(streams["num_input_streams"]),
        "windows_per_stream": windows_per_stream,
    }


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _capture_loss_db(offset_hz: float, detector_bin_hz: float) -> float:
    x = offset_hz / detector_bin_hz
    response = math.sin(math.pi * x) / (math.pi * x)
    return -20.0 * math.log10(response)


def render_specs() -> str:
    geometry = _geometry()
    facts = manifest_facts(_json_object(WEIGHT_MANIFEST_PATH))
    detector_bin_hz = geometry["detector_bin_hz"]
    bandwidth_spread_db = 10.0 * math.log10(
        geometry["dtv_bandwidth_hz"] / detector_bin_hz
    )
    adaptive_channels = ", ".join(str(value) for value in facts["adaptive_channels"])
    rows_per_frame = (
        geometry["num_input_streams"] * geometry["windows_per_stream"]
    )
    capture_300_hz, capture_500_hz = CAPTURE_LOSS_EXAMPLE_OFFSETS_HZ

    lines = [
        "% Generated by tools/generate_doc_specs.py -- DO NOT EDIT.",
        "% Regenerate with: make docs-specs",
        f"\\newcommand{{\\ppPackageVersion}}{{{package_version()}}}",
        f"\\newcommand{{\\ppKernelCoreVersion}}{{{kernel_core_version()}}}",
        "\\newcommand{\\ppPerPilotSchema}{"
        + _tex_escape(per_pilot_schema_token())
        + "}",
        f"\\newcommand{{\\ppDetectorWindowSamples}}{{{geometry['detector_window_samples']}}}",
        f"\\newcommand{{\\ppCoarseSpacingkHz}}{{{geometry['coarse_rate_hz'] / 1e3:.3f}}}",
        f"\\newcommand{{\\ppDetectorBinHz}}{{{detector_bin_hz:.11g}}}",
        f"\\newcommand{{\\ppFinePadFactor}}{{{geometry['fine_pad_factor']}}}",
        f"\\newcommand{{\\ppFineNumBins}}{{{geometry['fine_num_bins']}}}",
        f"\\newcommand{{\\ppFineBinHz}}{{{geometry['fine_bin_hz']:.6f}}}",
        f"\\newcommand{{\\ppWindowsPerStream}}{{{geometry['windows_per_stream']}}}",
        f"\\newcommand{{\\ppChimeStreams}}{{{geometry['num_input_streams']}}}",
        "\\newcommand{\\ppChimeRowsPerFrame}{"
        + f"{rows_per_frame:,}".replace(",", "{,}")
        + "}",
        f"\\newcommand{{\\ppBandwidthSpreadDb}}{{{bandwidth_spread_db:.3f}}}",
        f"\\newcommand{{\\ppMuZeroMin}}{{{facts['null_power_ratio_min']:.10f}}}",
        f"\\newcommand{{\\ppMuZeroMax}}{{{facts['null_power_ratio_max']:.10f}}}",
        f"\\newcommand{{\\ppWeightChannelCount}}{{{facts['n_channels']}}}",
        f"\\newcommand{{\\ppAdaptiveChannels}}{{{adaptive_channels}}}",
        "\\newcommand{\\ppCaptureLossThreeHundredDb}{"
        + f"{_capture_loss_db(capture_300_hz, detector_bin_hz):.4f}"
        + "}",
        "\\newcommand{\\ppCaptureLossFiveHundredDb}{"
        + f"{_capture_loss_db(capture_500_hz, detector_bin_hz):.4f}"
        + "}",
        "\\newcommand{\\ppCaptureLossEdgeDb}{"
        + f"{_capture_loss_db(detector_bin_hz / 2.0, detector_bin_hz):.2f}"
        + "}",
        "\\newcommand{\\ppCoherentGainDb}{"
        + f"{10.0 * math.log10(math.sqrt(geometry['windows_per_stream'])):.1f}"
        + "}",
        f"\\newcommand{{\\ppWeightBankShaShort}}{{{sha256_short(WEIGHT_BANK_PATH)}}}",
        f"\\newcommand{{\\ppWeightManifestShaShort}}{{{sha256_short(WEIGHT_MANIFEST_PATH)}}}",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate sources and an existing generated file without writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = render_specs()
    if args.check:
        if OUT.exists() and OUT.read_text(encoding="utf-8") != expected:
            print(
                "generate_doc_specs: docs/generated/specs.tex is stale; "
                "run 'make docs-specs'",
                file=sys.stderr,
            )
            return 1
        print("documentation specifications: PASS")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(expected, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(expected.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
