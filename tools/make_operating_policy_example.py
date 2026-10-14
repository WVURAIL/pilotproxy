#!/usr/bin/env python3
"""Create a complete synthetic policy example, never a telescope release.

The decisions are synthetic threshold-rule output for three channels: channel
14 not deployed, channel 29 deployed at Q16=65536, and the channel 37 control
band, which a DTV policy leaves out. The bundle is exported on the CHIME
receiver profile, and channel 29's profile is marked as carrying its word.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from pilot_proxy import operating_policy as op, runtime_calibration as rc
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
from pilot_proxy.runtime_bundle import (
    REQUIRED_RUNTIME_BUNDLE_FILES,
    export_runtime_weight_bundle,
)

REPO = Path(__file__).resolve().parents[1]
WORD = 65536


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_ref(path)


def file_ref(path):
    return {"path": str(path.resolve()), "sha256": rc.sha256(path)}


def synthetic_decisions():
    """Threshold-rule rows for the example: one channel per outcome and the control band."""
    return {"inputs": {"note": "synthetic example; no score cache or tolerance table was read"}, "channels": [
        {"channel": 14, "outcome": "not useful", "reason": "above tolerance", "deployed_at": None,
         "deployed_word": None, "knee_word": 70000},
        {"channel": 29, "outcome": "useful", "reason": None, "deployed_at": "knee", "deployed_word": WORD,
         "knee_word": WORD},
        {"channel": 37, "outcome": "useful", "reason": None, "deployed_at": "knee", "deployed_word": 72850,
         "knee_word": 72850},
    ]}


def make_example(output_dir):
    """Write new, self-contained inputs and the policy plan that compiles them."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receiver_path = out / "receiver_profile.json"
    core_path = out / "detector_core_profile.json"
    shutil.copyfile(REPO / "configs/receiver_profiles/chime_dtv_fengine.json", receiver_path)
    shutil.copyfile(REPO / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json", core_path)
    evidence = write_json(out / "synthetic_evidence.json", {
        "scope": "Command rehearsal only; no physical measurements or deployment approval",
        "decisions": "Synthetic threshold-rule rows, not a channel ruling",
        "consumer": "Fictitious consumer contract, not an implemented binary identity",
        "generator_sha256": rc.sha256(Path(__file__)),
    })
    decisions = write_json(out / "decisions.json", synthetic_decisions())
    bundle = out / "bundle"
    export_runtime_weight_bundle(
        receiver_profile_path=receiver_path, detector_core_profile_path=core_path,
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        physical_channels=[14, 29], output_dir=bundle,
    )
    mask = np.zeros(256, dtype=bool)
    mask[::2] = True
    mask[(62 + np.arange(-4, 5)) % 256] = False
    words = [hex(sum(int(mask[i * 64 + j]) << j for j in range(64))) for i in range(4)]
    profiles_path = bundle / "pilot_profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    by_channel = {p["physical_channel"]: p for p in profiles["profiles"]}
    by_channel[29]["fine_calibration"] = {
        "status": "calibrated", "decision_version": "fine_decision_v1", "anchor_bin": 62,
        "designated_half_width": 2, "bulk_mask_words_hex": words, "cfar_rank": 61,
        "cfar_multiplier_q16": WORD,
        "provenance": {"epochs": ["synthetic-era"], "null_quantile": .95,
                       "source_product_sha256": [decisions["sha256"]],
                       "note": "synthetic example: the word is the example decision's"},
    }
    write_json(profiles_path, profiles)
    names = [name for name in REQUIRED_RUNTIME_BUNDLE_FILES if name != "sha256sums.txt"]
    (bundle / "sha256sums.txt").write_text(
        "".join(f"{rc.sha256(bundle / name)}  {name}\n" for name in names), encoding="utf-8")
    plan = {
        "schema_version": op.PLAN_SCHEMA, "receiver_profile": file_ref(receiver_path),
        "receiver_channel_id_namespace": "chime_freq_id", "decisions": decisions,
        "bundle_files": {name: file_ref(bundle / name) for name in REQUIRED_RUNTIME_BUNDLE_FILES},
        "era": {"id": "synthetic-era", "scope": "Synthetic example only", "evidence": [evidence]},
        "consumer": {"name": op.CONSUMER_NAME, "version": "synthetic-contract-only",
                     "build_sha256": evidence["sha256"], "capabilities": sorted(op.REQUIRED_CAPABILITIES),
                     "evidence": [evidence]},
    }
    write_json(out / "policy_plan.json", plan)
    (out / "README.txt").write_text(
        "SYNTHETIC COMMAND REHEARSAL ONLY\n"
        "This directory has no telescope measurements, real consumer build, or science qualification.\n"
        "The decisions are synthetic and the bundle carries the deployed word by construction.\n"
        "Preserve this directory at its original absolute path for policy revalidation.\n"
        "Compile policy_plan.json with compile-operating-policy, then validate-operating-policy.\n",
        encoding="utf-8",
    )
    return out / "policy_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for synthetic inputs")
    args = parser.parse_args()
    print(make_example(args.output_dir))


if __name__ == "__main__":
    main()
