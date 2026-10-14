#!/usr/bin/env python3
"""Generate a small, deterministic native capture fixture, never telescope data.

The default pair contains one valid random block and one invalid all-zero block.
Use --valid-only for uniform synthetic draws suitable for testing the population
handoff. Neither fixture qualifies a threshold or exercises a GPU implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from pilot_proxy.capture_reference import (
    CHORD_METADATA_LIMITS,
    CHORD_METADATA_SIZE,
    REFERENCE_KOTEKAN_COMMIT,
    CaptureGeometry,
    expected_products,
)
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
from pilot_proxy.fine_decision import pack_bulk_mask
from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.runtime_bundle import (
    DEFAULT_SHA256SUMS_FILENAME,
    REQUIRED_RUNTIME_BUNDLE_FILES,
    export_runtime_weight_bundle,
)

REPO = Path(__file__).resolve().parents[1]
SEED = 20261006


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_ref(path, root):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def native_frame(payload, *, kind, sequence, counter, geometry):
    """Serialize the pinned little-endian metadata ABI for synthetic examples."""
    layouts = {
        "voltage": (
            "E",
            8,
            (geometry.num_times, geometry.num_freq, 2, geometry.num_dishes),
            ("T", "F", "P", "D"),
            1,
        ),
        "masks": ("dtv_mask", 9, (geometry.num_freq,), ("F",), geometry.block_samples),
        "powers": (
            "dtv_powers",
            6,
            (geometry.num_freq, 3),
            ("F", "W"),
            geometry.block_samples,
        ),
    }
    name, dtype, dims, names, downsampling = layouts[kind]
    metadata = bytearray(CHORD_METADATA_SIZE)
    struct.pack_into(
        "<7i4xqi", metadata, 0, *CHORD_METADATA_LIMITS, counter, sequence, downsampling
    )
    metadata[44 : 44 + len(name)] = name.encode("ascii")
    struct.pack_into("<2i", metadata, 68, dtype, len(dims))
    struct.pack_into(f"<{len(dims)}i", metadata, 76, *dims)
    for index, axis in enumerate(names):
        metadata[116 + 24 * index : 116 + 24 * index + len(axis)] = axis.encode("ascii")
    struct.pack_into(f"<{len(dims)}q", metadata, 360, *([1] * len(dims)))
    stride, strides = 1, []
    for extent in reversed(dims):
        strides.insert(0, stride)
        stride *= extent
    struct.pack_into(f"<{len(dims)}q", metadata, 440, *strides)
    struct.pack_into("<i", metadata, 528, geometry.num_freq)
    struct.pack_into(f"<{geometry.num_freq}i", metadata, 532, *geometry.frequency_ids)
    struct.pack_into(
        f"<{geometry.num_freq}i", metadata, 49684, *([1] * geometry.num_freq)
    )
    struct.pack_into("<i", metadata, 147988, -1)
    struct.pack_into("<2i", metadata, 148504, -1, -1)
    return struct.pack("<I", CHORD_METADATA_SIZE) + metadata + payload


def make_example(
    output_dir,
    *,
    valid_only=False,
    weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
):
    """Write a new example directory and return its replay plan path."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receiver = json.loads(
        (
            REPO / "configs/receiver_profiles/chord_pathfinder_dtv_fengine.json"
        ).read_text()
    )
    receiver["receiver_profile_id"] = "synthetic_capture_two_inputs"
    receiver["instrument_name"] = (
        "Synthetic capture fixture; not a telescope configuration"
    )
    receiver["input_streams"]["num_input_streams"] = 2
    receiver["metadata"]["kotekan_binding"]["num_dishes"] = 1
    receiver["metadata"]["digitizer"]["num_adc_inputs"] = 2
    receiver["metadata"]["profile_status"] = "synthetic_fixture_only"
    write_json(output / "receiver.json", receiver)
    evidence = {
        "kind": "synthetic_fixture_only",
        "generator": "numpy.random.SeedSequence(20261006).spawn(2), PCG64 child streams",
        "law": "independent uniform native uint8 blocks"
        if valid_only
        else "independent equal mixture of uniform native uint8 blocks and zero complex blocks",
        "limitations": "CPU-generated recorded products test serialization and replay plumbing only; no GPU parity, physical null, telescope timing or threshold qualification is established.",
    }
    write_json(output / "synthetic_evidence.json", evidence)
    write_json(
        output / "sampling_design.json",
        {
            "population": "synthetic pseudorandom draws only",
            "seed": SEED,
            "units": "One separate SeedSequence child per block; no samples selected from telescope data.",
        },
    )
    export_runtime_weight_bundle(
        receiver_profile_path=output / "receiver.json",
        detector_core_profile_path=REPO
        / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json",
        physical_channels=[14],
        weight_coordinate_system=weight_coordinate_system,
        output_dir=output / "bundle",
    )
    bundle_path = output / "bundle/pilot_profiles.json"
    bundle = json.loads(bundle_path.read_text())
    bulk = independent_bin_mask(256, designated_bins=list(range(60, 65)))
    bundle["profiles"][0]["fine_calibration"] = {
        "status": "calibrated",
        "decision_version": "fine_decision_v1",
        "anchor_bin": 62,
        "designated_half_width": 2,
        "bulk_mask_words_hex": [f"0x{word:016x}" for word in pack_bulk_mask(bulk)],
        "cfar_rank": int(sum(bulk) // 2),
        "cfar_multiplier_q16": 196608,
        "provenance": {
            "epochs": ["synthetic_fixture_only"],
            "null_quantile": 0.99,
            "source_product_sha256": [
                file_ref(output / "synthetic_evidence.json", output)["sha256"]
            ],
        },
    }
    write_json(bundle_path, bundle)
    sums = output / "bundle" / DEFAULT_SHA256SUMS_FILENAME
    sums.write_text(
        "".join(
            f"{file_ref(output / 'bundle' / name, output)['sha256']}  {name}\n"
            for name in REQUIRED_RUNTIME_BUNDLE_FILES
            if name != DEFAULT_SHA256SUMS_FILENAME
        )
    )
    geometry = CaptureGeometry(
        num_times=8192, k=64, frequency_ids=(2408, 1600), num_dishes=1
    )
    first_sequence = 1000000
    packing = {
        "schema_version": "pilotproxy_capture_packing_v1",
        "layout": "TFPD",
        "sample_encoding": "int4x2_swapped_withoffset",
        "pre_detector_transform": "none",
        "voltage_tap": "pre_dtv_mask",
        "num_times": geometry.num_times,
        "detector_window_samples": 64,
        "num_dishes": 1,
        "num_polarizations": 2,
        "windows_per_stream": 128,
        "frequency_ids": list(geometry.frequency_ids),
        "spectral_sense": "normal",
        "sample_rate_hz": "195312.5",
        "time_reverse_detector_windows_before_kernel": bundle["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ],
        "frequency_map": [
            {
                "receiver_channel_id": freq,
                "coarse_channel_index": freq,
                "rf_center_hz": str(freq * 195312.5),
            }
            for freq in geometry.frequency_ids
        ],
        "timing": {
            "time_downsampling_fpga": 1,
            "utc_reference": {
                "fpga_seq_num": 0,
                "unix_seconds": "0",
                "tick_period_seconds": "0.00000512",
            },
        },
    }
    write_json(output / "packing.json", packing)
    write_json(
        output / "synthetic_gains.json",
        {
            "kind": "synthetic_fixture_only",
            "gain": "unitless native integer generator; no hardware",
        },
    )
    state = {
        "schema_version": "pilotproxy_capture_input_state_v1",
        "input_population_id": "synthetic-two-inputs",
        "streams": [
            {
                "stream_index": p,
                "polarization_index": p,
                "dish_index": 0,
                "input_id": f"synthetic-input-{p}",
                "status": "live",
            }
            for p in range(2)
        ],
        "gain_evidence": file_ref(output / "synthetic_gains.json", output),
        "quantization": "Synthetic integer components [-8,7], offset-binary nibble packing, no hardware gains",
        "preprocessing": "None before the native voltage tap; all configured slots enter the detector",
        "valid_from_fpga_seq_num": 0,
        "valid_until_fpga_seq_num": 100000000,
    }
    write_json(output / "input_state.json", state)
    for filename in (
        "synthetic_binary.txt",
        "synthetic_configuration.txt",
        "synthetic_vendor.txt",
    ):
        (output / filename).write_text(
            "Synthetic fixture placeholder only. No telescope binary, configuration, or vendor build is represented.\n"
        )
    refs = {
        name: file_ref(output / filename, output)
        for name, filename in (
            ("receiver_profile", "receiver.json"),
            ("packing_contract", "packing.json"),
            ("input_state_contract", "input_state.json"),
        )
    }
    bundle_refs = {
        name: file_ref(output / "bundle" / name, output)
        for name in REQUIRED_RUNTIME_BUNDLE_FILES
    }
    producer = {
        "schema_version": "pilotproxy_capture_producer_v1",
        "source_revision": REFERENCE_KOTEKAN_COMMIT,
        "description": "Synthetic CPU fixture: source_revision identifies the reference ABI, not a running telescope build",
        "binary": file_ref(output / "synthetic_binary.txt", output),
        "configuration": file_ref(output / "synthetic_configuration.txt", output),
        "vendor_manifest": file_ref(output / "synthetic_vendor.txt", output),
        "binding": {
            **{f"{name}_sha256": ref["sha256"] for name, ref in refs.items()},
            "bundle_files_sha256": {
                name: ref["sha256"] for name, ref in bundle_refs.items()
            },
        },
        "detector": {
            "enabled": True,
            "samples_per_detector_frame": 8192,
            "num_input_streams": 2,
            "frequency_ids": list(geometry.frequency_ids),
        },
    }
    write_json(output / "producer.json", producer)
    recordings = {kind: bytearray() for kind in ("voltage", "masks", "powers")}
    weights = (output / "bundle/weights.bin").read_bytes()
    for block, seed in enumerate(np.random.SeedSequence(SEED).spawn(2)):
        rng = np.random.default_rng(seed)
        random_block = bool(rng.integers(2))
        voltage = (
            rng.integers(0, 256, geometry.voltage_frame_size, dtype=np.uint8)
            if valid_only or random_block
            else np.full(geometry.voltage_frame_size, 0x88, np.uint8)
        )
        products = expected_products(voltage, bundle, weights, geometry=geometry)
        payloads = {
            "voltage": voltage.tobytes(),
            "masks": products.masks.tobytes(),
            "powers": products.powers.astype("<u8").tobytes(),
        }
        for kind, payload in payloads.items():
            recordings[kind].extend(
                native_frame(
                    payload,
                    kind=kind,
                    sequence=first_sequence + block * 8192,
                    counter=block,
                    geometry=geometry,
                )
            )
    for kind, recording in recordings.items():
        (output / f"{kind}.raw").write_bytes(recording)
    null_ref = file_ref(output / "synthetic_evidence.json", output)
    design_ref = file_ref(output / "sampling_design.json", output)
    plan = {
        "schema_version": "pilotproxy_capture_fine_replay_plan_v1",
        "capture_id": "synthetic-capture-example",
        "first_fpga_seq_num": first_sequence,
        "expected_input_frames": 2,
        **refs,
        "producer": file_ref(output / "producer.json", output),
        "bundle_files": bundle_refs,
        "raw_files": {
            kind: [file_ref(output / f"{kind}.raw", output)] for kind in recordings
        },
        "profiles": [
            {
                "physical_channel": 14,
                "profile_id": "synthetic-CH14",
                "population": {
                    "role": "calibration",
                    "population_id": "synthetic-calibration-fixture",
                    "null_evidence": null_ref,
                    "sampling_design": design_ref,
                    "null_law": {
                        "id": "synthetic-example-model",
                        "kind": "synthetic_model",
                        "description": evidence["law"],
                        "evidence_sha256": null_ref["sha256"],
                    },
                    "sampling": {
                        "assumption": "independent_identically_distributed",
                        "independence_rationale": "Separate SeedSequence child streams simulate independent draws of the declared synthetic law; no telescope independence is inferred.",
                        "design_sha256": design_ref["sha256"],
                    },
                    "units": [
                        {
                            "block_index": block,
                            "trial_id": f"synthetic-seed-{SEED}-block-{block}",
                            "independent_unit_id": f"synthetic-seed-{SEED}-child-{block}",
                        }
                        for block in range(2)
                    ],
                },
            }
        ],
    }
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="generate two uniform blocks without the invalid-support example",
    )
    args = parser.parse_args()
    print(make_example(args.output_dir, valid_only=args.valid_only))


if __name__ == "__main__":
    main()
