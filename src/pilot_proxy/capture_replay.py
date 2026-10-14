"""Authenticated, offline fine replay of a declared native CHORD capture.

Contracts describe the recorded hardware state; replay checks their consistency,
not their physical truth. Invalid decisions and product mismatches are retained.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from pilot_proxy import (
    capture_reference,
    fine_decision,
    fine_reduction,
    fxfft,
    runtime_bundle,
    runtime_calibration,
)
from pilot_proxy.integration.receiver_profile import (
    load_receiver_profile,
    receiver_profile_hash,
)
from pilot_proxy.integration.weight_generation import (
    profile_requires_window_time_reversal,
)
from pilot_proxy.runtime_bundle import (
    REQUIRED_RUNTIME_BUNDLE_FILES,
    validate_runtime_weight_bundle,
)
from pilot_proxy.runtime_calibration import (
    POPULATION_SCHEMA,
    PROFILE_SCHEMA,
    SCORE_ENCODING,
    SUPPORT_RULE,
    _bound_file,
    _digest,
    _integer,
    _read,
    _require,
    _text,
    _write_json,
    sha256,
    validate_score_profile,
)

PLAN_SCHEMA = "pilotproxy_capture_fine_replay_plan_v1"
PACKING_SCHEMA = "pilotproxy_capture_packing_v1"
INPUT_SCHEMA = "pilotproxy_capture_input_state_v1"
PRODUCER_SCHEMA = "pilotproxy_capture_producer_v1"


def _fraction(value, name, positive=False):
    _require(type(value) in (str, int, float), f"{name} must be a finite decimal")
    try:
        number = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    _require(not positive or number > 0, f"{name} must be positive")
    return number


def _object(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _same(actual, expected, name):
    _require(type(actual) is type(expected) and json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), f"{name} mismatch")


def _source_identity():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "source_sha256": {
            module.__name__: sha256(module.__file__)
            for module in (capture_reference, fine_decision, fine_reduction, fxfft, runtime_bundle, runtime_calibration)
        },
        "exporter_sha256": sha256(__file__),
    }


def _load_contracts(plan, base, bindings):
    paths = {
        name: _bound_file(plan.get(name), base, bindings)
        for name in ("receiver_profile", "packing_contract", "input_state_contract", "producer")
    }
    # Parse even the receiver through the strict reader before its typed loader.
    data = {name: _read(path) for name, path in paths.items()}
    receiver = load_receiver_profile(paths["receiver_profile"])
    packing, state, producer = (data[k] for k in ("packing_contract", "input_state_contract", "producer"))
    _same(packing.get("schema_version"), PACKING_SCHEMA, "packing schema")
    _same(state.get("schema_version"), INPUT_SCHEMA, "input-state schema")
    _same(producer.get("schema_version"), PRODUCER_SCHEMA, "producer schema")
    _same(packing.get("layout"), "TFPD", "voltage layout")
    _same(packing.get("sample_encoding"), "int4x2_swapped_withoffset", "native packing")
    _same(packing.get("pre_detector_transform"), "none", "native voltage transform")
    _same(packing.get("voltage_tap"), "pre_dtv_mask", "voltage tap")
    freqs = packing.get("frequency_ids")
    _require(isinstance(freqs, list) and freqs, "frequency_ids must be a nonempty list")
    geometry = capture_reference.CaptureGeometry(
        num_times=_integer(packing.get("num_times"), "num_times", 1),
        k=_integer(packing.get("detector_window_samples"), "detector_window_samples", 1),
        frequency_ids=tuple(freqs),
        num_dishes=_integer(packing.get("num_dishes"), "num_dishes", 1),
        num_polarizations=_integer(packing.get("num_polarizations"), "num_polarizations", 1),
        windows_per_stream=_integer(packing.get("windows_per_stream"), "windows_per_stream", 1),
    )
    _same(geometry.k, receiver.detector_window_samples, "receiver window")
    _same(geometry.block_samples, receiver.frame_size_samples, "receiver detector frame")
    _same(geometry.num_streams, receiver.num_input_streams, "receiver input population")
    _same(packing.get("spectral_sense"), receiver.spectral_sense, "receiver spectral sense")
    _require(
        _fraction(packing.get("sample_rate_hz"), "sample_rate_hz", True)
        == _fraction(receiver.coarse_channel_width_hz, "receiver sample rate", True),
        "Receiver sample-rate mismatch",
    )
    channel_map = _object(data["receiver_profile"].get("metadata", {}).get("channel_id_map"), "receiver channel map")
    _same(channel_map.get("namespace"), "chord_freq_id", "receiver frequency namespace")
    offset = channel_map.get("offset_from_coarse_channel_index")
    _require(type(offset) is int, "Receiver frequency offset must be an integer")
    mapping = packing.get("frequency_map")
    _require(isinstance(mapping, list) and len(mapping) == len(freqs), "Missing frequency mapping")
    for freq, row in zip(freqs, mapping):
        _object(row, "frequency map row")
        _same(row.get("receiver_channel_id"), freq, "frequency mapping order")
        _same(row.get("coarse_channel_index"), freq - offset, "receiver coarse-channel mapping")
        _require(
            _fraction(row.get("rf_center_hz"), "rf_center_hz")
            == _fraction(receiver.coarse_channel_center_hz(freq - offset), "receiver RF center"),
            "Receiver RF-center mismatch",
        )
    _text(state.get("input_population_id"), "input_population_id")
    streams = state.get("streams")
    _require(isinstance(streams, list) and len(streams) == geometry.num_streams, "Input-state stream count mismatch")
    ids = []
    for index, stream in enumerate(streams):
        _object(stream, "input stream")
        _same(stream.get("stream_index"), index, "input order")
        _same(stream.get("polarization_index"), index // geometry.num_dishes, "polarization order")
        _same(stream.get("dish_index"), index % geometry.num_dishes, "dish order")
        ids.append(_text(stream.get("input_id"), "input_id"))
        _require(stream.get("status") in ("live", "missing", "flagged", "rfi_antenna"), "Unknown input status")
    _require(len(set(ids)) == len(ids), "Repeated input identity")
    _bound_file(state.get("gain_evidence"), paths["input_state_contract"].parent, bindings)
    for field in ("quantization", "preprocessing"):
        _text(state.get(field), f"input-state {field}")
    timing = _object(packing.get("timing"), "timing")
    start = _integer(plan.get("first_fpga_seq_num"), "first_fpga_seq_num")
    down = _integer(timing.get("time_downsampling_fpga"), "time_downsampling_fpga", 1)
    utc = _object(timing.get("utc_reference"), "UTC reference")
    _integer(utc.get("fpga_seq_num"), "UTC reference sequence")
    _fraction(utc.get("unix_seconds"), "UTC reference seconds")
    tick = _fraction(utc.get("tick_period_seconds"), "FPGA tick period", True)
    _require(tick * down * _fraction(packing["sample_rate_hz"], "sample rate") == 1, "FPGA/sample-rate mismatch")
    frame_count = _integer(plan.get("expected_input_frames"), "expected_input_frames", 1)
    end = start + frame_count * geometry.num_times * down
    valid_start = _integer(state.get("valid_from_fpga_seq_num"), "input-state validity start")
    valid_end = _integer(state.get("valid_until_fpga_seq_num"), "input-state validity end", 1)
    _require(valid_start <= start < end <= valid_end, "Capture outside input-state validity interval")
    _require(isinstance(producer.get("source_revision"), str) and re.fullmatch(r"[0-9a-f]{40}", producer["source_revision"]), "Producer requires a full source revision")
    for field in ("binary", "configuration", "vendor_manifest"):
        _bound_file(producer.get(field), paths["producer"].parent, bindings)
    return paths, data, receiver, geometry


def _load_bundle(plan, base, bindings, paths, data, receiver, geometry):
    refs = _object(plan.get("bundle_files"), "bundle_files")
    _require(set(refs) == set(REQUIRED_RUNTIME_BUNDLE_FILES), "Pin every required runtime bundle file")
    files = {name: _bound_file(refs[name], base, bindings) for name in refs}
    parents = {p.parent for p in files.values()}
    _require(len(parents) == 1 and all(p.name == name for name, p in files.items()), "Bundle files must retain names in one directory")
    loaded = {name: _read(path) for name, path in files.items() if name.endswith(".json")}
    result = validate_runtime_weight_bundle(bundle_dir=next(iter(parents)))
    _require(result["valid"], f"Invalid runtime bundle: {result['errors']}")
    bundle = loaded["pilot_profiles.json"]
    contract = loaded["detector_contract.json"]
    manifest = loaded["weights.manifest.json"]
    _same(bundle.get("receiver_profile_id"), receiver.receiver_profile_id, "bundle receiver")
    _same(manifest.get("receiver_profile_hash"), receiver_profile_hash(receiver), "bundle receiver content")
    _same(bundle.get("receiver_channel_id_namespace"), "chord_freq_id", "bundle namespace")
    _same(contract.get("detector_window_samples"), geometry.k, "bundle window")
    reverse = bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"]
    _require(type(reverse) is bool, "Bundle time reversal must be a boolean")
    _same(reverse, profile_requires_window_time_reversal(receiver, bundle["weight_coordinate_system"]), "receiver/weight preprocessing")
    _same(data["packing_contract"].get("time_reverse_detector_windows_before_kernel"), reverse, "packing preprocessing")
    producer = data["producer"]
    expected_binding = {
        **{f"{name}_sha256": sha256(paths[name]) for name in ("receiver_profile", "packing_contract", "input_state_contract")},
        "bundle_files_sha256": {name: sha256(path) for name, path in files.items()},
    }
    _same(producer.get("binding"), expected_binding, "recorded producer binding")
    expected_detector = {
        "enabled": True,
        "samples_per_detector_frame": geometry.block_samples,
        "num_input_streams": geometry.num_streams,
        "frequency_ids": list(geometry.frequency_ids),
    }
    _same(producer.get("detector"), expected_detector, "recorded detector configuration")
    for row in bundle["profiles"]:
        _integer(row.get("physical_channel"), "submitted physical channel", 14, 36)
    bound = [row for row in bundle["profiles"] if row["chord_channel_id"] in geometry.frequency_ids]
    _require(bound, "Capture has no bound fine profiles")
    for row in bound:
        _same(row["fine_calibration"].get("status"), "calibrated", "fine calibration status")
        maprow = data["packing_contract"]["frequency_map"][list(geometry.frequency_ids).index(row["chord_channel_id"])]
        _same(row["coarse_channel_index"], maprow["coarse_channel_index"], "bundle coarse channel")
        _require(_fraction(row["coarse_channel_center_hz"], "bundle center") == _fraction(maprow["rf_center_hz"], "capture center"), "Bundle RF-center mismatch")
    return bundle, files["weights.bin"].read_bytes(), bound


def _population(entry, profile, profile_sha, rows, base, bindings):
    declaration = entry.get("population")
    if declaration is None:
        return None
    _object(declaration, "population declaration")
    _require(declaration.get("role") in ("calibration", "evaluation"), "Unknown population role")
    _text(declaration.get("population_id"), "population_id")
    law = _object(declaration.get("null_law"), "null_law")
    _text(law.get("id"), "null_law.id")
    _text(law.get("description"), "null_law.description")
    _require(law.get("kind") in ("synthetic_model", "operational_background", "verified_signal_free"), "Unknown null law")
    evidence = _bound_file(declaration.get("null_evidence"), base, bindings)
    _same(law.get("evidence_sha256"), sha256(evidence), "null evidence")
    sampling = _object(declaration.get("sampling"), "sampling")
    _same(sampling.get("assumption"), "independent_identically_distributed", "declared sampling assumption")
    _text(sampling.get("independence_rationale"), "independence rationale")
    design = _bound_file(declaration.get("sampling_design"), base, bindings)
    _same(sampling.get("design_sha256"), sha256(design), "sampling design")
    units = declaration.get("units")
    _require(isinstance(units, list) and len(units) == len(rows), "Population must retain every captured block, including invalid rows")
    scores, trials, independent = [], set(), set()
    for index, (row, unit) in enumerate(zip(rows, units)):
        _object(unit, "population unit")
        _same(unit.get("block_index"), index, "population block order")
        trial = _text(unit.get("trial_id"), "trial_id")
        iid = _text(unit.get("independent_unit_id"), "independent_unit_id")
        _require(trial not in trials and iid not in independent, "Repeated population trial/independent unit")
        trials.add(trial)
        independent.add(iid)
        scores.append({**row, "declared_trial_id": trial, "independent_unit_id": iid})
    return {
        "schema_version": POPULATION_SCHEMA,
        "profile_id": profile["profile_id"],
        "profile_sha256": profile_sha,
        "score_encoding": SCORE_ENCODING,
        "role": declaration["role"],
        "population_id": declaration["population_id"],
        "null_law": law,
        "sampling": sampling,
        "scores": scores,
    }


def _encoded_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def replay_capture(plan_path):
    """Compute deterministic artifacts in memory; preserve failed parity as data."""
    path = Path(plan_path).resolve()
    plan_sha = sha256(path)
    plan = _read(path)
    _same(plan.get("schema_version"), PLAN_SCHEMA, "replay plan schema")
    capture_id = _text(plan.get("capture_id"), "capture_id")
    bindings = {str(path): plan_sha}
    paths, data, receiver, geometry = _load_contracts(plan, path.parent, bindings)
    bundle, weights, bound = _load_bundle(plan, path.parent, bindings, paths, data, receiver, geometry)
    raw = _object(plan.get("raw_files"), "raw_files")
    captures = {}
    seen = set()
    for kind, size in (("voltage", geometry.voltage_frame_size), ("masks", geometry.mask_frame_size), ("powers", geometry.power_frame_size)):
        refs = raw.get(kind)
        _require(isinstance(refs, list) and refs, f"Missing ordered {kind} files")
        files = [_bound_file(ref, path.parent, bindings) for ref in refs]
        _require(not seen.intersection(files) and len(set(files)) == len(files), "Repeated raw capture file")
        seen.update(files)
        captures[kind] = capture_reference.read_raw_frames(files, size)
    nblocks = capture_reference.validate_frames(captures["voltage"], captures["masks"], captures["powers"], plan["expected_input_frames"], geometry=geometry)
    first = captures["voltage"][0]
    timing = data["packing_contract"]["timing"]
    _same(first.fpga_seq_num, plan["first_fpga_seq_num"], "capture first FPGA sequence")
    _same(first.time_downsampling_fpga, timing["time_downsampling_fpga"], "capture FPGA downsampling")
    entries = plan.get("profiles")
    _require(isinstance(entries, list) and entries, "Missing capture profiles")
    by_channel = {}
    profile_ids = set()
    for entry in entries:
        _object(entry, "capture profile")
        channel = _integer(entry.get("physical_channel"), "physical_channel", 1)
        pid = _text(entry.get("profile_id"), "profile_id")
        _require(channel not in by_channel and pid not in profile_ids, "Repeated capture profile")
        by_channel[channel] = entry
        profile_ids.add(pid)
    _require(set(by_channel) == {row["physical_channel"] for row in bound}, "Export must cover every bound captured profile")
    producer_receipt = {"schema_version": "pilotproxy_capture_replay_producer_v1", "runtime_producer_sha256": sha256(paths["producer"]), "reference": _source_identity()}
    artifacts = {"producer.json": producer_receipt}
    producer_sha = hashlib.sha256(_encoded_json(producer_receipt)).hexdigest()
    profiles = {}
    for row in bound:
        channel = row["physical_channel"]
        calibration = row["fine_calibration"]
        profile = {
            "schema_version": PROFILE_SCHEMA,
            "profile_id": by_channel[channel]["profile_id"],
            "decision_version": fine_decision.FINE_DECISION_VERSION,
            "score_encoding": SCORE_ENCODING,
            "support_rule": SUPPORT_RULE,
            "physical_channel": channel,
            "receiver_profile_sha256": sha256(paths["receiver_profile"]),
            "weights_sha256": hashlib.sha256(weights).hexdigest(),
            "producer_sha256": producer_sha,
            "packing_contract_sha256": sha256(paths["packing_contract"]),
            "input_state_contract_sha256": sha256(paths["input_state_contract"]),
            "input_population_id": data["input_state_contract"]["input_population_id"],
            "num_input_streams": geometry.num_streams,
            "detector_window_samples": geometry.k,
            "windows_per_stream": geometry.windows_per_stream,
            "frame_size_samples": geometry.block_samples,
            "fine_bins": 256,
            **{key: calibration[key] for key in ("anchor_bin", "designated_half_width", "bulk_mask_words_hex", "cfar_rank")},
        }
        validate_score_profile(profile)
        profiles[channel] = profile
        artifacts[f"profile_CH{channel}.json"] = profile
    rows_by_channel = {channel: [] for channel in profiles}
    comparisons = []
    for block in range(nblocks):
        start = block * geometry.frames_per_block
        voltage = captures["voltage"][start:start + geometry.frames_per_block]
        payload = np.concatenate([frame.payload for frame in voltage])
        expected = capture_reference.expected_products(payload, bundle, weights, geometry=geometry)
        recorded_mask = captures["masks"][block].payload
        recorded_powers = np.frombuffer(captures["powers"][block].payload, dtype="<u8").reshape(geometry.num_freq, 3)
        provenance = {
            "capture_id": capture_id,
            "block_index": block,
            "fpga_seq_num": voltage[0].fpga_seq_num,
            "time_downsampling_fpga": voltage[0].time_downsampling_fpga,
            "voltage_frames": [frame.provenance() for frame in voltage],
            "mask_frame": captures["masks"][block].provenance(),
            "power_frame": captures["powers"][block].provenance(),
        }
        # Stable observational identity survives capture/file/trial relabeling.
        # Independent-unit labels still require the supplied sampling evidence.
        trial_identity = {
            "voltage_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
            "fpga_seq_num": voltage[0].fpga_seq_num,
            "time_downsampling_fpga": voltage[0].time_downsampling_fpga,
            "utc_reference": timing["utc_reference"],
            "frequency_ids": list(geometry.frequency_ids),
        }
        provenance["trial_id"] = "native-" + hashlib.sha256(_encoded_json(trial_identity)).hexdigest()
        for freq_index, freq in enumerate(geometry.frequency_ids):
            mask_ok = int(recorded_mask[freq_index]) == int(expected.masks[freq_index])
            powers_ok = np.array_equal(recorded_powers[freq_index], expected.powers[freq_index])
            comparison = {
                "block_index": block,
                "receiver_channel_id": freq,
                "recorded_mask": int(recorded_mask[freq_index]),
                "expected_mask": int(expected.masks[freq_index]),
                "recorded_coarse_powers": [str(int(v)) for v in recorded_powers[freq_index]],
                "expected_coarse_powers": [str(int(v)) for v in expected.powers[freq_index]],
                "mask_matches": mask_ok,
                "powers_match": bool(powers_ok),
            }
            comparisons.append(comparison)
            matched = next((row for row in bound if row["chord_channel_id"] == freq), None)
            if matched is None:
                continue
            decision = expected.decisions[freq_index]
            _require(decision is not None, "Missing fine reference decision")
            rows_by_channel[matched["physical_channel"]].append({
                **provenance,
                **comparison,
                "valid": bool(decision.valid),
                "n_bulk": int(decision.n_bulk),
                "rank_bin": int(decision.rank_bin),
                "limiting_bin": int(decision.limiting_bin),
                "required_multiplier_q16": None if decision.multiplier_q16 is None else str(decision.multiplier_q16),
                "bundle_multiplier_q16": str(matched["fine_calibration"]["cfar_multiplier_q16"]),
            })
    parity = all(row["mask_matches"] and row["powers_match"] for row in comparisons)
    invalid = sum(not row["valid"] for rows in rows_by_channel.values() for row in rows)
    # Authenticate declarations even when parity fails, but emit no calibrator input
    # that could conceal failed product agreement.
    for channel, profile in profiles.items():
        rows = rows_by_channel[channel]
        artifacts[f"scores_CH{channel}.json"] = {"schema_version": "pilotproxy_capture_fine_scores_v1", "profile_id": profile["profile_id"], "scores": rows}
        profile_sha = hashlib.sha256(_encoded_json(profile)).hexdigest()
        population = _population(by_channel[channel], profile, profile_sha, rows, path.parent, bindings)
        if population is not None and parity:
            artifacts[f"population_CH{channel}.json"] = population
    report = {
        "schema_version": "pilotproxy_capture_fine_replay_v1",
        "capture_id": capture_id,
        "status": "product_mismatch" if not parity else "invalid_support" if invalid else "matched",
        "product_agreement": parity,
        "num_blocks": nblocks,
        "num_frequencies": geometry.num_freq,
        "score_rows": sum(map(len, rows_by_channel.values())),
        "invalid_rows": invalid,
        "comparisons": comparisons,
        "input_files_sha256": bindings,
        "telescope_qualification": False,
        "interpretation": "Sampled replay checks declared contracts and recorded products. It does not verify hardware state, IID sampling, signal absence, full-rate validity, or telescope release acceptance.",
    }
    artifacts["replay.json"] = report
    # Recheck after parsing/computation so changed source files never seal a result.
    for name, digest in bindings.items():
        _require(sha256(name) == digest, f"Input changed during replay: {name}")
    _same(producer_receipt["reference"], _source_identity(), "reference changed during replay")
    return artifacts


def write_replay(plan_path, output_dir):
    """Write a new evidence directory, including invalid and mismatch outcomes."""
    output = Path(output_dir)
    _require(not output.exists(), "Replay output directory must not exist")
    artifacts = replay_capture(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        _write_json(output / name, value)
    _write_json(output / "manifest.json", {
        "schema_version": "pilotproxy_capture_fine_replay_manifest_v1",
        "plan_path": str(Path(plan_path).resolve()),
        "output_files_sha256": {name: sha256(output / name) for name in artifacts},
        "producer": _source_identity(),
    })
    return artifacts["replay.json"]


def validate_replay(output_dir):
    """Authenticate sources and recompute all exported rows and comparisons."""
    output = Path(output_dir)
    manifest = _read(output / "manifest.json")
    _same(manifest.get("schema_version"), "pilotproxy_capture_fine_replay_manifest_v1", "replay manifest schema")
    _same(manifest.get("producer"), _source_identity(), "replay reference environment")
    artifacts = replay_capture(manifest["plan_path"])
    digests = _object(manifest.get("output_files_sha256"), "replay output hashes")
    _require(set(digests) == set(artifacts), "Replay output inventory mismatch")
    for name, expected in artifacts.items():
        _digest(digests[name], "output digest")
        _require(sha256(output / name) == digests[name], f"Replay output identity mismatch: {name}")
        _require((output / name).read_bytes() == _encoded_json(expected), f"Replay result mismatch: {name}")
    return artifacts["replay.json"]
