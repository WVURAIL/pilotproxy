#!/usr/bin/env python3
"""Verify final inventory and product accounting for a local archive run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pilot_proxy.archive.sources.cadc_inventory import logical_unit_key
from pilot_proxy.archive.chime_coarse import source_event_key
from pilot_proxy.atomic_io import atomic_write_json
from pilot_proxy.product_contract import (
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    validate_current_product_identity,
)


DEFAULT_FREQ_IDS = (
    506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675,
    690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844,
)
EXPECTED_PHYSICAL_CHANNEL_BY_FREQ_ID = {
    freq_id: physical_channel
    for freq_id, physical_channel in zip(DEFAULT_FREQ_IDS, range(36, 13, -1))
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMON_SCALAR_FIELDS = (
    "weight_bank_sha256",
    "weight_manifest_sha256",
    "detector_version",
    "decision_contract_json",
    "mask_rule",
    "sample_rate_hz",
    "sense",
    "fine_pad_factor",
    "fine_num_bins",
    "fine_p_fa",
    "fine_guard_fine_bins",
    "pilot_below_data_db",
    "bin_enbw_hz",
    "dtv_bandwidth_hz",
    "pilot_capture_efficiency",
)


class CloseoutError(RuntimeError):
    """Raised when final archive accounting does not match the run contract."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise CloseoutError(message)


def scalar(product: Mapping[str, Any], field: str) -> Any:
    try:
        value = np.asarray(product[field])
    except KeyError as exc:
        raise CloseoutError(f"product is missing {field}") from exc
    if value.size != 1:
        raise CloseoutError(f"product field {field} is not scalar")
    return value.reshape(()).item()


def canonical_json(value: Any, field: str) -> str:
    try:
        document = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise CloseoutError(f"product field {field} is not valid JSON") from exc
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def common_product_identity(product: Mapping[str, Any]) -> dict[str, Any]:
    identity = {field: scalar(product, field) for field in COMMON_SCALAR_FIELDS}
    for field in ("weight_bank_sha256", "weight_manifest_sha256"):
        require(
            isinstance(identity[field], str) and SHA256_RE.fullmatch(identity[field]),
            f"product field {field} is not a lowercase SHA-256",
        )
    identity["decision_contract_json"] = canonical_json(
        identity["decision_contract_json"], "decision_contract_json"
    )
    try:
        detector_contract = json.loads(str(scalar(product, "detector_contract_json")))
    except json.JSONDecodeError as exc:
        raise CloseoutError("product field detector_contract_json is not valid JSON") from exc
    require(isinstance(detector_contract, dict), "detector contract is not a JSON object")
    fine_reduction = detector_contract.get("fine_reduction")
    if isinstance(fine_reduction, dict):
        detector_contract = dict(detector_contract)
        detector_contract["fine_reduction"] = dict(fine_reduction)
        detector_contract["fine_reduction"].pop("designated_bins", None)
    identity["detector_contract_json"] = json.dumps(
        detector_contract, sort_keys=True, separators=(",", ":")
    )
    return identity


def require_common_identity(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    product_name: str,
) -> None:
    for field in (*COMMON_SCALAR_FIELDS, "detector_contract_json"):
        require(
            candidate.get(field) == reference.get(field),
            f"{product_name}: common product identity differs for {field}",
        )


def parse_inventory(
    path: Path,
    expected_freq_ids: Sequence[int],
) -> tuple[str, dict[int, list[str]]]:
    digest = hashlib.sha256()
    expected: dict[int, list[str]] = defaultdict(list)
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
                freq_id = int(row["freq_id"])
                unit_key = logical_unit_key(row["scope"], row["event"], row["name"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CloseoutError(f"invalid inventory row {line_number}: {exc}") from exc
            require(freq_id in expected_freq_ids, f"unexpected freq_id {freq_id} on row {line_number}")
            expected[freq_id].append(unit_key)
    return digest.hexdigest(), dict(expected)


def load_scope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloseoutError(f"cannot read scan scope: {exc}") from exc
    require(isinstance(value, dict), "scan scope is not a JSON object")
    return value


def check_scope(
    scope: Mapping[str, Any],
    *,
    inventory_path: Path,
    inventory_sha256: str,
    expected: Mapping[int, Sequence[str]],
    expected_freq_ids: Sequence[int],
    expected_units: int,
    staging_dir: Path,
    download_workers: int,
    max_staged_files: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    require(scope.get("schema_version") == "pilotproxy_chime_scan_scope_v1", "scan scope schema differs")
    require(scope.get("complete") is True, "scan scope is not complete")
    require(scope.get("source") == "cadc-datatrail", "scan source differs")
    input_record = scope.get("input")
    require(isinstance(input_record, dict), "scan input record is missing")
    require(input_record.get("inventory_path") == str(inventory_path), "scan inventory path differs")
    require(input_record.get("inventory_sha256") == inventory_sha256, "scan inventory hash differs")
    selections = [[freq_id] for freq_id in expected_freq_ids]
    require(scope.get("requested_selections") == selections, "requested selections differ")
    require(scope.get("allow_partial") is False, "partial processing was allowed")
    require(scope.get("max_files") is None, "file cap was active")
    require(scope.get("max_chunks_per_file") is None, "chunk cap was active")
    require(
        scope.get("fine_retention") == {"requested": "on", "resolved": "enabled"},
        "fine retention differs",
    )
    execution = {
        "preserve_source_order": True,
        "download_workers": download_workers,
        "max_staged_files": max_staged_files,
        "checkpoint_every": checkpoint_every,
        "staging_dir": str(staging_dir),
    }
    require(scope.get("execution") == execution, "execution settings differ")
    attempts = scope.get("execution_attempts")
    require(isinstance(attempts, list) and attempts, "execution attempts are missing")
    require(all(attempt == execution for attempt in attempts), "an execution attempt used different settings")

    totals = scope.get("totals")
    require(isinstance(totals, dict), "scan totals are missing")
    require(totals.get("pilots_requested") == len(expected_freq_ids), "requested pilot total differs")
    require(totals.get("requested") == expected_units, "requested unit total differs")
    require(totals.get("enumerated") == expected_units, "enumerated unit total differs")
    require(totals.get("completed") == expected_units, "completed unit total differs")
    for field in ("capped", "failed", "quarantined", "unprocessed", "extra_completed"):
        require(totals.get(field) == 0, f"scan total {field} is not zero")

    pilots = scope.get("pilots")
    require(isinstance(pilots, list) and len(pilots) == len(expected_freq_ids), "pilot records differ")
    for entry, freq_id in zip(pilots, expected_freq_ids):
        require(isinstance(entry, dict), f"pilot {freq_id} record is invalid")
        require(entry.get("selection") == [freq_id], f"pilot {freq_id} selection differs")
        require(entry.get("status") == "complete", f"pilot {freq_id} is not complete")
        count = len(expected[freq_id])
        require(entry.get("enumerated") == count, f"pilot {freq_id} enumerated total differs")
        require(entry.get("completed") == count, f"pilot {freq_id} completed total differs")
        for field in ("capped", "failed", "quarantined", "unprocessed", "extra_completed"):
            require(entry.get(field) == 0, f"pilot {freq_id} {field} is not zero")

    terminal = scope.get("terminal_combine")
    require(isinstance(terminal, dict), "terminal combine record is missing")
    require(terminal.get("status") in {"combined", "skipped"}, "terminal combine status differs")
    if terminal.get("status") == "skipped":
        require(
            terminal.get("error") == "CombineEmptyIntersectionError",
            "terminal combine skip reason differs",
        )
    return dict(terminal)


def check_empty_paths(run_dir: Path, staging_dir: Path) -> None:
    require(staging_dir.is_dir(), f"staging directory is missing: {staging_dir}")
    staged_files = [path for path in staging_dir.rglob("*") if path.is_file()]
    require(not staged_files, f"staging directory contains files: {staged_files[0] if staged_files else ''}")
    quarantine = run_dir / "_per_pilot" / "quarantine.jsonl"
    if quarantine.exists():
        require(
            not any(line.strip() for line in quarantine.read_text(encoding="utf-8").splitlines()),
            "quarantine is not empty",
        )


def check_products(
    run_dir: Path,
    *,
    expected: Mapping[int, Sequence[str]],
    expected_freq_ids: Sequence[int],
    expected_units: int,
    package_sha256: str,
    kernel_sha256: str,
    weight_bank_sha256: str,
    weight_manifest_sha256: str,
) -> dict[str, Any]:
    product_dir = run_dir / "_per_pilot"
    try:
        products = sorted(product_dir.glob("*.npz"), key=lambda path: int(path.stem))
    except ValueError as exc:
        raise CloseoutError("a per-pilot product has a nonnumeric name") from exc
    require([int(path.stem) for path in products] == list(expected_freq_ids), "per-pilot product set differs")

    product_units = 0
    product_frames = 0
    per_freq_units: dict[str, int] = {}
    receiver_configurations: dict[tuple[str, str, str], int] = {}
    source_scopes: dict[str, int] = {}
    common_identity: dict[str, Any] | None = None
    for path, freq_id in zip(products, expected_freq_ids):
        try:
            with np.load(path, allow_pickle=False) as product:
                validate_current_product_identity(product)
                product_identity = common_product_identity(product)
                require(
                    product_identity["weight_bank_sha256"]
                    == weight_bank_sha256,
                    f"{path.name}: weight-bank hash differs",
                )
                require(
                    product_identity["weight_manifest_sha256"]
                    == weight_manifest_sha256,
                    f"{path.name}: weight-manifest hash differs",
                )
                if common_identity is None:
                    common_identity = product_identity
                else:
                    require_common_identity(common_identity, product_identity, path.name)
                require(str(scalar(product, "schema_version")) == PER_PILOT_PRODUCT_SCHEMA_TOKEN, f"{path.name}: schema differs")
                require(int(scalar(product, "freq_id")) == freq_id, f"{path.name}: freq_id differs")
                physical_channel = int(scalar(product, "physical_channel"))
                require(
                    physical_channel == EXPECTED_PHYSICAL_CHANNEL_BY_FREQ_ID.get(freq_id),
                    f"{path.name}: physical channel does not match freq_id {freq_id}",
                )
                require(int(scalar(product, "nfft")) == 16384, f"{path.name}: nfft differs")
                require(int(scalar(product, "detector_window_samples")) == 128, f"{path.name}: detector window differs")
                require(int(scalar(product, "num_input_streams")) == 2048, f"{path.name}: stream count differs")
                sample_rate_hz = float(scalar(product, "sample_rate_hz"))
                require(
                    float(scalar(product, "chime_frequency_hz"))
                    == 800e6 - freq_id * sample_rate_hz,
                    f"{path.name}: CHIME frequency does not match freq_id and sample rate",
                )
                require(int(scalar(product, "max_chunks_per_file")) == -1, f"{path.name}: chunk cap differs")
                require(str(scalar(product, "fine_status")) == "enabled", f"{path.name}: fine retention differs")
                fine = np.asarray(product["fine_power_u64"])
                require(fine.dtype == np.uint64 and fine.ndim == 3 and fine.shape[1:] == (3, 256), f"{path.name}: fine-power shape or type differs")
                version = str(scalar(product, "detector_version"))
                version_tokens = set(version.split())
                require(f"source={package_sha256}" in version_tokens, f"{path.name}: package source token differs")
                require(f"kernel_sha256={kernel_sha256}" in version_tokens, f"{path.name}: kernel token differs")
                unit_order = np.asarray(product["unit_order"]).astype(str).tolist()
                unit_keys = np.asarray(product["unit_keys"]).astype(str).tolist()
                source_event_keys = np.asarray(product["source_event_keys"]).astype(str).tolist()
                scopes = np.asarray(product["unit_scope"]).astype(str).tolist()
                archive_versions = np.asarray(product["archive_version"]).astype(str).tolist()
                tags = np.asarray(product["unit_git_version_tag"]).astype(str).tolist()
                input_hashes = np.asarray(product["unit_input_map_sha256"]).astype(str).tolist()
                collection_servers = np.asarray(product["unit_collection_server"]).astype(str).tolist()
                require(len(source_event_keys) == len(scopes) == len(tags) == len(input_hashes) == len(archive_versions) == len(collection_servers) == len(unit_order), f"{path.name}: per-unit receiver fields differ")
                require(
                    source_event_keys
                    == [source_event_key(unit_key, freq_id) for unit_key in unit_order],
                    f"{path.name}: source-event keys are not derived from unit order",
                )
                require(all(archive_versions) and all(tags) and all(input_hashes) and all(collection_servers) and all(scopes), f"{path.name}: nonlocal receiver identity is empty")
                require(all(SHA256_RE.fullmatch(value) for value in input_hashes), f"{path.name}: input-map hash is invalid")
                for scope_name in scopes:
                    source_scopes[scope_name] = source_scopes.get(scope_name, 0) + 1
                for receiver_state in zip(archive_versions, tags, input_hashes):
                    receiver_configurations[receiver_state] = receiver_configurations.get(receiver_state, 0) + 1
                require(unit_order == list(expected[freq_id]), f"{path.name}: unit order differs")
                require(unit_keys == sorted(expected[freq_id]), f"{path.name}: unit keys differ")
                per_freq_units[str(freq_id)] = len(unit_order)
                product_units += len(unit_order)
                product_frames += int(np.asarray(product["frame_index"]).size)
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise CloseoutError(f"cannot audit {path.name}: {exc}") from exc
    require(product_units == expected_units, "per-pilot product unit total differs")
    return {
        "per_pilot_products": len(products),
        "product_units": product_units,
        "product_frames": product_frames,
        "per_freq_units": per_freq_units,
        "source_scopes": [
            {"scope": scope_name, "units": count}
            for scope_name, count in sorted(source_scopes.items())
        ],
        "receiver_configurations": [
            {
                "archive_version": key[0],
                "git_version_tag": key[1],
                "input_map_sha256": key[2],
                "units": count,
            }
            for key, count in sorted(receiver_configurations.items())
        ],
        "common_product_identity": common_identity,
    }


def audit_run(
    *,
    inventory_path: Path,
    run_dir: Path,
    staging_dir: Path,
    package_sha256: str,
    kernel_sha256: str,
    weight_bank_sha256: str,
    weight_manifest_sha256: str,
    expected_inventory_sha256: str,
    expected_units: int,
    expected_freq_ids: Sequence[int] = DEFAULT_FREQ_IDS,
    download_workers: int = 4,
    max_staged_files: int = 8,
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    staging_dir = staging_dir.resolve(strict=True)
    require(SHA256_RE.fullmatch(package_sha256), "package source hash is invalid")
    require(SHA256_RE.fullmatch(kernel_sha256), "kernel hash is invalid")
    require(SHA256_RE.fullmatch(weight_bank_sha256), "weight-bank hash is invalid")
    require(SHA256_RE.fullmatch(weight_manifest_sha256), "weight-manifest hash is invalid")
    require(SHA256_RE.fullmatch(expected_inventory_sha256), "expected inventory hash is invalid")
    require(expected_units > 0, "expected unit count must be positive")
    require(len(set(expected_freq_ids)) == len(expected_freq_ids), "expected freq_ids contain duplicates")

    inventory_sha256, expected = parse_inventory(inventory_path, expected_freq_ids)
    require(inventory_sha256 == expected_inventory_sha256, "inventory hash differs")
    require(sum(len(values) for values in expected.values()) == expected_units, "inventory unit total differs")
    require(sorted(expected) == sorted(expected_freq_ids), "inventory freq_id set differs")
    terminal = check_scope(
        load_scope(run_dir / "scan_scope.json"),
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha256,
        expected=expected,
        expected_freq_ids=expected_freq_ids,
        expected_units=expected_units,
        staging_dir=staging_dir,
        download_workers=download_workers,
        max_staged_files=max_staged_files,
        checkpoint_every=checkpoint_every,
    )
    check_empty_paths(run_dir, staging_dir)
    product_summary = check_products(
        run_dir,
        expected=expected,
        expected_freq_ids=expected_freq_ids,
        expected_units=expected_units,
        package_sha256=package_sha256,
        kernel_sha256=kernel_sha256,
        weight_bank_sha256=weight_bank_sha256,
        weight_manifest_sha256=weight_manifest_sha256,
    )
    return {
        "schema_version": "chime_local_archive_closeout_v1",
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "inventory_units": expected_units,
        **product_summary,
        "terminal_combine": terminal,
    }


def parse_freq_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(field) for field in value.split(",") if field)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("freq-ids must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("freq-ids may not be empty")
    return result


def parser() -> argparse.ArgumentParser:
    result_parser = argparse.ArgumentParser(description=__doc__)
    result_parser.add_argument("--inventory", type=Path, required=True)
    result_parser.add_argument("--run-dir", type=Path, required=True)
    result_parser.add_argument("--staging-dir", type=Path, required=True)
    result_parser.add_argument("--package-source-sha256", required=True)
    result_parser.add_argument("--kernel-sha256", required=True)
    result_parser.add_argument("--weight-bank-sha256", required=True)
    result_parser.add_argument("--weight-manifest-sha256", required=True)
    result_parser.add_argument("--expected-inventory-sha256", required=True)
    result_parser.add_argument("--expected-units", type=int, required=True)
    result_parser.add_argument("--freq-ids", type=parse_freq_ids, default=DEFAULT_FREQ_IDS)
    result_parser.add_argument("--download-workers", type=int, default=4)
    result_parser.add_argument("--max-staged-files", type=int, default=8)
    result_parser.add_argument("--checkpoint-every", type=int, default=250)
    result_parser.add_argument("--output-json", type=Path)
    return result_parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        report = audit_run(
            inventory_path=arguments.inventory,
            run_dir=arguments.run_dir,
            staging_dir=arguments.staging_dir,
            package_sha256=arguments.package_source_sha256,
            kernel_sha256=arguments.kernel_sha256,
            weight_bank_sha256=arguments.weight_bank_sha256,
            weight_manifest_sha256=arguments.weight_manifest_sha256,
            expected_inventory_sha256=arguments.expected_inventory_sha256,
            expected_units=arguments.expected_units,
            expected_freq_ids=arguments.freq_ids,
            download_workers=arguments.download_workers,
            max_staged_files=arguments.max_staged_files,
            checkpoint_every=arguments.checkpoint_every,
        )
        if arguments.output_json is not None:
            atomic_write_json(arguments.output_json.resolve(), report)
            print(f"audit report: {arguments.output_json.resolve()}")
        print("final inventory and product accounting passes")
        return 0
    except (CloseoutError, OSError) as exc:
        print(f"archive closeout failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
