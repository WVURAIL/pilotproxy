#!/usr/bin/env python3
"""Freeze a production inventory from completed archive evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_DATA_SHELF_SNR_DB,
    ARCHIVED_FINE_NULL_BULK_EXCEEDANCE_FRACTION,
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB,
    ARCHIVED_NORMALIZED_PILOT_EXCESS,
    ARCHIVED_PILOT_EXCESS_DB,
)


SCHEMA = "chime_archive_inventory_freeze_v1"
PENDING_RESOLUTION_SCHEMA = "chime_pending_archive_resolution_v1"
PILOT_FREQ_IDS = [
    506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675,
    690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844,
]
EVIDENCE_FRAME_FIELDS = (
    "frame_index", "p_target_u64", "p_ref_sum_u64",
    ARCHIVED_COARSE_POWER_RATIO, ARCHIVED_FINE_POWER_RATIO,
    "fine_cfar_location", "fine_cfar_scale", "fine_cfar_threshold",
    ARCHIVED_FINE_NULL_BULK_EXCEEDANCE_FRACTION, "fine_cfar_mode",
    "fine_detected_count", ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB,
    ARCHIVED_PILOT_EXCESS_DB, ARCHIVED_DATA_SHELF_SNR_DB,
    ARCHIVED_NORMALIZED_PILOT_EXCESS, "reject_mask", "valid",
    "baseband_power_linear", "frame_unit_index", "frame_in_unit",
)
EVIDENCE_UNIT_FIELDS = (
    "unit_keys", "unit_order", "source_event_keys", "unit_time0_ctime",
    "unit_time0_fpga", "unit_event_id", "unit_delta_time", "archive_version",
)
SURVEY_ARTIFACTS = {
    "enum_cache": "/enum_cache.json",
    "no_files_events": "/no_files_events.jsonl",
    "attempts": "/attempts.json",
    "incomplete_events": "/incomplete_events.txt",
    "surveyed_events": "/surveyed_events.txt",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _single_member(archive: zipfile.ZipFile, suffix: str) -> str:
    members = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(members) != 1:
        raise ValueError(
            f"expected one {suffix!r} member, found {len(members)}"
        )
    return members[0]


def _scalar(product: Mapping[str, Any], field: str) -> Any:
    try:
        value = np.asarray(product[field])
    except KeyError as exc:
        raise ValueError(f"product is missing {field!r}") from exc
    if value.size != 1:
        raise ValueError(f"product field {field!r} is not scalar")
    return value.reshape(()).item()


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        scope = row["scope"]
        event = row["event"]
        name = row["name"]
        freq_id = row["freq_id"]
    except KeyError as exc:
        raise ValueError(f"inventory row is missing {exc.args[0]!r}") from exc
    if not all(isinstance(value, str) and value for value in (scope, event, name)):
        raise ValueError("scope, event, and name must be non-empty strings")
    if isinstance(freq_id, bool) or not isinstance(freq_id, int):
        raise ValueError("freq_id must be an integer")
    return scope, event, name, freq_id


def _uri(row: Mapping[str, Any]) -> str:
    common_path = row.get("common_path")
    name = row.get("name")
    if not isinstance(common_path, str) or not common_path:
        raise ValueError("common_path must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    return f"{common_path.rstrip('/')}/{name.lstrip('/')}"


def _read_inventory(data: bytes) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise ValueError("inventory must end with a newline")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, int]] = set()
    uris: set[str] = set()
    for line_number, raw in enumerate(data.splitlines(keepends=True), 1):
        if not raw.strip():
            raise ValueError(f"inventory line {line_number} is empty")
        try:
            row = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"inventory line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"inventory line {line_number} is not an object")
        identity = _identity(row)
        uri = _uri(row)
        if identity in identities:
            raise ValueError(f"duplicate inventory identity at line {line_number}")
        if uri in uris:
            raise ValueError(f"duplicate inventory URI at line {line_number}")
        identities.add(identity)
        uris.add(uri)
        records.append({
            "line_number": line_number,
            "raw": raw,
            "row": row,
            "identity": identity,
            "uri": uri,
        })
    return records


def _read_products(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, dict[str, Any]], set[str], list[str], dict[str, Any]]:
    members = sorted(name for name in archive.namelist() if name.endswith(".npz"))
    product_by_uri: dict[str, dict[str, Any]] = {}
    zero_frame_uris: set[str] = set()
    schemas: set[str] = set()
    detector_versions: set[str] = set()
    kernel_sha256s: set[str] = set()
    nffts: set[int] = set()
    detector_windows: set[int] = set()
    stream_counts: set[int] = set()
    caps: set[int] = set()
    freq_ids: set[int] = set()
    total_frames = 0
    for member in members:
        with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as product:
            schema = str(_scalar(product, "schema_version"))
            detector_version = str(_scalar(product, "detector_version"))
            nfft = int(_scalar(product, "nfft"))
            detector_window = int(_scalar(product, "detector_window_samples"))
            stream_count = int(_scalar(product, "num_input_streams"))
            cap = int(_scalar(product, "max_chunks_per_file"))
            freq_id = int(_scalar(product, "freq_id"))
            unit_order = [
                str(value)
                for value in np.asarray(product["unit_order"]).reshape(-1).tolist()
            ]
            unit_keys = [
                str(value)
                for value in np.asarray(product["unit_keys"]).reshape(-1).tolist()
            ]
            if not unit_order or len(set(unit_order)) != len(unit_order):
                raise ValueError(f"{member}: unit_order is empty or duplicated")
            if len(set(unit_keys)) != len(unit_keys) or set(unit_keys) != set(
                unit_order
            ):
                raise ValueError(f"{member}: unit_keys do not match unit_order")
            unit_lengths = {
                field: int(np.asarray(product[field]).reshape(-1).size)
                for field in EVIDENCE_UNIT_FIELDS
            }
            if set(unit_lengths.values()) != {len(unit_order)}:
                raise ValueError(f"{member}: per-unit arrays are not aligned")

            frame_index = np.asarray(product["frame_index"], dtype=np.int64).reshape(-1)
            frame_count = int(frame_index.size)
            if not np.array_equal(
                frame_index, np.arange(frame_count, dtype=np.int64)
            ):
                raise ValueError(f"{member}: frame_index is not contiguous")
            frame_lengths = {
                field: int(np.asarray(product[field]).shape[0])
                for field in EVIDENCE_FRAME_FIELDS
            }
            if set(frame_lengths.values()) != {frame_count}:
                raise ValueError(f"{member}: per-frame arrays are not aligned")
            frame_units = np.asarray(
                product["frame_unit_index"], dtype=np.int64
            ).reshape(-1)
            frame_in_unit = np.asarray(
                product["frame_in_unit"], dtype=np.int64
            ).reshape(-1)
        if np.any(frame_units < 0) or np.any(frame_units >= len(unit_order)):
            raise ValueError(f"{member}: frame_unit_index is out of range")
        if np.any(frame_in_unit < 0):
            raise ValueError(f"{member}: frame_in_unit is negative")
        if np.any(np.diff(frame_units) < 0):
            raise ValueError(f"{member}: frame_unit_index is not ordered")
        for index in np.unique(frame_units):
            positions = frame_in_unit[frame_units == index]
            if not np.array_equal(
                positions, np.arange(positions.size, dtype=np.int64)
            ):
                raise ValueError(f"{member}: frame_in_unit is not contiguous")
        member_stem = Path(member).stem
        if not member_stem.isdigit() or int(member_stem) != freq_id:
            raise ValueError(f"{member}: filename and freq_id disagree")
        if freq_id in freq_ids:
            raise ValueError(f"duplicate product freq_id {freq_id}")
        kernel_tokens = [
            token.split("=", 1)[1]
            for token in detector_version.split()
            if token.startswith("kernel_sha256=")
        ]
        if (
            len(kernel_tokens) != 1
            or len(kernel_tokens[0]) != 64
            or any(value not in "0123456789abcdef" for value in kernel_tokens[0])
        ):
            raise ValueError(f"{member}: detector_version has no kernel digest")
        if cap != -1:
            raise ValueError(f"{member}: evidence product is capped")
        schemas.add(schema)
        detector_versions.add(detector_version)
        kernel_sha256s.add(kernel_tokens[0])
        nffts.add(nfft)
        detector_windows.add(detector_window)
        stream_counts.add(stream_count)
        caps.add(cap)
        freq_ids.add(freq_id)
        total_frames += frame_count
        used = {int(value) for value in np.unique(frame_units)}
        for index, uri in enumerate(unit_order):
            if uri in product_by_uri:
                raise ValueError(f"duplicate product unit {uri!r}")
            product_by_uri[uri] = {"member": member, "freq_id": freq_id}
            if index not in used:
                zero_frame_uris.add(uri)
    summary = {
        "schema_versions": sorted(schemas),
        "detector_versions": sorted(detector_versions),
        "detector_version_sha256": sorted(
            _sha256(value.encode("utf-8")) for value in detector_versions
        ),
        "kernel_sha256": sorted(kernel_sha256s),
        "nfft": sorted(nffts),
        "detector_window_samples": sorted(detector_windows),
        "num_input_streams": sorted(stream_counts),
        "max_chunks_per_file": sorted(caps),
        "freq_ids": sorted(freq_ids),
        "frames": total_frames,
        "units": len(product_by_uri),
    }
    return product_by_uri, zero_frame_uris, members, summary


def _read_source_survey(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    artifacts: dict[str, bytes] = {}
    members: dict[str, str] = {}
    for key, suffix in SURVEY_ARTIFACTS.items():
        member = _single_member(archive, suffix)
        members[key] = member
        artifacts[key] = archive.read(member)

    try:
        enum_cache = json.loads(artifacts["enum_cache"])
        attempts = json.loads(artifacts["attempts"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source survey JSON: {exc}") from exc
    if not isinstance(enum_cache, dict) or any(
        not isinstance(key, str) or not key or "|" not in key
        for key in enum_cache
    ):
        raise ValueError("source enum cache is not an object")
    if not isinstance(attempts, dict):
        raise ValueError("source attempts record is not an object")
    if not all(
        isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        for key, value in attempts.items()
    ):
        raise ValueError("source attempts record is invalid")

    no_files = []
    for line_number, raw in enumerate(
        artifacts["no_files_events"].splitlines(), 1
    ):
        if not raw.strip():
            raise ValueError(f"no-files line {line_number} is empty")
        try:
            row = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"no-files line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"no-files line {line_number} is not an object")
        no_files.append(row)

    surveyed = [
        line.decode("utf-8")
        for line in artifacts["surveyed_events"].splitlines()
        if line.strip()
    ]
    incomplete = [
        line.decode("utf-8")
        for line in artifacts["incomplete_events"].splitlines()
        if line.strip()
    ]
    if len(set(surveyed)) != len(surveyed):
        raise ValueError("source surveyed events are duplicated")
    if incomplete:
        raise ValueError("source survey contains incomplete events")
    surveyed_keys = set(surveyed)
    attempt_keys = set(attempts)
    blocked_keys = {
        key
        for key, labels in enum_cache.items()
        if isinstance(labels, list)
        and any("outrigger" in str(label).lower() for label in labels)
    }
    eligible_keys = set(enum_cache) - blocked_keys
    if surveyed_keys & attempt_keys:
        raise ValueError("surveyed and pending event sets overlap")
    if eligible_keys != surveyed_keys | attempt_keys:
        raise ValueError("source survey event accounting does not close")

    summary = {
        "members": {
            key: {
                "name": members[key],
                "bytes": len(data),
                "sha256": _sha256(data),
            }
            for key, data in sorted(artifacts.items())
        },
        "enum_cache_entries": len(enum_cache),
        "no_files_events": len(no_files),
        "attempt_entries": len(attempts),
        "attempt_total": sum(attempts.values()),
        "incomplete_events": len(incomplete),
        "outrigger_excluded_events": len(blocked_keys),
        "eligible_events": len(eligible_keys),
        "pending_events": len(attempt_keys),
        "pending_event_keys": sorted(attempt_keys),
        "surveyed_events": len(surveyed),
    }
    return artifacts, summary


def _read_pending_resolution(
    data: bytes,
    pending_keys: set[str],
    selection: list[int],
) -> dict[str, Any]:
    try:
        resolution = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pending-event resolution: {exc}") from exc
    if not isinstance(resolution, dict):
        raise ValueError("pending-event resolution is not an object")
    if resolution.get("schema") != PENDING_RESOLUTION_SCHEMA:
        raise ValueError("pending-event resolution schema is unsupported")
    if resolution.get("selection") != selection:
        raise ValueError("pending-event resolution selection does not match")
    if resolution.get("method") != "authenticated_archive_metadata":
        raise ValueError("pending-event resolution method is unsupported")
    if resolution.get("reader_minimum_bytes") != 1048576:
        raise ValueError("pending-event resolution reader floor does not match")
    checked_at = resolution.get("checked_at")
    if not isinstance(checked_at, str) or not checked_at.endswith("Z"):
        raise ValueError("pending-event resolution has no UTC check time")
    events = resolution.get("events")
    if not isinstance(events, list):
        raise ValueError("pending-event resolution has no event list")
    resolved_keys: set[str] = set()
    for row in events:
        if not isinstance(row, dict):
            raise ValueError("pending-event resolution row is not an object")
        scope = row.get("scope")
        event = row.get("event")
        if not isinstance(scope, str) or not isinstance(event, str):
            raise ValueError("pending-event resolution row has no identity")
        key = f"{scope}|{event}"
        if key in resolved_keys:
            raise ValueError("pending-event resolution identity is duplicated")
        resolved_keys.add(key)
        if row.get("status") != "no_selected_files":
            raise ValueError(f"pending event {key} is not resolved as empty")
        common_path = row.get("common_path")
        if not isinstance(common_path, str) or not common_path.startswith(
            "cadc:CHIMEFRB/"
        ):
            raise ValueError(f"pending event {key} has no archive path")
        if row.get("absent_freq_ids") != selection:
            raise ValueError(f"pending event {key} has unresolved frequencies")
        for field in ("usable", "subfloor", "errors"):
            if row.get(field) != []:
                raise ValueError(f"pending event {key} has nonempty {field}")
    if resolved_keys != pending_keys:
        raise ValueError("pending-event resolution identities do not match")
    return resolution


def _read_quarantine(data: bytes) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"quarantine line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise ValueError(f"quarantine line {line_number} has no string key")
        uri = row["key"]
        if uri in rows:
            raise ValueError(f"duplicate quarantine key {uri!r}")
        rows[uri] = row
    return rows


def _write_or_verify(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink output {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"existing output differs: {path}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _copy_or_verify(source: Path, path: Path, expected_sha256: str) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink output {path}")
    if path.exists():
        if (
            path.stat().st_size != source.stat().st_size
            or _file_sha256(path) != expected_sha256
        ):
            raise ValueError(f"existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with source.open("rb") as source_stream, os.fdopen(fd, "wb") as stream:
            shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path = Path(temporary)
        if (
            temporary_path.stat().st_size != source.stat().st_size
            or _file_sha256(temporary_path) != expected_sha256
        ):
            raise ValueError(f"copied output differs: {path}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.stat().st_size != source.stat().st_size
                or _file_sha256(path) != expected_sha256
            ):
                raise ValueError(f"existing output differs: {path}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _assert_expected(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if value is not None and actual.get(key) != value:
            raise ValueError(
                f"expected {key}={value!r}, got {actual.get(key)!r}"
            )


def freeze_inventory(
    inventory_archive: Path,
    product_archive: Path,
    output_dir: Path,
    *,
    pending_resolution: Path | None = None,
    name: str = "chime-pilots-v5",
    selection: list[int] | None = None,
    expected: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
    invocation: list[str] | None = None,
) -> dict[str, Any]:
    inventory_archive = inventory_archive.resolve()
    product_archive = product_archive.resolve()
    output_dir = output_dir.resolve()
    selection = list(PILOT_FREQ_IDS if selection is None else selection)
    pending_resolution = (
        pending_resolution.resolve() if pending_resolution is not None else None
    )
    inventory_archive_sha256 = _file_sha256(inventory_archive)
    product_archive_sha256 = _file_sha256(product_archive)

    with zipfile.ZipFile(inventory_archive) as source_zip:
        inventory_member = _single_member(source_zip, "/inventory.jsonl")
        metadata_member = _single_member(source_zip, "/inventory.meta.json")
        inventory_data = source_zip.read(inventory_member)
        source_metadata_data = source_zip.read(metadata_member)
        survey_artifacts, survey_summary = _read_source_survey(source_zip)
    with zipfile.ZipFile(product_archive) as product_zip:
        quarantine_member = _single_member(product_zip, "/quarantine.jsonl")
        quarantine_data = product_zip.read(quarantine_member)
        (
            product_by_uri,
            zero_frame_uris,
            product_members,
            product_summary,
        ) = _read_products(product_zip)

    pending_keys = set(survey_summary["pending_event_keys"])
    pending_resolution_data: bytes | None = None
    pending_resolution_record: dict[str, Any] | None = None
    if pending_keys:
        if pending_resolution is None:
            raise ValueError(
                f"source survey has {len(pending_keys)} unresolved event(s)"
            )
        pending_resolution_data = pending_resolution.read_bytes()
        pending_resolution_record = _read_pending_resolution(
            pending_resolution_data,
            pending_keys,
            selection,
        )
    elif pending_resolution is not None:
        raise ValueError("pending-event resolution supplied for a complete survey")

    single_product_fields = (
        "schema_versions",
        "kernel_sha256",
        "nfft",
        "detector_window_samples",
        "num_input_streams",
        "max_chunks_per_file",
    )
    for field in single_product_fields:
        if len(product_summary[field]) != 1:
            raise ValueError(f"evidence products disagree on {field}")

    records = _read_inventory(inventory_data)
    inventory_by_uri = {record["uri"]: record for record in records}
    quarantine_by_uri = _read_quarantine(quarantine_data)
    inventory_uris = set(inventory_by_uri)
    product_uris = set(product_by_uri)
    quarantine_uris = set(quarantine_by_uri)

    if product_uris & quarantine_uris:
        raise ValueError("product and quarantine unit sets overlap")
    missing_evidence = inventory_uris - product_uris - quarantine_uris
    extra_evidence = (product_uris | quarantine_uris) - inventory_uris
    if missing_evidence or extra_evidence:
        raise ValueError(
            "inventory evidence does not close: "
            f"missing={len(missing_evidence)}, extra={len(extra_evidence)}"
        )
    if not zero_frame_uris <= product_uris:
        raise ValueError("zero-frame unit is absent from product evidence")
    for uri, evidence in product_by_uri.items():
        row_freq_id = inventory_by_uri[uri]["identity"][3]
        if row_freq_id != evidence["freq_id"]:
            raise ValueError(f"product freq_id does not match inventory for {uri}")

    excluded_uris = zero_frame_uris | quarantine_uris
    filtered_data = b"".join(
        record["raw"] for record in records if record["uri"] not in excluded_uris
    )
    frozen_records = [
        record for record in records if record["uri"] not in excluded_uris
    ]

    ledger_rows = []
    for uri in sorted(excluded_uris, key=lambda value: inventory_by_uri[value]["identity"]):
        record = inventory_by_uri[uri]
        row = record["row"]
        reasons = []
        evidence: dict[str, Any] = {}
        if uri in zero_frame_uris:
            reasons.append("prior_product_zero_frames")
            evidence["product_member"] = product_by_uri[uri]["member"]
        if uri in quarantine_uris:
            reasons.append("historical_quarantine")
            historical = quarantine_by_uri[uri]
            evidence["historical_quarantine_key"] = historical.get(
                "quarantine_key"
            )
            evidence["historical_quarantine_reason"] = historical.get("reason")
        estimate = row.get("n_frames_estimate", row.get("n_frames"))
        below_one = (
            isinstance(estimate, (int, float))
            and not isinstance(estimate, bool)
            and float(estimate) < 1.0
        )
        ledger_rows.append({
            "event": record["identity"][1],
            "freq_id": record["identity"][3],
            "name": record["identity"][2],
            "scope": record["identity"][0],
            "source_line": record["line_number"],
            "source_uri": uri,
            "size_bytes": row.get("size_bytes"),
            "n_frames_estimate": estimate,
            "below_one_frame_estimate": below_one,
            "reasons": reasons,
            "evidence": evidence,
        })
    ledger_data = b"".join(_json_bytes(row) for row in ledger_rows)

    identity_data = b"".join(
        _json_bytes(list(record["identity"]))
        for record in sorted(
            (inventory_by_uri[uri] for uri in excluded_uris),
            key=lambda record: record["identity"],
        )
    )
    source_events = {record["identity"][1] for record in records}
    frozen_events = {record["identity"][1] for record in frozen_records}
    source_freq_ids = sorted({record["identity"][3] for record in records})
    frozen_freq_ids = sorted({record["identity"][3] for record in frozen_records})
    if source_freq_ids != selection:
        raise ValueError(
            f"source freq_ids do not match the approved selection: {source_freq_ids}"
        )
    if frozen_freq_ids != selection:
        raise ValueError(
            f"frozen freq_ids do not match the approved selection: {frozen_freq_ids}"
        )
    if product_summary["freq_ids"] != selection:
        raise ValueError(
            "product freq_ids do not match the approved selection: "
            f"{product_summary['freq_ids']}"
        )
    low_estimate_units = sum(
        bool(row["below_one_frame_estimate"]) for row in ledger_rows
    )
    excluded_source_bytes = sum(
        int(record["row"].get("size_bytes", 0))
        for record in (inventory_by_uri[uri] for uri in excluded_uris)
    )

    counts = {
        "source_units": len(records),
        "source_events": len(source_events),
        "product_units": len(product_uris),
        "product_members": len(product_members),
        "zero_frame_units": len(zero_frame_uris),
        "quarantine_units": len(quarantine_uris),
        "low_estimate_exclusions": low_estimate_units,
        "excluded_units": len(excluded_uris),
        "affected_events": len({
            inventory_by_uri[uri]["identity"][1] for uri in excluded_uris
        }),
        "fully_excluded_events": len(source_events - frozen_events),
        "frozen_units": len(frozen_records),
        "frozen_events": len(frozen_events),
        "excluded_source_bytes": excluded_source_bytes,
        "source_pending_events": len(pending_keys),
        "resolved_pending_events": len(
            (pending_resolution_record or {}).get("events", [])
        ),
    }
    actual = {
        **counts,
        "source_freq_ids": source_freq_ids,
        "frozen_freq_ids": frozen_freq_ids,
        "frozen_sha256": _sha256(filtered_data),
        "frozen_bytes": len(filtered_data),
        "exclusion_identity_sha256": _sha256(identity_data),
        "inventory_archive_sha256": inventory_archive_sha256,
        "product_archive_sha256": product_archive_sha256,
        "source_inventory_sha256": _sha256(inventory_data),
        "source_metadata_sha256": _sha256(source_metadata_data),
        "source_quarantine_sha256": _sha256(quarantine_data),
        "product_schema_version": product_summary["schema_versions"][0],
        "product_detector_versions": len(product_summary["detector_versions"]),
        "product_detector_version_sha256": product_summary[
            "detector_version_sha256"
        ],
        "product_kernel_sha256": product_summary["kernel_sha256"][0],
        "product_nfft": product_summary["nfft"][0],
        "product_detector_window_samples": product_summary[
            "detector_window_samples"
        ][0],
        "product_num_input_streams": product_summary["num_input_streams"][0],
        "product_max_chunks_per_file": product_summary[
            "max_chunks_per_file"
        ][0],
        "product_frames": product_summary["frames"],
        "product_freq_ids": product_summary["freq_ids"],
        "source_enum_cache_entries": survey_summary["enum_cache_entries"],
        "source_no_files_events": survey_summary["no_files_events"],
        "source_attempt_entries": survey_summary["attempt_entries"],
        "source_attempt_total": survey_summary["attempt_total"],
        "source_incomplete_events": survey_summary["incomplete_events"],
        "source_surveyed_events": survey_summary["surveyed_events"],
        "source_eligible_events": survey_summary["eligible_events"],
        "source_outrigger_excluded_events": survey_summary[
            "outrigger_excluded_events"
        ],
        "source_pending_events": survey_summary["pending_events"],
        "effective_pending_events": 0,
    }
    if pending_resolution_data is not None:
        actual["pending_resolution_sha256"] = _sha256(pending_resolution_data)
    for key, member in survey_summary["members"].items():
        actual[f"source_{key}_sha256"] = member["sha256"]
    _assert_expected(actual, expected or {})

    try:
        source_metadata = json.loads(source_metadata_data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source inventory metadata: {exc}") from exc
    if not isinstance(source_metadata, dict):
        raise ValueError("source inventory metadata is not an object")
    production_metadata = dict(source_metadata)
    production_metadata["name"] = name
    production_metadata["derived_inventory"] = {
        "schema": SCHEMA,
        "source_inventory_sha256": _sha256(inventory_data),
        "exclusion_ledger_sha256": _sha256(ledger_data),
        "source_units": counts["source_units"],
        "excluded_units": counts["excluded_units"],
        "retained_units": counts["frozen_units"],
    }
    production_metadata_data = _json_bytes(production_metadata, pretty=True)

    output_payloads = {
        "attempts.source.json": survey_artifacts["attempts"],
        "enum_cache.source.json": survey_artifacts["enum_cache"],
        "incomplete_events.source.txt": survey_artifacts["incomplete_events"],
        "inventory.source.jsonl": inventory_data,
        "inventory.source.meta.json": source_metadata_data,
        "no_files_events.source.jsonl": survey_artifacts["no_files_events"],
        "quarantine.source.jsonl": quarantine_data,
        "surveyed_events.source.txt": survey_artifacts["surveyed_events"],
        "inventory.jsonl": filtered_data,
        "inventory.meta.json": production_metadata_data,
        "exclusions.jsonl": ledger_data,
    }
    if pending_resolution_data is not None:
        output_payloads["pending_resolution.json"] = pending_resolution_data
    archive_outputs = {
        "inventory.source.zip": {
            "bytes": inventory_archive.stat().st_size,
            "sha256": inventory_archive_sha256,
        },
        "products.source.zip": {
            "bytes": product_archive.stat().st_size,
            "sha256": product_archive_sha256,
        },
    }
    locked_assertions = {
        key: value
        for key, value in sorted((expected or {}).items())
        if value is not None
    }
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "selection": selection,
        "decisions": {
            "partial_run_acknowledgement": False,
            "terminal_product": "per_pilot_v5_products_for_selected_freq_ids",
            "terminal_product_count": len(selection),
            "combined_products": "derived_only",
            "evaluation_status": "historical_reprocessing_unblinded",
            "future_validation": "future_frozen_epoch",
            "historical_quarantine_exclusions": "approved_from_prior_direct_reads",
            "source_survey_completeness": (
                "complete_with_pending_event_resolution"
                if pending_keys
                else "complete"
            ),
        },
        "generator": {
            "script": {
                "name": script_path.name,
                "sha256": _file_sha256(script_path),
            },
            "source_revision": source_revision,
            "invocation": invocation,
            "assertions": locked_assertions,
        },
        "source": {
            "inventory_archive": {
                "name": inventory_archive.name,
                "bytes": inventory_archive.stat().st_size,
                "sha256": inventory_archive_sha256,
            },
            "product_archive": {
                "name": product_archive.name,
                "bytes": product_archive.stat().st_size,
                "sha256": product_archive_sha256,
            },
            "inventory_member": {
                "name": inventory_member,
                "bytes": len(inventory_data),
                "sha256": _sha256(inventory_data),
            },
            "metadata_member": {
                "name": metadata_member,
                "bytes": len(source_metadata_data),
                "sha256": _sha256(source_metadata_data),
            },
            "quarantine_member": {
                "name": quarantine_member,
                "bytes": len(quarantine_data),
                "sha256": _sha256(quarantine_data),
            },
            "survey": survey_summary,
            "product_evidence": product_summary,
        },
        "accounting": counts,
        "source_freq_ids": source_freq_ids,
        "frozen_freq_ids": frozen_freq_ids,
        "exclusion_identity_sha256": _sha256(identity_data),
        "outputs": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(output_payloads.items())
        } | archive_outputs,
    }
    if pending_resolution_data is not None and pending_resolution is not None:
        manifest["source"]["pending_resolution"] = {
            "name": pending_resolution.name,
            "bytes": len(pending_resolution_data),
            "sha256": _sha256(pending_resolution_data),
            "record": pending_resolution_record,
        }
    manifest_data = _json_bytes(manifest, pretty=True)
    for filename, data in output_payloads.items():
        _write_or_verify(output_dir / filename, data)
    _copy_or_verify(
        inventory_archive,
        output_dir / "inventory.source.zip",
        inventory_archive_sha256,
    )
    _copy_or_verify(
        product_archive,
        output_dir / "products.source.zip",
        product_archive_sha256,
    )
    _write_or_verify(output_dir / "inventory_manifest.json", manifest_data)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-archive", type=Path, required=True)
    parser.add_argument("--product-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pending-resolution", type=Path)
    parser.add_argument("--name", default="chime-pilots-v5")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--selection",
        default=",".join(str(value) for value in PILOT_FREQ_IDS),
    )
    parser.add_argument("--expect-source-units", type=int)
    parser.add_argument("--expect-source-events", type=int)
    parser.add_argument("--expect-product-units", type=int)
    parser.add_argument("--expect-product-members", type=int)
    parser.add_argument("--expect-product-frames", type=int)
    parser.add_argument("--expect-product-schema-version")
    parser.add_argument("--expect-product-detector-versions", type=int)
    parser.add_argument(
        "--expect-product-detector-version-sha256",
        action="append",
    )
    parser.add_argument("--expect-product-kernel-sha256")
    parser.add_argument("--expect-product-nfft", type=int)
    parser.add_argument("--expect-product-detector-window-samples", type=int)
    parser.add_argument("--expect-product-num-input-streams", type=int)
    parser.add_argument("--expect-product-max-chunks-per-file", type=int)
    parser.add_argument("--expect-zero-frame-units", type=int)
    parser.add_argument("--expect-quarantine-units", type=int)
    parser.add_argument("--expect-excluded-units", type=int)
    parser.add_argument("--expect-frozen-units", type=int)
    parser.add_argument("--expect-frozen-events", type=int)
    parser.add_argument("--expect-affected-events", type=int)
    parser.add_argument("--expect-fully-excluded-events", type=int)
    parser.add_argument("--expect-low-estimate-exclusions", type=int)
    parser.add_argument("--expect-frozen-bytes", type=int)
    parser.add_argument("--expect-frozen-sha256")
    parser.add_argument("--expect-exclusion-identity-sha256")
    parser.add_argument("--expect-inventory-archive-sha256")
    parser.add_argument("--expect-product-archive-sha256")
    parser.add_argument("--expect-source-inventory-sha256")
    parser.add_argument("--expect-source-metadata-sha256")
    parser.add_argument("--expect-source-quarantine-sha256")
    parser.add_argument("--expect-source-enum-cache-entries", type=int)
    parser.add_argument("--expect-source-no-files-events", type=int)
    parser.add_argument("--expect-source-attempt-entries", type=int)
    parser.add_argument("--expect-source-attempt-total", type=int)
    parser.add_argument("--expect-source-incomplete-events", type=int)
    parser.add_argument("--expect-source-surveyed-events", type=int)
    parser.add_argument("--expect-source-eligible-events", type=int)
    parser.add_argument("--expect-source-outrigger-excluded-events", type=int)
    parser.add_argument("--expect-source-pending-events", type=int)
    parser.add_argument("--expect-effective-pending-events", type=int)
    parser.add_argument("--expect-resolved-pending-events", type=int)
    parser.add_argument("--expect-source-enum-cache-sha256")
    parser.add_argument("--expect-source-no-files-events-sha256")
    parser.add_argument("--expect-source-attempts-sha256")
    parser.add_argument("--expect-source-incomplete-events-sha256")
    parser.add_argument("--expect-source-surveyed-events-sha256")
    parser.add_argument("--expect-pending-resolution-sha256")
    invocation_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(invocation_args)
    expected = {
        key.removeprefix("expect_"): value
        for key, value in vars(args).items()
        if key.startswith("expect_")
    }
    if expected.get("product_detector_version_sha256") is not None:
        expected["product_detector_version_sha256"] = sorted(
            expected["product_detector_version_sha256"]
        )
    manifest = freeze_inventory(
        args.inventory_archive,
        args.product_archive,
        args.output_dir,
        pending_resolution=args.pending_resolution,
        name=args.name,
        selection=[int(value) for value in args.selection.split(",")],
        expected=expected,
        source_revision=args.source_revision,
        invocation=[str(Path(__file__).resolve()), *invocation_args],
    )
    accounting = manifest["accounting"]
    frozen = manifest["outputs"]["inventory.jsonl"]
    print(
        f"frozen inventory: {accounting['frozen_units']} units, "
        f"{accounting['frozen_events']} events, sha256 {frozen['sha256']}"
    )
    print(
        f"exclusions: {accounting['excluded_units']} units, "
        f"{accounting['fully_excluded_events']} fully excluded events"
    )
    print(f"manifest: {args.output_dir.resolve() / 'inventory_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
