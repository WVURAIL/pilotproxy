#!/usr/bin/env python3
"""Freeze completed estimator-transfer evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pilot_proxy.testbench import plot_results as transfer_plot  # noqa: E402
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH  # noqa: E402


SCHEMA_VERSION = "estimator_transfer_release_v1"
PROVENANCE_SCHEMA = "estimator_transfer_source_provenance_v1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_825
RUN_BASE_REVISION = "e2f9f7fa0d2013822274b2f34a3420d26ca8225b"
ARCHIVE_SNAPSHOT_COMMIT = "4317198f159db110a42d18befe73b8be2ffa48db"
ARCHIVE_TRACKED_TREE = "6551a441613b16196e724e739c380fd823becfff"
ARCHIVE_UNTRACKED_TREE = "78e0987f49f5c7031a36091c944d5c68823cef2d"
HISTORICAL_POST_UPDATE_REVISION = "c36bf672ce59b0828f43f0fffa41a43c4c9602a5"
PLOT_SOURCE_ARCHIVE_TAG = "estimator-transfer-source-20260825"
PLOT_SOURCE_ARCHIVE_COMMIT = "33f4b725301a9a793609870cd7c0c00877a3a5e8"
PLOT_SOURCE_ARCHIVE_BLOB = "e061a0cc373688cca6846a9565b4703bc67962c1"
POST_RUN_PLOT_SOURCE_ARCHIVE_TAG = "estimator-transfer-post-run-source-20260825"
POST_RUN_PLOT_SOURCE_ARCHIVE_COMMIT = (
    "3b42f879adf04cd8ac7071960948f296f3f801e8"
)
POST_RUN_PLOT_SOURCE_ARCHIVE_BLOB = "46c07acef30c6b1f6c907ea4cd07f9706186ca33"
PUBLICATION_EXPORT_ARCHIVE_TAG = (
    "estimator-transfer-publication-source-20260826"
)
RUN_SOURCE_ARCHIVE_TAG = "estimator-transfer-run-source-20260825"
RUN_SOURCE_ARCHIVE_COMMIT = "c216e497ca3239a63b728b049ee9174eb18c8913"
DIGITAL_RUN_SCRIPT_ARCHIVE_BLOB = "01fc5f4d5148b597cf95743717ff7ff43f70c250"
OTA_RUNNER_ARCHIVE_BLOB = "8fb258868b6ef28c2e9ac66831be197ace5db4f7"
STREAM_WORKER_ARCHIVE_BLOB = "a6756b4c2ee0b7a2a615e00e900dc35a89469e7f"
RUNNER_ARCHIVE_SHA256 = (
    "9190098cfcbe8d705a8f196f5e474676e71b1334fe1498fda628eda922cdf1f5"
)
WORKER_SOURCE_SHA256 = (
    "612a042609a8a9d16b9cecff1f10177e73548cd9daad41f4e88af38ef2e56db6"
)
ORIGINAL_DIGITAL_PLOT_SOURCE_SHA256 = (
    "17ab05a1232a544580214b470c9a22bf50069aa9dd7e1a4ec5491ae4cd9ed60b"
)
POST_RUN_PLOT_SOURCE_SHA256 = (
    "20fe2102d02eea32f276422fdcbbead06931fc86266d2472245be20fd7a514fd"
)
RECOVERED_RUNTIME_PACKAGE_SHA256 = (
    "e602959854082763a8504cfd6f7bdc862611d6e1651a5f6970b83a21b8fcc023"
)
PRE_ARCHIVE_RUN_SCRIPT_SHA256 = (
    "33508fc3a405d84e8d1a1dd659374116321cbe8569955d49a826e7a600428d27"
)
CAPTURE_FILE_COUNT = 1980
CAPTURE_TOTAL_BYTES = 7_151_522_400
CANONICAL_CAPTURE_INVENTORY_BYTES = 229_643
CANONICAL_CAPTURE_INVENTORY_SHA256 = (
    "38dc85507f433bc79205ec2b6fe8b674bcbd6ba750b2828aa88bfc7a06fe0356"
)
LEGACY_CONDITIONING_WEIGHT_BANK_SHA256 = (
    "1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259"
)
LEGACY_CONDITIONING_WEIGHT_MANIFEST_SHA256 = (
    "d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_export_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "repository_revision": revision,
        "repository_dirty": bool(status.strip()),
    }


def _publication_source_identity(relative_path: str) -> dict[str, str]:
    source_path = PROJECT_ROOT / relative_path
    tag_ref = f"refs/tags/{PUBLICATION_EXPORT_ARCHIVE_TAG}"

    def git_output(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    tag_object = git_output("rev-parse", tag_ref)
    if git_output("cat-file", "-t", tag_object) != "tag":
        raise ValueError(
            f"Publication source reference is not an annotated tag: {tag_ref}"
        )
    commit = git_output("rev-parse", f"{tag_ref}^{{commit}}")
    archived_blob = git_output("rev-parse", f"{tag_ref}:{relative_path}")
    current_blob = git_output("hash-object", str(source_path))
    if current_blob != archived_blob:
        raise ValueError(
            f"Publication source differs from {PUBLICATION_EXPORT_ARCHIVE_TAG}: "
            f"{relative_path}"
        )
    return {
        "path": relative_path,
        "sha256": file_sha256(source_path),
        "archival_tag": PUBLICATION_EXPORT_ARCHIVE_TAG,
        "archival_tag_object": tag_object,
        "archival_commit": commit,
        "archival_blob": archived_blob,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return value


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_text_lf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )


def _strict_event_index(event: Mapping[str, Any]) -> int:
    event_index = event.get("event_index")
    if type(event_index) is not int:
        raise ValueError("Event indices must be integers.")
    return event_index


def _event_identity(event: Mapping[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _planned_events_by_index(
    planned_events: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    planned: dict[int, Mapping[str, Any]] = {}
    for event in planned_events:
        if not isinstance(event, Mapping):
            raise ValueError("Run plan events must be objects.")
        event_index = _strict_event_index(event)
        if event_index in planned:
            raise ValueError(f"Run plan repeats event index {event_index}.")
        planned[event_index] = event
    return planned


def _canonical_event_records(
    path: Path,
    *,
    planned_events: Sequence[Mapping[str, Any]] | None = None,
    require_complete: bool = False,
) -> list[tuple[dict[str, Any], str]]:
    if require_complete and planned_events is None:
        raise ValueError("A complete event ledger requires a run plan.")
    planned = (
        None
        if planned_events is None
        else _planned_events_by_index(planned_events)
    )
    records: dict[int, tuple[dict[str, Any], str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            event = row["event"]
            if not isinstance(event, Mapping):
                raise TypeError
            event_index = _strict_event_index(event)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid event ledger row at line {line_number}."
            ) from error
        previous = records.get(event_index)
        if previous is not None and _event_identity(
            previous[0]["event"]
        ) != _event_identity(event):
            raise ValueError(
                f"Event index {event_index} has conflicting planned identities."
            )
        if planned is not None:
            planned_event = planned.get(event_index)
            if planned_event is None:
                raise ValueError(f"Event index {event_index} is not in the run plan.")
            if _event_identity(event) != _event_identity(planned_event):
                raise ValueError(
                    f"Event index {event_index} does not match the run plan."
                )
        records[event_index] = (row, line + "\n")
    if require_complete and set(records) != set(planned or {}):
        missing = sorted(set(planned or {}) - set(records))
        raise ValueError(f"Event ledger is missing planned indices: {missing}.")
    return [records[index] for index in sorted(records)]


def read_canonical_event_rows(
    path: Path,
    *,
    planned_events: Sequence[Mapping[str, Any]] | None = None,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    return [
        row
        for row, _line in _canonical_event_records(
            path,
            planned_events=planned_events,
            require_complete=require_complete,
        )
    ]


def write_canonical_event_ledger(
    records: Sequence[tuple[Mapping[str, Any], str]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(line for _row, line in records),
        encoding="utf-8",
    )


def copy_canonical_event_ledger(
    source: Path,
    destination: Path,
    *,
    planned_events: Sequence[Mapping[str, Any]] | None = None,
    require_complete: bool = False,
) -> None:
    write_canonical_event_ledger(
        _canonical_event_records(
            source,
            planned_events=planned_events,
            require_complete=require_complete,
        ),
        destination,
    )


def digital_shard_dirs(root: Path) -> list[Path]:
    root = Path(root)
    paths = [root / "extreme_low" / f"snr_m{abs(value)}" for value in range(-60, -44, 3)]
    paths.append(root / "lower")
    for value in range(-45, -29, 3):
        for part in (1, 2):
            paths.append(
                root
                / "lower_additional"
                / f"snr_m{abs(value)}"
                / f"part_{part}"
            )
    paths.extend((root / "upper", root / "high"))
    paths.extend(root / "high" / f"snr_p{value}" for value in range(6, 61, 3))
    return paths


def digital_expected_allocation() -> dict[float, int]:
    allocation: dict[float, int] = {}
    allocation.update({float(value): 240 for value in range(-60, -47, 3)})
    allocation.update({float(value): 1000 for value in range(-45, -29, 3)})
    allocation.update({float(value): 60 for value in range(-27, 61, 3)})
    return allocation


def source_file_inventory(
    root: Path,
    paths: Sequence[tuple[Path, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, role in sorted(paths, key=lambda item: item[0].relative_to(root).as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _trial_allocation(rows: Sequence[Mapping[str, float]]) -> dict[float, int]:
    counts = Counter(float(row[transfer_plot.REQUESTED_SNR_COLUMN]) for row in rows)
    return dict(sorted(counts.items()))


def _curve_map(
    summary_rows: list[dict[str, float]],
    trial_rows: list[dict[str, float]],
    calibration: Any,
    *,
    radio: bool,
    bootstrap_samples: int,
) -> dict[str, dict[float, dict[str, float]]]:
    curves: dict[str, dict[float, dict[str, float]]] = {}
    for prefix in ("gpu", "cpu_float", "cpu_packed"):
        points = transfer_plot._curve_points(
            summary_rows,
            trial_rows,
            prefix=prefix,
            offset=0.0,
            calibration=calibration,
            bootstrap_samples=bootstrap_samples,
            radio=radio,
        )
        curves[prefix] = {float(point["x"]): point for point in points}
    return curves


def _point_interval(point: Mapping[str, float], calibration: Any) -> tuple[float, float]:
    return (
        transfer_plot._excess_to_shelf_db(float(point["low_excess"]), calibration),
        transfer_plot._excess_to_shelf_db(float(point["high_excess"]), calibration),
    )


def _positive_counts(
    rows: Sequence[Mapping[str, float]],
    *,
    key: str,
) -> dict[float, int]:
    counts: dict[float, int] = defaultdict(int)
    for row in rows:
        ratio = transfer_plot._direct_trial_ratio(dict(row), "gpu")
        if math.isfinite(ratio) and ratio > 1.0:
            counts[float(row[key])] += 1
    return dict(counts)


def _rms(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _digital_points(
    summary_rows: list[dict[str, float]],
    trial_rows: list[dict[str, float]],
    summary_paths: Sequence[Path],
    conditioning_paths: Sequence[Path],
    *,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    radio, _input_calibration, output_calibration = transfer_plot._plot_calibrations(
        summary_rows, summary_paths
    )
    if radio:
        raise ValueError("Digital inputs were identified as radio results.")
    conditioning_rows = transfer_plot._read_trial_rows(conditioning_paths)
    conditioned = transfer_plot._try_conditioned_transfer(
        summary_paths=summary_paths,
        conditioning_paths=conditioning_paths,
        rows=conditioning_rows,
        calibration=output_calibration,
        conditioning_weights_path=DEFAULT_WEIGHTS_PATH,
        conditioning_weights_sha256=LEGACY_CONDITIONING_WEIGHT_BANK_SHA256,
        conditioning_weight_manifest_sha256=(
            LEGACY_CONDITIONING_WEIGHT_MANIFEST_SHA256
        ),
    )
    if conditioned is None:
        raise ValueError("Waveform conditioning was not derived.")
    curves = _curve_map(
        summary_rows,
        trial_rows,
        output_calibration,
        radio=False,
        bootstrap_samples=bootstrap_samples,
    )
    allocation = _trial_allocation(trial_rows)
    positive = _positive_counts(
        trial_rows,
        key=transfer_plot.REQUESTED_SNR_COLUMN,
    )
    requested = sorted(allocation)
    rows: list[dict[str, Any]] = []
    for snr in requested:
        gpu = curves["gpu"][snr]
        cpu_float = curves["cpu_float"][snr]
        cpu_packed = curves["cpu_packed"][snr]
        gpu_low, gpu_high = _point_interval(gpu, output_calibration)
        float_low, float_high = _point_interval(cpu_float, output_calibration)
        rows.append(
            {
                "requested_data_shelf_snr_db": snr,
                "ideal_local_reference_db": float(
                    transfer_plot._reference_transfer_db([snr])[0]
                ),
                "waveform_conditioned_expected_db": float(
                    transfer_plot._conditioned_transfer_db([snr], conditioned)[0]
                ),
                "gpu_fixed_db": float(gpu["y"]),
                "gpu_ci95_low_db": gpu_low,
                "gpu_ci95_high_db": gpu_high,
                "cpu_float_db": float(cpu_float["y"]),
                "cpu_float_ci95_low_db": float_low,
                "cpu_float_ci95_high_db": float_high,
                "cpu_packed_db": float(cpu_packed["y"]),
                "trials": allocation[snr],
                "positive_excess_trials": positive.get(snr, 0),
                "positive_excess_fraction": positive.get(snr, 0) / allocation[snr],
            }
        )

    parity = [
        abs(float(curves["gpu"][snr]["excess"]) - float(curves["cpu_packed"][snr]["excess"]))
        for snr in requested
    ]
    float_differences = [
        abs(float(curves["gpu"][snr]["y"]) - float(curves["cpu_float"][snr]["y"]))
        for snr in requested
        if math.isfinite(float(curves["gpu"][snr]["y"]))
        and math.isfinite(float(curves["cpu_float"][snr]["y"]))
    ]
    comparison_rows = [
        row
        for row in rows
        if -30.0 <= float(row["requested_data_shelf_snr_db"]) <= 0.0
        and math.isfinite(float(row["gpu_fixed_db"]))
    ]
    comparison_residuals = [
        float(row["gpu_fixed_db"]) - float(row["waveform_conditioned_expected_db"])
        for row in comparison_rows
    ]
    analysis = {
        "schema_version": "digital_estimator_transfer_analysis_v1",
        "experiment": "digital_synthetic",
        "scope": {
            "validated": "threshold-free coarse F-statistic normalization and shelf-SNR estimation transfer",
            "not_validated": [
                "decision threshold calibration",
                "false-alarm or detection probability",
                "ROC performance",
                "full 2048-stream deployment geometry",
            ],
        },
        "grid": {
            "minimum_db": min(requested),
            "maximum_db": max(requested),
            "step_db": 3.0,
            "points": len(requested),
            "trials": len(trial_rows),
            "allocation": {format(key, ".0f"): value for key, value in allocation.items()},
        },
        "representation": {
            "gpu_fixed_vs_cpu_packed_max_abs_normalized_ratio_difference": max(parity),
            "gpu_fixed_vs_cpu_float_max_abs_estimate_difference_db": max(float_differences),
            "gpu_finite_points": [
                snr for snr in requested if math.isfinite(float(curves["gpu"][snr]["y"]))
            ],
            "gpu_nonpositive_pooled_excess_points": [
                snr for snr in requested if not math.isfinite(float(curves["gpu"][snr]["y"]))
            ],
            "cpu_float_nonpositive_pooled_excess_points": [
                snr
                for snr in requested
                if not math.isfinite(float(curves["cpu_float"][snr]["y"]))
            ],
        },
        "waveform_conditioned_comparison": {
            "requested_range_db": [-30.0, 0.0],
            "points": len(comparison_rows),
            "rms_residual_db": _rms(comparison_residuals),
            "max_abs_residual_db": max(abs(value) for value in comparison_residuals),
        },
        "plot": {
            "bootstrap_samples": bootstrap_samples,
            "interval": "95% trial bootstrap",
            "smoothing_points": 1,
        },
    }
    qc = {
        "schema_version": "digital_estimator_transfer_qc_v1",
        "checks": {
            "expected_grid": requested == sorted(digital_expected_allocation()),
            "expected_allocation": allocation == digital_expected_allocation(),
            "expected_point_count": len(requested) == 41,
            "expected_trial_count": len(trial_rows) == 9000,
            "gpu_matches_packed_cpu": max(parity) == 0.0,
            "no_display_smoothing": True,
        },
        "geometry": {
            "num_input_streams": 4,
            "detector_window_samples": 128,
            "detector_rows_per_frame": 512,
            "frame_size_samples": 16384,
            "quantization_bits_per_component": 4,
            "noise_source": "gnuradio",
        },
    }
    if not all(qc["checks"].values()):
        raise ValueError("Digital release checks did not pass.")
    return rows, analysis, qc


def _fit_line(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - np.mean(y)) ** 2)
    return {
        "slope": float(slope),
        "intercept_db": float(intercept),
        "r_squared": float(1.0 - np.sum(residual * residual) / total),
        "fit_line_rms_residual_db": _rms(list(residual)),
        "fit_line_max_abs_residual_db": float(np.max(np.abs(residual))),
    }


def _pass_bootstrap_fit(
    summary_rows: list[dict[str, float]],
    trial_rows: list[dict[str, float]],
    calibration: Any,
    *,
    command_min_db: float,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    expected_by_command = {
        float(row["commanded_data_shelf_snr_db"]): float(
            row[transfer_plot.CONTROL_EXPECTED_COLUMN]
        )
        for row in summary_rows
        if float(row["commanded_data_shelf_snr_db"]) >= command_min_db
    }
    commands = sorted(expected_by_command)
    grouped: dict[float, dict[int, list[float]]] = {
        command: {} for command in commands
    }
    passes: set[int] = set()
    for row in trial_rows:
        command = float(row["commanded_data_shelf_snr_db"])
        if command not in grouped:
            continue
        pass_index = int(row["pass_index"])
        passes.add(pass_index)
        ratio = transfer_plot._direct_trial_ratio(row, "gpu")
        weight = transfer_plot._trial_weight(row, "gpu")
        values = grouped[command].setdefault(pass_index, [0.0, 0.0])
        values[0] += ratio * weight
        values[1] += weight
    pass_values = sorted(passes)
    if not pass_values:
        raise ValueError("No pass clusters were found.")
    if any(set(by_pass) != passes for by_pass in grouped.values()):
        raise ValueError("Pass clusters are incomplete.")
    rng = np.random.default_rng(seed)
    fits: list[tuple[float, float]] = []
    for _ in range(samples):
        selected = rng.choice(pass_values, size=len(pass_values), replace=True)
        x_values: list[float] = []
        y_values: list[float] = []
        for command in commands:
            numerator = math.fsum(grouped[command][int(value)][0] for value in selected)
            denominator = math.fsum(grouped[command][int(value)][1] for value in selected)
            estimate = transfer_plot._excess_to_shelf_db(
                numerator / denominator - 1.0,
                calibration,
            )
            if math.isfinite(estimate):
                x_values.append(expected_by_command[command])
                y_values.append(estimate)
        if len(x_values) >= 3:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            fits.append((float(slope), float(intercept)))
    values = np.asarray(fits, dtype=np.float64)
    if values.shape[0] != samples:
        raise ValueError("Some pass-bootstrap fits were not finite.")
    return {
        "slope_ci95": [
            float(np.quantile(values[:, 0], 0.025)),
            float(np.quantile(values[:, 0], 0.975)),
        ],
        "intercept_db_ci95": [
            float(np.quantile(values[:, 1], 0.025)),
            float(np.quantile(values[:, 1], 0.975)),
        ],
    }


def _ota_points(
    summary_rows: list[dict[str, float]],
    trial_rows: list[dict[str, float]],
    summary_path: Path,
    *,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    radio, _input_calibration, output_calibration = transfer_plot._plot_calibrations(
        summary_rows, [summary_path]
    )
    if not radio:
        raise ValueError("OTA inputs were not identified as radio results.")
    curves = _curve_map(
        summary_rows,
        trial_rows,
        output_calibration,
        radio=True,
        bootstrap_samples=bootstrap_samples,
    )
    gpu_by_x = curves["gpu"]
    summary_by_x = {
        float(row[transfer_plot.REQUESTED_SNR_COLUMN]): row for row in summary_rows
    }
    allocation = _trial_allocation(trial_rows)
    positive = _positive_counts(
        trial_rows,
        key=transfer_plot.REQUESTED_SNR_COLUMN,
    )
    rows: list[dict[str, Any]] = []
    for received_snr in sorted(summary_by_x):
        source = summary_by_x[received_snr]
        point = gpu_by_x[received_snr]
        low, high = _point_interval(point, output_calibration)
        expected = float(source[transfer_plot.CONTROL_EXPECTED_COLUMN])
        measured = float(point["y"])
        rows.append(
            {
                "commanded_data_shelf_snr_db": float(
                    source["commanded_data_shelf_snr_db"]
                ),
                "received_input_data_shelf_snr_db": received_snr,
                "ideal_local_reference_db": float(
                    transfer_plot._reference_transfer_db([received_snr])[0]
                ),
                "control_conditioned_expected_db": expected,
                "gpu_fixed_db": measured,
                "gpu_ci95_low_db": low,
                "gpu_ci95_high_db": high,
                "output_minus_received_db": measured - received_snr,
                "output_minus_control_expected_db": measured - expected,
                "trials": allocation[received_snr],
                "positive_excess_trials": positive.get(received_snr, 0),
                "positive_excess_fraction": positive.get(received_snr, 0)
                / allocation[received_snr],
            }
        )

    fit_rows = [row for row in rows if float(row["commanded_data_shelf_snr_db"]) >= -27.0]
    fit = _fit_line(
        [float(row["control_conditioned_expected_db"]) for row in fit_rows],
        [float(row["gpu_fixed_db"]) for row in fit_rows],
    )
    control_differences = [
        float(row["gpu_fixed_db"])
        - float(row["control_conditioned_expected_db"])
        for row in fit_rows
    ]
    fit.update(
        {
            "control_model_rms_difference_db": _rms(control_differences),
            "control_model_max_abs_difference_db": max(
                abs(value) for value in control_differences
            ),
        }
    )
    fit.update(
        _pass_bootstrap_fit(
            summary_rows,
            trial_rows,
            output_calibration,
            command_min_db=-27.0,
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )
    )
    analysis = {
        "schema_version": "sdr_ota_estimator_transfer_analysis_v1",
        "experiment": "sdr_ota",
        "scope": {
            "validated": "threshold-free coarse F-statistic shelf-SNR estimation through the direct radio path",
            "not_validated": [
                "decision threshold calibration",
                "false-alarm or detection probability",
                "ROC performance",
                "field propagation or multipath performance",
            ],
        },
        "grid": {
            "commanded_minimum_db": min(float(row["commanded_data_shelf_snr_db"]) for row in rows),
            "commanded_maximum_db": max(float(row["commanded_data_shelf_snr_db"]) for row in rows),
            "received_minimum_db": min(float(row["received_input_data_shelf_snr_db"]) for row in rows),
            "received_maximum_db": max(float(row["received_input_data_shelf_snr_db"]) for row in rows),
            "step_db": 3.0,
            "points": len(rows),
            "mixtures": len(trial_rows),
            "passes": len({int(row["pass_index"]) for row in trial_rows}),
        },
        "unit_gain_fit": {
            "predictor": "control-conditioned expected output",
            "response": "pooled GPU fixed-point estimate",
            "commanded_range_db": [-27.0, 0.0],
            "received_range_db": [
                min(float(row["received_input_data_shelf_snr_db"]) for row in fit_rows),
                max(float(row["received_input_data_shelf_snr_db"]) for row in fit_rows),
            ],
            "points": len(fit_rows),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "pass",
            **fit,
        },
        "floor": {
            "description": "The low-input plateau is the received local-reference estimator floor.",
            "approximate_output_db": min(float(row["gpu_fixed_db"]) for row in rows[:3]),
        },
        "plot": {
            "bootstrap_samples": bootstrap_samples,
            "interval": "95% pass-cluster bootstrap",
            "smoothing_points": 1,
        },
    }
    return rows, analysis, output_calibration


def _capture_inventory(
    root: Path,
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for result in sorted(event_rows, key=lambda row: int(row["event"]["event_index"])):
        event = result["event"]
        recorded = Path(str(result["capture_path"]))
        local_capture = root / "captures" / recorded.name
        if local_capture.is_file():
            capture = local_capture.resolve()
        else:
            candidate = recorded if recorded.is_absolute() else root / recorded
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            capture = candidate.resolve()
        try:
            relative_path = capture.relative_to(root)
        except ValueError as error:
            raise ValueError("Capture path is outside the result root.") from error
        rows.append(
            {
                "event_index": int(event["event_index"]),
                "pass_index": int(event["pass_index"]),
                "kind": str(event["kind"]),
                "commanded_data_shelf_snr_db": event.get("snr_db"),
                "relative_path": relative_path.as_posix(),
                "size_bytes": capture.stat().st_size,
                "sha256": file_sha256(capture),
            }
        )
    return rows


def _ota_qc(
    root: Path,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    capture_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_counts = Counter(str(row["event"]["kind"]) for row in event_rows)
    statuses = Counter(str(row["status"]) for row in event_rows)
    stream_status = [session["stream_status"] for session in sessions]
    counter_fields = sorted(
        key
        for key in stream_status[0]
        if key.endswith(("overrun", "underrun", "dropped_packets"))
    )
    counter_values = [int(status[key]) for status in stream_status for key in counter_fields]
    alignments = [item for session in sessions for item in session["alignment"]]
    requested_rate = float(plan["radio"]["sample_rate_hz"])
    rate_error_ppm = max(
        abs(float(status[key]) - requested_rate) / requested_rate * 1.0e6
        for status in stream_status
        for key in ("rx_host_rate_hz", "tx_host_rate_hz")
    )

    def capture_clip(rows: Sequence[Mapping[str, Any]]) -> float:
        return max(float(row["capture_levels"]["clip_fraction"]) for row in rows)

    def quantization_clip(kind: str | None = None) -> float:
        values = [
            float((row.get("detector") or {}).get("quantization_clip_fraction", 0.0))
            for row in event_rows
            if kind is None or row["event"]["kind"] == kind
        ]
        return max(values)

    pilot_ratio_errors = [
        abs(float(calibration["pilot_ratio_error_db"]))
        for row in event_rows
        if isinstance((calibration := row.get("received_control_calibration")), Mapping)
        and "pilot_ratio_error_db" in calibration
    ]
    no_level_errors = all(row.get("level_error") is None for row in event_rows)
    checks = {
        "all_events_complete": statuses == {"complete": 1980},
        "expected_event_counts": event_counts
        == {
            "mixture": 1800,
            "drift": 60,
            "tx_off": 30,
            "tx_zero": 30,
            "signal_only": 30,
            "noise_only": 30,
        },
        "expected_completed_events": int(state["completed_events"]) == 1980,
        "expected_pass_count": len(sessions) == 30,
        "expected_capture_count": len(capture_inventory) == CAPTURE_FILE_COUNT,
        "expected_capture_bytes": sum(
            int(row["size_bytes"]) for row in capture_inventory
        )
        == CAPTURE_TOTAL_BYTES,
        "stream_counters_zero": max(abs(value) for value in counter_values) == 0,
        "capture_levels_pass": no_level_errors,
        "capture_clip_pass": capture_clip(event_rows)
        <= float(plan["capture"]["clip_fraction_limit"]),
        "quantization_clip_pass": quantization_clip()
        <= float(plan["quantization"]["clip_fraction_limit"]),
        "received_pilot_ratio_pass": max(pilot_ratio_errors)
        <= float(plan["capture"]["max_received_pilot_ratio_error_db"]),
    }
    if not all(checks.values()):
        raise ValueError("OTA release checks did not pass.")
    return {
        "schema_version": "sdr_ota_estimator_transfer_qc_v1",
        "checks": checks,
        "event_counts": dict(sorted(event_counts.items())),
        "stream": {
            "counter_fields": counter_fields,
            "counter_values_checked": len(counter_values),
            "maximum_abs_counter": max(abs(value) for value in counter_values),
            "maximum_host_rate_error_ppm": rate_error_ppm,
        },
        "alignment": {
            "events": len(alignments),
            "minimum_marker_correlation": min(float(row["correlation"]) for row in alignments),
            "minimum_separation_error_samples": min(int(row["separation_error_samples"]) for row in alignments),
            "maximum_separation_error_samples": max(int(row["separation_error_samples"]) for row in alignments),
            "minimum_stream_offset_samples": min(int(row["stream_offset_samples"]) for row in alignments),
            "maximum_stream_offset_samples": max(int(row["stream_offset_samples"]) for row in alignments),
        },
        "levels": {
            "maximum_rf_clip_fraction": capture_clip(event_rows),
            "maximum_quantization_clip_fraction": quantization_clip(),
            "maximum_signal_only_quantization_clip_fraction": quantization_clip("signal_only"),
            "maximum_mixture_quantization_clip_fraction": quantization_clip("mixture"),
            "maximum_received_pilot_ratio_error_db": max(pilot_ratio_errors),
        },
        "raw_captures": {
            "files": len(capture_inventory),
            "bytes": sum(int(row["size_bytes"]) for row in capture_inventory),
        },
    }


def _source_provenance(experiment: str, result_root: Path) -> dict[str, Any]:
    source_files = {
        "publication_plot_source": _publication_source_identity(
            "src/pilot_proxy/testbench/plot_results.py"
        ),
        "freezer": _publication_source_identity(
            "tools/freeze_estimator_transfer.py"
        ),
    }
    run_sources: dict[str, Any]
    if experiment == "sdr_ota":
        run_sources = {
            "run_source_archival_tag": RUN_SOURCE_ARCHIVE_TAG,
            "run_source_archival_commit": RUN_SOURCE_ARCHIVE_COMMIT,
            "runner_archival_snapshot_blob": OTA_RUNNER_ARCHIVE_BLOB,
            "stream_worker_source_blob": STREAM_WORKER_ARCHIVE_BLOB,
            "stream_worker_source_sha256": WORKER_SOURCE_SHA256,
            "runner_archival_snapshot_sha256": RUNNER_ARCHIVE_SHA256,
            "runner_note": "The run plan pins the worker source and binary. The Python runner was archived after the run and was not hashed by the run plan.",
        }
    else:
        run_sources = {
            "run_source_archival_tag": RUN_SOURCE_ARCHIVE_TAG,
            "run_source_archival_commit": RUN_SOURCE_ARCHIVE_COMMIT,
            "run_script_archival_snapshot_blob": DIGITAL_RUN_SCRIPT_ARCHIVE_BLOB,
            "conditioning_record_plot_source_archival_tag": PLOT_SOURCE_ARCHIVE_TAG,
            "conditioning_record_plot_source_archival_commit": PLOT_SOURCE_ARCHIVE_COMMIT,
            "conditioning_record_plot_source_blob": PLOT_SOURCE_ARCHIVE_BLOB,
            "conditioning_record_plot_source_sha256": ORIGINAL_DIGITAL_PLOT_SOURCE_SHA256,
            "conditioning_record_note": "This hash is embedded in the completed conditioning record. It identifies the plotter that wrote that record, not every raw shard producer.",
            "run_script_archival_snapshot_sha256": PRE_ARCHIVE_RUN_SCRIPT_SHA256,
            "run_script_note": "The completed sweep was extended in stages. The later shell snapshot records the layout but was not itself pinned by every shard.",
        }
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "experiment": experiment,
        "result_root_label": result_root.name,
        "run_worktree": {
            "base_revision": RUN_BASE_REVISION,
            "state": "uncommitted tracked and untracked source",
            "archival_stash_object": ARCHIVE_SNAPSHOT_COMMIT,
            "archival_snapshot_tracked_tree": ARCHIVE_TRACKED_TREE,
            "archival_snapshot_untracked_tree": ARCHIVE_UNTRACKED_TREE,
            "post_run_plotter_archival_tag": POST_RUN_PLOT_SOURCE_ARCHIVE_TAG,
            "post_run_plotter_archival_commit": (
                POST_RUN_PLOT_SOURCE_ARCHIVE_COMMIT
            ),
            "post_run_plotter_archival_blob": POST_RUN_PLOT_SOURCE_ARCHIVE_BLOB,
            "post_run_plotter_archival_snapshot_sha256": POST_RUN_PLOT_SOURCE_SHA256,
            "recovered_runtime_package_digest_sha256": RECOVERED_RUNTIME_PACKAGE_SHA256,
            **run_sources,
        },
        "publication_export": {
            "historical_post_update_revision": HISTORICAL_POST_UPDATE_REVISION,
            **_repository_export_state(),
            "state": "post-run archival regeneration from frozen raw results",
            "pdf_rendering": {
                "text": "LaTeX T1 Latin Modern",
                "content": "vector",
                "byte_note": "The release hashes pin the delivered PDF. A rebuild can differ at the byte level because embedded-font subset identifiers vary, even when the vector content and extracted text match.",
            },
            "source_files": source_files,
        },
    }


def _artifact_role(path: str) -> str:
    if path == "README.md":
        return "documentation"
    if path.startswith("figures/"):
        return "publication_figure" if path.endswith(".pdf") else "figure_preview"
    if path.startswith("raw/"):
        return "external_raw_inventory"
    if path.startswith("run/"):
        return "run_provenance"
    if path.endswith("plot_points.csv"):
        return "plot_data"
    if path.endswith("analysis.json"):
        return "analysis"
    if path.endswith("qc.json"):
        return "quality_control"
    return "supporting_data"


def write_release_index(
    release_dir: Path,
    *,
    experiment: str,
    title: str,
    counts: Mapping[str, int],
) -> tuple[str, str]:
    release_dir = Path(release_dir)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file() or path.name in {"release_manifest.json", "SHA256SUMS"}:
            continue
        relative = path.relative_to(release_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": _artifact_role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_date": "2026-08-25",
        "experiment": experiment,
        "title": title,
        "path_semantics": "release_root_relative_posix",
        "artifact_count": len(artifacts),
        "counts": dict(counts),
        "artifacts": artifacts,
    }
    manifest_path = release_dir / "release_manifest.json"
    write_json(manifest_path, manifest)
    checksum_paths = [
        path
        for path in sorted(release_dir.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    checksum_text = "".join(
        f"{file_sha256(path)}  {path.relative_to(release_dir).as_posix()}\n"
        for path in checksum_paths
    )
    checksum_path = release_dir / "SHA256SUMS"
    checksum_path.write_text(checksum_text, encoding="utf-8")
    return file_sha256(manifest_path), file_sha256(checksum_path)


def _install(staging: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)


def freeze_digital(
    result_root: Path,
    destination: Path,
    *,
    bootstrap_samples: int,
) -> tuple[str, str]:
    result_root = Path(result_root).resolve()
    shards = digital_shard_dirs(result_root)
    summary_paths = [path / "dtv_snr_summary.csv" for path in shards]
    trial_paths = [path / "dtv_snr_eval.csv" for path in shards]
    metadata_paths = [path / "dtv_snr_eval.json" for path in shards]
    for path in summary_paths + trial_paths + metadata_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_rows = transfer_plot._read_summary_rows(summary_paths)
    trial_rows = transfer_plot._read_trial_rows(trial_paths)
    points, analysis, qc = _digital_points(
        summary_rows,
        trial_rows,
        summary_paths,
        [result_root / "lower" / "dtv_snr_eval.csv"],
        bootstrap_samples=bootstrap_samples,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        inventory_sources = [
            *((path, "summary_csv") for path in summary_paths),
            *((path, "trial_csv") for path in trial_paths),
            *((path, "metadata_json") for path in metadata_paths),
        ]
        inventory = source_file_inventory(result_root, inventory_sources)
        write_csv(
            staging / "raw" / "raw_input_inventory.csv",
            ("relative_path", "role", "size_bytes", "sha256"),
            inventory,
        )
        write_csv(
            staging / "data" / "plot_points.csv",
            (
                "requested_data_shelf_snr_db",
                "ideal_local_reference_db",
                "waveform_conditioned_expected_db",
                "gpu_fixed_db",
                "gpu_ci95_low_db",
                "gpu_ci95_high_db",
                "cpu_float_db",
                "cpu_float_ci95_low_db",
                "cpu_float_ci95_high_db",
                "cpu_packed_db",
                "trials",
                "positive_excess_trials",
                "positive_excess_fraction",
            ),
            points,
        )
        qc["raw_source_files"] = len(inventory)
        qc["raw_source_bytes"] = sum(int(row["size_bytes"]) for row in inventory)
        write_json(staging / "data" / "analysis.json", analysis)
        write_json(staging / "data" / "qc.json", qc)
        conditioning = result_root / "estimator_transfer_m60_to_p60_conditioning.json"
        conditioning_record = json.loads(conditioning.read_text(encoding="utf-8"))
        if (
            conditioning_record.get("plot_source_sha256")
            != ORIGINAL_DIGITAL_PLOT_SOURCE_SHA256
        ):
            raise ValueError("The digital conditioning plot-source hash changed.")
        copy_file(conditioning, staging / "run" / "conditioning.json")
        write_json(
            staging / "run" / "source_provenance.json",
            _source_provenance("digital_synthetic", result_root),
        )
        transfer_plot.plot_summary(
            input_csv=summary_paths,
            trial_csv=trial_paths,
            conditioning_trial_csv=[result_root / "lower" / "dtv_snr_eval.csv"],
            conditioning_weights_path=DEFAULT_WEIGHTS_PATH,
            conditioning_weights_sha256=LEGACY_CONDITIONING_WEIGHT_BANK_SHA256,
            conditioning_weight_manifest_sha256=(
                LEGACY_CONDITIONING_WEIGHT_MANIFEST_SHA256
            ),
            output_png=staging / "figures" / "fig_estimator_transfer_digital.png",
            output_pdf=staging / "figures" / "fig_estimator_transfer_digital.pdf",
            title="Synthetic GNU Radio input: CPU/GPU estimator transfer",
            smooth_window=1,
            bootstrap_samples=bootstrap_samples,
            y_min_db=-60.0,
            dissertation_style=True,
        )
        readme = f"""# Digital estimator transfer — 2026-08-25

This release freezes the threshold-free synthetic GNU Radio sweep from -60 dB to +60 dB in 3 dB steps. It compares the CPU float, packed CPU, and GPU fixed-point implementations with the ideal local-reference benchmark and the waveform-conditioned expected transfer.

The 41 points contain 9,000 noise trials. Points from -60 dB through -48 dB have 240 trials each, points from -45 dB through -30 dB have 1,000 trials each, and points from -27 dB through +60 dB have 60 trials each. Error bars are deterministic 95% trial-bootstrap intervals from {bootstrap_samples:,} resamples. No display smoothing is used.

The raw 40 result shards remain outside the repository. `raw/raw_input_inventory.csv` pins the 120 summary, trial, and metadata files used here. `data/plot_points.csv` is sufficient to reproduce the publication figure without the raw shards. `run/conditioning.json` preserves the conditioning coefficients and their original source hashes.

The exact plotter source that wrote the conditioning record is preserved by the annotated tag `{PLOT_SOURCE_ARCHIVE_TAG}`. The later archival plotter is preserved by `{POST_RUN_PLOT_SOURCE_ARCHIVE_TAG}`, the sweep launcher by `{RUN_SOURCE_ARCHIVE_TAG}`, and the publication exporter by `{PUBLICATION_EXPORT_ARCHIVE_TAG}`. Their hashes and references are recorded in `run/source_provenance.json`.

The publication PDF is a vector figure rendered through LaTeX with embedded T1 Latin Modern fonts. Its exact delivered bytes are pinned here. Rebuilds can differ at the byte level because embedded-font subset identifiers vary, even when the vector content and extracted text match.

This validates coarse F-statistic normalization and SNR estimation. It does not test a decision threshold, Pfa, Pd, an ROC curve, or the full 2048-stream deployment geometry.

Repeat this sweep only after changes to estimator normalization or math, detector geometry or weights, numeric representation, or waveform and noise synthesis. Threshold-policy changes alone do not invalidate this threshold-free estimation evidence.

Rebuild the raw sweep with:

    scripts/run_estimator_transfer.sh /path/to/new_results

Freeze completed digital and radio results with:

    PYTHONPATH=src python3 tools/freeze_estimator_transfer.py --digital-results /path/to/digital_results --ota-results /path/to/radio_results
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        manifest_hash, checksum_hash = write_release_index(
            staging,
            experiment="digital_synthetic",
            title="Synthetic GNU Radio estimator transfer",
            counts={
                "plot_points": 41,
                "raw_source_files": 120,
                "raw_shards": 40,
                "trials": 9000,
            },
        )
        _install(staging, destination)
        return manifest_hash, checksum_hash
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def freeze_ota(
    result_root: Path,
    destination: Path,
    *,
    bootstrap_samples: int,
) -> tuple[str, str]:
    result_root = Path(result_root).resolve()
    summary_path = result_root / "sdr_transfer_summary.csv"
    trial_path = result_root / "sdr_transfer_trials.csv"
    events_path = result_root / "events.jsonl"
    plan_path = result_root / "run_plan.json"
    state_path = result_root / "run_state.json"
    session_paths = sorted((result_root / "sessions").glob("pass_*.json"))
    for path in (summary_path, trial_path, events_path, plan_path, state_path, *session_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_rows = transfer_plot._read_summary_rows([summary_path])
    trial_rows = transfer_plot._read_trial_rows([trial_path])
    points, analysis, _output_calibration = _ota_points(
        summary_rows,
        trial_rows,
        summary_path,
        bootstrap_samples=bootstrap_samples,
    )
    plan_text = plan_path.read_text(encoding="utf-8")
    plan = json.loads(plan_text)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    event_records = _canonical_event_records(
        events_path,
        planned_events=plan["events"],
        require_complete=True,
    )
    event_rows = [row for row, _line in event_records]
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in session_paths]
    capture_inventory = _capture_inventory(result_root, event_rows)
    qc = _ota_qc(result_root, plan, state, event_rows, sessions, capture_inventory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        canonical_inventory = [
            {
                "path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in capture_inventory
        ]
        capture_inventory_path = staging / "raw" / "raw_capture_inventory.csv"
        write_csv(
            capture_inventory_path,
            ("path", "size_bytes", "sha256"),
            canonical_inventory,
        )
        canonical_size = capture_inventory_path.stat().st_size
        canonical_hash = file_sha256(capture_inventory_path)
        if canonical_size != CANONICAL_CAPTURE_INVENTORY_BYTES:
            raise ValueError("The canonical capture inventory byte count changed.")
        if canonical_hash != CANONICAL_CAPTURE_INVENTORY_SHA256:
            raise ValueError("The canonical capture inventory hash changed.")
        qc["raw_captures"]["canonical_inventory"] = {
            "path": "raw/raw_capture_inventory.csv",
            "projection": ["path", "size_bytes", "sha256"],
            "path_semantics": "source_result_root_relative_posix",
            "row_order": "path ascending",
            "line_ending": "LF",
            "size_bytes": canonical_size,
            "sha256": canonical_hash,
        }
        write_csv(
            staging / "data" / "plot_points.csv",
            (
                "commanded_data_shelf_snr_db",
                "received_input_data_shelf_snr_db",
                "ideal_local_reference_db",
                "control_conditioned_expected_db",
                "gpu_fixed_db",
                "gpu_ci95_low_db",
                "gpu_ci95_high_db",
                "output_minus_received_db",
                "output_minus_control_expected_db",
                "trials",
                "positive_excess_trials",
                "positive_excess_fraction",
            ),
            points,
        )
        write_json(staging / "data" / "analysis.json", analysis)
        write_json(staging / "data" / "qc.json", qc)
        copy_text_lf(
            summary_path,
            staging / "data" / "sdr_transfer_summary.csv",
        )
        copy_text_lf(
            trial_path,
            staging / "data" / "sdr_transfer_trials.csv",
        )
        released_plan_path = staging / "run" / "run_plan.json"
        released_plan_path.parent.mkdir(parents=True, exist_ok=True)
        released_plan_path.write_text(
            plan_text,
            encoding="utf-8",
        )
        copy_file(state_path, staging / "run" / "run_state.json")
        write_canonical_event_ledger(
            event_records,
            staging / "run" / "events.jsonl",
        )
        for path in session_paths:
            copy_file(path, staging / "run" / "sessions" / path.name)
        write_json(
            staging / "run" / "source_provenance.json",
            _source_provenance("sdr_ota", result_root),
        )
        transfer_plot.plot_summary(
            input_csv=[summary_path],
            trial_csv=[trial_path],
            output_png=staging / "figures" / "fig_estimator_transfer_ota.png",
            output_pdf=staging / "figures" / "fig_estimator_transfer_ota.pdf",
            title="LimeSDR over-the-air estimator transfer",
            smooth_window=1,
            bootstrap_samples=bootstrap_samples,
            dissertation_style=True,
        )
        readme = f"""# LimeSDR over-the-air estimator transfer — 2026-08-25

This release freezes the direct LimeSDR transmit/receive sweep with the two radio ports separated by 1.45 cm. The commanded data-shelf SNR grid is -42 dB through 0 dB in 3 dB steps. Per-pass tx-zero, signal-only, and noise-only controls calibrate the received input axis and the expected detector output.

The run contains 30 passes, 1,800 mixture captures, and 1,980 total events. Error bars are deterministic 95% pass-cluster bootstrap intervals from {bootstrap_samples:,} resamples. No display smoothing is used. The unit-gain fit uses the control-conditioned expected output as its predictor over commanded -27 dB through 0 dB, corresponding to received -28.273 dB through -1.845 dB.

The 6.7 GiB capture set remains outside the repository. `raw/raw_capture_inventory.csv` pins all 1,980 captures by path, byte size, and SHA-256. The release includes the trial table, summary, event ledger, run plan and state, and all 30 session records. `data/plot_points.csv` is sufficient to reproduce the publication figure.

The exact runner and stream-worker sources are preserved by the annotated tag `{RUN_SOURCE_ARCHIVE_TAG}`. The later archival plotter is preserved by `{POST_RUN_PLOT_SOURCE_ARCHIVE_TAG}`, and the publication exporter by `{PUBLICATION_EXPORT_ARCHIVE_TAG}`. Their hashes and references are recorded in `run/source_provenance.json`.

The publication PDF is a vector figure rendered through LaTeX with embedded T1 Latin Modern fonts. Its exact delivered bytes are pinned here. Rebuilds can differ at the byte level because embedded-font subset identifiers vary, even when the vector content and extracted text match.

This validates threshold-free coarse F-statistic SNR estimation through the minimal direct radio path. It does not test a decision threshold, Pfa, Pd, an ROC curve, field propagation, or multipath performance.

Repeat this sweep only after changes to estimator normalization or math, detector geometry or weights, numeric representation, waveform synthesis, or the hardware and radio path. Threshold-policy changes alone do not invalidate this threshold-free estimation evidence.

Freeze completed digital and radio results with:

    PYTHONPATH=src python3 tools/freeze_estimator_transfer.py --digital-results /path/to/digital_results --ota-results /path/to/radio_results
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        manifest_hash, checksum_hash = write_release_index(
            staging,
            experiment="sdr_ota",
            title="LimeSDR over-the-air estimator transfer",
            counts={
                "events": 1980,
                "mixtures": 1800,
                "passes": 30,
                "plot_points": 15,
                "raw_captures": 1980,
            },
        )
        _install(staging, destination)
        return manifest_hash, checksum_hash
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze completed digital and over-the-air estimator-transfer evidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--digital-results",
        type=Path,
        default=PROJECT_ROOT.parent / "estimator_transfer_2026-08-25",
    )
    parser.add_argument(
        "--ota-results",
        type=Path,
        default=PROJECT_ROOT.parent
        / "sdr_transfer_2026-08-25"
        / "sdr_ch16_ota_tx38_rx8",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "docs" / "evidence",
    )
    parser.add_argument(
        "--only",
        choices=("all", "digital", "ota"),
        default="all",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    output_root = Path(args.output_root).resolve()
    results: list[tuple[str, tuple[str, str]]] = []
    if args.only in ("all", "digital"):
        results.append(
            (
                "digital",
                freeze_digital(
                    args.digital_results,
                    output_root / "estimator_transfer_2026-08-25",
                    bootstrap_samples=args.bootstrap_samples,
                ),
            )
        )
    if args.only in ("all", "ota"):
        results.append(
            (
                "ota",
                freeze_ota(
                    args.ota_results,
                    output_root / "sdr_ota_transfer_2026-08-25",
                    bootstrap_samples=args.bootstrap_samples,
                ),
            )
        )
    for label, (manifest_hash, checksum_hash) in results:
        print(f"{label} manifest sha256 {manifest_hash}")
        print(f"{label} checksums sha256 {checksum_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
