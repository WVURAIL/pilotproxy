#!/usr/bin/env python3
# coding=utf-8
"""Plot threshold-free DTV estimator transfer results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence
import zlib

import numpy as np

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    UNIT_NORMALIZED_POWER_RATIO,
    coarse_power_ratio_to_db,
    data_shelf_snr_db_to_pilot_excess_db,
    pilot_excess_db_to_data_shelf_snr_db,
    pilot_excess_db_to_normalized_power_ratio_threshold,
)
from pilot_proxy.detector_contract import weight_term_norms_sq
from pilot_proxy.detector_reference import coarse_power_ratio_cpu_reference_packed
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration import QUANTIZATION_SCALE_MODE_GLOBAL
from pilot_proxy.integration.packing import (
    estimate_complex_scale,
    pack_channelized_streams_for_detector,
)
from pilot_proxy.paths import (
    SOURCE_CHECKOUT_ROOT,
    resolve_user_path,
)
from pilot_proxy.plot_style import setup_matplotlib
from pilot_proxy.provenance import file_sha256, sidecar_manifest_path
from pilot_proxy.reference_channelizer import (
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    apply_reference_archive_phase_convention,
    channelize_real_blocks_to_reference_channels,
    complex_envelope_to_real_adc_blocks,
    sinc_hamming_pfb_response,
)

DEFAULT_INPUT_CSV = Path("generated/dtv_snr_eval/dtv_snr_summary.csv")
DEFAULT_OUTPUT_PNG = Path("generated/dtv_snr_eval/dtv_snr_sweep.png")
DEFAULT_PLOT_DPI = 300
FIGURE_WIDTH_IN = 8.0
FIGURE_HEIGHT_IN = 5.5
MARKER_SIZE = 4.0
REFERENCE_LINE_WIDTH = 1.4
BENCHMARK_LINE_WIDTH = 1.0
RESULT_LINE_WIDTH = 1.6
HZ_PER_KHZ = 1_000.0
HZ_PER_MHZ = 1_000_000.0
DB_AMPLITUDE_FACTOR = 20.0
FULL_CYCLE_RADIANS = 2.0 * math.pi
DEFAULT_SMOOTH_WINDOW = 1
MIN_SMOOTH_WINDOW = 1
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_825
PUBLICATION_DATE = datetime(2026, 8, 25, tzinfo=timezone.utc)

REQUESTED_SNR_COLUMN = "requested_data_shelf_snr_db"
FREQUENCY_OFFSET_COLUMN = "frequency_offset_hz"
FSTAT_LEVEL_TICKS_DB = np.asarray([0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 20.0])
CONDITIONED_TRANSFER_LABEL = "Waveform-conditioned expected transfer"
CONTROL_TRANSFER_LABEL = "Control-conditioned expected transfer"
IDEAL_BENCHMARK_LABEL = "Ideal local-reference benchmark"
CONTROL_EXPECTED_COLUMN = "gpu_control_expected_data_shelf_snr_db"
RECEIVED_INPUT_SNR_COLUMN = "received_input_data_shelf_snr_db"


@dataclass(frozen=True)
class _Calibration:
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB
    bin_enbw_hz: float = EFFECTIVE_BIN_BW_HZ
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY


@dataclass(frozen=True)
class _CurveSpec:
    prefix: str
    label: str
    linestyle: str
    marker: str


@dataclass(frozen=True)
class _CleanPackedTerms:
    scale: float
    p_target: int
    p_ref_lower: int
    p_ref_upper: int
    target_norm_sq: int
    reference_norm_sum_sq: int
    input_iq: Path
    weights_path: Path


@dataclass(frozen=True)
class _ConditionedTransfer:
    delta: float
    signal_contrast: float
    reference_loading: float
    conversion_offset_db: float
    clean_scale: float
    clean_p_target: int
    clean_p_ref_lower: int
    clean_p_ref_upper: int
    target_norm_sq: int
    reference_norm_sum_sq: int
    clean_target_normalized: float
    clean_reference_normalized: float
    target_noise_response: float
    reference_noise_response: float
    data_shelf_power: float
    frequency_offset_hz: float
    conditioning_rows: int
    conditioning_snr_min_db: float
    conditioning_snr_max_db: float
    metadata_path: Path
    input_iq: Path
    weights_path: Path


CURVE_SPECS = (
    _CurveSpec("cpu_float", "CPU float", ":", "o"),
    _CurveSpec("gpu", "GPU fixed-point", "-", "s"),
    _CurveSpec("cpu_packed", "Packed CPU reference", "None", "x"),
)


def _as_paths(value: Path | Sequence[Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths = [Path(path) for path in value]
    if not paths:
        raise ValueError("At least one input CSV is required.")
    return paths


def _read_numeric_csv(path: Path, *, required: Iterable[str]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(required).difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns ({path}): "
                + ", ".join(sorted(missing))
            )
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            row.setdefault(FREQUENCY_OFFSET_COLUMN, 0.0)
            rows.append(row)
    return rows


def _read_summary_rows(
    paths: Path | Sequence[Path],
) -> list[dict[str, float]]:
    """Read one or more estimator summary CSVs."""
    rows: list[dict[str, float]] = []
    for path in _as_paths(paths):
        rows.extend(_read_numeric_csv(path, required=(REQUESTED_SNR_COLUMN,)))
    if not rows:
        raise ValueError("Summary CSV has no data rows.")
    return rows


def _read_trial_rows(paths: Sequence[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for source_index, path in enumerate(paths):
        source_rows = _read_numeric_csv(path, required=(REQUESTED_SNR_COLUMN,))
        for row in source_rows:
            row["_trial_source_index"] = float(source_index)
        rows.extend(source_rows)
    return rows


def _default_trial_paths(summary_paths: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for summary_path in summary_paths:
        name = summary_path.name
        if name.endswith("_summary.csv"):
            candidates = (
                summary_path.with_name(name.replace("_summary.csv", "_eval.csv")),
                summary_path.with_name(name.replace("_summary.csv", ".csv")),
            )
            paths.extend(candidate for candidate in candidates if candidate.is_file())
    return paths


def _finite(row: dict[str, float], *names: str) -> float:
    for name in names:
        value = float(row.get(name, math.nan))
        if math.isfinite(value):
            return value
    return math.nan


def _consistent_value(values: Iterable[float], *, name: str, default: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float(default)
    first = finite[0]
    if any(not math.isclose(value, first, rel_tol=1e-10, abs_tol=1e-10) for value in finite[1:]):
        raise ValueError(f"Input shards disagree on {name}.")
    return first


def _metadata_documents(paths: Sequence[Path]) -> list[tuple[Path, dict]]:
    documents: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    for summary_path in paths:
        metadata_path = summary_path.with_name("dtv_snr_eval.json")
        if metadata_path in seen or not metadata_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata, dict):
            documents.append((metadata_path, metadata))
            seen.add(metadata_path)
    return documents


def _metadata_calibrations(paths: Sequence[Path]) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for _metadata_path, metadata in _metadata_documents(paths):
        calibration = metadata.get("calibration", {})
        geometry = metadata.get("detector_geometry", {})
        values.append(
            {
                "pilot_below_data_db": float(
                    calibration.get("pilot_below_data_db_assumed", math.nan)
                ),
                "bin_enbw_hz": float(geometry.get("bin_enbw_hz", math.nan)),
                "dtv_bandwidth_hz": float(
                    geometry.get("dtv_bandwidth_hz", math.nan)
                ),
                "pilot_capture_efficiency": float(
                    geometry.get("pilot_capture_efficiency", math.nan)
                ),
            }
        )
    return values


def _calibration_from_inputs(
    rows: list[dict[str, float]], summary_paths: Sequence[Path]
) -> _Calibration:
    metadata = _metadata_calibrations(summary_paths)

    def values(column: str) -> list[float]:
        return [float(row.get(column, math.nan)) for row in rows] + [
            float(item.get(column, math.nan)) for item in metadata
        ]

    return _Calibration(
        pilot_below_data_db=_consistent_value(
            values("pilot_below_data_db"),
            name="pilot_below_data_db",
            default=PILOT_BELOW_DATA_DB,
        ),
        bin_enbw_hz=_consistent_value(
            values("bin_enbw_hz"),
            name="bin_enbw_hz",
            default=EFFECTIVE_BIN_BW_HZ,
        ),
        dtv_bandwidth_hz=_consistent_value(
            values("dtv_bandwidth_hz"),
            name="dtv_bandwidth_hz",
            default=DTV_BANDWIDTH_HZ,
        ),
        pilot_capture_efficiency=_consistent_value(
            values("pilot_capture_efficiency"),
            name="pilot_capture_efficiency",
            default=PILOT_CAPTURE_EFFICIENCY,
        ),
    )


def _has_control_expected(rows: list[dict[str, float]]) -> bool:
    return any(CONTROL_EXPECTED_COLUMN in row for row in rows)


def _prefixed_calibration(
    rows: list[dict[str, float]], *, prefix: str, fallback: _Calibration
) -> _Calibration:
    def value(column: str, default: float) -> float:
        return _consistent_value(
            (row.get(f"{prefix}{column}", math.nan) for row in rows),
            name=f"{prefix}{column}",
            default=default,
        )

    return _Calibration(
        pilot_below_data_db=value(
            "pilot_below_data_db", fallback.pilot_below_data_db
        ),
        bin_enbw_hz=value("bin_enbw_hz", fallback.bin_enbw_hz),
        dtv_bandwidth_hz=value(
            "dtv_bandwidth_hz", fallback.dtv_bandwidth_hz
        ),
        pilot_capture_efficiency=value(
            "pilot_capture_efficiency", fallback.pilot_capture_efficiency
        ),
    )


def _plot_calibrations(
    rows: list[dict[str, float]], summary_paths: Sequence[Path]
) -> tuple[bool, _Calibration, _Calibration]:
    base = _calibration_from_inputs(rows, summary_paths)
    radio = _has_control_expected(rows)
    if not radio:
        return False, base, base
    output = _prefixed_calibration(
        rows, prefix="detector_output_", fallback=base
    )
    received_values = [
        float(row["received_input_pilot_below_data_db"])
        for row in rows
        if math.isfinite(
            float(row.get("received_input_pilot_below_data_db", math.nan))
        )
    ]
    received_pilot_below = (
        float(np.median(np.asarray(received_values, dtype=np.float64)))
        if received_values
        else output.pilot_below_data_db
    )
    received = _Calibration(
        pilot_below_data_db=received_pilot_below,
        bin_enbw_hz=output.bin_enbw_hz,
        dtv_bandwidth_hz=output.dtv_bandwidth_hz,
        pilot_capture_efficiency=output.pilot_capture_efficiency,
    )
    return True, received, output


def _frequency_offsets(rows: list[dict[str, float]]) -> list[float]:
    return sorted(
        {
            float(row.get(FREQUENCY_OFFSET_COLUMN, 0.0))
            for row in rows
            if math.isfinite(float(row.get(FREQUENCY_OFFSET_COLUMN, 0.0)))
        }
    )


def _rows_at(
    rows: list[dict[str, float]], *, frequency_offset_hz: float, requested_snr_db: float
) -> list[dict[str, float]]:
    return [
        row
        for row in rows
        if float(row.get(FREQUENCY_OFFSET_COLUMN, 0.0)) == float(frequency_offset_hz)
        and float(row[REQUESTED_SNR_COLUMN]) == float(requested_snr_db)
    ]


def _summary_input_snr(row: dict[str, float], *, radio: bool) -> float:
    if radio:
        received = _finite(row, RECEIVED_INPUT_SNR_COLUMN)
        if math.isfinite(received):
            return received
    return _finite(row, REQUESTED_SNR_COLUMN)


def _control_expected_points(
    rows: list[dict[str, float]], *, frequency_offset_hz: float
) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if float(row.get(FREQUENCY_OFFSET_COLUMN, 0.0)) != float(
            frequency_offset_hz
        ):
            continue
        x_value = _summary_input_snr(row, radio=True)
        if not math.isfinite(x_value):
            continue
        points.append((x_value, _finite(row, CONTROL_EXPECTED_COLUMN)))
    points.sort(key=lambda point: point[0])
    return (
        [point[0] for point in points],
        [point[1] for point in points],
    )


def _offset_label(prefix: str, frequency_offset_hz: float) -> str:
    if frequency_offset_hz == 0.0:
        return f"{prefix}, 0 Hz"
    if abs(frequency_offset_hz) >= HZ_PER_KHZ:
        return f"{prefix}, {frequency_offset_hz / HZ_PER_KHZ:+.1f} kHz"
    return f"{prefix}, {frequency_offset_hz:+.1f} Hz"


def _centered_moving_average(values: list[float], window: int) -> list[float]:
    if window <= MIN_SMOOTH_WINDOW:
        return list(values)
    smoothed: list[float] = []
    radius = int(window) // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        finite = [value for value in values[start:stop] if math.isfinite(value)]
        smoothed.append(sum(finite) / float(len(finite)) if finite else math.nan)
    return smoothed


def _curve_label(prefix: str, frequency_offset_hz: float, smooth_window: int) -> str:
    label = _offset_label(prefix, frequency_offset_hz)
    if smooth_window > MIN_SMOOTH_WINDOW:
        label += f", {smooth_window}-pt smooth"
    return label


def _reference_transfer_db(values) -> np.ndarray:
    """Return the ideal local-reference estimator transfer."""
    snr = np.asarray(values, dtype=np.float64)
    linear = DB_LINEAR_BASE ** (snr / DB_POWER_FACTOR)
    return snr - DB_POWER_FACTOR * np.log10(1.0 + linear)


def _conditioned_transfer_db(
    values, model: _ConditionedTransfer
) -> np.ndarray:
    """Return the expected transfer for the recorded waveform."""
    snr = np.asarray(values, dtype=np.float64)
    linear = DB_LINEAR_BASE ** (snr / DB_POWER_FACTOR)
    excess = (
        model.delta + model.signal_contrast * linear
    ) / (1.0 + model.reference_loading * linear)
    result = np.full_like(excess, np.nan, dtype=np.float64)
    keep = excess > 0.0
    result[keep] = (
        model.conversion_offset_db
        + DB_POWER_FACTOR * np.log10(excess[keep])
    )
    return result


def _artifact_path(value: object, metadata_path: Path) -> Path:
    base = SOURCE_CHECKOUT_ROOT or metadata_path.parent
    path = resolve_user_path(str(value), relative_to=base)
    if path.is_file():
        return path
    alternate = resolve_user_path(str(value), relative_to=metadata_path.parent)
    if alternate.is_file():
        return alternate
    raise FileNotFoundError(path)


def _expected_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} SHA256 is invalid.")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} SHA256 is invalid.") from exc
    return value.lower()


def _recorded_artifact_identity(metadata: dict, key: str, *, label: str) -> dict:
    identity = metadata.get(key)
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity is missing or invalid.")
    path = identity.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} path is missing.")
    _expected_sha256(identity.get("sha256"), label=label)
    return identity


def _verify_file_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: expected {expected}, got {actual}."
        )


def _conditioning_weight_bank_path(
    metadata: dict,
    metadata_path: Path,
    override_path: Path | None = None,
    override_sha256: str | None = None,
    override_manifest_sha256: str | None = None,
) -> Path:
    bank_identity = metadata.get("weight_bank")
    manifest_identity = metadata.get("weight_manifest")
    legacy = bank_identity is None and manifest_identity is None
    if legacy:
        if (
            override_path is None
            or override_sha256 is None
            or override_manifest_sha256 is None
        ):
            raise ValueError(
                "Legacy conditioning requires an explicit weight-bank path and "
                "expected bank and manifest SHA256 values."
            )
        expected_bank = _expected_sha256(
            override_sha256,
            label="Conditioning weight-bank",
        )
        expected_manifest = _expected_sha256(
            override_manifest_sha256,
            label="Conditioning weight manifest",
        )
        path = resolve_user_path(str(override_path))
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        bank_record = _recorded_artifact_identity(
            metadata,
            "weight_bank",
            label="Conditioning weight-bank",
        )
        manifest_record = _recorded_artifact_identity(
            metadata,
            "weight_manifest",
            label="Conditioning weight manifest",
        )
        expected_bank = _expected_sha256(
            bank_record["sha256"],
            label="Conditioning weight-bank",
        )
        expected_manifest = _expected_sha256(
            manifest_record["sha256"],
            label="Conditioning weight manifest",
        )
        if override_sha256 is not None:
            explicit_bank = _expected_sha256(
                override_sha256,
                label="Conditioning weight-bank override",
            )
            if explicit_bank != expected_bank:
                raise ValueError(
                    "Conditioning weight-bank override SHA256 disagrees with metadata."
                )
        if override_manifest_sha256 is not None:
            explicit_manifest = _expected_sha256(
                override_manifest_sha256,
                label="Conditioning weight manifest override",
            )
            if explicit_manifest != expected_manifest:
                raise ValueError(
                    "Conditioning weight manifest override SHA256 disagrees with "
                    "metadata."
                )
        if override_path is None:
            path = _artifact_path(bank_record["path"], metadata_path)
        else:
            path = resolve_user_path(str(override_path))
            if not path.is_file():
                raise FileNotFoundError(path)

    manifest_path = sidecar_manifest_path(path)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not legacy and override_path is None:
        recorded_manifest_path = _artifact_path(
            manifest_record["path"],
            metadata_path,
        )
        if recorded_manifest_path.resolve() != manifest_path.resolve():
            raise ValueError(
                "Conditioning weight manifest is not adjacent to the recorded bank."
            )
    _verify_file_sha256(path, expected_bank, label="Conditioning weight-bank")
    _verify_file_sha256(
        manifest_path,
        expected_manifest,
        label="Conditioning weight manifest",
    )
    return path


def _weight_coefficients_identity(weights: np.ndarray) -> dict[str, object]:
    values = np.ascontiguousarray(weights)
    return {
        "dtype": str(values.dtype),
        "shape": [int(value) for value in values.shape],
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _verify_selected_weight_provenance(
    metadata: dict,
    selected_layout: dict[str, object],
    weights: np.ndarray,
) -> None:
    recorded_layout = metadata.get("selected_weight_layout")
    if not isinstance(recorded_layout, dict):
        raise ValueError("Conditioning selected weight layout is missing or invalid.")
    if recorded_layout != selected_layout:
        raise ValueError(
            "Conditioning selected weight layout disagrees with the manifest."
        )
    recorded_coefficients = metadata.get("selected_weight_coefficients")
    if recorded_coefficients is None:
        return
    if not isinstance(recorded_coefficients, dict):
        raise ValueError("Conditioning selected weight coefficients are invalid.")
    actual = _weight_coefficients_identity(weights)
    expected_digest = _expected_sha256(
        recorded_coefficients.get("sha256"),
        label="Conditioning selected weight coefficients",
    )
    if recorded_coefficients.get("dtype") != actual["dtype"]:
        raise ValueError("Conditioning selected weight coefficient dtype changed.")
    if recorded_coefficients.get("shape") != actual["shape"]:
        raise ValueError("Conditioning selected weight coefficient shape changed.")
    if expected_digest != actual["sha256"]:
        raise ValueError("Conditioning selected weight coefficient SHA256 changed.")


def _required_iq_samples(
    *, iq_sample_rate_hz: float, adc_sample_rate_hz: float, frame_size: int
) -> int:
    n_blocks = int(frame_size) + REFERENCE_PFB_TAPS - 1
    total_adc_samples = n_blocks * REFERENCE_PFB_FFT_SIZE
    last_source_position = (
        (total_adc_samples - 1)
        * float(iq_sample_rate_hz)
        / float(adc_sample_rate_hz)
    )
    return int(math.ceil(last_source_position)) + 1


def _clean_packed_terms(
    metadata: dict,
    metadata_path: Path,
    *,
    frequency_offset_hz: float,
    channel_gain_db: float,
    channel_phase_deg: float,
    conditioning_weights_path: Path | None = None,
    conditioning_weights_sha256: str | None = None,
    conditioning_weight_manifest_sha256: str | None = None,
) -> _CleanPackedTerms:
    geometry = metadata["detector_geometry"]
    layout = geometry["input_layout"]
    testbench = metadata["testbench"]
    quantization = testbench["quantization"]
    audit = metadata["atsc_waveform_audit"]
    input_iq = _artifact_path(metadata["input_iq"], metadata_path)
    weights_path = _conditioning_weight_bank_path(
        metadata,
        metadata_path,
        conditioning_weights_path,
        conditioning_weights_sha256,
        conditioning_weight_manifest_sha256,
    )
    spec = ReferenceChannelizerSpec()
    frame_size = int(layout["frame_size_samples"])
    detector_window = int(layout["detector_window_samples"])
    num_inputs = int(layout["num_input_streams"])
    bits = int(quantization["bits_per_component"])
    clip_sigma = float(quantization["clip_sigma"])
    iq_sample_rate_hz = float(audit["sample_rate_hz"])
    required = _required_iq_samples(
        iq_sample_rate_hz=iq_sample_rate_hz,
        adc_sample_rate_hz=spec.adc_sample_rate_hz,
        frame_size=frame_size,
    )
    clean_iq = np.fromfile(input_iq, dtype=np.complex64, count=required)
    if clean_iq.size != required:
        raise ValueError(f"Input IQ is too short: {input_iq}")
    if (
        frequency_offset_hz != 0.0
        or channel_gain_db != 0.0
        or channel_phase_deg != 0.0
    ):
        sample_index = np.arange(clean_iq.size, dtype=np.float64)
        phase = (
            FULL_CYCLE_RADIANS
            * frequency_offset_hz
            * sample_index
            / iq_sample_rate_hz
            + math.radians(channel_phase_deg)
        )
        gain = DB_LINEAR_BASE ** (channel_gain_db / DB_AMPLITUDE_FACTOR)
        clean_iq = np.ascontiguousarray(
            clean_iq * np.asarray(gain * np.exp(1j * phase), dtype=np.complex64)
        )

    raw_blocks = complex_envelope_to_real_adc_blocks(
        clean_iq,
        iq_sample_rate_hz=iq_sample_rate_hz,
        rf_center_hz=float(geometry["rf_center_hz"]),
        adc_sample_rate_hz=spec.adc_sample_rate_hz,
        band_lower_hz=spec.band_lower_hz,
        n_blocks=frame_size + REFERENCE_PFB_TAPS - 1,
        block_size=REFERENCE_PFB_FFT_SIZE,
    )
    response = sinc_hamming_pfb_response(
        REFERENCE_PFB_TAPS, REFERENCE_PFB_FFT_SIZE
    )
    channel_streams = channelize_real_blocks_to_reference_channels(
        raw_blocks,
        channel_indices=[int(geometry["selected_channel_index"])],
        response=response,
        spec=spec,
    )
    if bool(geometry.get("reference_archive_phase", False)):
        channel_streams = apply_reference_archive_phase_convention(channel_streams)
    feed_streams = np.repeat(channel_streams[np.newaxis, :, :], num_inputs, axis=0)
    scale = estimate_complex_scale(
        feed_streams,
        bits_per_component=bits,
        clip_sigma=clip_sigma,
    )
    packed = pack_channelized_streams_for_detector(
        feed_streams,
        frame_size_samples=frame_size,
        detector_window_samples=detector_window,
        spectral_sense=str(geometry["spectral_sense"]),
        quantization_scale_mode=QUANTIZATION_SCALE_MODE_GLOBAL,
        clip_sigma=clip_sigma,
        bits_per_component=bits,
        scale=scale,
    ).packed[0]
    bank = DetectorWeightBank(explicit_path=weights_path)
    selected_layout = bank.layout_for_pilot_frequency(
        float(geometry["dtv_pilot_hz"]) / HZ_PER_MHZ
    )
    weights, valid = bank.get_weights_for_pilot_frequency(
        float(geometry["dtv_pilot_hz"]) / HZ_PER_MHZ
    )
    if weights is None or not valid:
        raise ValueError("No matching detector weights were found.")
    _verify_selected_weight_provenance(metadata, selected_layout, weights)
    _ratio, powers = coarse_power_ratio_cpu_reference_packed(packed, weights, bits)
    target_norm, lower_norm, upper_norm = weight_term_norms_sq(weights)
    return _CleanPackedTerms(
        scale=float(scale),
        p_target=int(round(float(powers[0]))),
        p_ref_lower=int(round(float(powers[1]))),
        p_ref_upper=int(round(float(powers[2]))),
        target_norm_sq=int(target_norm),
        reference_norm_sum_sq=int(lower_norm + upper_norm),
        input_iq=input_iq,
        weights_path=weights_path,
    )


def _conditioned_transfer_from_clean_terms(
    rows: list[dict[str, float]],
    calibration: _Calibration,
    clean: _CleanPackedTerms,
    *,
    metadata_path: Path,
    frequency_offset_hz: float,
) -> _ConditionedTransfer:
    clean_target = (
        clean.p_target / (clean.target_norm_sq * clean.scale * clean.scale)
    )
    clean_reference = (
        (clean.p_ref_lower + clean.p_ref_upper)
        / (clean.reference_norm_sum_sq * clean.scale * clean.scale)
    )
    target_residuals: list[float] = []
    reference_residuals: list[float] = []
    noise_powers: list[float] = []
    shelf_powers: list[float] = []
    snr_values: list[float] = []
    for row in rows:
        scale = _finite(row, "quantization_scale")
        p_target = _finite(row, "p_target_u64", "cpu_packed_p_target")
        p_reference = _finite(row, "p_ref_sum_u64", "cpu_packed_p_ref_sum")
        noise_power = _finite(row, "measured_in_band_noise_power")
        shelf_power = _finite(row, "measured_data_shelf_power")
        target_norm = _finite(row, "target_weight_norm_sq")
        reference_norm = _finite(row, "reference_weight_norm_sum_sq")
        snr = _finite(row, REQUESTED_SNR_COLUMN)
        required = (
            scale,
            p_target,
            p_reference,
            noise_power,
            shelf_power,
            target_norm,
            reference_norm,
            snr,
        )
        if not all(math.isfinite(value) for value in required):
            continue
        if scale <= 0.0 or noise_power <= 0.0 or shelf_power <= 0.0:
            continue
        if not math.isclose(target_norm, clean.target_norm_sq):
            raise ValueError("Target weight norms do not match.")
        if not math.isclose(reference_norm, clean.reference_norm_sum_sq):
            raise ValueError("Reference weight norms do not match.")
        scale_sq = scale * scale
        target = p_target / (clean.target_norm_sq * scale_sq)
        reference = p_reference / (clean.reference_norm_sum_sq * scale_sq)
        target_residuals.append(target - clean_target)
        reference_residuals.append(reference - clean_reference)
        noise_powers.append(noise_power)
        shelf_powers.append(shelf_power)
        snr_values.append(snr)
    if len(noise_powers) < 2:
        raise ValueError("Conditioning requires at least two raw trials.")
    noise_sq_sum = math.fsum(value * value for value in noise_powers)
    target_noise_response = math.fsum(
        noise * residual
        for noise, residual in zip(noise_powers, target_residuals, strict=True)
    ) / noise_sq_sum
    reference_noise_response = math.fsum(
        noise * residual
        for noise, residual in zip(noise_powers, reference_residuals, strict=True)
    ) / noise_sq_sum
    data_shelf_power = math.fsum(shelf_powers) / len(shelf_powers)
    if target_noise_response <= 0.0 or reference_noise_response <= 0.0:
        raise ValueError("Conditioned noise responses must be positive.")
    denominator = reference_noise_response * data_shelf_power
    delta = target_noise_response / reference_noise_response - 1.0
    signal_contrast = (clean_target - clean_reference) / denominator
    reference_loading = clean_reference / denominator
    conversion_offset_db = float(_pilot_to_shelf(0.0, calibration))
    return _ConditionedTransfer(
        delta=float(delta),
        signal_contrast=float(signal_contrast),
        reference_loading=float(reference_loading),
        conversion_offset_db=conversion_offset_db,
        clean_scale=clean.scale,
        clean_p_target=clean.p_target,
        clean_p_ref_lower=clean.p_ref_lower,
        clean_p_ref_upper=clean.p_ref_upper,
        target_norm_sq=clean.target_norm_sq,
        reference_norm_sum_sq=clean.reference_norm_sum_sq,
        clean_target_normalized=float(clean_target),
        clean_reference_normalized=float(clean_reference),
        target_noise_response=float(target_noise_response),
        reference_noise_response=float(reference_noise_response),
        data_shelf_power=float(data_shelf_power),
        frequency_offset_hz=float(frequency_offset_hz),
        conditioning_rows=len(noise_powers),
        conditioning_snr_min_db=min(snr_values),
        conditioning_snr_max_db=max(snr_values),
        metadata_path=metadata_path,
        input_iq=clean.input_iq,
        weights_path=clean.weights_path,
    )


def _estimated_data_shelf_snr_db_to_normalized_coarse_power_ratio_db(values) -> np.ndarray:
    """Map shelf-SNR coordinates to normalized ratio coordinates."""
    snr = np.asarray(values, dtype=np.float64)
    pnr = data_shelf_snr_db_to_pilot_excess_db(snr)
    raw = pilot_excess_db_to_normalized_power_ratio_threshold(pnr)
    return np.asarray(coarse_power_ratio_to_db(raw), dtype=np.float64)


def _normalized_coarse_power_ratio_db_to_estimated_data_shelf_snr_db(values) -> np.ndarray:
    """Map normalized ratio coordinates back to shelf-SNR coordinates."""
    level = np.asarray(values, dtype=np.float64)
    raw = DB_LINEAR_BASE ** (level / DB_POWER_FACTOR)
    excess = np.maximum(
        raw - UNIT_NORMALIZED_POWER_RATIO, np.finfo(np.float64).tiny
    )
    pnr = DB_POWER_FACTOR * np.log10(excess)
    return np.asarray(pilot_excess_db_to_data_shelf_snr_db(pnr), dtype=np.float64)


def _shelf_to_pilot(values, calibration: _Calibration) -> np.ndarray:
    return np.asarray(
        data_shelf_snr_db_to_pilot_excess_db(
            values,
            pilot_below_data_db=calibration.pilot_below_data_db,
            bin_enbw_hz=calibration.bin_enbw_hz,
            dtv_bandwidth_hz=calibration.dtv_bandwidth_hz,
            pilot_capture_efficiency=calibration.pilot_capture_efficiency,
        ),
        dtype=np.float64,
    )


def _pilot_to_shelf(values, calibration: _Calibration) -> np.ndarray:
    return np.asarray(
        pilot_excess_db_to_data_shelf_snr_db(
            values,
            pilot_below_data_db=calibration.pilot_below_data_db,
            bin_enbw_hz=calibration.bin_enbw_hz,
            dtv_bandwidth_hz=calibration.dtv_bandwidth_hz,
            pilot_capture_efficiency=calibration.pilot_capture_efficiency,
        ),
        dtype=np.float64,
    )


def _conditioning_metadata(
    summary_paths: Sequence[Path], conditioning_paths: Sequence[Path]
) -> tuple[Path, dict] | None:
    documents = _metadata_documents(summary_paths)
    if not documents:
        return None
    conditioning_parents = {path.resolve().parent for path in conditioning_paths}
    for metadata_path, metadata in documents:
        if metadata_path.resolve().parent in conditioning_parents:
            return metadata_path, metadata
    return documents[0]


def _try_conditioned_transfer(
    *,
    summary_paths: Sequence[Path],
    conditioning_paths: Sequence[Path],
    rows: list[dict[str, float]],
    calibration: _Calibration,
    conditioning_weights_path: Path | None = None,
    conditioning_weights_sha256: str | None = None,
    conditioning_weight_manifest_sha256: str | None = None,
) -> _ConditionedTransfer | None:
    if not conditioning_paths:
        return None
    selected = _conditioning_metadata(summary_paths, conditioning_paths)
    if selected is None:
        raise ValueError("Conditioning metadata was not found.")
    if not rows:
        raise ValueError("Conditioning trial CSV has no rows.")
    metadata_path, metadata = selected
    try:
        frequency_offset_hz = _consistent_value(
            (row.get(FREQUENCY_OFFSET_COLUMN, math.nan) for row in rows),
            name=FREQUENCY_OFFSET_COLUMN,
            default=0.0,
        )
        channel_gain_db = _consistent_value(
            (row.get("channel_gain_db", math.nan) for row in rows),
            name="channel_gain_db",
            default=0.0,
        )
        channel_phase_deg = _consistent_value(
            (row.get("channel_phase_deg", math.nan) for row in rows),
            name="channel_phase_deg",
            default=0.0,
        )
        clean = _clean_packed_terms(
            metadata,
            metadata_path,
            frequency_offset_hz=frequency_offset_hz,
            channel_gain_db=channel_gain_db,
            channel_phase_deg=channel_phase_deg,
            conditioning_weights_path=conditioning_weights_path,
            conditioning_weights_sha256=conditioning_weights_sha256,
            conditioning_weight_manifest_sha256=(
                conditioning_weight_manifest_sha256
            ),
        )
        return _conditioned_transfer_from_clean_terms(
            rows,
            calibration,
            clean,
            metadata_path=metadata_path,
            frequency_offset_hz=frequency_offset_hz,
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not derive the waveform-conditioned transfer: {exc}"
        ) from exc


def _conditioning_record(
    model: _ConditionedTransfer, conditioning_paths: Sequence[Path]
) -> dict[str, object]:
    weight_manifest_path = sidecar_manifest_path(model.weights_path)
    return {
        "schema_version": "pilotproxy_waveform_conditioning_v1",
        "plot_source_sha256": file_sha256(Path(__file__)),
        "comparator_label": CONDITIONED_TRANSFER_LABEL,
        "formula": "y=C+10log10((delta+a*10^(x/10))/(1+b*10^(x/10)))",
        "coefficients": {
            "C_db": model.conversion_offset_db,
            "delta": model.delta,
            "a": model.signal_contrast,
            "b": model.reference_loading,
        },
        "clean_waveform": {
            "input_iq": str(model.input_iq),
            "input_iq_sha256": file_sha256(model.input_iq),
            "weights": str(model.weights_path),
            "weights_sha256": file_sha256(model.weights_path),
            "weight_manifest": (
                None if weight_manifest_path is None else str(weight_manifest_path)
            ),
            "weight_manifest_sha256": file_sha256(weight_manifest_path),
            "quantization_scale": model.clean_scale,
            "p_target": model.clean_p_target,
            "p_ref_lower": model.clean_p_ref_lower,
            "p_ref_upper": model.clean_p_ref_upper,
            "target_norm_sq": model.target_norm_sq,
            "reference_norm_sum_sq": model.reference_norm_sum_sq,
            "target_normalized": model.clean_target_normalized,
            "reference_normalized": model.clean_reference_normalized,
        },
        "lower_trial_conditioning": {
            "trial_csv": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in conditioning_paths
            ],
            "metadata_json": str(model.metadata_path),
            "metadata_sha256": file_sha256(model.metadata_path),
            "rows": model.conditioning_rows,
            "snr_min_db": model.conditioning_snr_min_db,
            "snr_max_db": model.conditioning_snr_max_db,
            "frequency_offset_hz": model.frequency_offset_hz,
            "data_shelf_power": model.data_shelf_power,
            "target_noise_response": model.target_noise_response,
            "reference_noise_response": model.reference_noise_response,
            "power_scale_rule": "raw_power/(weight_norm_sq*quantization_scale^2)",
            "noise_response_rule": "sum(noise_power*(normalized_trial-clean_normalized))/sum(noise_power^2)",
            "power_scale_note": "First-order correction for adaptive 4-bit scaling.",
        },
        "benchmark": {
            "label": IDEAL_BENCHMARK_LABEL,
            "formula": "y=x-10log10(1+10^(x/10))",
        },
    }


def _write_conditioning_record(
    path: Path,
    model: _ConditionedTransfer,
    conditioning_paths: Sequence[Path],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(
            _conditioning_record(model, conditioning_paths),
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        f.write("\n")


def _add_pilot_axis(
    axis,
    *,
    orientation: str,
    calibration: _Calibration,
    received_input: bool = False,
):
    functions = (
        lambda values: _shelf_to_pilot(values, calibration),
        lambda values: _pilot_to_shelf(values, calibration),
    )
    if orientation == "x":
        secondary = axis.secondary_xaxis("top", functions=functions)
        if received_input:
            secondary.set_xlabel(
                r"Received input pilot-bin excess, $\rho_{\mathrm{pilot}}\;[\mathrm{dB}]$"
            )
        else:
            secondary.set_xlabel(
                r"Known pilot-bin excess, $\rho_{\mathrm{pilot}}\;[\mathrm{dB}]$"
            )
    elif orientation == "y":
        secondary = axis.secondary_yaxis("right", functions=functions)
        secondary.set_ylabel(
            r"Measured pilot-bin excess, $\hat{\rho}_{\mathrm{pilot}}\;[\mathrm{dB}]$"
        )
    else:
        raise ValueError(f"unknown secondary-axis orientation: {orientation!r}")
    return secondary


def _summary_ratio(rows: list[dict[str, float]], prefix: str) -> float:
    excess_names = [f"{prefix}_pooled_normalized_pilot_excess"]
    ratio_names = [f"{prefix}_pooled_normalized_coarse_power_ratio"]
    weight_names = [f"{prefix}_pooled_p_ref_sum"]
    if prefix == "gpu":
        excess_names.append("pooled_normalized_pilot_excess")
        ratio_names.append("pooled_normalized_coarse_power_ratio")
        weight_names.append("pooled_p_ref_sum")

    pairs: list[tuple[float, float]] = []
    for row in rows:
        excess = _finite(row, *excess_names)
        ratio = (
            excess + UNIT_NORMALIZED_POWER_RATIO
            if math.isfinite(excess)
            else _finite(row, *ratio_names)
        )
        if not math.isfinite(ratio):
            continue
        weight = _finite(row, *weight_names)
        pairs.append((ratio, weight))
    if not pairs:
        return math.nan
    if len(pairs) == 1:
        return float(pairs[0][0])
    if any(not math.isfinite(weight) or weight <= 0.0 for _, weight in pairs):
        raise ValueError(
            f"Overlapping {prefix} summary shards need pooled reference powers."
        )
    return float(
        sum(ratio * weight for ratio, weight in pairs)
        / sum(weight for _, weight in pairs)
    )


def _trial_weight(row: dict[str, float], prefix: str) -> float:
    if prefix == "gpu":
        direct = _finite(row, "p_ref_sum_u64", "p_ref_sum")
        if math.isfinite(direct):
            return direct
        lower = _finite(row, "p_ref_lower_u64", "p_ref_lower")
        upper = _finite(row, "p_ref_upper_u64", "p_ref_upper")
    elif prefix == "cpu_float":
        direct = _finite(row, "cpu_float_p_ref_sum")
        if math.isfinite(direct):
            return direct
        lower = _finite(row, "cpu_float_p_ref_lower")
        upper = _finite(row, "cpu_float_p_ref_upper")
    else:
        direct = _finite(row, "cpu_packed_p_ref_sum", "p_ref_sum_u64")
        if math.isfinite(direct):
            return direct
        lower = _finite(row, "cpu_packed_p_ref_lower", "p_ref_lower_u64")
        upper = _finite(row, "cpu_packed_p_ref_upper", "p_ref_upper_u64")
    if math.isfinite(lower) and math.isfinite(upper):
        return lower + upper
    return math.nan


def _direct_trial_ratio(row: dict[str, float], prefix: str) -> float:
    if prefix == "gpu":
        ratio = _finite(row, "normalized_coarse_power_ratio")
        if math.isfinite(ratio):
            return ratio
        excess = _finite(row, "normalized_pilot_excess")
        if math.isfinite(excess):
            return excess + UNIT_NORMALIZED_POWER_RATIO
        level = _finite(row, "normalized_coarse_power_ratio_db")
        if math.isfinite(level):
            return float(DB_LINEAR_BASE ** (level / DB_POWER_FACTOR))
    elif prefix == "cpu_float":
        ratio = _finite(row, "cpu_float_normalized_coarse_power_ratio")
        if math.isfinite(ratio):
            return ratio
        excess = _finite(row, "cpu_float_normalized_pilot_excess")
        if math.isfinite(excess):
            return excess + UNIT_NORMALIZED_POWER_RATIO
    else:
        ratio = _finite(row, "cpu_packed_normalized_coarse_power_ratio")
        if math.isfinite(ratio):
            return ratio
        excess = _finite(row, "cpu_packed_normalized_pilot_excess")
        if math.isfinite(excess):
            return excess + UNIT_NORMALIZED_POWER_RATIO
    return math.nan


def _raw_trial_ratio(row: dict[str, float], prefix: str) -> float:
    if prefix == "gpu":
        return _finite(row, "coarse_power_ratio")
    if prefix == "cpu_float":
        return _finite(row, "cpu_float_coarse_power_ratio")
    return _finite(row, "cpu_packed_coarse_power_ratio", "cpu_coarse_power_ratio")


def _null_ratio(rows: list[dict[str, float]], prefix: str) -> float:
    candidates: list[float] = []
    for row in rows:
        raw = _raw_trial_ratio(row, prefix)
        ratio = _direct_trial_ratio(row, prefix)
        if prefix == "cpu_float" and not math.isfinite(ratio):
            excess_db = _finite(row, "cpu_float_pilot_excess_db")
            if math.isfinite(excess_db):
                ratio = 1.0 + DB_LINEAR_BASE ** (excess_db / DB_POWER_FACTOR)
        if prefix == "cpu_packed" and not math.isfinite(ratio):
            ratio = _direct_trial_ratio(row, "gpu")
        if math.isfinite(raw) and math.isfinite(ratio) and ratio > 0.0:
            candidates.append(raw / ratio)
    if not candidates:
        return math.nan
    return float(np.median(np.asarray(candidates, dtype=np.float64)))


def _trial_ratio_weight_records(
    rows: list[dict[str, float]], prefix: str
) -> list[tuple[dict[str, float], float, float]]:
    null_ratio = _null_ratio(rows, prefix)
    records: list[tuple[dict[str, float], float, float]] = []
    for row in rows:
        ratio = _direct_trial_ratio(row, prefix)
        if not math.isfinite(ratio) and math.isfinite(null_ratio) and null_ratio > 0.0:
            raw = _raw_trial_ratio(row, prefix)
            if math.isfinite(raw):
                ratio = raw / null_ratio
        weight = _trial_weight(row, prefix)
        if math.isfinite(ratio) and math.isfinite(weight) and weight > 0.0:
            records.append((row, ratio, weight))
    return records


def _trial_ratio_weight_pairs(
    rows: list[dict[str, float]], prefix: str
) -> list[tuple[float, float]]:
    return [
        (ratio, weight)
        for _row, ratio, weight in _trial_ratio_weight_records(rows, prefix)
    ]


def _pooled_ratio_from_trials(rows: list[dict[str, float]], prefix: str) -> float:
    pairs = _trial_ratio_weight_pairs(rows, prefix)
    if not pairs:
        return math.nan
    return float(
        sum(ratio * weight for ratio, weight in pairs)
        / sum(weight for _, weight in pairs)
    )


def _bootstrap_pooled_excess_interval(
    rows: list[dict[str, float]],
    prefix: str,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap signed pooled excess in linear coordinates."""
    records = _trial_ratio_weight_records(rows, prefix)
    if len(records) < 2 or int(samples) < 1:
        return math.nan, math.nan

    pass_values = [_finite(row, "pass_index") for row, _ratio, _weight in records]
    if all(math.isfinite(value) for value in pass_values):
        grouped: dict[tuple[float, int], list[float]] = {}
        for (row, ratio, weight), pass_value in zip(
            records, pass_values, strict=True
        ):
            source = _finite(row, "_trial_source_index")
            source = source if math.isfinite(source) else 0.0
            key = (source, int(pass_value))
            values = grouped.setdefault(key, [0.0, 0.0])
            values[0] += ratio * weight
            values[1] += weight
        units = np.asarray(list(grouped.values()), dtype=np.float64)
    else:
        units = np.asarray(
            [(ratio * weight, weight) for _row, ratio, weight in records],
            dtype=np.float64,
        )
    if units.shape[0] < 2:
        return math.nan, math.nan

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, units.shape[0], size=(int(samples), units.shape[0])
    )
    sampled = units[indices]
    pooled = np.sum(sampled[:, :, 0], axis=1) / np.sum(
        sampled[:, :, 1], axis=1
    )
    excess = pooled - UNIT_NORMALIZED_POWER_RATIO
    low, high = np.quantile(excess, [0.025, 0.975])
    return float(low), float(high)


def _excess_to_shelf_db(excess: float, calibration: _Calibration) -> float:
    if not math.isfinite(excess) or excess <= 0.0:
        return math.nan
    pilot_excess_db = DB_POWER_FACTOR * math.log10(excess)
    return float(_pilot_to_shelf(pilot_excess_db, calibration))


def _point_seed(prefix: str, offset: float, snr: float) -> int:
    token = f"{prefix}:{offset:.9f}:{snr:.9f}".encode("ascii")
    return DEFAULT_BOOTSTRAP_SEED + int(zlib.crc32(token))


def _curve_points(
    summary_rows: list[dict[str, float]],
    trial_rows: list[dict[str, float]],
    *,
    prefix: str,
    offset: float,
    calibration: _Calibration,
    bootstrap_samples: int,
    radio: bool = False,
) -> list[dict[str, float]]:
    requested = sorted(
        {
            float(row[REQUESTED_SNR_COLUMN])
            for row in summary_rows
            if float(row.get(FREQUENCY_OFFSET_COLUMN, 0.0)) == float(offset)
            and math.isfinite(float(row[REQUESTED_SNR_COLUMN]))
        }
    )
    points: list[dict[str, float]] = []
    for snr in requested:
        summaries = _rows_at(
            summary_rows, frequency_offset_hz=offset, requested_snr_db=snr
        )
        trials = _rows_at(
            trial_rows, frequency_offset_hz=offset, requested_snr_db=snr
        )
        ratio = _summary_ratio(summaries, prefix)
        if not math.isfinite(ratio):
            ratio = _pooled_ratio_from_trials(trials, prefix)
        if not math.isfinite(ratio):
            continue
        x_value = _summary_input_snr(summaries[0], radio=radio)
        if not math.isfinite(x_value):
            continue
        low, high = _bootstrap_pooled_excess_interval(
            trials,
            prefix,
            samples=bootstrap_samples,
            seed=_point_seed(prefix, offset, snr),
        )
        points.append(
            {
                "x": x_value,
                "excess": ratio - UNIT_NORMALIZED_POWER_RATIO,
                "y": _excess_to_shelf_db(
                    ratio - UNIT_NORMALIZED_POWER_RATIO, calibration
                ),
                "low_excess": low,
                "high_excess": high,
            }
        )
    return points


def _plot_control_expected(
    ax,
    rows: list[dict[str, float]],
    *,
    offset: float,
    y_floor: float,
) -> bool:
    x_values, y_values = _control_expected_points(
        rows, frequency_offset_hz=offset
    )
    if not x_values:
        return False
    ax.plot(
        x_values,
        y_values,
        linestyle="--",
        color="black",
        linewidth=REFERENCE_LINE_WIDTH,
        label=CONTROL_TRANSFER_LABEL,
    )
    nonfinite_x = [
        x_value
        for x_value, y_value in zip(x_values, y_values, strict=True)
        if not math.isfinite(y_value)
    ]
    if nonfinite_x:
        ax.scatter(
            nonfinite_x,
            [y_floor] * len(nonfinite_x),
            marker="v",
            facecolors="none",
            edgecolors="black",
            s=28,
        )
    return True


def _plot_intervals(
    ax,
    points: list[dict[str, float]],
    *,
    color: str,
    y_floor: float,
    calibration: _Calibration,
    label: str | None = None,
) -> None:
    x_values: list[float] = []
    y_values: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    crossed: list[float] = []
    censored: list[float] = []
    for point in points:
        y = float(point["y"])
        if not math.isfinite(y):
            censored.append(float(point["x"]))
            continue
        low_excess = float(point["low_excess"])
        high_excess = float(point["high_excess"])
        if not math.isfinite(low_excess) or not math.isfinite(high_excess):
            continue
        low_y = _excess_to_shelf_db(low_excess, calibration)
        high_y = _excess_to_shelf_db(high_excess, calibration)
        if not math.isfinite(high_y):
            continue
        if not math.isfinite(low_y):
            low_y = y_floor
            crossed.append(float(point["x"]))
        low_y = min(low_y, y)
        high_y = max(high_y, y)
        x_values.append(float(point["x"]))
        y_values.append(y)
        lower_errors.append(y - low_y)
        upper_errors.append(high_y - y)
    if x_values:
        ax.errorbar(
            x_values,
            y_values,
            yerr=np.asarray([lower_errors, upper_errors]),
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.75,
            label=label,
        )
    if crossed:
        ax.scatter(crossed, [y_floor] * len(crossed), marker="v", color=color, s=18)
    if censored:
        ax.scatter(
            censored,
            [y_floor] * len(censored),
            marker="v",
            facecolors="none",
            edgecolors=color,
            s=28,
        )


def _parity_text(curves: dict[tuple[str, float], list[dict[str, float]]]) -> str | None:
    differences: list[float] = []
    for (prefix, offset), gpu_points in curves.items():
        if prefix != "gpu":
            continue
        packed = curves.get(("cpu_packed", offset), [])
        packed_by_x = {float(point["x"]): point for point in packed}
        for point in gpu_points:
            other = packed_by_x.get(float(point["x"]))
            if other is not None:
                differences.append(abs(float(point["excess"]) - float(other["excess"])))
    if not differences:
        return None
    maximum = max(differences)
    if maximum == 0.0:
        return "Packed CPU and GPU agree exactly"
    return rf"Packed CPU/GPU max $|\Delta Q|={maximum:.3g}$"


def _y_axis_upper(x_max: float, plotted_y: Iterable[float]) -> float:
    baseline = max(min(float(x_max) + 1.0, 1.0), -2.0)
    finite = [float(value) for value in plotted_y if math.isfinite(float(value))]
    return max(baseline, max(finite, default=-math.inf) + 1.0)


def plot_summary(
    *,
    input_csv: Path | Sequence[Path],
    output_png: Path,
    output_pdf: Path | None = None,
    title: str,
    trial_csv: Path | Sequence[Path] | None = None,
    conditioning_trial_csv: Path | Sequence[Path] | None = None,
    conditioning_weights_path: Path | None = None,
    conditioning_weights_sha256: str | None = None,
    conditioning_weight_manifest_sha256: str | None = None,
    conditioning_json: Path | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    y_min_db: float | None = None,
    dissertation_style: bool = False,
    show: bool = False,
) -> Path:
    """Render a pooled estimator transfer plot."""
    if int(smooth_window) < MIN_SMOOTH_WINDOW:
        raise ValueError(
            f"smooth_window must be >= {MIN_SMOOTH_WINDOW}; got {smooth_window}."
        )
    if int(bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    if y_min_db is not None and not math.isfinite(float(y_min_db)):
        raise ValueError("y_min_db must be finite.")
    smooth_window = int(smooth_window)

    try:
        plt = setup_matplotlib(
            force_agg=not bool(show),
            dissertation_style=bool(dissertation_style),
        )
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install the optional plot "
            "dependency, for example: python -m pip install matplotlib"
        ) from exc
    if dissertation_style:
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"
            r"\usepackage{amsmath}"
        )

    summary_paths = _as_paths(input_csv)
    rows = _read_summary_rows(summary_paths)
    trial_paths = (
        _default_trial_paths(summary_paths)
        if trial_csv is None
        else _as_paths(trial_csv)
    )
    trial_rows = _read_trial_rows(trial_paths) if trial_paths else []
    radio, input_calibration, output_calibration = _plot_calibrations(
        rows, summary_paths
    )
    conditioning_paths = (
        []
        if conditioning_trial_csv is None
        else _as_paths(conditioning_trial_csv)
    )
    conditioning_rows = (
        _read_trial_rows(conditioning_paths) if conditioning_paths else []
    )
    conditioned = _try_conditioned_transfer(
        summary_paths=summary_paths,
        conditioning_paths=conditioning_paths,
        rows=conditioning_rows,
        calibration=output_calibration,
        conditioning_weights_path=conditioning_weights_path,
        conditioning_weights_sha256=conditioning_weights_sha256,
        conditioning_weight_manifest_sha256=conditioning_weight_manifest_sha256,
    )

    requested_values = [
        _summary_input_snr(row, radio=radio)
        for row in rows
        if math.isfinite(_summary_input_snr(row, radio=radio))
    ]
    if not requested_values:
        raise ValueError("No finite requested SNR values found in summary CSV.")

    x_min = min(requested_values)
    x_max = max(requested_values)
    y_floor = x_min - 4.0 if y_min_db is None else float(y_min_db)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    theory_x = np.linspace(x_min, x_max, 600)
    if conditioned is not None:
        ax.plot(
            theory_x,
            _conditioned_transfer_db(theory_x, conditioned),
            linestyle="--",
            color="black",
            linewidth=REFERENCE_LINE_WIDTH,
            label=CONDITIONED_TRANSFER_LABEL,
        )
    if radio:
        for offset in _frequency_offsets(rows):
            _plot_control_expected(
                ax,
                rows,
                offset=offset,
                y_floor=y_floor,
            )
    ax.plot(
        theory_x,
        _reference_transfer_db(theory_x),
        linestyle=":",
        color="0.35",
        linewidth=BENCHMARK_LINE_WIDTH,
        alpha=0.4,
        label=IDEAL_BENCHMARK_LABEL,
    )

    curves: dict[tuple[str, float], list[dict[str, float]]] = {}
    pass_intervals = bool(trial_rows) and all(
        math.isfinite(_finite(row, "pass_index")) for row in trial_rows
    )
    percent = r"95\%" if dissertation_style else "95%"
    interval_label = (
        f"{percent} pass-bootstrap CI"
        if pass_intervals
        else f"{percent} bootstrap CI"
    )
    for offset in _frequency_offsets(rows):
        for spec in CURVE_SPECS:
            points = _curve_points(
                rows,
                trial_rows,
                prefix=spec.prefix,
                offset=offset,
                calibration=output_calibration,
                bootstrap_samples=int(bootstrap_samples),
                radio=radio,
            )
            if not points:
                continue
            curves[(spec.prefix, offset)] = points
            finite = [point for point in points if math.isfinite(float(point["y"]))]
            if finite:
                x_values = [float(point["x"]) for point in finite]
                raw_y = [float(point["y"]) for point in finite]
                line = ax.plot(
                    x_values,
                    _centered_moving_average(raw_y, smooth_window),
                    linestyle=spec.linestyle,
                    marker=spec.marker,
                    markersize=MARKER_SIZE,
                    linewidth=RESULT_LINE_WIDTH,
                    label=_curve_label(spec.label, offset, smooth_window),
                )[0]
                if spec.prefix != "cpu_packed":
                    _plot_intervals(
                        ax,
                        points,
                        color=line.get_color(),
                        y_floor=y_floor,
                        calibration=output_calibration,
                        label=interval_label,
                    )
                    interval_label = None
            else:
                ax.scatter(
                    [float(point["x"]) for point in points],
                    [y_floor] * len(points),
                    marker="v",
                    facecolors="none",
                    label=_curve_label(spec.label, offset, smooth_window),
                )

    if not curves:
        raise ValueError(
            "No pooled linear estimator fields were found. Provide the raw trial CSV "
            "or a summary containing pooled normalized ratios."
        )

    parity = _parity_text(curves)
    if parity:
        ax.text(
            0.98,
            0.02,
            parity,
            transform=ax.transAxes,
            fontsize="small",
            ha="right",
            va="bottom",
        )
    ax.set_title(title)
    if radio:
        ax.set_xlabel(
            r"Received input data-shelf SNR, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
        )
    else:
        ax.set_xlabel(
            r"Known data-shelf SNR, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
        )
    ax.set_ylabel(
        r"Estimated data-shelf SNR, $\widehat{\mathrm{SNR}}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
    )
    ax.set_xlim(x_min - 0.6, x_max + 0.6)
    plotted_y = (
        value
        for line in ax.get_lines()
        for value in np.asarray(line.get_ydata(), dtype=float).ravel()
    )
    y_upper = _y_axis_upper(x_max, plotted_y)
    if y_floor >= y_upper:
        raise ValueError("y_min_db must be below the plotted data.")
    ax.set_ylim(y_floor, y_upper)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    _add_pilot_axis(
        ax,
        orientation="x",
        calibration=input_calibration,
        received_input=radio,
    )
    _add_pilot_axis(
        ax,
        orientation="y",
        calibration=output_calibration,
    )
    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=DEFAULT_PLOT_DPI)
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_pdf,
            metadata={
                "Creator": "PilotProxy",
                "CreationDate": PUBLICATION_DATE,
                "ModDate": PUBLICATION_DATE,
            },
        )
    if conditioning_json is not None and conditioned is not None:
        _write_conditioning_record(
            Path(conditioning_json), conditioned, conditioning_paths
        )
    if show:
        plt.show()
    plt.close(fig)
    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pooled threshold-free estimator transfer results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        action="append",
        default=None,
        help="Summary CSV. Repeat for disjoint run shards.",
    )
    parser.add_argument(
        "--trial-csv",
        type=Path,
        action="append",
        default=None,
        help="Raw trial CSV. Repeat for disjoint run shards.",
    )
    parser.add_argument(
        "--conditioning-trial-csv",
        type=Path,
        action="append",
        default=None,
        help="Lower raw trial CSV used only to derive the expected transfer.",
    )
    parser.add_argument(
        "--conditioning-json",
        type=Path,
        default=None,
        help="Write the derived transfer coefficients and provenance.",
    )
    parser.add_argument(
        "--conditioning-weights-path",
        type=Path,
        default=None,
        help="Explicit weight bank used for conditioning.",
    )
    parser.add_argument(
        "--conditioning-weights-sha256",
        default=None,
        help="Expected weight-bank SHA256 for legacy conditioning metadata.",
    )
    parser.add_argument(
        "--conditioning-weight-manifest-sha256",
        default=None,
        help="Expected adjacent weight-manifest SHA256 for legacy metadata.",
    )
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--title", default="PilotProxy estimator transfer")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Centered display smoothing across adjacent SNR points.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples; pass clusters are used when available.",
    )
    parser.add_argument(
        "--y-min-db",
        type=float,
        default=None,
        help="Optional lower output-axis limit in dB.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--dissertation-style",
        action="store_true",
        help="Use embedded Latin Modern/T1 text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = plot_summary(
        input_csv=args.input_csv or [DEFAULT_INPUT_CSV],
        trial_csv=args.trial_csv,
        conditioning_trial_csv=args.conditioning_trial_csv,
        conditioning_weights_path=args.conditioning_weights_path,
        conditioning_weights_sha256=args.conditioning_weights_sha256,
        conditioning_weight_manifest_sha256=(
            args.conditioning_weight_manifest_sha256
        ),
        conditioning_json=args.conditioning_json,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        title=str(args.title),
        smooth_window=int(args.smooth_window),
        bootstrap_samples=int(args.bootstrap_samples),
        y_min_db=args.y_min_db,
        dissertation_style=bool(args.dissertation_style),
        show=bool(args.show),
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
