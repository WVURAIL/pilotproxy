#!/usr/bin/env python3
# coding=utf-8
"""Run a guarded two-antenna LimeSDR transfer sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_geometry import flatten_feed_channel_streams
from pilot_proxy.detector_contract import weight_term_norms_sq
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import (
    DB_POWER_FACTOR,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_CAPTURE_EFFICIENCY,
    composite_to_data_shelf_snr_correction_db,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
)
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.integration.packing import estimate_complex_scale
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH
from pilot_proxy.reference_channelizer import (
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    channelize_real_blocks_to_reference_channels_gpu,
    complex_envelope_to_real_adc_blocks_gpu,
    nearest_reference_channel_index,
    sinc_hamming_pfb_response,
)
from pilot_proxy.testbench.evaluate_snr import (
    _kernel_measurements,
    _pack_streams_for_kernel,
    required_iq_samples,
)
from pilot_proxy.testbench.generate_atsc_signal import (
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
)
from pilot_proxy.testbench.quantize import (
    DEFAULT_FRAME_SIZE_SAMPLES,
    LOCKED_BITS_PER_COMPONENT,
    LOCKED_DETECTOR_WINDOW_SAMPLES,
)

UTC = timezone.utc
SCHEMA_VERSION = "sdr_transfer_v1"
STREAM_STATUS_SCHEMA = "sdr_stream_v1"
RADIO_DRIVER = "limesuite_direct"
STREAM_WORKER_SOURCE = REPO_ROOT / "tools" / "sdr_stream_worker.cpp"
LIME_NATIVE_GAIN_OFFSET_DB = 12
DEFAULT_SEED = 20260824
DEFAULT_PASSES = 3
DEFAULT_DRIFT_SNR_DB = -18.0
DEFAULT_DRIFT_INTERVAL = 100
DEFAULT_SETTLE_SAMPLES = 65_536
DEFAULT_CAPTURE_GUARD_SAMPLES = 16_384
DEFAULT_SYNC_MARKER_SAMPLES = 16_384
DEFAULT_MARKER_SEARCH_SAMPLES = 8_192
DEFAULT_SESSION_GUARD_SAMPLES = 1_048_576
DEFAULT_PILOT_BELOW_DATA_DB = 11.918446870168612
DEFAULT_WORKER_TIMEOUT_SECONDS = 300.0
DEFAULT_STREAM_BUFFER_SAMPLES = 8_388_608
DEFAULT_STREAM_START_DELAY_SECONDS = 0.5
DEFAULT_RADIO_FILTER_BANDWIDTH_HZ = 8_000_000.0
SNR_VALUES_DB = tuple(float(value) for value in range(-42, 1, 3))
LOW_SNR_MAX_DB = -30.0
LOW_SNR_TRIALS = 240
HIGH_SNR_TRIALS = 60
CONTROL_KINDS = ("tx_off", "tx_zero", "signal_only", "noise_only")
TRANSMIT_KINDS = ("tx_zero", "signal_only", "noise_only", "mixture", "drift")
MAX_DIGITAL_RMS = 0.1
MAX_TX_COMPONENT = 0.95
FREQUENCY_TOLERANCE_HZ = 100.0
TV_TX_ANTENNA = "BAND2"
TV_RX_ANTENNA = "LNAW"
ALIGNMENT_BLOCK_SAMPLES = 4_096
MAX_MARKER_SEPARATION_ERROR_SAMPLES = 64
MIN_MARKER_CORRELATION = 0.15
SPECTRUM_SEGMENT_SAMPLES = 16_384
PILOT_WINDOW_HALF_WIDTH_HZ = 10_000.0
PILOT_SHELF_INNER_HZ = 30_000.0
PILOT_SHELF_OUTER_HZ = 150_000.0
DEFAULT_MAX_RECEIVED_PILOT_RATIO_ERROR_DB = 1.5
DEFAULT_QUANTIZATION_CLIP_FRACTION_LIMIT = 0.01


@dataclass(frozen=True)
class SweepEvent:
    event_index: int
    pass_index: int
    kind: str
    snr_db: float | None
    trial_index: int | None
    seed: int
    note: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def trials_for_snr(snr_db: float) -> int:
    return LOW_SNR_TRIALS if float(snr_db) <= LOW_SNR_MAX_DB else HIGH_SNR_TRIALS


def build_schedule(
    *,
    seed: int = DEFAULT_SEED,
    passes: int = DEFAULT_PASSES,
    drift_snr_db: float = DEFAULT_DRIFT_SNR_DB,
    drift_interval: int = DEFAULT_DRIFT_INTERVAL,
) -> list[SweepEvent]:
    """Build a pinned, randomized schedule with controls and sentinels."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    if drift_interval <= 0:
        raise ValueError("drift_interval must be positive")
    if float(drift_snr_db) not in SNR_VALUES_DB:
        raise ValueError("drift_snr_db must be on the sweep grid")

    rng = np.random.default_rng(int(seed))
    pending: list[dict[str, Any]] = []
    for pass_index in range(1, int(passes) + 1):
        for kind in CONTROL_KINDS:
            pending.append(
                {
                    "pass_index": pass_index,
                    "kind": kind,
                    "snr_db": None,
                    "trial_index": None,
                    "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
                    "note": "control",
                }
            )

        pending.append(
            {
                "pass_index": pass_index,
                "kind": "drift",
                "snr_db": float(drift_snr_db),
                "trial_index": None,
                "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
                "note": "pass_start",
            }
        )

        mixture: list[dict[str, Any]] = []
        for snr_db in SNR_VALUES_DB:
            trial_indices = list(range(trials_for_snr(snr_db)))[pass_index - 1 :: passes]
            for trial_index in trial_indices:
                mixture.append(
                    {
                        "pass_index": pass_index,
                        "kind": "mixture",
                        "snr_db": float(snr_db),
                        "trial_index": int(trial_index),
                        "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
                        "note": "sweep",
                    }
                )
        rng.shuffle(mixture)
        for offset, item in enumerate(mixture, start=1):
            pending.append(item)
            if offset % int(drift_interval) == 0 and offset != len(mixture):
                pending.append(
                    {
                        "pass_index": pass_index,
                        "kind": "drift",
                        "snr_db": float(drift_snr_db),
                        "trial_index": None,
                        "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
                        "note": "periodic",
                    }
                )
        pending.append(
            {
                "pass_index": pass_index,
                "kind": "drift",
                "snr_db": float(drift_snr_db),
                "trial_index": None,
                "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
                "note": "pass_end",
            }
        )

    return [
        SweepEvent(event_index=index, **item)
        for index, item in enumerate(pending, start=1)
    ]


def _unit_power(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.complex64).reshape(-1)
    power = float(np.mean(np.abs(data) ** 2))
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("waveform power must be positive and finite")
    return np.asarray(data / math.sqrt(power), dtype=np.complex64)


def _unit_noise(
    signal: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size)
    noise = np.asarray(noise, dtype=np.complex128)
    signal_128 = np.asarray(signal, dtype=np.complex128)
    projection = np.vdot(signal_128, noise) / np.vdot(signal_128, signal_128)
    noise -= projection * signal_128
    return _unit_power(noise)


def make_tx_waveform(
    clean_iq: np.ndarray,
    *,
    kind: str,
    snr_db: float | None,
    target_rms: float,
    sample_rate_hz: float,
    bandwidth_hz: float,
    seed: int,
    pilot_below_data_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mix one constant-level transmit waveform."""
    if kind not in (*CONTROL_KINDS, "mixture", "drift"):
        raise ValueError(f"unknown event kind: {kind}")
    if not 0.0 < float(target_rms) <= MAX_DIGITAL_RMS:
        raise ValueError(f"target_rms must be in (0, {MAX_DIGITAL_RMS}]")
    if sample_rate_hz <= 0.0 or bandwidth_hz <= 0.0:
        raise ValueError("sample rate and bandwidth must be positive")
    if bandwidth_hz > sample_rate_hz:
        raise ValueError("bandwidth must not exceed sample rate")

    clean = _unit_power(clean_iq)
    rng = np.random.default_rng(int(seed))
    noise = _unit_noise(clean, rng=rng)
    total_power = float(target_rms) ** 2
    signal_power = 0.0
    noise_power = 0.0

    if kind in ("tx_off", "tx_zero"):
        mixed = np.zeros_like(clean)
    elif kind == "signal_only":
        signal_power = total_power
        mixed = clean * np.float32(math.sqrt(signal_power))
    elif kind == "noise_only":
        noise_power = total_power
        mixed = noise * np.float32(math.sqrt(noise_power))
    else:
        if snr_db is None:
            raise ValueError("mixture events require snr_db")
        correction_db = composite_to_data_shelf_snr_correction_db(
            pilot_below_data_db=float(pilot_below_data_db)
        )
        composite_snr_db = float(snr_db) - float(correction_db)
        in_band_ratio = 10.0 ** (composite_snr_db / 10.0)
        noise_to_signal = (float(sample_rate_hz) / float(bandwidth_hz)) / in_band_ratio
        signal_power = total_power / (1.0 + noise_to_signal)
        noise_power = total_power - signal_power
        mixed = (
            clean * np.float32(math.sqrt(signal_power))
            + noise * np.float32(math.sqrt(noise_power))
        )

    mixed = np.asarray(mixed, dtype=np.complex64)
    measured_power = float(np.mean(np.abs(mixed) ** 2))
    measured_rms = math.sqrt(measured_power)
    actual_composite_snr_db: float | None = None
    actual_shelf_snr_db: float | None = None
    if signal_power > 0.0 and noise_power > 0.0:
        in_band_noise_power = noise_power * float(bandwidth_hz) / float(sample_rate_hz)
        actual_composite_snr_db = 10.0 * math.log10(signal_power / in_band_noise_power)
        actual_shelf_snr_db = actual_composite_snr_db + float(
            composite_to_data_shelf_snr_correction_db(
                pilot_below_data_db=float(pilot_below_data_db)
            )
        )
    metadata = {
        "kind": kind,
        "requested_data_shelf_snr_db": snr_db,
        "actual_data_shelf_snr_db": actual_shelf_snr_db,
        "actual_composite_snr_db": actual_composite_snr_db,
        "target_rms": float(target_rms),
        "measured_rms": float(measured_rms),
        "signal_power": float(signal_power),
        "noise_power": float(noise_power),
        "sample_rate_hz": float(sample_rate_hz),
        "bandwidth_hz": float(bandwidth_hz),
        "seed": int(seed),
    }
    return mixed, metadata


def capture_level_stats(
    iq: np.ndarray,
    *,
    clip_level: float,
    sample_rate_hz: float | None = None,
    bandwidth_hz: float | None = None,
) -> dict[str, float | int]:
    data = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if data.size == 0:
        raise ValueError("capture is empty")
    components = np.concatenate((np.abs(data.real), np.abs(data.imag)))
    stats: dict[str, float | int] = {
        "samples": int(data.size),
        "rms": float(np.sqrt(np.mean(np.abs(data) ** 2))),
        "peak_component": float(np.max(components)),
        "clip_fraction": float(np.mean(components >= float(clip_level))),
    }
    if sample_rate_hz is not None and bandwidth_hz is not None:
        if not 0.0 < float(bandwidth_hz) <= float(sample_rate_hz):
            raise ValueError("bandwidth must be in (0, sample_rate]")
        spectrum = np.fft.fft(data)
        frequencies = np.fft.fftfreq(data.size, d=1.0 / float(sample_rate_hz))
        keep = np.abs(frequencies) <= float(bandwidth_hz) / 2.0
        stats["in_band_power"] = float(
            np.sum(np.abs(spectrum[keep]) ** 2) / float(data.size * data.size)
        )
    return stats


def averaged_complex_psd(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    segment_samples: int = SPECTRUM_SEGMENT_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(iq, dtype=np.complex64).reshape(-1)
    segment_samples = int(segment_samples)
    count = data.size // segment_samples
    if count < 2:
        raise ValueError("capture is too short for spectral calibration")
    blocks = data[: count * segment_samples].reshape(count, segment_samples)
    window = np.hanning(segment_samples).astype(np.float64)
    spectra = np.fft.fftshift(
        np.fft.fft(blocks * window[np.newaxis, :], axis=1),
        axes=1,
    )
    psd = np.mean(np.abs(spectra) ** 2, axis=0) / (
        float(sample_rate_hz) * float(np.sum(window**2))
    )
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(segment_samples, d=1.0 / float(sample_rate_hz))
    )
    return frequencies, np.asarray(psd, dtype=np.float64)


def calibrate_received_controls(
    *,
    tx_zero: np.ndarray,
    signal_only: np.ndarray,
    noise_only: np.ndarray,
    sample_rate_hz: float,
    bandwidth_hz: float,
    expected_pilot_below_data_db: float,
) -> dict[str, float]:
    """Separate received data, pilot, leakage, and injected noise."""
    frequency, zero_psd = averaged_complex_psd(
        tx_zero,
        sample_rate_hz=float(sample_rate_hz),
    )
    signal_frequency, signal_psd = averaged_complex_psd(
        signal_only,
        sample_rate_hz=float(sample_rate_hz),
    )
    noise_frequency, noise_psd = averaged_complex_psd(
        noise_only,
        sample_rate_hz=float(sample_rate_hz),
    )
    if not np.array_equal(frequency, signal_frequency) or not np.array_equal(
        frequency,
        noise_frequency,
    ):
        raise RuntimeError("Control spectra do not share one frequency grid.")
    signal_excess = np.maximum(signal_psd - zero_psd, 0.0)
    noise_excess = np.maximum(noise_psd - zero_psd, 0.0)
    half_band = float(bandwidth_hz) / 2.0
    in_band = np.abs(frequency) <= half_band
    pilot_hz = -ATSC_CHANNEL_WIDTH_HZ / 2.0 + ATSC_PILOT_OFFSET_HZ
    pilot_distance = np.abs(frequency - pilot_hz)
    pilot_window = pilot_distance <= PILOT_WINDOW_HALF_WIDTH_HZ
    pilot_shelf = (
        (pilot_distance >= PILOT_SHELF_INNER_HZ)
        & (pilot_distance <= PILOT_SHELF_OUTER_HZ)
        & in_band
    )
    if np.count_nonzero(pilot_window) < 2 or np.count_nonzero(pilot_shelf) < 10:
        raise RuntimeError("Control spectrum cannot resolve the pilot shelf.")
    bin_width_hz = float(sample_rate_hz) / float(frequency.size)
    baseline_power = float(np.sum(zero_psd[in_band]) * bin_width_hz)
    composite_power = float(np.sum(signal_excess[in_band]) * bin_width_hz)
    injected_noise_power = float(np.sum(noise_excess[in_band]) * bin_width_hz)
    local_shelf_psd = float(np.median(signal_excess[pilot_shelf]))
    pilot_window_power = float(
        np.sum(signal_excess[pilot_window]) * bin_width_hz
    )
    pilot_baseline_power = float(
        local_shelf_psd * np.count_nonzero(pilot_window) * bin_width_hz
    )
    pilot_excess_power = max(0.0, pilot_window_power - pilot_baseline_power)
    data_shelf_power = composite_power - pilot_excess_power
    if (
        baseline_power < 0.0
        or data_shelf_power <= 0.0
        or injected_noise_power <= 0.0
        or pilot_excess_power <= 0.0
    ):
        raise RuntimeError("Control spectra do not resolve positive path powers.")
    received_pilot_below_data_db = -DB_POWER_FACTOR * math.log10(
        pilot_excess_power / data_shelf_power
    )
    return {
        "receiver_and_leakage_power": baseline_power,
        "signal_composite_power": composite_power,
        "signal_data_shelf_power": data_shelf_power,
        "signal_pilot_excess_power": pilot_excess_power,
        "injected_noise_power": injected_noise_power,
        "received_pilot_below_data_db": received_pilot_below_data_db,
        "pilot_ratio_error_db": (
            received_pilot_below_data_db - float(expected_pilot_below_data_db)
        ),
        "psd_bin_width_hz": bin_width_hz,
    }


def event_slot_samples(args: argparse.Namespace) -> int:
    return (
        int(args.sync_marker_samples)
        + int(args.capture_guard_samples)
        + int(args.settle_samples)
        + int(args.capture_samples)
    )


def make_sync_marker(
    *,
    samples: int,
    sample_rate_hz: float,
    target_rms: float,
) -> np.ndarray:
    """Build a tapered in-band chirp for slot alignment."""
    count = int(samples)
    if count < 256:
        raise ValueError("sync marker must contain at least 256 samples")
    time = np.arange(count, dtype=np.float64) / float(sample_rate_hz)
    duration = count / float(sample_rate_hz)
    lower_hz = -2.25e6
    upper_hz = 2.25e6
    sweep_rate = (upper_hz - lower_hz) / duration
    phase = 2.0 * math.pi * (lower_hz * time + 0.5 * sweep_rate * time**2)
    taper = np.sin(math.pi * (np.arange(count) + 0.5) / count) ** 2
    marker = taper * np.exp(1j * phase)
    return np.asarray(
        _unit_power(marker) * np.float32(float(target_rms)),
        dtype=np.complex64,
    )


def _normalized_marker_match(
    values: np.ndarray,
    marker: np.ndarray,
) -> tuple[int, float]:
    data = np.asarray(values, dtype=np.complex64).reshape(-1)
    template = np.asarray(marker, dtype=np.complex64).reshape(-1)
    if data.size < template.size:
        raise RuntimeError("Marker search window is too short.")
    full_size = data.size + template.size - 1
    fft_size = 1 << (full_size - 1).bit_length()
    kernel = np.conjugate(template[::-1])
    correlation = np.fft.ifft(
        np.fft.fft(data, fft_size) * np.fft.fft(kernel, fft_size)
    )
    valid = correlation[template.size - 1 : data.size]
    power = np.abs(data.astype(np.complex128)) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    window_power = cumulative[template.size :] - cumulative[: -template.size]
    template_power = float(np.sum(np.abs(template) ** 2))
    denominator = np.sqrt(np.maximum(window_power * template_power, 0.0))
    score = np.divide(
        np.abs(valid),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    index = int(np.argmax(score))
    return index, float(score[index])


def align_session_slots(
    capture: np.ndarray,
    *,
    slots: Sequence[Mapping[str, Any]],
    marker: np.ndarray,
    initial_search_samples: int,
    local_search_samples: int,
) -> list[dict[str, float | int]]:
    """Locate every marker and reject stream slips."""
    data = np.asarray(capture, dtype=np.complex64).reshape(-1)
    aligned: list[dict[str, float | int]] = []
    stream_offset: int | None = None
    previous_expected: int | None = None
    previous_observed: int | None = None
    for slot in slots:
        expected = int(slot["marker_start_sample"])
        if stream_offset is None:
            center = expected
            radius = int(initial_search_samples)
        else:
            center = expected + stream_offset
            radius = int(local_search_samples)
        start = max(0, center - radius)
        stop = min(data.size, center + radius + marker.size)
        relative, score = _normalized_marker_match(data[start:stop], marker)
        observed = start + relative
        if score < MIN_MARKER_CORRELATION:
            raise RuntimeError(
                f"Marker correlation {score:.3f} is below the session limit "
                f"at event {int(slot['event_index'])}."
            )
        if stream_offset is None:
            stream_offset = observed - expected
        separation_error = 0
        if previous_expected is not None and previous_observed is not None:
            separation_error = (observed - previous_observed) - (
                expected - previous_expected
            )
            if abs(separation_error) > MAX_MARKER_SEPARATION_ERROR_SAMPLES:
                raise RuntimeError(
                    "Session marker spacing shows a sample slip at event "
                    f"{int(slot['event_index'])}: {separation_error:+d} samples."
                )
        aligned.append(
            {
                "event_index": int(slot["event_index"]),
                "expected_marker_sample": expected,
                "observed_marker_sample": observed,
                "stream_offset_samples": observed - expected,
                "separation_error_samples": separation_error,
                "correlation": score,
            }
        )
        previous_expected = expected
        previous_observed = observed
    return aligned


def calibrated_received_shelf_snr(
    *,
    tx_metadata: Mapping[str, Any],
    control_calibration: Mapping[str, float],
) -> dict[str, float]:
    """Calibrate input SNR from the received control spectra."""
    required = (
        "receiver_and_leakage_power",
        "signal_composite_power",
        "signal_data_shelf_power",
        "injected_noise_power",
    )
    missing = [name for name in required if name not in control_calibration]
    if missing:
        raise RuntimeError("Missing pass controls: " + ", ".join(missing))

    off_power = float(control_calibration["receiver_and_leakage_power"])
    signal_path_power = float(control_calibration["signal_composite_power"])
    data_path_power = float(control_calibration["signal_data_shelf_power"])
    noise_path_power = float(control_calibration["injected_noise_power"])
    if (
        off_power <= 0.0
        or signal_path_power <= 0.0
        or data_path_power <= 0.0
        or noise_path_power <= 0.0
    ):
        raise RuntimeError("Pass controls do not resolve signal and noise power.")

    tx_signal_power = float(tx_metadata["signal_power"])
    tx_noise_power = float(tx_metadata["noise_power"])
    tx_total_power = tx_signal_power + tx_noise_power
    if tx_total_power <= 0.0:
        raise RuntimeError("Transmit mixture has no power.")
    signal_fraction = tx_signal_power / tx_total_power
    noise_fraction = tx_noise_power / tx_total_power
    received_composite_power = signal_fraction * signal_path_power
    received_noise_power = off_power + noise_fraction * noise_path_power
    data_power = signal_fraction * data_path_power
    if data_power <= 0.0 or received_noise_power <= 0.0:
        raise RuntimeError("Calibrated received powers must be positive.")
    snr_linear = data_power / received_noise_power
    result = {
        "data_shelf_snr_linear": float(snr_linear),
        "data_shelf_snr_db": float(DB_POWER_FACTOR * math.log10(snr_linear)),
        "receiver_noise_power": off_power,
        "signal_path_power": signal_path_power,
        "noise_path_power": noise_path_power,
        "data_shelf_path_power": data_path_power,
        "received_composite_power": received_composite_power,
        "received_noise_power": received_noise_power,
        "received_data_shelf_power": data_power,
    }
    if "received_pilot_below_data_db" in control_calibration:
        result["received_pilot_below_data_db"] = float(
            control_calibration["received_pilot_below_data_db"]
        )
    return result


def expected_center_frequency_hz(physical_channel: int) -> float:
    pilot_hz = physical_channel_to_pilot_hz(int(physical_channel))
    return float(
        pilot_hz + ATSC_CHANNEL_WIDTH_HZ / 2.0 - ATSC_PILOT_OFFSET_HZ
    )


def resolve_pilot_calibration(args: argparse.Namespace) -> None:
    path = args.waveform_audit_json
    if path is None:
        if args.transmit:
            raise SystemExit("Transmit mode requires --waveform-audit-json.")
        if args.pilot_below_data_db is None:
            args.pilot_below_data_db = DEFAULT_PILOT_BELOW_DATA_DB
        args.pilot_calibration_source = "default"
        return
    audit_path = Path(path).resolve()
    if not audit_path.is_file():
        raise SystemExit(f"Waveform audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "pilotproxy_atsc_waveform_audit_v1":
        raise SystemExit("Waveform audit has the wrong schema.")
    if not bool(audit.get("quality_passed")):
        raise SystemExit("Waveform audit did not pass its quality checks.")
    audited_input = audit.get("input_iq")
    if not isinstance(audited_input, str) or not audited_input.strip():
        raise SystemExit("Waveform audit does not identify its input waveform.")
    audited_path = Path(audited_input)
    if not audited_path.is_absolute():
        audited_path = (Path.cwd() / audited_path).resolve()
    else:
        audited_path = audited_path.resolve()
    if audited_path != Path(args.input_iq).resolve():
        raise SystemExit("Waveform audit does not match --input-iq.")
    if not math.isclose(
        float(audit.get("sample_rate_hz", math.nan)),
        float(args.sample_rate_hz),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise SystemExit("Waveform audit sample rate does not match the run.")
    measured = float(audit.get("measured_pilot_below_data_db", math.nan))
    if not math.isfinite(measured) or measured <= 0.0:
        raise SystemExit("Waveform audit has no valid pilot calibration.")
    if args.pilot_below_data_db is not None and not math.isclose(
        float(args.pilot_below_data_db),
        measured,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise SystemExit("Requested pilot calibration does not match the audit.")
    args.pilot_below_data_db = measured
    args.pilot_calibration_source = str(audit_path)


def validate_transmit_args(args: argparse.Namespace) -> None:
    if not bool(args.transmit):
        return
    required = {
        "--frequency-hz": args.frequency_hz,
        "--physical-channel": args.physical_channel,
        "--tx-gain-db": args.tx_gain_db,
        "--rx-gain-db": args.rx_gain_db,
        "--tx-rms": args.tx_rms,
        "--device-serial": args.device_serial,
        "--tx-antenna": args.tx_antenna,
        "--rx-antenna": args.rx_antenna,
        "--antenna-separation-cm": args.antenna_separation_cm,
        "--stream-worker": args.stream_worker,
        "--limesuite-library": args.limesuite_library,
        "--worker-python": args.worker_python,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("Transmit mode requires " + ", ".join(missing) + ".")
    if not bool(args.rf_authorized):
        raise SystemExit("Transmit mode requires --rf-authorized.")
    if not -12.0 <= float(args.tx_gain_db) <= 61.0:
        raise SystemExit("--tx-gain-db must be in [-12, 61].")
    if not -12.0 <= float(args.rx_gain_db) <= 61.0:
        raise SystemExit("--rx-gain-db must be in [-12, 61].")
    if not float(args.tx_gain_db).is_integer() or not float(args.rx_gain_db).is_integer():
        raise SystemExit("LimeSDR gains must use whole decibels.")
    if not 0.0 < float(args.tx_rms) <= MAX_DIGITAL_RMS:
        raise SystemExit(f"--tx-rms must be in (0, {MAX_DIGITAL_RMS}].")
    if float(args.antenna_separation_cm) <= 0.0:
        raise SystemExit("--antenna-separation-cm must be positive.")
    if str(args.tx_antenna) != TV_TX_ANTENNA:
        raise SystemExit("--tx-antenna must be BAND2 for this TV-band sweep.")
    if str(args.rx_antenna) != TV_RX_ANTENNA:
        raise SystemExit("--rx-antenna must be LNAW for this TV-band sweep.")
    if not math.isclose(
        float(args.bandwidth_hz),
        DTV_BANDWIDTH_HZ,
        rel_tol=0.0,
        abs_tol=1.0,
    ):
        raise SystemExit("--bandwidth-hz must remain 6000000 for this sweep.")
    if not float(args.bandwidth_hz) < float(args.radio_filter_bandwidth_hz) < float(
        args.sample_rate_hz
    ):
        raise SystemExit(
            "--radio-filter-bandwidth-hz must be between the signal and sample rates."
        )
    expected_hz = expected_center_frequency_hz(int(args.physical_channel))
    error_hz = abs(float(args.frequency_hz) - expected_hz)
    if error_hz > FREQUENCY_TOLERANCE_HZ:
        raise SystemExit(
            "--frequency-hz does not match the selected physical channel: "
            f"expected {expected_hz:.3f} Hz, got {float(args.frequency_hz):.3f} Hz."
        )
    if int(args.frame_size_samples) % LOCKED_DETECTOR_WINDOW_SAMPLES != 0:
        raise SystemExit(
            "--frame-size-samples must be a multiple of "
            f"{LOCKED_DETECTOR_WINDOW_SAMPLES}."
        )
    required_capture = required_iq_samples(
        iq_sample_rate_hz=float(args.sample_rate_hz),
        adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        num_output_samples=int(args.frame_size_samples),
    )
    if int(args.capture_samples) < required_capture:
        raise SystemExit(
            f"--capture-samples must be at least {required_capture} for this frame."
        )
    if int(args.capture_samples) < 2 * SPECTRUM_SEGMENT_SAMPLES:
        raise SystemExit(
            "--capture-samples is too short for received spectral calibration."
        )
    if int(args.capture_guard_samples) < ALIGNMENT_BLOCK_SAMPLES:
        raise SystemExit(
            f"--capture-guard-samples must be at least {ALIGNMENT_BLOCK_SAMPLES}."
        )
    if int(args.sync_marker_samples) < 256:
        raise SystemExit("--sync-marker-samples must be at least 256.")
    if int(args.marker_search_samples) < MAX_MARKER_SEPARATION_ERROR_SAMPLES:
        raise SystemExit("--marker-search-samples is too small.")
    if int(args.session_guard_samples) < int(args.settle_samples):
        raise SystemExit(
            "--session-guard-samples must be at least --settle-samples."
        )
    if float(args.worker_timeout_seconds) < 60.0:
        raise SystemExit("--worker-timeout-seconds must be at least 60.")
    if not 1_048_576 <= int(args.stream_buffer_samples) <= 67_108_864:
        raise SystemExit("--stream-buffer-samples must be in [1048576, 67108864].")
    worker = Path(args.stream_worker).resolve()
    library = Path(args.limesuite_library).resolve()
    worker_python = Path(args.worker_python).resolve()
    if not worker.is_file() or not os.access(worker, os.X_OK):
        raise SystemExit("--stream-worker must be an executable file.")
    if not library.is_file():
        raise SystemExit("--limesuite-library must be a regular file.")
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise SystemExit("--worker-python must be an executable file.")
    if not STREAM_WORKER_SOURCE.is_file():
        raise SystemExit("The stream worker source is missing.")


def native_lime_gain_db(gain_db: float) -> int:
    return int(round(float(gain_db))) + LIME_NATIVE_GAIN_OFFSET_DB


class GpuCaptureDetector:
    """Measure one received stream with the CUDA detector."""

    def __init__(self, args: argparse.Namespace):
        import cupy as cp

        self.cp = cp
        self.args = args
        self.kernel = FStatKernel(Path(args.lib_path))
        pilot_hz = physical_channel_to_pilot_hz(int(args.physical_channel))
        self.spec = ReferenceChannelizerSpec(
            adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
            band_lower_hz=REFERENCE_BAND_LOWER_HZ,
        )
        self.channel_index = nearest_reference_channel_index(pilot_hz, self.spec)
        self.response = sinc_hamming_pfb_response(
            REFERENCE_PFB_TAPS,
            REFERENCE_PFB_FFT_SIZE,
        )
        bank = DetectorWeightBank(
            explicit_path=Path(args.weights_path),
            expected_kernel=self.kernel.specs,
        )
        self.weights, valid = bank.get_weights_for_pilot_frequency(pilot_hz / 1.0e6)
        if self.weights is None or not valid:
            raise SystemExit(f"No detector weights for physical channel {args.physical_channel}.")
        target_norm, lower_norm, upper_norm = weight_term_norms_sq(self.weights)
        self.target_weight_norm_sq = int(target_norm)
        self.reference_weight_norm_sum_sq = int(lower_norm + upper_norm)
        self.n_blocks = int(args.frame_size_samples) + REFERENCE_PFB_TAPS - 1

    def measure(self, capture: np.ndarray) -> dict[str, Any]:
        raw_blocks = complex_envelope_to_real_adc_blocks_gpu(
            capture,
            iq_sample_rate_hz=float(self.args.sample_rate_hz),
            rf_center_hz=float(self.args.frequency_hz),
            adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
            band_lower_hz=REFERENCE_BAND_LOWER_HZ,
            n_blocks=self.n_blocks,
            block_size=REFERENCE_PFB_FFT_SIZE,
        )
        channel_streams = channelize_real_blocks_to_reference_channels_gpu(
            raw_blocks,
            channel_indices=[self.channel_index],
            response=self.response,
            spec=self.spec,
        )
        streams = flatten_feed_channel_streams(
            np.stack([channel_streams], axis=0)
        )
        scale = (
            float(self.args.quantization_scale)
            if self.args.quantization_scale is not None
            else estimate_complex_scale(
                streams,
                bits_per_component=LOCKED_BITS_PER_COMPONENT,
                clip_sigma=float(self.args.clip_sigma),
            )
        )
        packed = _pack_streams_for_kernel(
            streams,
            frame_size_samples=int(self.args.frame_size_samples),
            detector_window_samples=LOCKED_DETECTOR_WINDOW_SAMPLES,
            bits=LOCKED_BITS_PER_COMPONENT,
            scale=scale,
            spectral_sense="normal",
        )
        max_int = (1 << (LOCKED_BITS_PER_COMPONENT - 1)) - 1
        scaled = np.asarray(streams) * float(scale)
        quantization_clip_fraction = float(
            np.mean(
                (scaled.real > max_int)
                | (scaled.real < -max_int)
                | (scaled.imag > max_int)
                | (scaled.imag < -max_int)
            )
        )
        if quantization_clip_fraction > float(
            self.args.quantization_clip_fraction_limit
        ):
            raise RuntimeError("Detector quantization clipping exceeds the limit.")
        measured = _kernel_measurements(
            cp=self.cp,
            kernel=self.kernel,
            packed=packed,
            weights=self.weights,
            pilot_below_data_db=float(self.args.pilot_below_data_db),
            bin_enbw_hz=EFFECTIVE_BIN_BW_HZ,
            pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
            dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
            threshold=None,
        )
        measured.update(
            {
                "backend": "cuda",
                "num_input_streams": 1,
                "detector_rows": int(packed.shape[0]),
                "quantization_scale": float(scale),
                "quantization_clip_fraction": quantization_clip_fraction,
                "target_weight_norm_sq": self.target_weight_norm_sq,
                "reference_weight_norm_sum_sq": (
                    self.reference_weight_norm_sum_sq
                ),
            }
        )
        return measured


def _worker_config(
    args: argparse.Namespace,
    *,
    pass_index: int,
    slots: Sequence[Mapping[str, Any]],
    tx_iq_path: Path,
    session_capture_path: Path,
    tx_off_capture_path: Path,
    stream_status_path: Path,
    session_samples: int,
    tx_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "driver": RADIO_DRIVER,
        "stream_worker": str(Path(args.stream_worker).resolve()),
        "stream_worker_sha256": _file_sha256(Path(args.stream_worker).resolve()),
        "limesuite_library": str(Path(args.limesuite_library).resolve()),
        "limesuite_library_sha256": _file_sha256(
            Path(args.limesuite_library).resolve()
        ),
        "worker_python": str(Path(args.worker_python).resolve()),
        "device_serial": str(args.device_serial),
        "rf_authorized": bool(args.rf_authorized),
        "physical_channel": int(args.physical_channel),
        "frequency_hz": float(args.frequency_hz),
        "sample_rate_hz": float(args.sample_rate_hz),
        "bandwidth_hz": float(args.bandwidth_hz),
        "radio_filter_bandwidth_hz": float(args.radio_filter_bandwidth_hz),
        "tx_gain_db": float(args.tx_gain_db),
        "rx_gain_db": float(args.rx_gain_db),
        "native_tx_gain_db": native_lime_gain_db(args.tx_gain_db),
        "native_rx_gain_db": native_lime_gain_db(args.rx_gain_db),
        "tx_antenna": str(args.tx_antenna),
        "rx_antenna": str(args.rx_antenna),
        "tx_rms": float(args.tx_rms),
        "max_tx_component": MAX_TX_COMPONENT,
        "stream_buffer_samples": int(args.stream_buffer_samples),
        "session_guard_samples": int(args.session_guard_samples),
        "stream_start_delay_samples": max(
            int(args.session_guard_samples),
            int(round(float(args.sample_rate_hz) * DEFAULT_STREAM_START_DELAY_SECONDS)),
        ),
        "settle_samples": int(args.settle_samples),
        "capture_samples": int(args.capture_samples),
        "sync_marker_samples": int(args.sync_marker_samples),
        "capture_guard_samples": int(args.capture_guard_samples),
        "pass_index": int(pass_index),
        "slots": [_json_ready(slot) for slot in slots],
        "session_samples": int(session_samples),
        "tx_iq_path": str(tx_iq_path),
        "tx_sha256": str(tx_sha256),
        "session_capture_path": str(session_capture_path),
        "tx_off_capture_path": str(tx_off_capture_path),
        "stream_status_path": str(stream_status_path),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "stream_worker_timeout_seconds": float(args.worker_timeout_seconds) - 30.0,
    }


def _scan_tx_file(request: Mapping[str, Any]) -> None:
    path = Path(str(request["tx_iq_path"]))
    expected = int(request["session_samples"])
    if not path.is_file() or path.stat().st_size != expected * np.dtype(np.complex64).itemsize:
        raise SystemExit("Session transmit file has the wrong size.")
    samples = np.memmap(path, dtype=np.complex64, mode="r", shape=(expected,))
    peak = 0.0
    digest = hashlib.sha256()
    for start in range(0, expected, 1_000_000):
        chunk = np.asarray(samples[start : start + 1_000_000])
        if not np.all(np.isfinite(chunk)):
            raise SystemExit("Session transmit file contains non-finite samples.")
        peak = max(
            peak,
            float(np.max(np.abs(chunk.real))),
            float(np.max(np.abs(chunk.imag))),
        )
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    if peak > float(request["max_tx_component"]):
        raise SystemExit("Session transmit file exceeds the component limit.")
    if digest.hexdigest() != str(request.get("tx_sha256", "")):
        raise SystemExit("Session transmit file hash does not match the request.")

    for slot in request["slots"]:
        start = int(slot["event_start_sample"])
        count = int(slot["event_samples"])
        stop = start + count
        if start < 0 or stop > expected or count <= 0:
            raise SystemExit("Session slot is outside the transmit file.")
        values = np.asarray(samples[start:stop])
        rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
        if str(slot["kind"]) == "tx_zero":
            if rms != 0.0:
                raise SystemExit("tx_zero slot is not silent.")
        elif not math.isclose(
            rms,
            float(request["tx_rms"]),
            rel_tol=5.0e-4,
            abs_tol=1.0e-7,
        ):
            raise SystemExit("Session slot RMS does not match --tx-rms.")


def validate_worker_request(request: Mapping[str, Any]) -> None:
    """Recheck the session before opening the radio."""
    if request.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("Hardware worker received the wrong schema.")
    if str(request.get("driver")) != RADIO_DRIVER:
        raise SystemExit(f"Hardware worker only accepts driver={RADIO_DRIVER}.")
    if not bool(request.get("rf_authorized")):
        raise SystemExit("Hardware worker refused an unauthorized transmit request.")
    for path_key, hash_key, executable in (
        ("stream_worker", "stream_worker_sha256", True),
        ("limesuite_library", "limesuite_library_sha256", False),
    ):
        path = Path(str(request.get(path_key, ""))).resolve()
        if not path.is_file() or (executable and not os.access(path, os.X_OK)):
            raise SystemExit(f"Hardware worker requires a valid {path_key}.")
        if _file_sha256(path) != str(request.get(hash_key, "")):
            raise SystemExit(f"Hardware worker rejected the {path_key} hash.")
    if not str(request.get("device_serial", "")).strip():
        raise SystemExit("Hardware worker requires a device serial.")
    if str(request.get("tx_antenna")) != TV_TX_ANTENNA:
        raise SystemExit("Hardware worker requires BAND2 for TV-band transmit.")
    if str(request.get("rx_antenna")) != TV_RX_ANTENNA:
        raise SystemExit("Hardware worker requires LNAW for TV-band receive.")
    if not -12.0 <= float(request.get("tx_gain_db", math.nan)) <= 61.0:
        raise SystemExit("Hardware worker rejected the TX gain.")
    if not -12.0 <= float(request.get("rx_gain_db", math.nan)) <= 61.0:
        raise SystemExit("Hardware worker rejected the RX gain.")
    if not float(request["tx_gain_db"]).is_integer() or not float(
        request["rx_gain_db"]
    ).is_integer():
        raise SystemExit("Hardware worker requires whole-decibel gains.")
    if int(request.get("native_tx_gain_db", -1)) != native_lime_gain_db(
        float(request["tx_gain_db"])
    ) or int(request.get("native_rx_gain_db", -1)) != native_lime_gain_db(
        float(request["rx_gain_db"])
    ):
        raise SystemExit("Hardware worker rejected the native gain mapping.")
    if not 0.0 < float(request.get("tx_rms", math.nan)) <= MAX_DIGITAL_RMS:
        raise SystemExit("Hardware worker rejected the TX RMS.")
    if not 1_048_576 <= int(request.get("stream_buffer_samples", 0)) <= 67_108_864:
        raise SystemExit("Hardware worker rejected the stream buffer size.")
    if not math.isclose(
        float(request.get("max_tx_component", math.nan)),
        MAX_TX_COMPONENT,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SystemExit("Hardware worker rejected the component limit.")
    sample_rate_hz = float(request.get("sample_rate_hz", math.nan))
    bandwidth_hz = float(request.get("bandwidth_hz", math.nan))
    filter_bandwidth_hz = float(
        request.get("radio_filter_bandwidth_hz", math.nan)
    )
    if not sample_rate_hz > 0.0 or not math.isclose(
        bandwidth_hz,
        DTV_BANDWIDTH_HZ,
        rel_tol=0.0,
        abs_tol=1.0,
    ):
        raise SystemExit("Hardware worker rejected the sample rate or bandwidth.")
    if not bandwidth_hz < filter_bandwidth_hz < sample_rate_hz:
        raise SystemExit("Hardware worker rejected the radio filter bandwidth.")
    channel = int(request.get("physical_channel", -1))
    expected_hz = expected_center_frequency_hz(channel)
    if abs(float(request.get("frequency_hz", math.nan)) - expected_hz) > FREQUENCY_TOLERANCE_HZ:
        raise SystemExit("Hardware worker rejected the channel frequency.")
    if int(request.get("settle_samples", 0)) < ALIGNMENT_BLOCK_SAMPLES:
        raise SystemExit("Hardware worker rejected the settle length.")
    if int(request.get("capture_samples", 0)) <= 0:
        raise SystemExit("Hardware worker rejected the capture length.")
    if int(request["capture_samples"]) < 2 * SPECTRUM_SEGMENT_SAMPLES:
        raise SystemExit("Hardware worker rejected the spectral capture length.")
    if int(request.get("sync_marker_samples", 0)) < 256:
        raise SystemExit("Hardware worker rejected the marker length.")
    if int(request.get("capture_guard_samples", 0)) < ALIGNMENT_BLOCK_SAMPLES:
        raise SystemExit("Hardware worker rejected the marker guard.")
    if int(request.get("session_guard_samples", 0)) < int(
        request.get("settle_samples", 0)
    ):
        raise SystemExit("Hardware worker rejected the session guard.")
    if int(request.get("stream_start_delay_samples", 0)) < int(
        request.get("session_guard_samples", 0)
    ):
        raise SystemExit("Hardware worker rejected the stream start delay.")
    inner_timeout = float(request.get("stream_worker_timeout_seconds", math.nan))
    outer_timeout = float(request.get("worker_timeout_seconds", math.nan))
    if not 0.0 < inner_timeout <= outer_timeout - 20.0:
        raise SystemExit("Hardware worker rejected the timeout margin.")
    session_samples = int(request.get("session_samples", 0))
    if session_samples <= 0 or session_samples / sample_rate_hz > 600.0:
        raise SystemExit("Hardware worker rejected the session duration.")
    slots = request.get("slots")
    if not isinstance(slots, list) or len(slots) < 3:
        raise SystemExit("Hardware worker requires the session controls.")
    if [str(slot.get("kind")) for slot in slots[:3]] != [
        "tx_zero",
        "signal_only",
        "noise_only",
    ]:
        raise SystemExit("Hardware worker requires controls first.")
    if any(str(slot.get("kind")) not in TRANSMIT_KINDS for slot in slots):
        raise SystemExit("Hardware worker received an unknown slot kind.")
    expected_slot_samples = (
        int(request["sync_marker_samples"])
        + int(request["capture_guard_samples"])
        + int(request["settle_samples"])
        + int(request["capture_samples"])
    )
    for slot in slots:
        marker_start = int(slot["marker_start_sample"])
        event_start = int(slot["event_start_sample"])
        if int(slot["slot_samples"]) != expected_slot_samples:
            raise SystemExit("Hardware worker rejected the slot length.")
        if event_start != (
            marker_start
            + int(request["sync_marker_samples"])
            + int(request["capture_guard_samples"])
        ):
            raise SystemExit("Hardware worker rejected the event offset.")
        if int(slot["event_samples"]) != (
            int(request["settle_samples"]) + int(request["capture_samples"])
        ):
            raise SystemExit("Hardware worker rejected the event length.")
    for previous, slot in zip(slots, slots[1:]):
        separation = int(slot["marker_start_sample"]) - int(
            previous["marker_start_sample"]
        )
        if separation != expected_slot_samples:
            raise SystemExit("Hardware worker rejected the slot spacing.")
    first_marker = int(slots[0]["marker_start_sample"])
    final_stop = int(slots[-1]["marker_start_sample"]) + expected_slot_samples
    if first_marker <= 0 or final_stop >= session_samples:
        raise SystemExit("Hardware worker rejected the session margins.")
    for name in (
        "session_capture_path",
        "tx_off_capture_path",
        "stream_status_path",
    ):
        if not str(request.get(name, "")).strip():
            raise SystemExit(f"Hardware worker requires {name}.")
    _scan_tx_file(request)


def _load_stream_status(request: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(request["stream_status_path"]))
    if not path.is_file():
        raise RuntimeError("Stream status is missing.")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("schema_version") != STREAM_STATUS_SCHEMA or not bool(
        status.get("valid")
    ):
        raise RuntimeError("Stream status is invalid.")
    if int(status.get("tx_off_samples", -1)) != int(request["capture_samples"]):
        raise RuntimeError("Transmitter-off status has the wrong sample count.")
    if int(status.get("session_samples", -1)) != int(request["session_samples"]):
        raise RuntimeError("Session status has the wrong sample count.")
    for field in (
        "tx_off_underrun",
        "tx_off_overrun",
        "tx_off_dropped_packets",
        "rx_underrun",
        "rx_overrun",
        "rx_dropped_packets",
        "tx_underrun",
        "tx_overrun",
        "tx_dropped_packets",
    ):
        if int(status.get(field, -1)) != 0:
            raise RuntimeError(f"Stream status reported {field}.")
    for field in ("rx_host_rate_hz", "tx_host_rate_hz"):
        if not math.isclose(
            float(status.get(field, math.nan)),
            float(request["sample_rate_hz"]),
            rel_tol=0.0,
            abs_tol=1.0,
        ):
            raise RuntimeError(f"Stream status reported the wrong {field}.")
    requested_filter = float(request["radio_filter_bandwidth_hz"])
    if not math.isclose(
        float(status.get("requested_filter_bandwidth_hz", math.nan)),
        requested_filter,
        rel_tol=0.0,
        abs_tol=1.0,
    ) or not bool(status.get("gfir_enabled")):
        raise RuntimeError("Stream status reported the wrong filter settings.")
    for field in ("rx_lpf_bandwidth_hz", "tx_lpf_bandwidth_hz"):
        if not math.isclose(
            float(status.get(field, math.nan)),
            requested_filter,
            rel_tol=0.05,
            abs_tol=1.0,
        ):
            raise RuntimeError(f"Stream status reported the wrong {field}.")
    return status


def run_hardware_worker(request: Mapping[str, Any]) -> None:
    """Capture one timestamped full-duplex session."""
    validate_worker_request(request)
    command = [
        str(request["stream_worker"]),
        "--serial",
        str(request["device_serial"]),
        "--frequency-hz",
        str(request["frequency_hz"]),
        "--sample-rate-hz",
        str(request["sample_rate_hz"]),
        "--bandwidth-hz",
        str(request["radio_filter_bandwidth_hz"]),
        "--tx-gain-db",
        str(request["native_tx_gain_db"]),
        "--rx-gain-db",
        str(request["native_rx_gain_db"]),
        "--fifo-samples",
        str(request["stream_buffer_samples"]),
        "--settle-samples",
        str(request["settle_samples"]),
        "--capture-samples",
        str(request["capture_samples"]),
        "--session-samples",
        str(request["session_samples"]),
        "--start-delay-samples",
        str(request["stream_start_delay_samples"]),
        "--tx-file",
        str(request["tx_iq_path"]),
        "--session-rx-file",
        str(request["session_capture_path"]),
        "--tx-off-rx-file",
        str(request["tx_off_capture_path"]),
        "--status-file",
        str(request["stream_status_path"]),
    ]
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(request["limesuite_library"])
    completed = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(request["stream_worker_timeout_seconds"]),
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if "Calibration: MCU error" in completed.stdout:
        raise RuntimeError("Radio calibration failed.")
    _load_stream_status(request)


def subprocess_capture_runner(request: Mapping[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="sdr_worker_") as temp_dir:
        config_path = Path(temp_dir) / "request.json"
        _write_json(config_path, _json_ready(request))
        command = [
            str(request["worker_python"]),
            str(Path(__file__).resolve()),
            "--worker-config",
            str(config_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(request["worker_timeout_seconds"]),
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if "Calibration: MCU error" in completed.stdout:
            raise RuntimeError("Radio calibration failed.")


def _manifest(args: argparse.Namespace, schedule: Sequence[SweepEvent]) -> dict[str, Any]:
    input_path = Path(args.input_iq).resolve()
    input_metadata_path = input_path.with_suffix(input_path.suffix + ".json")
    weights_path = Path(args.weights_path).resolve()
    lib_path = Path(args.lib_path).resolve()
    calibration_path = (
        None
        if args.pilot_calibration_source == "default"
        else Path(args.pilot_calibration_source)
    )
    worker = None if args.stream_worker is None else Path(args.stream_worker).resolve()
    library = (
        None if args.limesuite_library is None else Path(args.limesuite_library).resolve()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "mode": "transmit" if args.transmit else "dry_run",
        "driver": RADIO_DRIVER,
        "radio_runtime": {
            "worker": None if worker is None else str(worker),
            "worker_sha256": None if worker is None else _file_sha256(worker),
            "worker_source": str(STREAM_WORKER_SOURCE),
            "worker_source_sha256": _file_sha256(STREAM_WORKER_SOURCE),
            "library": None if library is None else str(library),
            "library_sha256": None if library is None else _file_sha256(library),
        },
        "detector_backend": "cuda",
        "num_input_streams": 1,
        "input_axis": (
            "received data-shelf SNR calibrated per pass from tx_zero, "
            "signal_only, and noise_only captures"
        ),
        "tx_off_role": "receiver-only diagnostic; not used for SNR calibration",
        "quantization": {
            "policy": (
                "fixed_user_scale"
                if args.quantization_scale is not None
                else "fixed_from_first_pass_controls"
            ),
            "requested_scale": args.quantization_scale,
            "clip_fraction_limit": float(
                args.quantization_clip_fraction_limit
            ),
        },
        "detector": {
            "frame_size_samples": int(args.frame_size_samples),
            "clip_sigma": float(args.clip_sigma),
            "weights_path": str(weights_path),
            "weights_sha256": _file_sha256(weights_path),
            "lib_path": str(lib_path),
            "lib_sha256": _file_sha256(lib_path),
        },
        "pilot_calibration": {
            "pilot_below_data_db": float(args.pilot_below_data_db),
            "source": str(args.pilot_calibration_source),
            "source_sha256": (
                None if calibration_path is None else _file_sha256(calibration_path)
            ),
        },
        "sweep": {
            "snr_start_db": SNR_VALUES_DB[0],
            "snr_stop_db": SNR_VALUES_DB[-1],
            "snr_step_db": 3.0,
            "low_snr_max_db": LOW_SNR_MAX_DB,
            "low_snr_trials": LOW_SNR_TRIALS,
            "high_snr_trials": HIGH_SNR_TRIALS,
            "passes": int(args.passes),
            "seed": int(args.seed),
            "drift_snr_db": float(args.drift_snr_db),
            "drift_interval": int(args.drift_interval),
        },
        "radio": {
            "device_serial": args.device_serial,
            "frequency_hz": args.frequency_hz,
            "physical_channel": args.physical_channel,
            "sample_rate_hz": float(args.sample_rate_hz),
            "bandwidth_hz": float(args.bandwidth_hz),
            "filter_bandwidth_hz": float(args.radio_filter_bandwidth_hz),
            "tx_gain_db": args.tx_gain_db,
            "rx_gain_db": args.rx_gain_db,
            "native_tx_gain_db": (
                None
                if args.tx_gain_db is None
                else native_lime_gain_db(args.tx_gain_db)
            ),
            "native_rx_gain_db": (
                None
                if args.rx_gain_db is None
                else native_lime_gain_db(args.rx_gain_db)
            ),
            "stream_buffer_samples": int(args.stream_buffer_samples),
            "stream_start_delay_seconds": DEFAULT_STREAM_START_DELAY_SECONDS,
            "tx_rms": args.tx_rms,
            "tx_antenna": args.tx_antenna,
            "rx_antenna": args.rx_antenna,
            "antenna_separation_cm": args.antenna_separation_cm,
            "agc": False,
            "frequency_correction_ppm": 0.0,
        },
        "capture": {
            "settle_samples": int(args.settle_samples),
            "capture_samples": int(args.capture_samples),
            "capture_guard_samples": int(args.capture_guard_samples),
            "sync_marker_samples": int(args.sync_marker_samples),
            "marker_search_samples": int(args.marker_search_samples),
            "session_guard_samples": int(args.session_guard_samples),
            "worker_timeout_seconds": float(args.worker_timeout_seconds),
            "max_received_pilot_ratio_error_db": float(
                args.max_received_pilot_ratio_error_db
            ),
            "clip_level": float(args.clip_level),
            "clip_fraction_limit": float(args.clip_fraction_limit),
            "rx_rms_min": float(args.rx_rms_min),
            "rx_rms_max": float(args.rx_rms_max),
        },
        "input_iq": str(input_path),
        "input_iq_sha256": _file_sha256(input_path),
        "input_metadata": str(input_metadata_path),
        "input_metadata_sha256": _file_sha256(input_metadata_path),
        "events": [asdict(event) for event in schedule],
    }


def _event_stem(event: SweepEvent) -> str:
    snr = "none" if event.snr_db is None else f"{event.snr_db:+.0f}".replace("+", "p").replace("-", "m")
    return f"event_{event.event_index:05d}_{event.kind}_snr_{snr}"


def _append_result(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) + "\n")


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "event_index",
        "pass_index",
        "kind",
        "note",
        "requested_data_shelf_snr_db",
        "received_input_data_shelf_snr_db",
        "trial_index",
        "rx_rms",
        "rx_peak_component",
        "rx_clip_fraction",
        "estimated_data_shelf_snr_db",
        "p_target_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "normalized_coarse_power_ratio",
        "normalized_pilot_excess",
        "quantization_scale",
        "quantization_clip_fraction",
        "received_pilot_below_data_db",
        "capture_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            detector = row.get("detector") or {}
            levels = row.get("capture_levels") or {}
            control = row.get("received_control_calibration") or {}
            event = row["event"]
            writer.writerow(
                {
                    "event_index": event["event_index"],
                    "pass_index": event["pass_index"],
                    "kind": event["kind"],
                    "note": event["note"],
                    "requested_data_shelf_snr_db": event["snr_db"],
                    "received_input_data_shelf_snr_db": row.get(
                        "received_input_data_shelf_snr_db"
                    ),
                    "trial_index": event["trial_index"],
                    "rx_rms": levels.get("rms"),
                    "rx_peak_component": levels.get("peak_component"),
                    "rx_clip_fraction": levels.get("clip_fraction"),
                    "estimated_data_shelf_snr_db": detector.get(
                        "estimated_data_shelf_snr_db"
                    ),
                    "p_target_u64": detector.get("p_target_u64"),
                    "p_ref_lower_u64": detector.get("p_ref_lower_u64"),
                    "p_ref_upper_u64": detector.get("p_ref_upper_u64"),
                    "normalized_coarse_power_ratio": detector.get(
                        "normalized_coarse_power_ratio"
                    ),
                    "normalized_pilot_excess": detector.get(
                        "normalized_pilot_excess"
                    ),
                    "quantization_scale": detector.get("quantization_scale"),
                    "quantization_clip_fraction": detector.get(
                        "quantization_clip_fraction"
                    ),
                    "received_pilot_below_data_db": control.get(
                        "received_pilot_below_data_db"
                    ),
                    "capture_path": row.get("capture_path"),
                }
            )


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _control_rows_by_pass(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    controls: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        event = row.get("event", {})
        kind = str(event.get("kind", ""))
        if row.get("status") != "complete" or kind not in (
            "tx_zero",
            "signal_only",
            "noise_only",
        ):
            continue
        pass_index = int(event["pass_index"])
        pass_controls = controls.setdefault(pass_index, {})
        if kind in pass_controls:
            raise RuntimeError("A pass contains duplicate received controls.")
        pass_controls[kind] = row
    return controls


def _control_expected_transfer(
    group: Sequence[Mapping[str, Any]],
    *,
    controls_by_pass: Mapping[int, Mapping[str, Mapping[str, Any]]],
    pilot_below_data_db: float,
) -> dict[str, float]:
    if not controls_by_pass:
        return {}

    power_fields = ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")
    expected_terms: dict[str, list[float]] = {name: [] for name in power_fields}
    scales: set[float] = set()
    target_norms: set[int] = set()
    reference_norms: set[int] = set()
    received_pilot_power = 0.0
    received_data_power = 0.0

    for row in group:
        pass_index = int(row["event"]["pass_index"])
        pass_controls = controls_by_pass.get(pass_index, {})
        missing = [
            kind
            for kind in ("tx_zero", "signal_only", "noise_only")
            if kind not in pass_controls
        ]
        if missing:
            raise RuntimeError(
                f"Pass {pass_index} is missing received controls: "
                + ", ".join(missing)
            )
        tx = row.get("tx")
        if not isinstance(tx, Mapping):
            raise RuntimeError("Mixture transmit metadata is missing.")
        signal_power = float(tx["signal_power"])
        noise_power = float(tx["noise_power"])
        total_power = signal_power + noise_power
        if total_power <= 0.0:
            raise RuntimeError("Mixture transmit power must be positive.")
        signal_fraction = signal_power / total_power
        noise_fraction = noise_power / total_power

        detectors: dict[str, Mapping[str, Any]] = {}
        for kind, control in pass_controls.items():
            detector = control.get("detector")
            if not isinstance(detector, Mapping):
                raise RuntimeError(f"Pass {pass_index} {kind} detector data is missing.")
            detectors[kind] = detector
            scales.add(float(detector["quantization_scale"]))
            target_norms.add(int(detector["target_weight_norm_sq"]))
            reference_norms.add(int(detector["reference_weight_norm_sum_sq"]))
        mixture_detector = row["detector"]
        scales.add(float(mixture_detector["quantization_scale"]))
        target_norms.add(int(mixture_detector["target_weight_norm_sq"]))
        reference_norms.add(int(mixture_detector["reference_weight_norm_sum_sq"]))

        for name in power_fields:
            zero = float(detectors["tx_zero"][name])
            signal = float(detectors["signal_only"][name])
            noise = float(detectors["noise_only"][name])
            expected_terms[name].append(
                zero
                + signal_fraction * (signal - zero)
                + noise_fraction * (noise - zero)
            )

        calibration = pass_controls["noise_only"].get(
            "received_control_calibration"
        )
        if not isinstance(calibration, Mapping):
            raise RuntimeError(
                f"Pass {pass_index} received control calibration is missing."
            )
        received_pilot_power += signal_fraction * float(
            calibration["signal_pilot_excess_power"]
        )
        received_data_power += signal_fraction * float(
            calibration["signal_data_shelf_power"]
        )

    if len(scales) != 1:
        raise RuntimeError("Received controls and mixtures need one quantization scale.")
    if len(target_norms) != 1 or len(reference_norms) != 1:
        raise RuntimeError("Detector weight norms changed during the sweep.")
    target_norm = target_norms.pop()
    reference_norm = reference_norms.pop()
    expected_target = math.fsum(expected_terms["p_target_u64"])
    expected_lower = math.fsum(expected_terms["p_ref_lower_u64"])
    expected_upper = math.fsum(expected_terms["p_ref_upper_u64"])
    expected_reference = expected_lower + expected_upper
    if expected_target <= 0.0 or expected_reference <= 0.0:
        raise RuntimeError("Control-derived detector powers must be positive.")
    expected_ratio = float(
        expected_target * reference_norm / (expected_reference * target_norm)
    )
    expected_excess = expected_ratio - 1.0
    expected_excess_db = float(normalized_pilot_excess_to_db(expected_excess))
    expected_output_db = float(
        pilot_excess_db_to_data_shelf_snr_db(
            expected_excess_db,
            pilot_below_data_db=float(pilot_below_data_db),
            bin_enbw_hz=EFFECTIVE_BIN_BW_HZ,
            dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
            pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
        )
    )
    if received_pilot_power <= 0.0 or received_data_power <= 0.0:
        raise RuntimeError("Received pilot and data powers must be positive.")
    received_pilot_below_data_db = -DB_POWER_FACTOR * math.log10(
        received_pilot_power / received_data_power
    )
    return {
        "gpu_control_expected_p_target": expected_target,
        "gpu_control_expected_p_ref_lower": expected_lower,
        "gpu_control_expected_p_ref_upper": expected_upper,
        "gpu_control_expected_p_ref_sum": expected_reference,
        "gpu_control_expected_normalized_coarse_power_ratio": expected_ratio,
        "gpu_control_expected_normalized_pilot_excess": expected_excess,
        "gpu_control_expected_pilot_excess_db": expected_excess_db,
        "gpu_control_expected_data_shelf_snr_db": expected_output_db,
        "received_input_pilot_below_data_db": received_pilot_below_data_db,
    }


def build_transfer_tables(
    rows: Sequence[Mapping[str, Any]],
    *,
    pilot_below_data_db: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build pooled GPU transfer rows from completed mixtures."""
    controls_by_pass = _control_rows_by_pass(rows)
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        event = row.get("event", {})
        detector = row.get("detector")
        calibration = row.get("received_input_calibration")
        if (
            row.get("status") != "complete"
            or event.get("kind") != "mixture"
            or not isinstance(detector, Mapping)
            or not isinstance(calibration, Mapping)
        ):
            continue
        commanded = float(event["snr_db"])
        grouped.setdefault(commanded, []).append(row)

    summaries: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for commanded, group in sorted(grouped.items()):
        pooled_input_data_power = sum(
            float(row["received_input_calibration"]["received_data_shelf_power"])
            for row in group
        )
        pooled_input_noise_power = sum(
            float(row["received_input_calibration"]["received_noise_power"])
            for row in group
        )
        if pooled_input_data_power <= 0.0 or pooled_input_noise_power <= 0.0:
            raise RuntimeError("Pooled received input powers are not positive.")
        input_snr_linear = pooled_input_data_power / pooled_input_noise_power
        input_snr_db = float(DB_POWER_FACTOR * math.log10(input_snr_linear))
        trial_input_snr_db = np.asarray(
            [
                float(row["received_input_calibration"]["data_shelf_snr_db"])
                for row in group
            ],
            dtype=np.float64,
        )
        p_target = sum(int(row["detector"]["p_target_u64"]) for row in group)
        p_ref_lower = sum(int(row["detector"]["p_ref_lower_u64"]) for row in group)
        p_ref_upper = sum(int(row["detector"]["p_ref_upper_u64"]) for row in group)
        p_ref_sum = p_ref_lower + p_ref_upper
        if p_ref_sum <= 0:
            raise RuntimeError("Pooled detector reference power is not positive.")
        scales = {
            float(row["detector"]["quantization_scale"]) for row in group
        }
        if len(scales) != 1:
            raise RuntimeError("Mixture captures do not share one quantization scale.")
        quantization_scale = scales.pop()
        target_norms = {
            int(row["detector"]["target_weight_norm_sq"]) for row in group
        }
        reference_norms = {
            int(row["detector"]["reference_weight_norm_sum_sq"])
            for row in group
        }
        if len(target_norms) != 1 or len(reference_norms) != 1:
            raise RuntimeError("Detector weight norms changed during the sweep.")
        target_norm = target_norms.pop()
        reference_norm = reference_norms.pop()
        ratio = float(p_target * reference_norm / (p_ref_sum * target_norm))
        excess = ratio - 1.0
        excess_db = float(normalized_pilot_excess_to_db(excess))
        output_snr_db = float(
            pilot_excess_db_to_data_shelf_snr_db(
                excess_db,
                pilot_below_data_db=float(pilot_below_data_db),
                bin_enbw_hz=EFFECTIVE_BIN_BW_HZ,
                dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
                pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
            )
        )
        expected = _control_expected_transfer(
            group,
            controls_by_pass=controls_by_pass,
            pilot_below_data_db=float(pilot_below_data_db),
        )
        summary = {
            "requested_data_shelf_snr_db": input_snr_db,
            "commanded_data_shelf_snr_db": commanded,
            "frequency_offset_hz": 0.0,
            "pilot_below_data_db": float(pilot_below_data_db),
            "detector_output_pilot_below_data_db": float(pilot_below_data_db),
            "bin_enbw_hz": EFFECTIVE_BIN_BW_HZ,
            "detector_output_bin_enbw_hz": EFFECTIVE_BIN_BW_HZ,
            "dtv_bandwidth_hz": DTV_BANDWIDTH_HZ,
            "detector_output_dtv_bandwidth_hz": DTV_BANDWIDTH_HZ,
            "pilot_capture_efficiency": PILOT_CAPTURE_EFFICIENCY,
            "detector_output_pilot_capture_efficiency": (
                PILOT_CAPTURE_EFFICIENCY
            ),
            "quantization_scale": quantization_scale,
            "pooled_received_data_shelf_power": pooled_input_data_power,
            "pooled_received_noise_power": pooled_input_noise_power,
            "received_input_snr_min_db": float(np.min(trial_input_snr_db)),
            "received_input_snr_max_db": float(np.max(trial_input_snr_db)),
            "received_input_snr_std_db": float(np.std(trial_input_snr_db)),
            "gpu_pooled_p_target": p_target,
            "gpu_pooled_p_ref_lower": p_ref_lower,
            "gpu_pooled_p_ref_upper": p_ref_upper,
            "gpu_pooled_p_ref_sum": p_ref_sum,
            "gpu_pooled_normalized_coarse_power_ratio": ratio,
            "gpu_pooled_normalized_pilot_excess": excess,
            "gpu_pooled_pilot_excess_db": excess_db,
            "gpu_pooled_estimated_data_shelf_snr_db": output_snr_db,
            "gpu_pooled_snr_error_db": output_snr_db - input_snr_db,
            "trials": len(group),
        }
        summary.update(expected)
        summaries.append(summary)
        for row in group:
            detector = row["detector"]
            calibration = row["received_input_calibration"]
            trials.append(
                {
                    "requested_data_shelf_snr_db": input_snr_db,
                    "commanded_data_shelf_snr_db": commanded,
                    "frequency_offset_hz": 0.0,
                    "pass_index": int(row["event"]["pass_index"]),
                    "trial": int(row["event"]["trial_index"]),
                    "pilot_below_data_db": float(pilot_below_data_db),
                    "detector_output_pilot_below_data_db": float(
                        pilot_below_data_db
                    ),
                    "detector_output_bin_enbw_hz": EFFECTIVE_BIN_BW_HZ,
                    "detector_output_dtv_bandwidth_hz": DTV_BANDWIDTH_HZ,
                    "detector_output_pilot_capture_efficiency": (
                        PILOT_CAPTURE_EFFICIENCY
                    ),
                    "quantization_scale": float(detector["quantization_scale"]),
                    "quantization_clip_fraction": float(
                        detector.get("quantization_clip_fraction", math.nan)
                    ),
                    "trial_received_input_data_shelf_snr_db": float(
                        calibration["data_shelf_snr_db"]
                    ),
                    "trial_received_input_data_shelf_snr_linear": float(
                        calibration["data_shelf_snr_linear"]
                    ),
                    "trial_received_data_shelf_power": float(
                        calibration["received_data_shelf_power"]
                    ),
                    "trial_received_noise_power": float(
                        calibration["received_noise_power"]
                    ),
                    "received_pilot_below_data_db": float(
                        calibration.get("received_pilot_below_data_db", math.nan)
                    ),
                    "received_input_pilot_below_data_db": float(
                        calibration.get("received_pilot_below_data_db", math.nan)
                    ),
                    "p_target_u64": int(detector["p_target_u64"]),
                    "p_ref_lower_u64": int(detector["p_ref_lower_u64"]),
                    "p_ref_upper_u64": int(detector["p_ref_upper_u64"]),
                    "p_ref_sum_u64": int(detector["p_ref_sum_u64"]),
                    "normalized_coarse_power_ratio": float(
                        detector["normalized_coarse_power_ratio"]
                    ),
                    "normalized_pilot_excess": float(
                        detector["normalized_pilot_excess"]
                    ),
                    "coarse_power_ratio": float(detector["coarse_power_ratio"]),
                    "target_weight_norm_sq": int(
                        detector["target_weight_norm_sq"]
                    ),
                    "reference_weight_norm_sum_sq": int(
                        detector["reference_weight_norm_sum_sq"]
                    ),
                }
            )
    return summaries, trials


def load_generated_waveform(input_path: Path, *, sample_rate_hz: float) -> np.ndarray:
    metadata_path = input_path.with_suffix(input_path.suffix + ".json")
    if not metadata_path.is_file():
        raise SystemExit(f"Generated-waveform metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "fstat_atsc_clean_iq_v1":
        raise SystemExit(f"Unexpected generated-waveform metadata: {metadata_path}")
    recorded_rate = float(metadata.get("sample_rate_hz", 0.0))
    if not math.isclose(recorded_rate, float(sample_rate_hz), rel_tol=0.0, abs_tol=1.0e-6):
        raise SystemExit(
            "Generated-waveform sample rate does not match --sample-rate-hz."
        )
    input_iq = np.fromfile(input_path, dtype=np.complex64)
    if input_iq.size != int(metadata.get("num_iq_samples", -1)):
        raise SystemExit("Generated-waveform sample count does not match its metadata.")
    if input_iq.size < 2:
        raise SystemExit("--input-iq must contain complex64 samples.")
    return input_iq


def prepare_pass_session(
    args: argparse.Namespace,
    *,
    events: Sequence[SweepEvent],
    clean_iq: np.ndarray,
    tx_path: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], int, str]:
    """Write one finite pass waveform and its slot map."""
    transmitted = [event for event in events if event.kind in TRANSMIT_KINDS]
    if [event.kind for event in transmitted[:3]] != [
        "tx_zero",
        "signal_only",
        "noise_only",
    ]:
        raise RuntimeError("Each pass must begin with the three active controls.")
    slot_samples = event_slot_samples(args)
    event_samples = int(args.settle_samples) + int(args.capture_samples)
    if clean_iq.size < event_samples:
        raise SystemExit(
            "Generated waveform is too short for one session slot; increase the "
            "waveform length or reduce the guarded slot length."
        )
    clean = np.asarray(clean_iq[:event_samples], dtype=np.complex64)
    marker = make_sync_marker(
        samples=int(args.sync_marker_samples),
        sample_rate_hz=float(args.sample_rate_hz),
        target_rms=float(args.tx_rms),
    )
    marker_guard = np.zeros(int(args.capture_guard_samples), dtype=np.complex64)
    lead = np.zeros(int(args.session_guard_samples), dtype=np.complex64)
    slots: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    cursor = int(lead.size)
    digest = hashlib.sha256()

    def write_samples(stream: Any, values: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(values, dtype=np.complex64)
        contiguous.tofile(stream)
        digest.update(memoryview(contiguous).cast("B"))

    with tx_path.open("wb") as stream:
        write_samples(stream, lead)
        for event in transmitted:
            marker_start = cursor
            write_samples(stream, marker)
            write_samples(stream, marker_guard)
            event_start = cursor + int(marker.size) + int(marker_guard.size)
            values, tx_metadata = make_tx_waveform(
                clean,
                kind=event.kind,
                snr_db=event.snr_db,
                target_rms=float(args.tx_rms),
                sample_rate_hz=float(args.sample_rate_hz),
                bandwidth_hz=float(args.bandwidth_hz),
                seed=int(event.seed),
                pilot_below_data_db=float(args.pilot_below_data_db),
            )
            write_samples(stream, values)
            slots.append(
                {
                    "event_index": int(event.event_index),
                    "kind": event.kind,
                    "marker_start_sample": int(marker_start),
                    "event_start_sample": int(event_start),
                    "event_samples": int(event_samples),
                    "slot_samples": int(slot_samples),
                }
            )
            metadata[event.event_index] = tx_metadata
            cursor += slot_samples
        write_samples(stream, lead)
    return slots, metadata, cursor + int(lead.size), digest.hexdigest()


def _strict_event_index(event: Mapping[str, Any]) -> int:
    event_index = event.get("event_index")
    if type(event_index) is not int:
        raise RuntimeError("Event indices must be integers.")
    return event_index


def _event_identity(event: Mapping[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _planned_events_by_index(
    planned_events: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    planned: dict[int, Mapping[str, Any]] = {}
    for event in planned_events:
        if not isinstance(event, Mapping):
            raise RuntimeError("Run plan events must be objects.")
        event_index = _strict_event_index(event)
        if event_index in planned:
            raise RuntimeError(f"Run plan repeats event index {event_index}.")
        planned[event_index] = event
    return planned


def _load_result_rows(
    path: Path,
    *,
    planned_events: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    planned = _planned_events_by_index(planned_events)
    if not path.exists():
        return rows
    payload = path.read_text(encoding="utf-8")
    lines = payload.splitlines()
    needs_rewrite = bool(payload) and not payload.endswith("\n")
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                needs_rewrite = True
                break
            raise
        try:
            event = row["event"]
            if not isinstance(event, Mapping):
                raise RuntimeError("Result events must be objects.")
            event_index = _strict_event_index(event)
        except (KeyError, TypeError) as error:
            raise RuntimeError("Result row has no valid event identity.") from error
        previous = rows.get(event_index)
        if previous is not None and _event_identity(previous["event"]) != _event_identity(
            event
        ):
            raise RuntimeError(
                f"Event index {event_index} has conflicting planned identities."
            )
        planned_event = planned.get(event_index)
        if planned_event is None:
            raise RuntimeError(f"Event index {event_index} is not in the run plan.")
        if _event_identity(event) != _event_identity(planned_event):
            raise RuntimeError(f"Event index {event_index} does not match the run plan.")
        if previous is not None:
            needs_rewrite = True
        rows[event_index] = row
    if needs_rewrite:
        _rewrite_result_rows(
            path,
            [rows[event_index] for event_index in sorted(rows)],
        )
    return rows


def _rewrite_result_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(
                    json.dumps(
                        _json_ready(row),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stable_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(payload)
    stable.pop("created_utc", None)
    return stable


def run(
    args: argparse.Namespace,
    *,
    capture_runner: Callable[[Mapping[str, Any]], None] = subprocess_capture_runner,
    detector: Any | None = None,
) -> int:
    resolve_pilot_calibration(args)
    validate_transmit_args(args)
    if float(args.bandwidth_hz) > float(args.sample_rate_hz):
        raise SystemExit("--bandwidth-hz must not exceed --sample-rate-hz.")
    if (
        int(args.capture_samples) <= 0
        or int(args.settle_samples) < 0
        or int(args.capture_guard_samples) < 0
    ):
        raise SystemExit("Capture sample counts are invalid.")
    if not 0.0 < float(args.clip_level) <= 1.0:
        raise SystemExit("--clip-level must be in (0, 1].")
    if not 0.0 <= float(args.clip_fraction_limit) <= 1.0:
        raise SystemExit("--clip-fraction-limit must be in [0, 1].")
    if not 0.0 <= float(args.rx_rms_min) < float(args.rx_rms_max):
        raise SystemExit("RX RMS limits are invalid.")
    if args.quantization_scale is not None and (
        not math.isfinite(float(args.quantization_scale))
        or float(args.quantization_scale) <= 0.0
    ):
        raise SystemExit("--quantization-scale must be positive and finite.")
    if not 0.0 <= float(args.quantization_clip_fraction_limit) <= 1.0:
        raise SystemExit("--quantization-clip-fraction-limit must be in [0, 1].")
    if (
        not math.isfinite(float(args.max_received_pilot_ratio_error_db))
        or float(args.max_received_pilot_ratio_error_db) < 0.0
    ):
        raise SystemExit(
            "--max-received-pilot-ratio-error-db must be finite and nonnegative."
        )

    full_schedule = build_schedule(
        seed=int(args.seed),
        passes=int(args.passes),
        drift_snr_db=float(args.drift_snr_db),
        drift_interval=int(args.drift_interval),
    )
    schedule = full_schedule
    if args.event_limit is not None:
        if int(args.event_limit) <= 0:
            raise SystemExit("--event-limit must be positive.")
        schedule = full_schedule[: int(args.event_limit)]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_plan.json"
    expected_manifest = _json_ready(_manifest(args, full_schedule))
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Output already contains {manifest_path.name}; use --resume."
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _event_identity(_stable_manifest(existing_manifest)) != _event_identity(
            _stable_manifest(expected_manifest)
        ):
            raise SystemExit("The existing run plan does not match this command.")
        active_manifest = existing_manifest
    else:
        if args.resume:
            raise SystemExit("--resume requires an existing run plan.")
        _write_json(manifest_path, expected_manifest)
        active_manifest = expected_manifest
    if not args.transmit:
        print(f"Dry run only. Wrote {manifest_path}")
        print(f"Full planned events: {len(full_schedule)}")
        print(f"Selected events: {len(schedule)}")
        return 0

    input_iq = load_generated_waveform(
        Path(args.input_iq),
        sample_rate_hz=float(args.sample_rate_hz),
    )
    calibration_start = int(args.settle_samples)
    calibration_stop = calibration_start + int(args.capture_samples)
    if input_iq.size < calibration_stop:
        raise SystemExit("Generated waveform is too short for spectral calibration.")
    calibration_signal = np.asarray(
        input_iq[calibration_start:calibration_stop],
        dtype=np.complex64,
    )
    calibration_noise = _unit_noise(
        calibration_signal,
        rng=np.random.default_rng(int(args.seed)),
    )
    transmitted_spectral_profile = calibrate_received_controls(
        tx_zero=np.zeros_like(calibration_signal),
        signal_only=calibration_signal,
        noise_only=calibration_noise,
        sample_rate_hz=float(args.sample_rate_hz),
        bandwidth_hz=float(args.bandwidth_hz),
        expected_pilot_below_data_db=float(args.pilot_below_data_db),
    )
    transmitted_spectral_pilot_below_data_db = float(
        transmitted_spectral_profile["received_pilot_below_data_db"]
    )
    captures_dir = output_dir / "captures"
    metadata_dir = output_dir / "metadata"
    sessions_dir = output_dir / "sessions"
    captures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "events.jsonl"
    rows_by_event = (
        _load_result_rows(
            results_path,
            planned_events=active_manifest["events"],
        )
        if args.resume
        else {}
    )
    quantization_scale_source = (
        "requested" if args.quantization_scale is not None else None
    )
    completed_scales = {
        float(row["detector"]["quantization_scale"])
        for row in rows_by_event.values()
        if row.get("status") == "complete"
        and row.get("event", {}).get("kind")
        in ("tx_off", "tx_zero", "signal_only", "noise_only", "mixture", "drift")
        and isinstance(row.get("detector"), Mapping)
        and row["detector"].get("quantization_scale") is not None
    }
    if args.quantization_scale is None and completed_scales:
        if len(completed_scales) != 1:
            raise SystemExit("Completed events do not share one quantization scale.")
        args.quantization_scale = completed_scales.pop()
        quantization_scale_source = "resumed"
    if detector is None:
        detector = GpuCaptureDetector(args)

    selected_by_pass: dict[int, list[SweepEvent]] = {}
    for event in schedule:
        selected_by_pass.setdefault(event.pass_index, []).append(event)

    for pass_index, pass_events in selected_by_pass.items():
        if all(
            rows_by_event.get(event.event_index, {}).get("status") == "complete"
            for event in pass_events
        ):
            continue
        with tempfile.TemporaryDirectory(
            dir=output_dir,
            prefix=f"pass_{pass_index:02d}_",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            tx_path = temp_path / "session_tx.cfile"
            session_capture_path = temp_path / "session_rx.cfile"
            tx_off_capture_path = temp_path / "tx_off_rx.cfile"
            stream_status_path = temp_path / "stream_status.json"
            (
                slots,
                tx_metadata_by_event,
                session_samples,
                tx_sha256,
            ) = prepare_pass_session(
                args,
                events=pass_events,
                clean_iq=input_iq,
                tx_path=tx_path,
            )
            request = _worker_config(
                args,
                pass_index=pass_index,
                slots=slots,
                tx_iq_path=tx_path,
                session_capture_path=session_capture_path,
                tx_off_capture_path=tx_off_capture_path,
                stream_status_path=stream_status_path,
                session_samples=session_samples,
                tx_sha256=tx_sha256,
            )
            capture_runner(request)
            stream_status = _load_stream_status(request)
            expected_bytes = session_samples * np.dtype(np.complex64).itemsize
            if (
                not session_capture_path.is_file()
                or session_capture_path.stat().st_size != expected_bytes
            ):
                raise RuntimeError("Full-duplex session capture has the wrong size.")
            session_capture = np.memmap(
                session_capture_path,
                dtype=np.complex64,
                mode="r",
                shape=(session_samples,),
            )
            tx_off_capture = np.fromfile(tx_off_capture_path, dtype=np.complex64)
            if tx_off_capture.size != int(args.capture_samples):
                raise RuntimeError("Transmitter-off capture has the wrong size.")
            marker = make_sync_marker(
                samples=int(args.sync_marker_samples),
                sample_rate_hz=float(args.sample_rate_hz),
                target_rms=float(args.tx_rms),
            )
            alignments = align_session_slots(
                session_capture,
                slots=slots,
                marker=marker,
                initial_search_samples=min(
                    int(args.session_guard_samples),
                    event_slot_samples(args) // 3,
                ),
                local_search_samples=int(args.marker_search_samples),
            )
            alignment_by_event = {
                int(item["event_index"]): item for item in alignments
            }

            def event_capture(
                event: SweepEvent,
            ) -> tuple[np.ndarray, dict[str, Any] | None]:
                if event.kind == "tx_off":
                    return tx_off_capture, None
                event_marker = alignment_by_event[event.event_index]
                start = (
                    int(event_marker["observed_marker_sample"])
                    + int(args.sync_marker_samples)
                    + int(args.capture_guard_samples)
                    + int(args.settle_samples)
                )
                stop = start + int(args.capture_samples)
                if start < 0 or stop > session_capture.size:
                    raise RuntimeError("Aligned event capture is outside the session.")
                alignment = {
                    **event_marker,
                    "start_sample": start,
                    "stop_sample": stop,
                }
                return session_capture[start:stop], alignment

            if args.quantization_scale is None:
                controls = [
                    event
                    for event in pass_events
                    if event.kind in ("tx_zero", "signal_only", "noise_only")
                ]
                if [event.kind for event in controls] != [
                    "tx_zero",
                    "signal_only",
                    "noise_only",
                ]:
                    raise RuntimeError("Three pass controls are required to set the scale.")
                proposed_scales: list[float] = []
                for control in controls:
                    control_capture, _ = event_capture(control)
                    probe = detector.measure(control_capture)
                    proposed_scales.append(
                        float(probe.get("quantization_scale", math.nan))
                    )
                if any(
                    not math.isfinite(scale) or scale <= 0.0
                    for scale in proposed_scales
                ):
                    raise RuntimeError("A pass control returned an invalid scale.")
                args.quantization_scale = min(proposed_scales)
                quantization_scale_source = "pass_controls"

            control_captures: dict[str, np.ndarray] = {}
            control_calibration: dict[str, float] | None = None
            _write_json(
                sessions_dir / f"pass_{pass_index:02d}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "completed_utc": _utc_now(),
                    "pass_index": pass_index,
                    "alignment": alignments,
                    "slots": slots,
                    "session_samples": session_samples,
                    "tx_sha256": tx_sha256,
                    "stream_status": stream_status,
                    "transmitted_spectral_pilot_below_data_db": (
                        transmitted_spectral_pilot_below_data_db
                    ),
                    "quantization_scale": float(args.quantization_scale),
                    "quantization_scale_source": quantization_scale_source,
                },
            )
            for event in pass_events:
                stem = _event_stem(event)
                capture_path = captures_dir / f"{stem}.cfile"
                tx_metadata = tx_metadata_by_event.get(event.event_index)
                capture, event_alignment = event_capture(event)
                np.asarray(capture, dtype=np.complex64).tofile(capture_path)
                levels = capture_level_stats(
                    capture,
                    clip_level=float(args.clip_level),
                    sample_rate_hz=float(args.sample_rate_hz),
                    bandwidth_hz=float(args.bandwidth_hz),
                )
                level_error: str | None = None
                if float(levels["clip_fraction"]) > float(args.clip_fraction_limit):
                    level_error = "clip_fraction_limit"
                elif float(levels["rms"]) > float(args.rx_rms_max):
                    level_error = "rx_rms_max"
                elif event.kind not in ("tx_off", "tx_zero") and float(
                    levels["rms"]
                ) < float(args.rx_rms_min):
                    level_error = "rx_rms_min"

                received_calibration: dict[str, float] | None = None
                if level_error is None and event.kind in (
                    "tx_zero",
                    "signal_only",
                    "noise_only",
                ):
                    control_captures[event.kind] = np.asarray(
                        capture,
                        dtype=np.complex64,
                    ).copy()
                    if event.kind == "noise_only":
                        control_calibration = calibrate_received_controls(
                            tx_zero=control_captures["tx_zero"],
                            signal_only=control_captures["signal_only"],
                            noise_only=control_captures["noise_only"],
                            sample_rate_hz=float(args.sample_rate_hz),
                            bandwidth_hz=float(args.bandwidth_hz),
                            expected_pilot_below_data_db=float(
                                transmitted_spectral_pilot_below_data_db
                            ),
                        )
                        if abs(
                            control_calibration["pilot_ratio_error_db"]
                        ) > float(args.max_received_pilot_ratio_error_db):
                            level_error = "received_pilot_ratio_error"
                elif level_error is None and event.kind in ("mixture", "drift"):
                    if tx_metadata is None:
                        raise RuntimeError("Transmit metadata is missing for a mixture.")
                    if control_calibration is None:
                        raise RuntimeError("Received control calibration is missing.")
                    received_calibration = calibrated_received_shelf_snr(
                        tx_metadata=tx_metadata,
                        control_calibration=control_calibration,
                    )

                result: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "completed_utc": _utc_now(),
                    "event": asdict(event),
                    "tx": tx_metadata,
                    "radio": {
                        "driver": RADIO_DRIVER,
                        "frequency_hz": float(args.frequency_hz),
                        "sample_rate_hz": float(args.sample_rate_hz),
                        "bandwidth_hz": float(args.bandwidth_hz),
                        "tx_gain_db": float(args.tx_gain_db),
                        "rx_gain_db": float(args.rx_gain_db),
                        "native_tx_gain_db": native_lime_gain_db(args.tx_gain_db),
                        "native_rx_gain_db": native_lime_gain_db(args.rx_gain_db),
                        "tx_antenna": str(args.tx_antenna),
                        "rx_antenna": str(args.rx_antenna),
                        "agc": False,
                        "frequency_correction_ppm": 0.0,
                    },
                    "session_alignment": event_alignment,
                    "capture_levels": levels,
                    "capture_path": str(capture_path),
                    "received_input_calibration": received_calibration,
                    "detector_output_calibration": {
                        "pilot_below_data_db": float(args.pilot_below_data_db),
                        "bin_enbw_hz": EFFECTIVE_BIN_BW_HZ,
                        "dtv_bandwidth_hz": DTV_BANDWIDTH_HZ,
                        "pilot_capture_efficiency": PILOT_CAPTURE_EFFICIENCY,
                    },
                    "received_control_calibration": (
                        control_calibration if event.kind == "noise_only" else None
                    ),
                    "quantization_scale_source": quantization_scale_source,
                    "received_input_data_shelf_snr_db": (
                        None
                        if received_calibration is None
                        else received_calibration["data_shelf_snr_db"]
                    ),
                    "detector": None,
                    "status": "level_guard_failed" if level_error else "complete",
                    "level_error": level_error,
                }
                if level_error is None and event.kind != "tx_off":
                    measured = _json_ready(detector.measure(capture))
                    result["detector"] = measured
                metadata_path = metadata_dir / f"{stem}.json"
                _write_json(metadata_path, _json_ready(result))
                _append_result(results_path, result)
                rows_by_event[event.event_index] = result
                if level_error is not None:
                    raise RuntimeError(
                        f"Level guard {level_error} failed at event "
                        f"{event.event_index}; see {metadata_path}."
                    )

    rows = [rows_by_event[index] for index in sorted(rows_by_event)]
    _rewrite_result_rows(results_path, rows)
    _write_summary_csv(output_dir / "events.csv", rows)
    summaries, trials = build_transfer_tables(
        rows,
        pilot_below_data_db=float(args.pilot_below_data_db),
    )
    summary_path = output_dir / "sdr_transfer_summary.csv"
    trial_path = output_dir / "sdr_transfer_trials.csv"
    _write_rows_csv(summary_path, summaries)
    _write_rows_csv(trial_path, trials)
    if summaries:
        from pilot_proxy.testbench.plot_results import plot_summary

        plot_summary(
            input_csv=summary_path,
            trial_csv=trial_path,
            output_png=output_dir / "sdr_estimator_transfer.png",
            title="LimeSDR two-antenna GPU detector transfer",
            show=False,
        )
    _write_json(
        output_dir / "run_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "updated_utc": _utc_now(),
            "completed_events": len(rows),
            "quantization_scale": args.quantization_scale,
            "quantization_scale_source": quantization_scale_source,
            "detector_output_calibration": {
                "pilot_below_data_db": float(args.pilot_below_data_db),
                "bin_enbw_hz": EFFECTIVE_BIN_BW_HZ,
                "dtv_bandwidth_hz": DTV_BANDWIDTH_HZ,
                "pilot_capture_efficiency": PILOT_CAPTURE_EFFICIENCY,
            },
            "transmitted_spectral_pilot_below_data_db": (
                transmitted_spectral_pilot_below_data_db
            ),
        },
    )
    print(f"Completed events: {len(rows)}")
    print(f"Wrote {output_dir / 'events.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or run a guarded LimeSDR two-antenna transfer sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-iq",
        type=Path,
        default=None,
        help="Clean complex64 waveform from the GNU Radio ATSC generator.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--transmit", action="store_true")
    parser.add_argument("--rf-authorized", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--drift-snr-db", type=float, default=DEFAULT_DRIFT_SNR_DB)
    parser.add_argument("--drift-interval", type=int, default=DEFAULT_DRIFT_INTERVAL)
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--physical-channel", type=int, default=None)
    parser.add_argument("--tx-gain-db", type=float, default=None)
    parser.add_argument("--rx-gain-db", type=float, default=None)
    parser.add_argument("--tx-rms", type=float, default=None)
    parser.add_argument("--device-serial", default=None)
    parser.add_argument("--tx-antenna", default=None)
    parser.add_argument("--rx-antenna", default=None)
    parser.add_argument("--antenna-separation-cm", type=float, default=None)
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    )
    parser.add_argument("--bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument(
        "--radio-filter-bandwidth-hz",
        type=float,
        default=DEFAULT_RADIO_FILTER_BANDWIDTH_HZ,
    )
    parser.add_argument(
        "--settle-samples",
        type=int,
        default=DEFAULT_SETTLE_SAMPLES,
    )
    parser.add_argument(
        "--capture-samples",
        type=int,
        default=required_iq_samples(
            iq_sample_rate_hz=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
            adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
            num_output_samples=DEFAULT_FRAME_SIZE_SAMPLES,
        ),
    )
    parser.add_argument(
        "--capture-guard-samples",
        type=int,
        default=DEFAULT_CAPTURE_GUARD_SAMPLES,
    )
    parser.add_argument(
        "--sync-marker-samples",
        type=int,
        default=DEFAULT_SYNC_MARKER_SAMPLES,
    )
    parser.add_argument(
        "--marker-search-samples",
        type=int,
        default=DEFAULT_MARKER_SEARCH_SAMPLES,
    )
    parser.add_argument(
        "--session-guard-samples",
        type=int,
        default=DEFAULT_SESSION_GUARD_SAMPLES,
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--stream-buffer-samples",
        type=int,
        default=DEFAULT_STREAM_BUFFER_SAMPLES,
    )
    parser.add_argument("--clip-level", type=float, default=0.98)
    parser.add_argument("--clip-fraction-limit", type=float, default=1.0e-4)
    parser.add_argument("--rx-rms-min", type=float, default=1.0e-3)
    parser.add_argument("--rx-rms-max", type=float, default=0.35)
    parser.add_argument("--frame-size-samples", type=int, default=DEFAULT_FRAME_SIZE_SAMPLES)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--quantization-scale", type=float, default=None)
    parser.add_argument(
        "--quantization-clip-fraction-limit",
        type=float,
        default=DEFAULT_QUANTIZATION_CLIP_FRACTION_LIMIT,
    )
    parser.add_argument(
        "--max-received-pilot-ratio-error-db",
        type=float,
        default=DEFAULT_MAX_RECEIVED_PILOT_RATIO_ERROR_DB,
    )
    parser.add_argument(
        "--pilot-below-data-db",
        type=float,
        default=None,
    )
    parser.add_argument("--waveform-audit-json", type=Path, default=None)
    parser.add_argument("--worker-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--stream-worker", type=Path, default=None)
    parser.add_argument("--limesuite-library", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_config is not None:
        request = json.loads(args.worker_config.read_text(encoding="utf-8"))
        run_hardware_worker(request)
        return 0
    if args.input_iq is None or args.output_dir is None:
        raise SystemExit("--input-iq and --output-dir are required.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
