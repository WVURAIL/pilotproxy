#!/usr/bin/env python3
# coding=utf-8
"""Run the staged current-geometry synthetic sensitivity experiment.

This is a resumable, shard-oriented experiment, not a one-command source of a
publication number.  The production sequence is::

    # Generate/audit the deterministic GNU Radio waveform with the existing
    # testbench first, then cache its reference-PFB output for every profile.
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage prepare --input-iq generated/atsc/atsc_8vsb_complex64.cfile

    # Production defaults to an empirical per-stream sufficient-statistic
    # model of the 2048-stream sum. Null shards may be split over machines or
    # seeds. Calibrate only after the planned null population is present.
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage null --seed 1 --trials 1000 --input-iq ...
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage calibrate --input-iq ...

    # H1 shards are independent files and can be resumed safely.
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage sweep --seed 7 --trials 1500 --input-iq ...
    # A stratified audit traverses literal 2048-stream packed frames and the
    # selected CUDA artifact; it is required before precision review.
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage audit --gpu --input-iq ...
    python3 tools/current_geometry_sensitivity.py --mode production \
        --stage report --input-iq ...

``--mode smoke --stage smoke`` runs the same code at reduced stream/trial
count and marks the result as code validation, not current-geometry evidence.
Production defaults use all physical channels 14--36, the circularly wrapped
channel-14 weight profile, centered and half-fine-bin offsets, modeled
2048-stream sums, empirical Pfa calibration, and a grid intended to bracket Pd
crossings. The accelerated backend and literal audit are labeled separately.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_geometry import (
    apply_spectral_sense_to_detector_matrix,
    predicted_pilot_fine_bin,
    stream_time_block_to_detector_matrix,
)
from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import (
    DTV_BANDWIDTH_HZ,
    PILOT_BELOW_DATA_DB,
    pilot_to_data_power_ratio,
)
from pilot_proxy.fine_decision import pack_bulk_mask
from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.fxfft import fxfft256, fine_power_fx
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import (
    CONFIGS_DIR,
    DEFAULT_LIB_PATH,
    DEFAULT_WEIGHTS_PATH,
)
from pilot_proxy.provenance import file_sha256
from pilot_proxy.reference_channelizer import (
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    channelize_real_blocks_to_reference_channels,
    complex_envelope_to_real_adc_blocks,
    nearest_reference_channel_index,
    sinc_hamming_pfb_response,
)
from pilot_proxy.testbench.evaluate_snr import (
    _ideal_float_weights_from_layout,
    apply_channel_impairments,
    required_iq_samples,
)
from pilot_proxy.testbench.generate_atsc_signal import (
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
)
from pilot_proxy.testbench.sensitivity_study import (
    ExactResponseComponents,
    FLOAT_RESPONSE_STAGES,
    REPORT_STAGES,
    SENSITIVITY_STUDY_SCHEMA,
    STAGE_ATSC_FLOAT,
    STAGE_DEFINITIONS,
    STAGE_FIXED_FLOAT_DECISION,
    STAGE_FIXED_Q16_CPU,
    STAGE_FULL_GPU,
    STAGE_IDEAL_TONE_FLOAT,
    STAGE_INPUT_INT4,
    STAGE_JOINT_INT4_FLOAT,
    STAGE_WEIGHT_INT4,
    components_from_columns,
    crossing_bracket,
    designated_bins,
    exact_columns,
    exact_q16_decision,
    exact_response_components,
    float_fine_power_ratio,
    float_fine_powers_by_stream,
    float_response_ratio,
    order_statistic_threshold,
    paired_crossing_bootstrap,
    q16_ceil_multiplier,
    stage_seed,
    wilson_interval,
)

UTC = timezone.utc
MODE_SMOKE = "smoke"
MODE_PRODUCTION = "production"
STAGES = ("prepare", "null", "calibrate", "sweep", "audit", "report", "smoke")
SIMULATION_FULL_FRAME = "full-frame"
SIMULATION_SUFFICIENT = "sufficient-statistic"
SIMULATION_BACKENDS = (SIMULATION_FULL_FRAME, SIMULATION_SUFFICIENT)
BITS = 4
K = 128
FRAME_SAMPLES = 16_384
WINDOWS = FRAME_SAMPLES // K
FINE_BINS = 2 * WINDOWS
OUTPUT_SAMPLE_RATE_HZ = REFERENCE_ADC_SAMPLE_RATE_HZ / REFERENCE_PFB_FFT_SIZE
FINE_BIN_HZ = OUTPUT_SAMPLE_RATE_HZ / (K * FINE_BINS)
WEIGHT_DEQUANTIZATION_SCALE = 7.0
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_INPUT_SCALE = 7.0 / (DEFAULT_CLIP_SIGMA / math.sqrt(2.0))
DEFAULT_P_FA = 1.0e-3
DEFAULT_NULL_QUANTILE = 0.5
DEFAULT_DESIGNATED_HALF_WIDTH = 2
DEFAULT_GUARD_FINE_BINS = 1
DEFAULT_BOOTSTRAP_TARGETS = (0.5, 0.9)
SMOKE_CHANNELS = (14,)
PRODUCTION_CHANNELS = tuple(range(14, 37))
DEFAULT_OFFSETS_FINE_BINS = (0.0, 0.5)
SMOKE_SNRS_DB = (-36.0, -32.0, -28.0, -24.0)
PRODUCTION_SNRS_DB = tuple(float(value) for value in range(-60, -29, 2))
SMOKE_STREAMS = 32
PRODUCTION_STREAMS = 2048
SMOKE_TRIALS = 4
PRODUCTION_NULL_TRIALS = 4000
PRODUCTION_H1_TRIALS = 1500
SMOKE_BOOTSTRAP = 100
PRODUCTION_BOOTSTRAP = 2000
PRODUCTION_SUFFICIENT_POOL_STREAMS = 256
PRODUCTION_AUDIT_CHANNELS = (14, 25, 36)
PRODUCTION_AUDIT_SNRS_DB = (-54.0, -50.0, -46.0, -42.0)
PRODUCTION_AUDIT_TRIALS = 16
PFB_GAIN_SEED_BASE = 20260820
SHARD_META_KEY = "meta_json"


@dataclass(frozen=True)
class StudyProfile:
    physical_channel: int
    offset_fine_bins: float
    anchor_bin: int
    designated: np.ndarray
    bulk_mask: np.ndarray
    cfar_rank: int
    ideal_weights: np.ndarray
    packed_weights: np.ndarray
    float_packed_weights: np.ndarray
    layout: dict[str, Any]


@dataclass
class GpuContext:
    cp: Any
    kernel: FStatKernel
    execution_form: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_provenance() -> dict[str, Any]:
    dirty = _git_value("status", "--short") or ""
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "dirty": bool(dirty),
        "dirty_paths": dirty.splitlines(),
    }


def _safe_offset_label(value: float) -> str:
    return (
        f"{float(value):+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
    )


def _safe_snr_label(value: float) -> str:
    return (
        f"{float(value):+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    )


def _unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _mode_defaults(args: argparse.Namespace) -> None:
    if args.simulation_backend is None:
        args.simulation_backend = (
            SIMULATION_FULL_FRAME
            if args.mode == MODE_SMOKE
            else SIMULATION_SUFFICIENT
        )
    if args.physical_channel is None:
        args.physical_channel = list(
            SMOKE_CHANNELS if args.mode == MODE_SMOKE else PRODUCTION_CHANNELS
        )
    if args.offset_fine_bins is None:
        args.offset_fine_bins = list(DEFAULT_OFFSETS_FINE_BINS)
    if args.snr_db is None:
        args.snr_db = list(
            SMOKE_SNRS_DB if args.mode == MODE_SMOKE else PRODUCTION_SNRS_DB
        )
    if args.num_streams is None:
        args.num_streams = (
            SMOKE_STREAMS if args.mode == MODE_SMOKE else PRODUCTION_STREAMS
        )
    if args.trials is None:
        if args.mode == MODE_SMOKE:
            args.trials = SMOKE_TRIALS
        elif args.stage == "audit":
            args.trials = PRODUCTION_AUDIT_TRIALS
        elif args.stage == "null":
            args.trials = PRODUCTION_NULL_TRIALS
        else:
            args.trials = PRODUCTION_H1_TRIALS
    if args.bootstrap_replicates is None:
        args.bootstrap_replicates = (
            SMOKE_BOOTSTRAP
            if args.mode == MODE_SMOKE
            else PRODUCTION_BOOTSTRAP
        )
    args.physical_channel = [int(v) for v in _unique(args.physical_channel)]
    args.offset_fine_bins = [float(v) for v in _unique(args.offset_fine_bins)]
    args.snr_db = [float(v) for v in _unique(args.snr_db)]
    args.run_physical_channel = (
        list(args.physical_channel)
        if args.run_physical_channel is None
        else [int(v) for v in _unique(args.run_physical_channel)]
    )
    args.run_offset_fine_bins = (
        list(args.offset_fine_bins)
        if args.run_offset_fine_bins is None
        else [float(v) for v in _unique(args.run_offset_fine_bins)]
    )
    args.sweep_snr_db = (
        list(args.snr_db)
        if args.sweep_snr_db is None
        else [float(v) for v in _unique(args.sweep_snr_db)]
    )
    if args.sufficient_pool_streams is None:
        args.sufficient_pool_streams = PRODUCTION_SUFFICIENT_POOL_STREAMS
    default_audit_channels = [
        channel
        for channel in PRODUCTION_AUDIT_CHANNELS
        if channel in args.physical_channel
    ]
    if not default_audit_channels and args.physical_channel:
        default_audit_channels = [args.physical_channel[0]]
    args.audit_physical_channel = (
        default_audit_channels
        if args.audit_physical_channel is None
        else [int(v) for v in _unique(args.audit_physical_channel)]
    )
    args.audit_offset_fine_bins = (
        list(args.offset_fine_bins)
        if args.audit_offset_fine_bins is None
        else [float(v) for v in _unique(args.audit_offset_fine_bins)]
    )
    default_audit_snrs = [
        snr for snr in PRODUCTION_AUDIT_SNRS_DB if snr in args.snr_db
    ]
    if not default_audit_snrs and args.snr_db:
        ordered = sorted(args.snr_db)
        default_audit_snrs = [ordered[len(ordered) // 2]]
    args.audit_snr_db = (
        default_audit_snrs
        if args.audit_snr_db is None
        else [float(v) for v in _unique(args.audit_snr_db)]
    )
    args.run_audit_physical_channel = (
        list(args.audit_physical_channel)
        if args.run_audit_physical_channel is None
        else [int(v) for v in _unique(args.run_audit_physical_channel)]
    )
    args.run_audit_offset_fine_bins = (
        list(args.audit_offset_fine_bins)
        if args.run_audit_offset_fine_bins is None
        else [float(v) for v in _unique(args.run_audit_offset_fine_bins)]
    )
    args.run_audit_snr_db = (
        list(args.audit_snr_db)
        if args.run_audit_snr_db is None
        else [float(v) for v in _unique(args.run_audit_snr_db)]
    )


def _validate_args(args: argparse.Namespace, bank: DetectorWeightBank) -> None:
    supported = set(bank.supported_physical_channels())
    unknown = sorted(set(args.physical_channel) - supported)
    if unknown:
        raise SystemExit(f"weight bank does not support physical channels {unknown}")
    if not set(args.run_physical_channel).issubset(set(args.physical_channel)):
        raise SystemExit("--run-physical-channel must be a subset of the study channels")
    if not set(args.run_offset_fine_bins).issubset(set(args.offset_fine_bins)):
        raise SystemExit("--run-offset-fine-bins must be a subset of study offsets")
    if not set(args.sweep_snr_db).issubset(set(args.snr_db)):
        raise SystemExit("--sweep-snr-db must be a subset of the planned SNR grid")
    if not set(args.audit_physical_channel).issubset(set(args.physical_channel)):
        raise SystemExit("--audit-physical-channel must be a subset of study channels")
    if not set(args.audit_offset_fine_bins).issubset(set(args.offset_fine_bins)):
        raise SystemExit("--audit-offset-fine-bins must be a subset of study offsets")
    if not set(args.audit_snr_db).issubset(set(args.snr_db)):
        raise SystemExit("--audit-snr-db must be a subset of the planned SNR grid")
    if not set(args.run_audit_physical_channel).issubset(
        set(args.audit_physical_channel)
    ):
        raise SystemExit(
            "--run-audit-physical-channel must be a subset of declared audit channels"
        )
    if not set(args.run_audit_offset_fine_bins).issubset(
        set(args.audit_offset_fine_bins)
    ):
        raise SystemExit(
            "--run-audit-offset-fine-bins must be a subset of declared audit offsets"
        )
    if not set(args.run_audit_snr_db).issubset(set(args.audit_snr_db)):
        raise SystemExit("--run-audit-snr-db must be a subset of declared audit SNRs")
    if int(args.num_streams) <= 0 or int(args.trials) <= 0:
        raise SystemExit("--num-streams and --trials must be positive")
    if int(args.bootstrap_replicates) <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    if int(args.sufficient_pool_streams) < 2:
        raise SystemExit("--sufficient-pool-streams must be at least 2")
    if int(args.num_streams) != PRODUCTION_STREAMS and not args.allow_reduced_geometry:
        raise SystemExit(
            "a non-2048 stream count is code validation only; pass "
            "--allow-reduced-geometry to acknowledge that scope"
        )
    if args.mode == MODE_PRODUCTION and int(args.num_streams) != PRODUCTION_STREAMS:
        raise SystemExit("production mode requires 2048 streams")
    if not 0.0 < float(args.p_fa) < 0.5:
        raise SystemExit("--p-fa must be in (0, 0.5)")
    if not 0.0 < float(args.null_quantile) < 1.0:
        raise SystemExit("--null-quantile must be in (0, 1)")
    if args.input_iq is None:
        raise SystemExit("--input-iq is required so every shard binds to one waveform")
    if not args.input_iq.is_file():
        raise SystemExit(f"input IQ file does not exist: {args.input_iq}")
    if args.mode == MODE_PRODUCTION and args.waveform_audit is None:
        raise SystemExit("production mode requires --waveform-audit")
    if args.waveform_audit is not None:
        _read_passed_waveform_audit(args.waveform_audit)
    if args.gpu and not args.lib_path.is_file():
        raise SystemExit(f"CUDA library does not exist: {args.lib_path}")
    if args.stage == "audit" and not args.gpu:
        raise SystemExit("--stage audit requires --gpu for full-frame parity evidence")


def _artifact(path: Path | None, *, loaded: bool) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
        "loaded_by_this_study": bool(loaded),
    }


def _read_passed_waveform_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"waveform audit does not exist: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read waveform audit: {exc}") from exc
    if audit.get("schema_version") != "pilotproxy_atsc_waveform_audit_v1":
        raise SystemExit("waveform audit has an unsupported schema_version")
    if audit.get("quality", {}).get("quality_passed") is not True:
        raise SystemExit("waveform audit did not pass all declared quality gates")
    return audit


def _initialize_gpu(args: argparse.Namespace) -> GpuContext | None:
    if not args.gpu:
        return None
    import cupy as cp

    kernel = FStatKernel(args.lib_path)
    if not kernel.supports_row_projections() or not kernel.supports_fine_powers():
        raise SystemExit(
            "--gpu requires a kernel with exact row projections and fine powers"
        )
    if kernel.supports_fused_fine() and kernel.supports_fused_fine_mask():
        form = "fused_gpu_fine_powers_and_device_q16_epilogue"
    else:
        form = (
            "composed_gpu_row_projections_and_fine_powers_plus_"
            "host_exact_q16_decision"
        )
    return GpuContext(cp=cp, kernel=kernel, execution_form=form)


def _study_config(
    args: argparse.Namespace,
    *,
    bank: DetectorWeightBank,
    gpu: GpuContext | None,
) -> dict[str, Any]:
    input_iq = args.input_iq.resolve()
    generation_metadata = Path(f"{input_iq}.json")
    audit = (
        _read_passed_waveform_audit(args.waveform_audit)
        if args.waveform_audit is not None
        else None
    )
    weights_path = args.weights_path.resolve()
    manifest_path = weights_path.with_suffix(weights_path.suffix + ".manifest.json")
    gpu_info = None
    if gpu is not None:
        props = gpu.cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props["name"]
        gpu_info = {
            "device_name": raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name),
            "kernel_version": gpu.kernel.version.as_string(),
            "kernel_specs": gpu.kernel.specs.as_descriptive_dict(),
            "kernel_artifact": _artifact(args.lib_path, loaded=True),
            "execution_form": gpu.execution_form,
            "supports_fused_fine": bool(gpu.kernel.supports_fused_fine()),
            "supports_device_q16_epilogue": bool(
                gpu.kernel.supports_fused_fine_mask()
            ),
        }
    selected_kernel = None
    if args.lib_path.is_file():
        selected_kernel = _artifact(args.lib_path, loaded=False)
    scientific_scope = (
        "current_chime_profile_geometry"
        if int(args.num_streams) == PRODUCTION_STREAMS
        else "reduced_stream_code_validation_not_scientific_evidence"
    )
    config: dict[str, Any] = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "created_utc": _utc_now(),
        "mode": str(args.mode),
        "scientific_scope": scientific_scope,
        "physical_channels": list(args.physical_channel),
        "offsets_fine_bins": list(args.offset_fine_bins),
        "offsets_hz": [float(v) * FINE_BIN_HZ for v in args.offset_fine_bins],
        "snr_grid_db": sorted(args.snr_db),
        "geometry": {
            "num_streams": int(args.num_streams),
            "frame_size_samples": FRAME_SAMPLES,
            "detector_window_samples": K,
            "windows_per_stream": WINDOWS,
            "fine_bins": FINE_BINS,
            "fine_bin_width_hz": FINE_BIN_HZ,
            "synthetic_reference_pfb_output_spectral_sense": str(
                args.spectral_sense
            ),
            "deployment_raw_input_spectral_sense": "inverted",
            "weight_coordinate_system": "post_spectral_sense_normalized",
        },
        "simulation": {
            "primary_backend": str(args.simulation_backend),
            "common_random_numbers": True,
            "seed_rule": (
                "SHA256(base_seed,purpose,physical_channel,offset_microbins,"
                "trial_index); SNR and ablation stage deliberately omitted"
            ),
            "noise_model": (
                "independent circular complex Gaussian at the selected "
                "reference-PFB output, normalized by a deterministic PFB-gain "
                "calibration; the 8-VSB signal itself passes through the full "
                "reference ADC/PFB model"
            ),
            "reference_pfb_coordinate_note": (
                "The reference rFFT channelizer emits the post-normalization "
                "coordinate consumed by the current weight bank. The CHIME "
                "archive adapter's raw-input time reversal and the legacy "
                "lower-edge archive-phase conversion are therefore not "
                "applied a second time in this synthetic path."
            ),
            "pfb_gain_seed_base": PFB_GAIN_SEED_BASE,
            "noise_variance_complex": 1.0,
            "sufficient_statistic": {
                "pool_streams": int(args.sufficient_pool_streams),
                "additive_boundary": (
                    "per-stream [term,fine-bin] power after the complete stage "
                    "transform and before the 2048-stream sum"
                ),
                "aggregate_model": (
                    "multivariate Gaussian multiplier CLT using the empirical "
                    "per-stream mean/covariance; one multiplier vector is shared "
                    "across bins and ablation stages"
                ),
                "fixed_power_projection": (
                    "negative approximations clipped to zero, then rounded to "
                    "uint64 before exact rational/Q16 decisions"
                ),
                "uncertainty_limit": (
                    "curve/bootstrap uncertainty is conditional on the finite "
                    "per-stream pool; the declared full-frame audit is required"
                ),
            },
            "input_quantization_bits_per_component": BITS,
            "input_quantization_scale": float(args.input_scale),
            "input_scale_policy": (
                "fixed before H1 from unit-noise component sigma; never "
                "estimated from an SNR point or trial"
            ),
            "weight_dequantization_scale": WEIGHT_DEQUANTIZATION_SCALE,
            "pilot_below_data_db": float(args.pilot_below_data_db),
            "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
            "iq_sample_rate_hz": float(args.iq_sample_rate_hz),
        },
        "decision": {
            "requested_p_fa": float(args.p_fa),
            "null_rank_quantile": float(args.null_quantile),
            "designated_half_width": int(args.designated_half_width),
            "guard_fine_bins": int(args.guard_fine_bins),
            "threshold_selection": "observed_order_statistic_strict_greater_than",
            "q16_policy": "ceil(float_fixed_threshold * 2**16)",
            "crossing_targets": list(DEFAULT_BOOTSTRAP_TARGETS),
            "crossing_policy": "adjacent sampled bracket; no extrapolation",
        },
        "full_frame_audit_design": {
            "required_for_sufficient_statistic_precision_review": True,
            "physical_channels": list(args.audit_physical_channel),
            "offsets_fine_bins": list(args.audit_offset_fine_bins),
            "snr_db": sorted(args.audit_snr_db),
            "minimum_trials_per_null_and_h1_point": PRODUCTION_AUDIT_TRIALS,
            "geometry": "literal 2048-stream packed frame",
            "selected_gpu_artifact_required": True,
            "acceptance": {
                "gpu_fine_power_mismatches": 0,
                "gpu_q16_decision_mismatches": 0,
                "two_sample_ks_rule": (
                    "D <= 1.36 * sqrt((n_full+n_primary)/(n_full*n_primary)); "
                    "large-sample alpha approximately 0.05"
                ),
                "detection_probability_rule": (
                    "full-frame and primary Wilson 95% intervals overlap at "
                    "every declared SNR for float and fixed-Q16 stages"
                ),
                "maximum_sufficient_negative_power_clip_fraction": 1.0e-6,
            },
        },
        "stage_definitions": STAGE_DEFINITIONS,
        "waveform": {
            "input_iq": str(input_iq),
            "sha256": file_sha256(input_iq),
            "bytes": int(input_iq.stat().st_size),
            "format": "complex64 GNU Radio ATSC 8-VSB waveform",
            "generation_metadata": (
                _artifact(generation_metadata, loaded=True)
                if generation_metadata.is_file()
                else None
            ),
            "quality_audit": (
                {
                    "artifact": _artifact(args.waveform_audit, loaded=True),
                    "schema_version": audit["schema_version"],
                    "quality_passed": True,
                    "checks_passed": int(
                        audit["quality"]["num_quality_checks_passed"]
                    ),
                    "checks_total": int(audit["quality"]["num_quality_checks"]),
                    "pilot_frequency_error_hz": float(
                        audit["pilot_frequency_error_hz"]
                    ),
                    "measured_pilot_below_data_db": float(
                        audit["measured_pilot_below_data_db"]
                    ),
                    "occupied_bandwidth_hz": float(
                        audit["occupied_bandwidth_hz"]
                    ),
                }
                if audit is not None
                else None
            ),
        },
        "weights": {
            "artifact": _artifact(weights_path, loaded=True),
            "manifest": _artifact(manifest_path, loaded=True),
            "supported_physical_channels": bank.supported_physical_channels(),
        },
        "gpu_kernel_selection": selected_kernel,
        "gpu_execution_at_config_creation": gpu_info,
        "historical_kernel_artifact": _artifact(
            args.historical_kernel_artifact, loaded=False
        ),
        "software": {
            "git": _git_provenance(),
            "study_driver": _artifact(Path(__file__), loaded=True),
            "study_arithmetic": _artifact(
                SRC_ROOT / "pilot_proxy" / "testbench" / "sensitivity_study.py",
                loaded=True,
            ),
            "packed_cpu_reference": _artifact(
                SRC_ROOT / "pilot_proxy" / "detector_reference.py", loaded=True
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    identity_payload = dict(config)
    identity_payload.pop("created_utc", None)
    # Untracked paths are build products, not scientific identity. The study's
    # own output directory is created by --stage prepare, so binding identity to
    # the untracked list makes every later stage refuse its own run directory.
    # Modified *tracked* sources still bind, so shards cannot be pooled across
    # an edited working tree.
    identity_payload["software"] = _identity_software(config["software"])
    # Device name and whether this particular command executed CUDA are shard
    # provenance, not scientific-study identity.  The selected library hash is
    # retained above, so mixed machines cannot silently mix kernel artifacts.
    identity_payload.pop("gpu_execution_at_config_creation", None)
    config["config_sha256"] = _sha256_bytes(_canonical_json_bytes(identity_payload))
    return config


def _identity_software(software: Mapping[str, Any]) -> dict[str, Any]:
    """Return the software block with untracked paths removed from identity."""
    identity = dict(software)
    git = identity.get("git")
    if isinstance(git, Mapping):
        tracked = [
            line
            for line in git.get("dirty_paths", [])
            if not line.lstrip().startswith("??")
        ]
        identity["git"] = dict(git) | {
            "dirty": bool(tracked),
            "dirty_paths": tracked,
        }
    return identity


def _write_or_validate_config(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "study_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config.get("config_sha256"):
            raise SystemExit(
                f"output directory is bound to a different config: {path}"
            )
        return existing
    write_json_strict(path, config, indent=2, sort_keys=True)
    return config


def _profile(
    args: argparse.Namespace,
    bank: DetectorWeightBank,
    physical_channel: int,
    offset_fine_bins: float,
) -> StudyProfile:
    layout = bank.layout_for_physical_channel(int(physical_channel))
    packed, valid = bank.get_weights_for_physical_channel(int(physical_channel))
    if packed is None or not valid:
        raise RuntimeError(f"invalid packed weight profile for channel {physical_channel}")
    pilot_hz = physical_channel_to_pilot_hz(int(physical_channel))
    shifted_pilot_hz = pilot_hz + float(offset_fine_bins) * FINE_BIN_HZ
    anchor = predicted_pilot_fine_bin(
        pilot_rf_hz=shifted_pilot_hz,
        coarse_center_hz=float(layout["coarse_channel_center_hz"]),
        sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ,
        detector_window_samples=K,
        nfft=FRAME_SAMPLES,
        spectral_sense=args.spectral_sense,
        pad_factor=2,
    )
    designated = designated_bins(anchor, int(args.designated_half_width))
    bulk = independent_bin_mask(
        FINE_BINS,
        pad_factor=2,
        designated_bins=designated,
        guard_fine_bins=int(args.guard_fine_bins),
    )
    n_bulk = int(np.count_nonzero(bulk))
    rank = min(
        n_bulk - 1,
        max(0, int(math.floor(float(args.null_quantile) * (n_bulk - 1)))),
    )
    ideal = _ideal_float_weights_from_layout(
        layout, detector_window_samples=K
    )
    unpacked = unpack_packed_complex(packed, BITS, dtype=np.float64)
    unpacked = np.ascontiguousarray(unpacked / WEIGHT_DEQUANTIZATION_SCALE)
    return StudyProfile(
        physical_channel=int(physical_channel),
        offset_fine_bins=float(offset_fine_bins),
        anchor_bin=int(anchor),
        designated=designated,
        bulk_mask=bulk,
        cfar_rank=int(rank),
        ideal_weights=ideal,
        packed_weights=np.ascontiguousarray(packed),
        float_packed_weights=unpacked,
        layout=dict(layout),
    )


def _cache_path(output_dir: Path, profile: StudyProfile) -> Path:
    return output_dir / "signal_cache" / (
        f"ch{profile.physical_channel:02d}_off_{_safe_offset_label(profile.offset_fine_bins)}.npz"
    )


def _channelize_one(
    iq: np.ndarray,
    *,
    rf_center_hz: float,
    channel_index: int,
) -> np.ndarray:
    n_blocks = FRAME_SAMPLES + REFERENCE_PFB_TAPS - 1
    blocks = complex_envelope_to_real_adc_blocks(
        iq,
        iq_sample_rate_hz=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
        rf_center_hz=float(rf_center_hz),
        adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        band_lower_hz=REFERENCE_BAND_LOWER_HZ,
        n_blocks=n_blocks,
        block_size=REFERENCE_PFB_FFT_SIZE,
    )
    response = sinc_hamming_pfb_response(REFERENCE_PFB_TAPS, REFERENCE_PFB_FFT_SIZE)
    selected = channelize_real_blocks_to_reference_channels(
        blocks,
        channel_indices=[int(channel_index)],
        response=response,
        spec=ReferenceChannelizerSpec(),
    )
    # The checked-in current weights live in the
    # post_spectral_sense_normalized coordinate.  The reference rFFT output is
    # already in that centered coordinate; applying the archive lower-edge
    # conversion or the CHIME raw-input reversal here would move the pilot away
    # from the manifest target.  ``_prepare_one_cache`` gates the resulting
    # line against the geometry-predicted fine anchor.
    return np.ascontiguousarray(selected[0])


def _prepare_one_cache(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    profile: StudyProfile,
) -> Path:
    path = _cache_path(args.output_dir, profile)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive[SHARD_META_KEY].item()))
        if meta.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError(f"signal cache config mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    required = required_iq_samples(
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        num_output_samples=FRAME_SAMPLES,
    )
    clean_iq = np.fromfile(args.input_iq, dtype=np.complex64, count=required)
    if clean_iq.size < required:
        raise RuntimeError(
            f"input waveform too short: need {required}, got {clean_iq.size}"
        )
    pilot_hz = physical_channel_to_pilot_hz(profile.physical_channel)
    rf_center_hz = pilot_hz + (
        ATSC_CHANNEL_WIDTH_HZ / 2.0 - ATSC_PILOT_OFFSET_HZ
    )
    shifted = apply_channel_impairments(
        clean_iq,
        sample_rate_hz=float(args.iq_sample_rate_hz),
        frequency_offset_hz=profile.offset_fine_bins * FINE_BIN_HZ,
        gain_db=0.0,
        phase_deg=0.0,
    )
    spec = ReferenceChannelizerSpec()
    channel_index = nearest_reference_channel_index(pilot_hz, spec)
    clean_stream = _channelize_one(
        shifted, rf_center_hz=rf_center_hz, channel_index=channel_index
    )

    gain_seed = stage_seed(
        PFB_GAIN_SEED_BASE,
        purpose="pfb_gain",
        physical_channel=profile.physical_channel,
        offset_fine_bins=0.0,
        trial_index=0,
    )
    rng = np.random.default_rng(gain_seed)
    gain_noise = (
        rng.standard_normal(required, dtype=np.float32)
        + 1j * rng.standard_normal(required, dtype=np.float32)
    ) / math.sqrt(2.0)
    noise_stream = _channelize_one(
        np.asarray(gain_noise, dtype=np.complex64),
        rf_center_hz=rf_center_hz,
        channel_index=channel_index,
    )
    pfb_gain = float(np.mean(np.abs(noise_stream.astype(np.complex128)) ** 2))
    if not math.isfinite(pfb_gain) or pfb_gain <= 0.0:
        raise RuntimeError("reference PFB noise-gain calibration failed")
    clean_stream = np.asarray(clean_stream / math.sqrt(pfb_gain), dtype=np.complex64)
    one_stream_rows = stream_time_block_to_detector_matrix(
        clean_stream[np.newaxis, :], detector_window_samples=K
    )
    one_stream_rows = apply_spectral_sense_to_detector_matrix(
        one_stream_rows, spectral_sense=args.spectral_sense
    )
    z_target = one_stream_rows.astype(np.complex128) @ np.conjugate(
        profile.ideal_weights[0]
    )
    target_spectrum = np.fft.fft(z_target, n=FINE_BINS)
    observed_anchor = int(np.argmax(np.abs(target_spectrum)))
    circular_distance = min(
        (observed_anchor - profile.anchor_bin) % FINE_BINS,
        (profile.anchor_bin - observed_anchor) % FINE_BINS,
    )
    if circular_distance > int(args.designated_half_width):
        raise RuntimeError(
            "geometry-predicted fine anchor does not contain the synthetic "
            f"8-VSB line: predicted={profile.anchor_bin}, observed={observed_anchor}"
        )
    coherent_projection = target_spectrum[profile.anchor_bin] / WINDOWS
    m = np.arange(WINDOWS, dtype=np.float64)
    target_sequence = coherent_projection * np.exp(
        2j * math.pi * profile.anchor_bin * m / FINE_BINS
    )
    tone_rows = (
        target_sequence[:, np.newaxis] * profile.ideal_weights[0][np.newaxis, :] / K
    )

    clean_iq_power = float(np.mean(np.abs(clean_iq.astype(np.complex128)) ** 2))
    meta = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "config_sha256": config["config_sha256"],
        "created_utc": _utc_now(),
        "physical_channel": profile.physical_channel,
        "num_input_streams": int(args.num_streams),
        "offset_fine_bins": profile.offset_fine_bins,
        "offset_hz": profile.offset_fine_bins * FINE_BIN_HZ,
        "pilot_hz": pilot_hz,
        "rf_center_hz": rf_center_hz,
        "reference_channel_index": channel_index,
        "predicted_anchor_bin": profile.anchor_bin,
        "observed_clean_line_bin": observed_anchor,
        "predicted_observed_circular_distance_bins": int(circular_distance),
        "pfb_noise_gain": pfb_gain,
        "pfb_gain_seed": int(gain_seed),
        "clean_iq_power": clean_iq_power,
        "coherent_target_projection_abs": float(abs(coherent_projection)),
        "normalization": (
            "reference-PFB clean output divided by sqrt(PFB output power for "
            "unit-variance deterministic complex Gaussian IQ)"
        ),
    }
    np.savez_compressed(
        path,
        atsc_rows=np.asarray(one_stream_rows, dtype=np.complex64),
        ideal_tone_rows=np.asarray(tone_rows, dtype=np.complex64),
        **{SHARD_META_KEY: np.asarray(json.dumps(meta, sort_keys=True))},
    )
    return path


def _load_signal_cache(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    profile: StudyProfile,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = _cache_path(args.output_dir, profile)
    if not path.exists():
        _prepare_one_cache(args, config, profile)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive[SHARD_META_KEY].item()))
        atsc = np.asarray(archive["atsc_rows"], dtype=np.complex64)
        tone = np.asarray(archive["ideal_tone_rows"], dtype=np.complex64)
    if meta.get("config_sha256") != config["config_sha256"]:
        raise RuntimeError(f"signal cache config mismatch: {path}")
    if atsc.shape != (WINDOWS, K) or tone.shape != (WINDOWS, K):
        raise RuntimeError(f"signal cache has wrong shape: {path}")
    return atsc, tone, meta


def _signal_amplitude_for_snr(
    snr_db: float,
    *,
    clean_iq_power: float,
    args: argparse.Namespace,
) -> float:
    pilot_ratio = pilot_to_data_power_ratio(
        pilot_below_data_db=float(args.pilot_below_data_db)
    )
    noise_in_dtv_band = float(args.dtv_bandwidth_hz) / float(args.iq_sample_rate_hz)
    amplitude_sq = (
        10.0 ** (float(snr_db) / 10.0)
        * noise_in_dtv_band
        * (1.0 + pilot_ratio)
        / float(clean_iq_power)
    )
    return float(math.sqrt(amplitude_sq))


def _noise_rows(
    args: argparse.Namespace,
    profile: StudyProfile,
    *,
    purpose: str,
    trial_index: int,
    num_streams: int | None = None,
) -> tuple[np.ndarray, int]:
    seed = stage_seed(
        int(args.seed),
        purpose=purpose,
        physical_channel=profile.physical_channel,
        offset_fine_bins=profile.offset_fine_bins,
        trial_index=int(trial_index),
    )
    rng = np.random.default_rng(seed)
    streams = int(args.num_streams) if num_streams is None else int(num_streams)
    shape = (streams, WINDOWS, K)
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    rows = np.asarray((real + 1j * imag) / math.sqrt(2.0), dtype=np.complex64)
    return np.ascontiguousarray(rows.reshape(-1, K)), seed


def _clip_fraction(rows: np.ndarray, scale: float) -> float:
    bound = 7.0
    return float(
        np.mean(
            (np.abs(rows.real * float(scale)) > bound)
            | (np.abs(rows.imag * float(scale)) > bound)
        )
    )


def _gpu_fine_and_mask(
    gpu: GpuContext,
    *,
    packed: np.ndarray,
    profile: StudyProfile,
    multiplier_q16: int | None,
) -> tuple[np.ndarray, int | None]:
    cp = gpu.cp
    kernel = gpu.kernel
    d_input = cp.asarray(packed)
    d_diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(int(packed.shape[0]), d_input.data.ptr, d_diag.data.ptr)
    d_fine = cp.zeros((3, FINE_BINS), dtype=cp.uint64)
    d_powers = cp.zeros(3, dtype=cp.uint64)
    try:
        if gpu.execution_form == "fused_gpu_fine_powers_and_device_q16_epilogue":
            if multiplier_q16 is None:
                kernel.compute_fused_fine_u64(
                    handle,
                    profile.packed_weights.ctypes.data,
                    int(d_fine.data.ptr),
                    int(d_powers.data.ptr),
                    0,
                )
                cp.cuda.Device().synchronize()
                mask = None
            else:
                d_mask = cp.zeros(1, dtype=cp.int32)
                kernel.compute_fused_fine_mask_u64(
                    handle,
                    profile.packed_weights.ctypes.data,
                    profile.anchor_bin,
                    int(len(profile.designated) // 2),
                    pack_bulk_mask(profile.bulk_mask),
                    profile.cfar_rank,
                    int(multiplier_q16),
                    int(d_fine.data.ptr),
                    int(d_mask.data.ptr),
                    int(d_powers.data.ptr),
                    0,
                )
                cp.cuda.Device().synchronize()
                mask = int(cp.asnumpy(d_mask)[0])
        else:
            d_rows = cp.zeros((3, int(packed.shape[0]), 2), dtype=cp.int32)
            kernel.compute_row_projections_i32(
                handle,
                profile.packed_weights.ctypes.data,
                int(d_rows.data.ptr),
            )
            kernel.compute_fine_powers_u64(
                int(d_rows.data.ptr),
                int(packed.shape[0]) // WINDOWS,
                WINDOWS,
                1,
                int(d_fine.data.ptr),
            )
            cp.cuda.Device().synchronize()
            mask = None
        fine = cp.asnumpy(d_fine).astype(np.uint64, copy=False)
        if (
            multiplier_q16 is not None
            and gpu.execution_form
            != "fused_gpu_fine_powers_and_device_q16_epilogue"
        ):
            exact = exact_response_components(
                fine,
                designated=profile.designated,
                bulk_mask=profile.bulk_mask,
                cfar_rank=profile.cfar_rank,
            )
            mask = exact_q16_decision(
                exact, multiplier_q16=int(multiplier_q16)
            )
    finally:
        kernel.destroy(handle)
    return fine, mask


def _evaluate_trial(
    args: argparse.Namespace,
    profile: StudyProfile,
    *,
    noise_rows: np.ndarray,
    atsc_signal_rows: np.ndarray,
    tone_signal_rows: np.ndarray,
    signal_amplitude: float,
    gpu: GpuContext | None,
    multiplier_q16: int | None,
) -> dict[str, Any]:
    streams = int(args.num_streams)
    shaped_noise = noise_rows.reshape(streams, WINDOWS, K)
    atsc = np.asarray(
        shaped_noise + float(signal_amplitude) * atsc_signal_rows[np.newaxis, :, :],
        dtype=np.complex64,
    ).reshape(-1, K)
    tone = np.asarray(
        shaped_noise + float(signal_amplitude) * tone_signal_rows[np.newaxis, :, :],
        dtype=np.complex64,
    ).reshape(-1, K)

    packed = quantize_complex_numpy(atsc, BITS, float(args.input_scale))
    dequantized = unpack_packed_complex(packed, BITS, dtype=np.float64)
    dequantized = np.asarray(dequantized / float(args.input_scale), dtype=np.complex128)

    ratios: dict[str, float] = {}
    for stage, rows, weights in (
        (STAGE_IDEAL_TONE_FLOAT, tone, profile.ideal_weights),
        (STAGE_ATSC_FLOAT, atsc, profile.ideal_weights),
        (STAGE_INPUT_INT4, dequantized, profile.ideal_weights),
        (STAGE_WEIGHT_INT4, atsc, profile.float_packed_weights),
        (STAGE_JOINT_INT4_FLOAT, dequantized, profile.float_packed_weights),
    ):
        f2 = float_fine_power_ratio(
            rows, weights, num_streams=streams, windows_per_stream=WINDOWS
        )
        ratios[stage] = float_response_ratio(
            f2,
            designated=profile.designated,
            bulk_mask=profile.bulk_mask,
            cfar_rank=profile.cfar_rank,
        )

    projections = matched_filter_row_projections_cpu_reference_packed(
        packed, profile.packed_weights, BITS
    )
    fixed_powers = fine_power_fx(projections, num_streams=streams)
    exact = exact_response_components(
        fixed_powers,
        designated=profile.designated,
        bulk_mask=profile.bulk_mask,
        cfar_rank=profile.cfar_rank,
    )
    ratios[STAGE_FIXED_FLOAT_DECISION] = exact.response_ratio()
    cpu_q16 = (
        None
        if multiplier_q16 is None
        else exact_q16_decision(exact, multiplier_q16=int(multiplier_q16))
    )
    gpu_mask = None
    gpu_fine_mismatch = 0
    gpu_mask_mismatch = 0
    if gpu is not None:
        gpu_powers, gpu_mask = _gpu_fine_and_mask(
            gpu,
            packed=packed,
            profile=profile,
            multiplier_q16=multiplier_q16,
        )
        gpu_fine_mismatch = int(not np.array_equal(gpu_powers, fixed_powers))
        if gpu_fine_mismatch:
            raise AssertionError("GPU fused fine powers differ from CPU fxfft reference")
        if multiplier_q16 is not None:
            gpu_mask_mismatch = int(int(gpu_mask) != int(cpu_q16))
            if gpu_mask_mismatch:
                raise AssertionError("GPU Q16 mask differs from exact CPU decision")
    return {
        "ratios": ratios,
        "exact": exact,
        "clip_fraction": _clip_fraction(atsc, float(args.input_scale)),
        "cpu_q16": cpu_q16,
        "gpu_mask": gpu_mask,
        "gpu_fine_mismatch": gpu_fine_mismatch,
        "gpu_mask_mismatch": gpu_mask_mismatch,
    }


def _fixed_fine_powers_by_stream(
    projections: np.ndarray, *, num_streams: int
) -> np.ndarray:
    """Exact fixed powers at the additive stream-sum boundary."""
    arr = np.asarray(projections)
    streams = int(num_streams)
    if arr.shape != (3, streams * WINDOWS, 2):
        raise ValueError("fixed projections do not match the stream geometry")
    transformed = fxfft256(arr.reshape(3, streams, WINDOWS, 2)).astype(np.int64)
    powers = (
        transformed[..., 0] * transformed[..., 0]
        + transformed[..., 1] * transformed[..., 1]
    )
    return np.ascontiguousarray(powers.transpose(1, 0, 2), dtype=np.uint64)


def _power_response_ratio(
    powers: np.ndarray, *, profile: StudyProfile
) -> float:
    p = np.asarray(powers, dtype=np.float64)
    den = p[1] + p[2]
    f2 = np.divide(
        2.0 * p[0],
        den,
        out=np.zeros(FINE_BINS, dtype=np.float64),
        where=den > 0,
    )
    return float_response_ratio(
        f2,
        designated=profile.designated,
        bulk_mask=profile.bulk_mask,
        cfar_rank=profile.cfar_rank,
    )


def _evaluate_sufficient_pool(
    args: argparse.Namespace,
    profile: StudyProfile,
    *,
    noise_rows: np.ndarray,
    atsc_signal_rows: np.ndarray,
    tone_signal_rows: np.ndarray,
    signal_amplitude: float,
    gpu: GpuContext | None,
) -> dict[str, Any]:
    """Evaluate every ablation for a pool of independent single streams."""
    pool_streams = int(args.sufficient_pool_streams)
    shaped_noise = noise_rows.reshape(pool_streams, WINDOWS, K)
    atsc = np.asarray(
        shaped_noise + float(signal_amplitude) * atsc_signal_rows[np.newaxis, :, :],
        dtype=np.complex64,
    ).reshape(-1, K)
    tone = np.asarray(
        shaped_noise + float(signal_amplitude) * tone_signal_rows[np.newaxis, :, :],
        dtype=np.complex64,
    ).reshape(-1, K)
    packed = quantize_complex_numpy(atsc, BITS, float(args.input_scale))
    dequantized = unpack_packed_complex(packed, BITS, dtype=np.float64)
    dequantized = np.asarray(
        dequantized / float(args.input_scale), dtype=np.complex128
    )

    powers_by_stage: dict[str, np.ndarray] = {}
    for stage, rows, weights in (
        (STAGE_IDEAL_TONE_FLOAT, tone, profile.ideal_weights),
        (STAGE_ATSC_FLOAT, atsc, profile.ideal_weights),
        (STAGE_INPUT_INT4, dequantized, profile.ideal_weights),
        (STAGE_WEIGHT_INT4, atsc, profile.float_packed_weights),
        (STAGE_JOINT_INT4_FLOAT, dequantized, profile.float_packed_weights),
    ):
        powers = float_fine_powers_by_stream(
            rows,
            weights,
            num_streams=pool_streams,
            windows_per_stream=WINDOWS,
        )
        powers_by_stage[stage] = np.ascontiguousarray(
            powers.transpose(1, 0, 2), dtype=np.float64
        )

    projections = matched_filter_row_projections_cpu_reference_packed(
        packed, profile.packed_weights, BITS
    )
    fixed = _fixed_fine_powers_by_stream(
        projections, num_streams=pool_streams
    )
    powers_by_stage[STAGE_FIXED_FLOAT_DECISION] = fixed

    gpu_pool_mismatch: int | None = None
    if gpu is not None:
        gpu_powers, _ = _gpu_fine_and_mask(
            gpu, packed=packed, profile=profile, multiplier_q16=None
        )
        expected = fixed.sum(axis=0, dtype=np.uint64)
        gpu_pool_mismatch = int(not np.array_equal(gpu_powers, expected))
        if gpu_pool_mismatch:
            raise AssertionError(
                "GPU sufficient-pool fine powers differ from exact CPU sum"
            )
    return {
        "powers_by_stage": powers_by_stage,
        "clip_fraction": _clip_fraction(atsc, float(args.input_scale)),
        "gpu_pool_mismatch": gpu_pool_mismatch,
    }


def _aggregate_sufficient_pool(
    args: argparse.Namespace,
    profile: StudyProfile,
    *,
    purpose: str,
    powers_by_stage: Mapping[str, np.ndarray],
    multiplier_q16: int | None,
) -> dict[str, Any]:
    """Draw current-geometry sums from empirical per-stream sufficient vectors.

    This is a multivariate multiplier-CLT approximation. The same multiplier
    vector is used for every fine bin and every stage, retaining the empirical
    cross-bin and cross-ablation covariance learned from the common-noise pool.
    """
    stage_order = list(FLOAT_RESPONSE_STAGES)
    pool_streams = int(args.sufficient_pool_streams)
    target_streams = int(args.num_streams)
    trials = int(args.trials)
    matrices: list[np.ndarray] = []
    for stage in stage_order:
        values = np.asarray(powers_by_stage[stage], dtype=np.float64)
        if values.shape != (pool_streams, 3, FINE_BINS):
            raise ValueError(f"sufficient pool has wrong shape for {stage}")
        matrices.append(values.reshape(pool_streams, -1))
    joint = np.concatenate(matrices, axis=1)
    mean = np.mean(joint, axis=0, dtype=np.float64)
    centered = joint - mean[np.newaxis, :]

    coefficients = np.empty((trials, pool_streams), dtype=np.float64)
    trial_indices: list[int] = []
    aggregate_seeds: list[int] = []
    for local_index in range(trials):
        trial_index = int(args.trial_start) + local_index
        seed = stage_seed(
            int(args.seed),
            purpose=f"{purpose}_sufficient_aggregate",
            physical_channel=profile.physical_channel,
            offset_fine_bins=profile.offset_fine_bins,
            trial_index=trial_index,
        )
        coefficients[local_index] = np.random.default_rng(seed).standard_normal(
            pool_streams
        )
        trial_indices.append(trial_index)
        aggregate_seeds.append(seed)
    scale = math.sqrt(target_streams / (pool_streams - 1.0))
    aggregate = target_streams * mean[np.newaxis, :] + scale * (
        coefficients @ centered
    )

    ratios = {stage: [] for stage in FLOAT_RESPONSE_STAGES}
    exact_values: list[ExactResponseComponents] = []
    cpu_masks: list[int] = []
    negative_fraction_by_stage: dict[str, float] = {}
    width = 3 * FINE_BINS
    for stage_index, stage in enumerate(stage_order):
        block = aggregate[:, stage_index * width : (stage_index + 1) * width]
        negative_fraction_by_stage[stage] = float(np.mean(block < 0.0))
        block = np.maximum(block, 0.0).reshape(trials, 3, FINE_BINS)
        if stage != STAGE_FIXED_FLOAT_DECISION:
            for trial in range(trials):
                ratios[stage].append(
                    _power_response_ratio(block[trial], profile=profile)
                )
            continue
        uint = np.rint(block)
        if np.any(uint > np.iinfo(np.uint64).max):
            raise OverflowError("simulated aggregate fixed power exceeds uint64")
        fixed_uint = np.asarray(uint, dtype=np.uint64)
        for trial in range(trials):
            exact = exact_response_components(
                fixed_uint[trial],
                designated=profile.designated,
                bulk_mask=profile.bulk_mask,
                cfar_rank=profile.cfar_rank,
            )
            exact_values.append(exact)
            ratios[stage].append(exact.response_ratio())
            cpu_masks.append(
                -1
                if multiplier_q16 is None
                else exact_q16_decision(exact, multiplier_q16=multiplier_q16)
            )
    return {
        "ratios": ratios,
        "exact": exact_values,
        "cpu_q16": cpu_masks,
        "trial_index": trial_indices,
        "trial_seed": aggregate_seeds,
        "negative_fraction_by_stage": negative_fraction_by_stage,
    }


def _shard_path(
    args: argparse.Namespace,
    profile: StudyProfile,
    *,
    purpose: str,
    snr_db: float | None,
) -> Path:
    directory = args.output_dir / "shards" / purpose
    if snr_db is not None:
        directory = directory / f"snr_{_safe_snr_label(snr_db)}"
    filename = (
        f"ch{profile.physical_channel:02d}_off_{_safe_offset_label(profile.offset_fine_bins)}_"
        f"seed{int(args.seed)}_start{int(args.trial_start)}_n{int(args.trials)}.npz"
    )
    return directory / filename


def _write_shard(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        if resume:
            print(f"resume: keeping existing {path}")
            return
        raise RuntimeError(f"refusing to overwrite existing shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}")


def _run_full_shard(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    profile: StudyProfile,
    *,
    purpose: str,
    snr_db: float | None,
    gpu: GpuContext | None,
    calibration: Mapping[str, Any] | None,
) -> None:
    started = time.perf_counter()
    path = _shard_path(args, profile, purpose=purpose, snr_db=snr_db)
    if path.exists() and args.resume:
        print(f"resume: keeping existing {path}")
        return
    is_h1 = purpose in ("h1", "audit_h1")
    if is_h1:
        atsc, tone, signal_meta = _load_signal_cache(args, config, profile)
        amplitude = _signal_amplitude_for_snr(
            float(snr_db), clean_iq_power=float(signal_meta["clean_iq_power"]), args=args
        )
    else:
        atsc = np.zeros((WINDOWS, K), dtype=np.complex64)
        tone = np.zeros((WINDOWS, K), dtype=np.complex64)
        signal_meta = None
        amplitude = 0.0
    calibration_row = None
    multiplier_q16 = None
    if calibration is not None:
        calibration_row = _calibration_row(calibration, profile)
        multiplier_q16 = int(calibration_row["fixed_q16"]["multiplier_q16"])

    ratios = {stage: [] for stage in FLOAT_RESPONSE_STAGES}
    exact_values: list[ExactResponseComponents] = []
    seeds: list[int] = []
    trial_indices: list[int] = []
    clips: list[float] = []
    cpu_q16: list[int] = []
    gpu_masks: list[int] = []
    gpu_fine_mismatches: list[int] = []
    gpu_mask_mismatches: list[int] = []
    for local_index in range(int(args.trials)):
        trial_index = int(args.trial_start) + local_index
        noise, noise_seed = _noise_rows(
            args,
            profile,
            purpose=("null" if not is_h1 else "h1"),
            trial_index=trial_index,
        )
        result = _evaluate_trial(
            args,
            profile,
            noise_rows=noise,
            atsc_signal_rows=atsc,
            tone_signal_rows=tone,
            signal_amplitude=amplitude,
            gpu=gpu,
            multiplier_q16=multiplier_q16,
        )
        for stage in FLOAT_RESPONSE_STAGES:
            ratios[stage].append(float(result["ratios"][stage]))
        exact_values.append(result["exact"])
        seeds.append(int(noise_seed))
        trial_indices.append(trial_index)
        clips.append(float(result["clip_fraction"]))
        cpu_q16.append(-1 if result["cpu_q16"] is None else int(result["cpu_q16"]))
        gpu_masks.append(-1 if result["gpu_mask"] is None else int(result["gpu_mask"]))
        gpu_fine_mismatches.append(int(result["gpu_fine_mismatch"]))
        gpu_mask_mismatches.append(int(result["gpu_mask_mismatch"]))
        print(
            f"{purpose} ch{profile.physical_channel} off={profile.offset_fine_bins:+g} "
            f"snr={snr_db} trial={trial_index} clip={clips[-1]:.3e}"
        )
    meta = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "config_sha256": config["config_sha256"],
        "created_utc": _utc_now(),
        "purpose": purpose,
        "simulation_backend": SIMULATION_FULL_FRAME,
        "physical_channel": profile.physical_channel,
        "num_input_streams": int(args.num_streams),
        "offset_fine_bins": profile.offset_fine_bins,
        "snr_db": snr_db,
        "base_seed": int(args.seed),
        "trial_start": int(args.trial_start),
        "trials": int(args.trials),
        "anchor_bin": profile.anchor_bin,
        "designated_bins": profile.designated.tolist(),
        "cfar_rank": profile.cfar_rank,
        "n_bulk": int(np.count_nonzero(profile.bulk_mask)),
        "signal_amplitude": amplitude,
        "signal_cache_metadata": signal_meta,
        "calibration_applied": calibration_row,
        "gpu_executed": bool(gpu is not None),
        "gpu_full_frame_executed": bool(gpu is not None),
        "gpu_pool_executed": False,
        "gpu_validation_scope": (
            "full_frame_each_trial" if gpu is not None else "not_executed"
        ),
        "gpu_execution_form": None if gpu is None else gpu.execution_form,
        "gpu_kernel_version": (
            None if gpu is None else gpu.kernel.version.as_string()
        ),
        "gpu_kernel_artifact_sha256": (
            None if gpu is None else file_sha256(args.lib_path)
        ),
        "wall_seconds_before_npz_write": float(time.perf_counter() - started),
    }
    payload: dict[str, Any] = {
        SHARD_META_KEY: np.asarray(json.dumps(meta, sort_keys=True)),
        "trial_index": np.asarray(trial_indices, dtype=np.int64),
        "trial_seed": np.asarray(seeds, dtype=np.uint64),
        "clip_fraction": np.asarray(clips, dtype=np.float64),
        "cpu_q16_mask": np.asarray(cpu_q16, dtype=np.int8),
        "gpu_q16_mask": np.asarray(gpu_masks, dtype=np.int8),
        "gpu_fine_mismatch": np.asarray(gpu_fine_mismatches, dtype=np.int8),
        "gpu_mask_mismatch": np.asarray(gpu_mask_mismatches, dtype=np.int8),
    }
    payload.update(
        {
            f"ratio__{stage}": np.asarray(values, dtype=np.float64)
            for stage, values in ratios.items()
        }
    )
    payload.update(
        {f"fixed__{key}": value for key, value in exact_columns(exact_values).items()}
    )
    _write_shard(path, payload, resume=bool(args.resume))


def _run_sufficient_shard(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    profile: StudyProfile,
    *,
    purpose: str,
    snr_db: float | None,
    gpu: GpuContext | None,
    calibration: Mapping[str, Any] | None,
) -> None:
    started = time.perf_counter()
    if purpose not in ("null", "h1"):
        raise ValueError("sufficient-statistic shards support only null and h1")
    path = _shard_path(args, profile, purpose=purpose, snr_db=snr_db)
    if path.exists() and args.resume:
        print(f"resume: keeping existing {path}")
        return
    if purpose == "h1":
        atsc, tone, signal_meta = _load_signal_cache(args, config, profile)
        amplitude = _signal_amplitude_for_snr(
            float(snr_db),
            clean_iq_power=float(signal_meta["clean_iq_power"]),
            args=args,
        )
    else:
        atsc = np.zeros((WINDOWS, K), dtype=np.complex64)
        tone = np.zeros((WINDOWS, K), dtype=np.complex64)
        signal_meta = None
        amplitude = 0.0
    calibration_row = None
    multiplier_q16 = None
    if calibration is not None:
        calibration_row = _calibration_row(calibration, profile)
        multiplier_q16 = int(calibration_row["fixed_q16"]["multiplier_q16"])

    pool_noise, pool_seed = _noise_rows(
        args,
        profile,
        purpose=f"{purpose}_sufficient_pool",
        trial_index=0,
        num_streams=int(args.sufficient_pool_streams),
    )
    pool = _evaluate_sufficient_pool(
        args,
        profile,
        noise_rows=pool_noise,
        atsc_signal_rows=atsc,
        tone_signal_rows=tone,
        signal_amplitude=amplitude,
        gpu=gpu,
    )
    aggregate = _aggregate_sufficient_pool(
        args,
        profile,
        purpose=purpose,
        powers_by_stage=pool["powers_by_stage"],
        multiplier_q16=multiplier_q16,
    )
    trials = int(args.trials)
    print(
        f"{purpose} sufficient ch{profile.physical_channel} "
        f"off={profile.offset_fine_bins:+g} snr={snr_db} trials={trials} "
        f"pool={int(args.sufficient_pool_streams)} streams"
    )
    meta = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "config_sha256": config["config_sha256"],
        "created_utc": _utc_now(),
        "purpose": purpose,
        "simulation_backend": SIMULATION_SUFFICIENT,
        "physical_channel": profile.physical_channel,
        "offset_fine_bins": profile.offset_fine_bins,
        "snr_db": snr_db,
        "base_seed": int(args.seed),
        "trial_start": int(args.trial_start),
        "trials": trials,
        "num_input_streams": int(args.num_streams),
        "sufficient_pool_streams": int(args.sufficient_pool_streams),
        "sufficient_pool_seed": int(pool_seed),
        "sufficient_aggregate_method": (
            "empirical_multivariate_multiplier_clt_at_per_stream_fine_power_boundary"
        ),
        "negative_power_clip_fraction_by_stage": aggregate[
            "negative_fraction_by_stage"
        ],
        "anchor_bin": profile.anchor_bin,
        "designated_bins": profile.designated.tolist(),
        "cfar_rank": profile.cfar_rank,
        "n_bulk": int(np.count_nonzero(profile.bulk_mask)),
        "signal_amplitude": amplitude,
        "signal_cache_metadata": signal_meta,
        "calibration_applied": calibration_row,
        "gpu_executed": bool(gpu is not None),
        "gpu_full_frame_executed": False,
        "gpu_pool_executed": bool(gpu is not None),
        "gpu_validation_scope": (
            "sufficient_pool_aggregate_only" if gpu is not None else "not_executed"
        ),
        "gpu_pool_fine_mismatch": pool["gpu_pool_mismatch"],
        "gpu_execution_form": None if gpu is None else gpu.execution_form,
        "gpu_kernel_version": (
            None if gpu is None else gpu.kernel.version.as_string()
        ),
        "gpu_kernel_artifact_sha256": (
            None if gpu is None else file_sha256(args.lib_path)
        ),
        "wall_seconds_before_npz_write": float(time.perf_counter() - started),
    }
    payload: dict[str, Any] = {
        SHARD_META_KEY: np.asarray(json.dumps(meta, sort_keys=True)),
        "trial_index": np.asarray(aggregate["trial_index"], dtype=np.int64),
        "trial_seed": np.asarray(aggregate["trial_seed"], dtype=np.uint64),
        "clip_fraction": np.full(
            trials, float(pool["clip_fraction"]), dtype=np.float64
        ),
        "cpu_q16_mask": np.asarray(aggregate["cpu_q16"], dtype=np.int8),
        "gpu_q16_mask": np.full(trials, -1, dtype=np.int8),
        "gpu_fine_mismatch": np.full(trials, -1, dtype=np.int8),
        "gpu_mask_mismatch": np.full(trials, -1, dtype=np.int8),
    }
    payload.update(
        {
            f"ratio__{stage}": np.asarray(values, dtype=np.float64)
            for stage, values in aggregate["ratios"].items()
        }
    )
    payload.update(
        {
            f"fixed__{key}": value
            for key, value in exact_columns(aggregate["exact"]).items()
        }
    )
    _write_shard(path, payload, resume=bool(args.resume))


def _run_shard(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    profile: StudyProfile,
    *,
    purpose: str,
    snr_db: float | None,
    gpu: GpuContext | None,
    calibration: Mapping[str, Any] | None,
) -> None:
    runner = (
        _run_sufficient_shard
        if args.simulation_backend == SIMULATION_SUFFICIENT
        else _run_full_shard
    )
    runner(
        args,
        config,
        profile,
        purpose=purpose,
        snr_db=snr_db,
        gpu=gpu,
        calibration=calibration,
    )


def _iter_shards(output_dir: Path, purpose: str) -> Iterable[Path]:
    root = output_dir / "shards" / purpose
    if not root.exists():
        return []
    return sorted(root.rglob("*.npz"))


def _load_shards(
    output_dir: Path,
    *,
    purpose: str,
    config_sha256: str,
    physical_channel: int,
    offset_fine_bins: float,
    snr_db: float | None,
) -> dict[str, Any]:
    ratio_lists = {stage: [] for stage in FLOAT_RESPONSE_STAGES}
    exact_lists: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "designated_num",
            "designated_den",
            "rank_num",
            "rank_den",
            "designated_bin",
            "rank_bin",
        )
    }
    trial_keys: list[tuple[int, int]] = []
    clips: list[np.ndarray] = []
    cpu_masks: list[np.ndarray] = []
    gpu_masks: list[np.ndarray] = []
    gpu_fine_mismatch: list[np.ndarray] = []
    gpu_mask_mismatch: list[np.ndarray] = []
    paths: list[str] = []
    gpu_execution_forms: set[str] = set()
    gpu_kernel_versions: set[str] = set()
    gpu_kernel_artifact_sha256s: set[str] = set()
    gpu_executed_trials = 0
    gpu_full_frame_trials = 0
    gpu_pool_shards = 0
    gpu_pool_fine_mismatches = 0
    gpu_validation_scopes: set[str] = set()
    simulation_backends: set[str] = set()
    sufficient_negative_power_clip_max: dict[str, float] = {}
    recorded_wall_seconds = 0.0
    seen: set[tuple[int, int]] = set()
    for path in _iter_shards(output_dir, purpose):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive[SHARD_META_KEY].item()))
            if meta.get("config_sha256") != config_sha256:
                continue
            if int(meta["physical_channel"]) != int(physical_channel):
                continue
            if not math.isclose(
                float(meta["offset_fine_bins"]), float(offset_fine_bins), abs_tol=1e-12
            ):
                continue
            meta_snr = meta.get("snr_db")
            if snr_db is None:
                if meta_snr is not None:
                    continue
            elif meta_snr is None or not math.isclose(
                float(meta_snr), float(snr_db), abs_tol=1e-12
            ):
                continue
            trial_index = np.asarray(archive["trial_index"], dtype=np.int64)
            base_seed = int(meta["base_seed"])
            keys = [(base_seed, int(index)) for index in trial_index]
            duplicate = [key for key in keys if key in seen]
            if duplicate:
                raise RuntimeError(
                    f"overlapping trial coordinates in {path}: first {duplicate[0]}"
                )
            seen.update(keys)
            trial_keys.extend(keys)
            for stage in FLOAT_RESPONSE_STAGES:
                ratio_lists[stage].append(
                    np.asarray(archive[f"ratio__{stage}"], dtype=np.float64)
                )
            for key in exact_lists:
                exact_lists[key].append(np.asarray(archive[f"fixed__{key}"]))
            clips.append(np.asarray(archive["clip_fraction"], dtype=np.float64))
            cpu_masks.append(np.asarray(archive["cpu_q16_mask"], dtype=np.int8))
            gpu_masks.append(np.asarray(archive["gpu_q16_mask"], dtype=np.int8))
            gpu_fine_mismatch.append(
                np.asarray(archive["gpu_fine_mismatch"], dtype=np.int8)
            )
            gpu_mask_mismatch.append(
                np.asarray(archive["gpu_mask_mismatch"], dtype=np.int8)
            )
            if meta.get("gpu_execution_form"):
                gpu_execution_forms.add(str(meta["gpu_execution_form"]))
            if meta.get("gpu_executed"):
                gpu_executed_trials += len(keys)
            if meta.get("gpu_full_frame_executed"):
                gpu_full_frame_trials += len(keys)
            if meta.get("gpu_pool_executed"):
                gpu_pool_shards += 1
                gpu_pool_fine_mismatches += int(
                    meta.get("gpu_pool_fine_mismatch", 0)
                )
            if meta.get("gpu_validation_scope"):
                gpu_validation_scopes.add(str(meta["gpu_validation_scope"]))
            if meta.get("simulation_backend"):
                simulation_backends.add(str(meta["simulation_backend"]))
            for stage, fraction in meta.get(
                "negative_power_clip_fraction_by_stage", {}
            ).items():
                sufficient_negative_power_clip_max[str(stage)] = max(
                    sufficient_negative_power_clip_max.get(str(stage), 0.0),
                    float(fraction),
                )
            recorded_wall_seconds += float(meta.get("wall_seconds_before_npz_write", 0.0))
            if meta.get("gpu_kernel_version"):
                gpu_kernel_versions.add(str(meta["gpu_kernel_version"]))
            if meta.get("gpu_kernel_artifact_sha256"):
                gpu_kernel_artifact_sha256s.add(
                    str(meta["gpu_kernel_artifact_sha256"])
                )
            paths.append(str(path))
    if not paths:
        return {"n": 0, "paths": []}
    order = np.asarray(
        sorted(range(len(trial_keys)), key=lambda index: trial_keys[index]),
        dtype=np.int64,
    )

    def joined(parts: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts)[order]

    exact_columns_joined = {key: joined(parts) for key, parts in exact_lists.items()}
    return {
        "n": int(len(order)),
        "paths": paths,
        "trial_keys": [trial_keys[int(index)] for index in order],
        "ratios": {stage: joined(parts) for stage, parts in ratio_lists.items()},
        "exact": components_from_columns(exact_columns_joined),
        "clip_fraction": joined(clips),
        "cpu_q16_mask": joined(cpu_masks),
        "gpu_q16_mask": joined(gpu_masks),
        "gpu_fine_mismatch": joined(gpu_fine_mismatch),
        "gpu_mask_mismatch": joined(gpu_mask_mismatch),
        "gpu_execution_forms": sorted(gpu_execution_forms),
        "gpu_kernel_versions": sorted(gpu_kernel_versions),
        "gpu_kernel_artifact_sha256s": sorted(gpu_kernel_artifact_sha256s),
        "gpu_executed_trials": int(gpu_executed_trials),
        "gpu_full_frame_trials": int(gpu_full_frame_trials),
        "gpu_pool_shards": int(gpu_pool_shards),
        "gpu_pool_fine_mismatches": int(gpu_pool_fine_mismatches),
        "gpu_validation_scopes": sorted(gpu_validation_scopes),
        "simulation_backends": sorted(simulation_backends),
        "sufficient_negative_power_clip_fraction_max_by_stage": (
            sufficient_negative_power_clip_max
        ),
        "recorded_wall_seconds_before_npz_write": float(recorded_wall_seconds),
    }


def _calibration_key(profile: StudyProfile) -> str:
    return f"ch{profile.physical_channel:02d}_off_{_safe_offset_label(profile.offset_fine_bins)}"


def _calibration_row(
    calibration: Mapping[str, Any], profile: StudyProfile
) -> Mapping[str, Any]:
    key = _calibration_key(profile)
    rows = calibration.get("profiles", {})
    if key not in rows:
        raise RuntimeError(f"calibration lacks profile {key}")
    return rows[key]


def _build_calibration(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    bank: DetectorWeightBank,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for channel in args.physical_channel:
        for offset in args.offset_fine_bins:
            profile = _profile(args, bank, channel, offset)
            data = _load_shards(
                args.output_dir,
                purpose="null",
                config_sha256=str(config["config_sha256"]),
                physical_channel=channel,
                offset_fine_bins=offset,
                snr_db=None,
            )
            if data["n"] == 0:
                raise RuntimeError(
                    f"no null shards for channel {channel}, offset {offset}"
                )
            stage_calibration = {
                stage: order_statistic_threshold(
                    data["ratios"][stage], p_fa=float(args.p_fa)
                )
                for stage in FLOAT_RESPONSE_STAGES
            }
            fixed_float = stage_calibration[STAGE_FIXED_FLOAT_DECISION]
            multiplier_q16 = q16_ceil_multiplier(
                float(fixed_float["threshold_multiplier"])
            )
            q16_decisions = np.asarray(
                [
                    exact_q16_decision(item, multiplier_q16=multiplier_q16)
                    for item in data["exact"]
                ],
                dtype=np.int8,
            )
            profiles[_calibration_key(profile)] = {
                "physical_channel": channel,
                "offset_fine_bins": offset,
                "anchor_bin": profile.anchor_bin,
                "designated_bins": profile.designated.tolist(),
                "bulk_mask_words_hex": [
                    f"0x{word:016x}" for word in pack_bulk_mask(profile.bulk_mask)
                ],
                "n_bulk": int(np.count_nonzero(profile.bulk_mask)),
                "cfar_rank": profile.cfar_rank,
                "stage_float_thresholds": stage_calibration,
                "fixed_q16": {
                    "multiplier_q16": multiplier_q16,
                    "multiplier_float_equivalent": multiplier_q16 / 65536.0,
                    "rounding_loss_upper_bound_multiplier": 1.0 / 65536.0,
                    "null_trials": int(data["n"]),
                    "null_exceedances": int(np.sum(q16_decisions)),
                    "empirical_p_fa": float(np.mean(q16_decisions)),
                },
                "null_shards": data["paths"],
                "gpu_fine_mismatches": int(
                    np.count_nonzero(data["gpu_fine_mismatch"] > 0)
                ),
                "gpu_sufficient_pool_fine_mismatches": int(
                    data["gpu_pool_fine_mismatches"]
                ),
                "calibration_scope": (
                    "synthetic-null operating point for this sensitivity study; "
                    "not a substitute for the pending archive/live bundle calibration"
                ),
            }
    calibration = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "created_utc": _utc_now(),
        "config_sha256": config["config_sha256"],
        "requested_p_fa": float(args.p_fa),
        "profiles": profiles,
    }
    write_json_strict(
        args.output_dir / "calibration.json", calibration, indent=2, sort_keys=True
    )
    return calibration


def _load_calibration(
    output_dir: Path, config_sha256: str
) -> dict[str, Any]:
    path = output_dir / "calibration.json"
    if not path.exists():
        raise RuntimeError("calibration.json is missing; run --stage calibrate first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("config_sha256") != config_sha256:
        raise RuntimeError("calibration.json belongs to a different study config")
    return data


def _decisions_for_stage(
    data: Mapping[str, Any],
    calibration_row: Mapping[str, Any],
    stage: str,
) -> np.ndarray | None:
    if stage in FLOAT_RESPONSE_STAGES:
        threshold = float(
            calibration_row["stage_float_thresholds"][stage]["threshold_multiplier"]
        )
        return np.asarray(data["ratios"][stage] > threshold, dtype=np.int8)
    if stage == STAGE_FIXED_Q16_CPU:
        q16 = int(calibration_row["fixed_q16"]["multiplier_q16"])
        return np.asarray(
            [exact_q16_decision(item, multiplier_q16=q16) for item in data["exact"]],
            dtype=np.int8,
        )
    if stage == STAGE_FULL_GPU:
        masks = np.asarray(data["gpu_q16_mask"], dtype=np.int8)
        return None if np.any(masks < 0) else masks
    raise KeyError(stage)


def _write_report_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _empirical_ks_distance(a: Sequence[float], b: Sequence[float]) -> float | None:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    left = np.sort(left[np.isfinite(left)])
    right = np.sort(right[np.isfinite(right)])
    if left.size == 0 or right.size == 0:
        return None
    support = np.sort(np.concatenate((left, right)))
    cdf_left = np.searchsorted(left, support, side="right") / left.size
    cdf_right = np.searchsorted(right, support, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))


def _ks_comparison(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    n_left = int(np.count_nonzero(np.isfinite(left)))
    n_right = int(np.count_nonzero(np.isfinite(right)))
    distance = _empirical_ks_distance(left, right)
    critical = (
        None
        if n_left == 0 or n_right == 0
        else float(1.36 * math.sqrt((n_left + n_right) / (n_left * n_right)))
    )
    return {
        "distance": distance,
        "critical_value": critical,
        "finite_full_frame_trials": n_left,
        "finite_primary_trials": n_right,
        "passes_declared_rule": bool(
            distance is not None and critical is not None and distance <= critical
        ),
        "rule": (
            "D <= 1.36*sqrt((n_full+n_primary)/(n_full*n_primary)); "
            "large-sample alpha approximately 0.05"
        ),
    }


def _wilson_intervals_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return bool(
        float(a["wilson95_lo"]) <= float(b["wilson95_hi"])
        and float(b["wilson95_lo"]) <= float(a["wilson95_hi"])
    )


def _decision_summary(
    data: Mapping[str, Any],
    calibration_row: Mapping[str, Any],
    stage: str,
) -> dict[str, Any] | None:
    decisions = _decisions_for_stage(data, calibration_row, stage)
    if decisions is None:
        return None
    detected = int(np.sum(decisions))
    lo, hi = wilson_interval(detected, int(data["n"]))
    return {
        "trials": int(data["n"]),
        "detected": detected,
        "detection_probability": float(detected / data["n"]),
        "wilson95_lo": lo,
        "wilson95_hi": hi,
    }


def _build_report(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    bank: DetectorWeightBank,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    curve_rows: list[dict[str, Any]] = []
    profile_reports: dict[str, Any] = {}
    for channel in args.physical_channel:
        for offset in args.offset_fine_bins:
            profile = _profile(args, bank, channel, offset)
            key = _calibration_key(profile)
            cal_row = _calibration_row(calibration, profile)
            null = _load_shards(
                args.output_dir,
                purpose="null",
                config_sha256=str(config["config_sha256"]),
                physical_channel=channel,
                offset_fine_bins=offset,
                snr_db=None,
            )
            by_snr: dict[float, dict[str, Any]] = {}
            for snr in sorted(args.snr_db):
                data = _load_shards(
                    args.output_dir,
                    purpose="h1",
                    config_sha256=str(config["config_sha256"]),
                    physical_channel=channel,
                    offset_fine_bins=offset,
                    snr_db=snr,
                )
                if data["n"] == 0:
                    continue
                by_snr[float(snr)] = data
                for stage in REPORT_STAGES:
                    decisions = _decisions_for_stage(data, cal_row, stage)
                    if decisions is None:
                        continue
                    detected = int(np.sum(decisions))
                    lo, hi = wilson_interval(detected, int(data["n"]))
                    curve_rows.append(
                        {
                            "physical_channel": channel,
                            "num_input_streams": int(args.num_streams),
                            "scientific_scope": config["scientific_scope"],
                            "simulation_backend": config["simulation"][
                                "primary_backend"
                            ],
                            "offset_fine_bins": offset,
                            "offset_hz": offset * FINE_BIN_HZ,
                            "snr_db": snr,
                            "stage": stage,
                            "trials": int(data["n"]),
                            "detected": detected,
                            "detection_probability": float(detected / data["n"]),
                            "wilson95_lo": lo,
                            "wilson95_hi": hi,
                            "clip_fraction_mean": float(np.mean(data["clip_fraction"])),
                            "gpu_fine_mismatches": int(
                                np.count_nonzero(data["gpu_fine_mismatch"] > 0)
                            ),
                            "gpu_mask_mismatches": int(
                                np.count_nonzero(data["gpu_mask_mismatch"] > 0)
                            ),
                        }
                    )
            audit_design = config["full_frame_audit_design"]
            audit_declared = bool(
                channel in audit_design["physical_channels"]
                and offset in audit_design["offsets_fine_bins"]
            )
            audit_report: dict[str, Any] = {
                "declared_for_profile": audit_declared,
                "required_trials_per_point": int(
                    audit_design["minimum_trials_per_null_and_h1_point"]
                ),
                "snr_db": list(audit_design["snr_db"]),
                "points": {},
                "complete_for_precision_review": False,
            }
            if audit_declared:
                audit_null = _load_shards(
                    args.output_dir,
                    purpose="audit_null",
                    config_sha256=str(config["config_sha256"]),
                    physical_channel=channel,
                    offset_fine_bins=offset,
                    snr_db=None,
                )
                required_audit_trials = int(
                    audit_design["minimum_trials_per_null_and_h1_point"]
                )
                audit_report["null_trials"] = int(audit_null["n"])
                if audit_null["n"]:
                    audit_report["null"] = {
                        "response_ks_full_vs_primary": {
                            STAGE_ATSC_FLOAT: _ks_comparison(
                                audit_null["ratios"][STAGE_ATSC_FLOAT],
                                null["ratios"][STAGE_ATSC_FLOAT],
                            ),
                            STAGE_FIXED_FLOAT_DECISION: _ks_comparison(
                                audit_null["ratios"][STAGE_FIXED_FLOAT_DECISION],
                                null["ratios"][STAGE_FIXED_FLOAT_DECISION],
                            ),
                        },
                        "decisions_at_primary_calibration": {
                            stage: summary
                            for stage in (
                                STAGE_ATSC_FLOAT,
                                STAGE_FIXED_Q16_CPU,
                                STAGE_FULL_GPU,
                            )
                            if (
                                summary := _decision_summary(
                                    audit_null, cal_row, stage
                                )
                            )
                            is not None
                        },
                    }
                audit_point_complete: list[bool] = []
                for audit_snr in audit_design["snr_db"]:
                    audit_data = _load_shards(
                        args.output_dir,
                        purpose="audit_h1",
                        config_sha256=str(config["config_sha256"]),
                        physical_channel=channel,
                        offset_fine_bins=offset,
                        snr_db=float(audit_snr),
                    )
                    primary = by_snr.get(float(audit_snr))
                    point: dict[str, Any] = {
                        "full_frame_trials": int(audit_data["n"]),
                        "primary_trials": 0 if primary is None else int(primary["n"]),
                    }
                    if audit_data["n"]:
                        point["full_frame_decisions"] = {
                            stage: summary
                            for stage in (
                                STAGE_ATSC_FLOAT,
                                STAGE_FIXED_Q16_CPU,
                                STAGE_FULL_GPU,
                            )
                            if (
                                summary := _decision_summary(
                                    audit_data, cal_row, stage
                                )
                            )
                            is not None
                        }
                    if audit_data["n"] and primary is not None:
                        point["response_ks_full_vs_primary"] = {
                            STAGE_ATSC_FLOAT: _ks_comparison(
                                audit_data["ratios"][STAGE_ATSC_FLOAT],
                                primary["ratios"][STAGE_ATSC_FLOAT],
                            ),
                            STAGE_FIXED_FLOAT_DECISION: _ks_comparison(
                                audit_data["ratios"][STAGE_FIXED_FLOAT_DECISION],
                                primary["ratios"][STAGE_FIXED_FLOAT_DECISION],
                            ),
                        }
                        point["detection_probability_full_minus_primary"] = {}
                        point["detection_wilson95_overlap"] = {}
                        for stage in (STAGE_ATSC_FLOAT, STAGE_FIXED_Q16_CPU):
                            full_summary = _decision_summary(audit_data, cal_row, stage)
                            primary_summary = _decision_summary(primary, cal_row, stage)
                            if full_summary is not None and primary_summary is not None:
                                point["detection_probability_full_minus_primary"][stage] = float(
                                    full_summary["detection_probability"]
                                    - primary_summary["detection_probability"]
                                )
                                point["detection_wilson95_overlap"][stage] = (
                                    _wilson_intervals_overlap(
                                        full_summary, primary_summary
                                    )
                                )
                    audit_report["points"][str(audit_snr)] = point
                    audit_point_complete.append(
                        audit_data["n"] >= required_audit_trials
                        and primary is not None
                    )
                audit_data_all = [audit_null]
                for audit_snr in audit_design["snr_db"]:
                    loaded = _load_shards(
                        args.output_dir,
                        purpose="audit_h1",
                        config_sha256=str(config["config_sha256"]),
                        physical_channel=channel,
                        offset_fine_bins=offset,
                        snr_db=float(audit_snr),
                    )
                    if loaded["n"]:
                        audit_data_all.append(loaded)
                audit_report["gpu_execution_forms"] = sorted(
                    {
                        form
                        for data in audit_data_all
                        if data["n"]
                        for form in data["gpu_execution_forms"]
                    }
                )
                audit_report["gpu_kernel_artifact_sha256s"] = sorted(
                    {
                        digest
                        for data in audit_data_all
                        if data["n"]
                        for digest in data["gpu_kernel_artifact_sha256s"]
                    }
                )
                audit_report["gpu_fine_power_mismatches"] = int(
                    sum(
                        np.count_nonzero(data["gpu_fine_mismatch"] > 0)
                        for data in audit_data_all
                        if data["n"]
                    )
                )
                audit_report["gpu_q16_decision_mismatches"] = int(
                    sum(
                        np.count_nonzero(data["gpu_mask_mismatch"] > 0)
                        for data in audit_data_all
                        if data["n"]
                    )
                )
                audit_report["recorded_wall_seconds_before_npz_write"] = float(
                    sum(
                        data["recorded_wall_seconds_before_npz_write"]
                        for data in audit_data_all
                        if data["n"]
                    )
                )
                ks_passes: list[bool] = []
                if "null" in audit_report:
                    ks_passes.extend(
                        bool(item["passes_declared_rule"])
                        for item in audit_report["null"][
                            "response_ks_full_vs_primary"
                        ].values()
                    )
                wilson_passes: list[bool] = []
                for point in audit_report["points"].values():
                    ks_passes.extend(
                        bool(item["passes_declared_rule"])
                        for item in point.get(
                            "response_ks_full_vs_primary", {}
                        ).values()
                    )
                    wilson_passes.extend(
                        bool(value)
                        for value in point.get(
                            "detection_wilson95_overlap", {}
                        ).values()
                    )
                audit_report["comparison_acceptance"] = {
                    "ks_checks": len(ks_passes),
                    "ks_all_pass": bool(ks_passes and all(ks_passes)),
                    "wilson_overlap_checks": len(wilson_passes),
                    "wilson_all_pass": bool(
                        wilson_passes and all(wilson_passes)
                    ),
                    "passes": bool(
                        ks_passes
                        and all(ks_passes)
                        and wilson_passes
                        and all(wilson_passes)
                    ),
                }
                audit_report["complete_for_precision_review"] = bool(
                    audit_null["n"] >= required_audit_trials
                    and audit_point_complete
                    and all(audit_point_complete)
                    and audit_report["gpu_execution_forms"]
                    and audit_report["gpu_fine_power_mismatches"] == 0
                    and audit_report["gpu_q16_decision_mismatches"] == 0
                    and audit_report["comparison_acceptance"]["passes"]
                )
            stages_report: dict[str, Any] = {}
            for stage in REPORT_STAGES:
                stage_rows = [
                    row
                    for row in curve_rows
                    if row["physical_channel"] == channel
                    and math.isclose(row["offset_fine_bins"], offset)
                    and row["stage"] == stage
                ]
                if not stage_rows:
                    stages_report[stage] = {
                        "status": "not_executed_or_no_complete_shards"
                    }
                    continue
                stage_rows.sort(key=lambda row: row["snr_db"])
                stages_report[stage] = {
                    "status": "evaluated",
                    "crossings": {
                        str(target): crossing_bracket(
                            [row["snr_db"] for row in stage_rows],
                            [row["detection_probability"] for row in stage_rows],
                            target=target,
                        )
                        for target in DEFAULT_BOOTSTRAP_TARGETS
                    },
                }
            bootstrap: dict[str, Any] = {}
            common_snrs = sorted(by_snr)
            if common_snrs and null["n"] > 0:
                for target in DEFAULT_BOOTSTRAP_TARGETS:
                    bootstrap[str(target)] = paired_crossing_bootstrap(
                        snr_db=common_snrs,
                        float_null_ratios=null["ratios"][STAGE_ATSC_FLOAT],
                        fixed_null_ratios=null["ratios"][STAGE_FIXED_FLOAT_DECISION],
                        null_trial_keys=null["trial_keys"],
                        float_h1_ratios_by_snr={
                            snr: by_snr[snr]["ratios"][STAGE_ATSC_FLOAT]
                            for snr in common_snrs
                        },
                        fixed_h1_components_by_snr={
                            snr: by_snr[snr]["exact"] for snr in common_snrs
                        },
                        trial_keys_by_snr={
                            snr: by_snr[snr]["trial_keys"] for snr in common_snrs
                        },
                        p_fa=float(args.p_fa),
                        target_pd=target,
                        replicates=int(args.bootstrap_replicates),
                        seed=stage_seed(
                            int(args.seed),
                            purpose=f"bootstrap_pd_{target}",
                            physical_channel=channel,
                            offset_fine_bins=offset,
                            trial_index=0,
                        ),
                    )
            float_cross = stages_report.get(STAGE_ATSC_FLOAT, {}).get("crossings", {})
            fixed_cross = stages_report.get(STAGE_FIXED_Q16_CPU, {}).get(
                "crossings", {}
            )
            point_losses: dict[str, Any] = {}
            for target in DEFAULT_BOOTSTRAP_TARGETS:
                f = float_cross.get(str(target), {})
                q = fixed_cross.get(str(target), {})
                point_losses[str(target)] = {
                    "bracketed": bool(f.get("bracketed") and q.get("bracketed")),
                    "fixed_minus_float_sensitivity_loss_db": (
                        float(q["estimate_db"] - f["estimate_db"])
                        if f.get("bracketed") and q.get("bracketed")
                        else None
                    ),
                }
            precision_targets_bracketed = all(
                bool(float_cross.get(str(target), {}).get("bracketed"))
                and bool(fixed_cross.get(str(target), {}).get("bracketed"))
                for target in DEFAULT_BOOTSTRAP_TARGETS
            )
            planned_snr_grid_complete = set(common_snrs) == {
                float(snr) for snr in args.snr_db
            }
            all_data = [null, *by_snr.values()]
            gpu_execution_forms = sorted(
                {
                    form
                    for data in all_data
                    for form in data["gpu_execution_forms"]
                }
            )
            negative_power_clip_max: dict[str, float] = {}
            for data in all_data:
                for stage, fraction in data[
                    "sufficient_negative_power_clip_fraction_max_by_stage"
                ].items():
                    negative_power_clip_max[stage] = max(
                        negative_power_clip_max.get(stage, 0.0), float(fraction)
                    )
            negative_clip_limit = float(
                audit_design["acceptance"][
                    "maximum_sufficient_negative_power_clip_fraction"
                ]
            )
            negative_clip_pass = bool(
                not negative_power_clip_max
                or max(negative_power_clip_max.values()) <= negative_clip_limit
            )
            if not negative_clip_pass:
                for target in DEFAULT_BOOTSTRAP_TARGETS:
                    point_losses[str(target)] = {
                        "bracketed": False,
                        "fixed_minus_float_sensitivity_loss_db": None,
                        "withheld_reason": (
                            "sufficient-statistic negative-power clipping exceeds "
                            "the declared acceptance limit"
                        ),
                    }
                    if str(target) in bootstrap:
                        bootstrap[str(target)][
                            "fixed_minus_float_sensitivity_loss_median_db"
                        ] = None
                        bootstrap[str(target)][
                            "fixed_minus_float_sensitivity_loss_bootstrap95_lo_db"
                        ] = None
                        bootstrap[str(target)][
                            "fixed_minus_float_sensitivity_loss_bootstrap95_hi_db"
                        ] = None
                        bootstrap[str(target)]["sensitivity_loss_withheld_reason"] = (
                            "sufficient-statistic negative-power clipping exceeds "
                            "the declared acceptance limit"
                        )
            audit_requirement_satisfied = bool(
                config["simulation"]["primary_backend"] == SIMULATION_FULL_FRAME
                or audit_report["complete_for_precision_review"]
            )
            profile_reports[key] = {
                "physical_channel": channel,
                "num_input_streams": int(args.num_streams),
                "scientific_scope": config["scientific_scope"],
                "simulation_backend": config["simulation"]["primary_backend"],
                "offset_fine_bins": offset,
                "anchor_bin": profile.anchor_bin,
                "channel14_circular_wrap_profile": bool(
                    profile.layout.get("edge_reference_wrapped", False)
                ),
                "null_trials": int(null["n"]),
                "null_base_seeds": sorted(
                    {int(base_seed) for base_seed, _ in null["trial_keys"]}
                ),
                "h1_snrs_present_db": common_snrs,
                "planned_snr_grid_complete": bool(planned_snr_grid_complete),
                "h1_base_seeds_by_snr": {
                    str(snr): sorted(
                        {
                            int(base_seed)
                            for base_seed, _ in by_snr[snr]["trial_keys"]
                        }
                    )
                    for snr in common_snrs
                },
                "gpu_execution_forms": gpu_execution_forms,
                "gpu_kernel_versions": sorted(
                    {
                        version
                        for data in all_data
                        for version in data["gpu_kernel_versions"]
                    }
                ),
                "gpu_kernel_artifact_sha256s": sorted(
                    {
                        digest
                        for data in all_data
                        for digest in data["gpu_kernel_artifact_sha256s"]
                    }
                ),
                "gpu_parity": {
                    "fine_power_mismatches": int(
                        sum(
                            np.count_nonzero(data["gpu_fine_mismatch"] > 0)
                            for data in all_data
                        )
                    ),
                    "q16_decision_mismatches": int(
                        sum(
                            np.count_nonzero(data["gpu_mask_mismatch"] > 0)
                            for data in all_data
                        )
                    ),
                    "sufficient_pool_fine_mismatches": int(
                        sum(data["gpu_pool_fine_mismatches"] for data in all_data)
                    ),
                    "sufficient_pool_shards_checked": int(
                        sum(data["gpu_pool_shards"] for data in all_data)
                    ),
                    "full_frame_null_trials_executed": (
                        int(null["gpu_full_frame_trials"])
                    ),
                    "full_frame_h1_trials_executed": int(
                        sum(data["gpu_full_frame_trials"] for data in by_snr.values())
                    ),
                    "validation_scopes": sorted(
                        {
                            scope
                            for data in all_data
                            for scope in data["gpu_validation_scopes"]
                        }
                    ),
                },
                "stages": stages_report,
                "fixed_minus_float_point_estimates": point_losses,
                "paired_bootstrap": bootstrap,
                "primary_shard_wall_seconds_before_npz_write": float(
                    sum(
                        data["recorded_wall_seconds_before_npz_write"]
                        for data in all_data
                    )
                ),
                "sufficient_statistic_diagnostics": {
                    "pool_streams": int(args.sufficient_pool_streams),
                    "negative_power_clip_fraction_max_by_stage": (
                        negative_power_clip_max
                    ),
                    "declared_maximum_negative_power_clip_fraction": (
                        negative_clip_limit
                    ),
                    "negative_power_clip_acceptance_pass": negative_clip_pass,
                    "uncertainty_scope": (
                        "conditional on the finite per-stream sufficient pool; "
                        "full-frame audit is reported separately"
                    ),
                },
                "full_frame_audit": audit_report,
                "precision_review_prerequisites": {
                    "current_2048_stream_geometry": bool(
                        config["scientific_scope"]
                        == "current_chime_profile_geometry"
                    ),
                    "planned_snr_grid_complete": bool(planned_snr_grid_complete),
                    "at_least_1000_h1_trials_per_snr": bool(
                        common_snrs
                        and all(
                            int(by_snr[snr]["n"]) >= 1000
                            for snr in common_snrs
                        )
                    ),
                    "requested_pfa_resolved_by_null_count": bool(
                        int(null["n"]) * float(args.p_fa) >= 1.0
                    ),
                    "both_primary_curves_bracket_all_targets": bool(
                        precision_targets_bracketed
                    ),
                    "negative_power_clip_acceptance": negative_clip_pass,
                    "stratified_full_frame_audit": audit_requirement_satisfied,
                },
                "claim_status": "preliminary_or_code_validation",
            }
    declared_audits = [
        profile["full_frame_audit"]
        for profile in profile_reports.values()
        if profile["full_frame_audit"]["declared_for_profile"]
    ]
    expected_audit_profiles = int(
        len(config["full_frame_audit_design"]["physical_channels"])
        * len(config["full_frame_audit_design"]["offsets_fine_bins"])
    )
    stratified_audit_complete = bool(
        config["simulation"]["primary_backend"] == SIMULATION_FULL_FRAME
        or (
            len(declared_audits) == expected_audit_profiles
            and all(
                audit["complete_for_precision_review"]
                for audit in declared_audits
            )
        )
    )
    for profile in profile_reports.values():
        prerequisites = profile["precision_review_prerequisites"]
        prerequisites["stratified_full_frame_audit"] = stratified_audit_complete
        profile["claim_status"] = (
            "eligible_for_precision_review"
            if all(bool(value) for value in prerequisites.values())
            else "preliminary_or_code_validation"
        )
    report = {
        "schema_version": SENSITIVITY_STUDY_SCHEMA,
        "created_utc": _utc_now(),
        "config_sha256": config["config_sha256"],
        "scientific_scope": config["scientific_scope"],
        "num_input_streams": int(args.num_streams),
        "geometry": config["geometry"],
        "simulation": config["simulation"],
        "full_frame_audit_design": config["full_frame_audit_design"],
        "stratified_full_frame_audit": {
            "expected_profile_offset_combinations": expected_audit_profiles,
            "reported_profile_offset_combinations": len(declared_audits),
            "complete_for_precision_review": stratified_audit_complete,
        },
        "bootstrap_reporting": {
            "replicates": int(args.bootstrap_replicates),
            "base_seed": int(args.seed),
            "profile_seed_rule": (
                "stage_seed(base_seed, purpose=bootstrap_pd_TARGET, channel, "
                "offset, trial_index=0)"
            ),
        },
        "stage_definitions": STAGE_DEFINITIONS,
        "profiles": profile_reports,
        "curve_csv": str(args.output_dir / "sensitivity_curves.csv"),
        "interpretation": {
            "fixed_minus_float_loss": (
                "Positive means the full fixed/Q16 candidate requires more "
                "input data-shelf SNR than the ATSC ideal-float stage."
            ),
            "unbracketed": (
                "No sensitivity number is emitted unless adjacent sampled SNR "
                "points bracket the requested detection probability."
            ),
            "gpu": (
                "GPU curves appear only for sweep shards that executed the selected "
                "artifact. Kernel 2.3+ uses its device Q16 epilogue; kernel 2.1 "
                "composes GPU row projections and fine powers with the host exact "
                "Q16 decision. Every GPU fine-power result and emitted mask is "
                "parity-checked against the exact CPU reference."
            ),
            "sufficient_statistic": (
                "Production curves use a declared empirical multivariate CLT "
                "at the additive per-stream fine-power boundary. Their intervals "
                "are conditional on the finite pool and cannot pass precision "
                "review unless the stratified literal full-frame audit is complete, "
                "passes the declared KS/Wilson rules, has zero GPU parity errors, "
                "and negative-power clipping stays below its declared limit."
            ),
            "activation": (
                "The synthetic Q16 operating point evaluates the implemented "
                "candidate. It does not activate the fine mask or replace the "
                "pending archive/live per-channel calibration campaign."
            ),
        },
    }
    _write_report_csv(args.output_dir / "sensitivity_curves.csv", curve_rows)
    write_json_strict(
        args.output_dir / "sensitivity_report.json", report, indent=2, sort_keys=True
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=(MODE_SMOKE, MODE_PRODUCTION), default=MODE_SMOKE)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument(
        "--simulation-backend",
        choices=SIMULATION_BACKENDS,
        default=None,
        help=(
            "Primary null/H1 engine. Production defaults to the accelerated "
            "per-stream sufficient-statistic model; audit is always full-frame."
        ),
    )
    parser.add_argument(
        "--sufficient-pool-streams",
        type=int,
        default=None,
        help="Independent per-stream vectors used to fit the aggregate CLT model.",
    )
    parser.add_argument("--input-iq", type=Path, required=True)
    parser.add_argument(
        "--waveform-audit",
        type=Path,
        default=None,
        help=(
            "Passed audit JSON for --input-iq. Required in production and "
            "hashed into the immutable study configuration."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "generated" / "current_geometry_sensitivity"
    )
    parser.add_argument("--physical-channel", type=int, action="append", default=None)
    parser.add_argument(
        "--run-physical-channel",
        type=int,
        action="append",
        default=None,
        help="Execute only this subset while retaining the full study identity.",
    )
    parser.add_argument(
        "--audit-physical-channel", type=int, action="append", default=None
    )
    parser.add_argument(
        "--run-audit-physical-channel", type=int, action="append", default=None
    )
    parser.add_argument("--offset-fine-bins", type=float, action="append", default=None)
    parser.add_argument(
        "--run-offset-fine-bins",
        type=float,
        action="append",
        default=None,
        help="Execute only this offset subset while retaining the study identity.",
    )
    parser.add_argument(
        "--audit-offset-fine-bins", type=float, action="append", default=None
    )
    parser.add_argument(
        "--run-audit-offset-fine-bins", type=float, action="append", default=None
    )
    parser.add_argument("--snr-db", type=float, action="append", default=None)
    parser.add_argument(
        "--sweep-snr-db",
        type=float,
        action="append",
        default=None,
        help=(
            "Execute only this subset of the planned --snr-db grid. This is "
            "the job-array/resume control; --snr-db remains the immutable study grid."
        ),
    )
    parser.add_argument("--audit-snr-db", type=float, action="append", default=None)
    parser.add_argument("--run-audit-snr-db", type=float, action="append", default=None)
    parser.add_argument("--num-streams", type=int, default=None)
    parser.add_argument("--allow-reduced-geometry", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--historical-kernel-artifact",
        type=Path,
        default=None,
        help=(
            "Optional historical .so to hash as read-only provenance. It is never "
            "loaded or executed by this tool."
        ),
    )
    parser.add_argument("--p-fa", type=float, default=DEFAULT_P_FA)
    parser.add_argument("--null-quantile", type=float, default=DEFAULT_NULL_QUANTILE)
    parser.add_argument(
        "--designated-half-width", type=int, default=DEFAULT_DESIGNATED_HALF_WIDTH
    )
    parser.add_argument("--guard-fine-bins", type=int, default=DEFAULT_GUARD_FINE_BINS)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--input-scale", type=float, default=DEFAULT_INPUT_SCALE)
    parser.add_argument(
        "--spectral-sense",
        choices=("normal", "inverted"),
        default="normal",
        help=(
            "Sense of the synthetic reference-PFB output before the current "
            "post-normalization weights. The reference channelizer is already "
            "normalized, so normal is the validated current-weight path."
        ),
    )
    parser.add_argument("--pilot-below-data-db", type=float, default=PILOT_BELOW_DATA_DB)
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument("--iq-sample-rate-hz", type=float, default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ)
    return parser


def run(args: argparse.Namespace) -> int:
    args.input_iq = args.input_iq.resolve()
    if args.waveform_audit is not None:
        args.waveform_audit = args.waveform_audit.resolve()
    args.output_dir = args.output_dir.resolve()
    args.lib_path = args.lib_path.resolve()
    args.weights_path = args.weights_path.resolve()
    if args.historical_kernel_artifact is not None:
        args.historical_kernel_artifact = args.historical_kernel_artifact.resolve()
    _mode_defaults(args)
    bank = DetectorWeightBank(explicit_path=args.weights_path)
    _validate_args(args, bank)
    receiver = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    if receiver.detector_window_samples != K or receiver.frame_size_samples != FRAME_SAMPLES:
        raise SystemExit("checked-in CHIME profile no longer matches frozen study geometry")
    gpu = _initialize_gpu(args)
    config = _study_config(args, bank=bank, gpu=gpu)
    config = _write_or_validate_config(args.output_dir, config)
    if args.stage == "prepare":
        for channel in args.run_physical_channel:
            for offset in args.run_offset_fine_bins:
                path = _prepare_one_cache(args, config, _profile(args, bank, channel, offset))
                print(f"prepared {path}")
        return 0
    if args.stage == "null":
        for channel in args.run_physical_channel:
            for offset in args.run_offset_fine_bins:
                _run_shard(
                    args,
                    config,
                    _profile(args, bank, channel, offset),
                    purpose="null",
                    snr_db=None,
                    gpu=gpu,
                    calibration=None,
                )
        return 0
    if args.stage == "calibrate":
        calibration = _build_calibration(args, config, bank)
        print(f"wrote {args.output_dir / 'calibration.json'} ({len(calibration['profiles'])} profiles)")
        return 0
    if args.stage == "sweep":
        calibration = _load_calibration(args.output_dir, str(config["config_sha256"]))
        for channel in args.run_physical_channel:
            for offset in args.run_offset_fine_bins:
                profile = _profile(args, bank, channel, offset)
                for snr in sorted(args.sweep_snr_db):
                    _run_shard(
                        args,
                        config,
                        profile,
                        purpose="h1",
                        snr_db=snr,
                        gpu=gpu,
                        calibration=calibration,
                    )
        return 0
    if args.stage == "audit":
        calibration = _load_calibration(args.output_dir, str(config["config_sha256"]))
        for channel in args.run_audit_physical_channel:
            for offset in args.run_audit_offset_fine_bins:
                profile = _profile(args, bank, channel, offset)
                _run_full_shard(
                    args,
                    config,
                    profile,
                    purpose="audit_null",
                    snr_db=None,
                    gpu=gpu,
                    calibration=calibration,
                )
                for snr in sorted(args.run_audit_snr_db):
                    _run_full_shard(
                        args,
                        config,
                        profile,
                        purpose="audit_h1",
                        snr_db=snr,
                        gpu=gpu,
                        calibration=calibration,
                    )
        return 0
    if args.stage == "report":
        calibration = _load_calibration(args.output_dir, str(config["config_sha256"]))
        report = _build_report(args, config, bank, calibration)
        print(f"wrote {args.output_dir / 'sensitivity_report.json'}")
        print(f"profiles={len(report['profiles'])}, scope={report['scientific_scope']}")
        return 0
    if args.stage == "smoke":
        if args.mode != MODE_SMOKE:
            raise SystemExit("--stage smoke requires --mode smoke")
        for channel in args.physical_channel:
            for offset in args.offset_fine_bins:
                profile = _profile(args, bank, channel, offset)
                _prepare_one_cache(args, config, profile)
                _run_shard(
                    args,
                    config,
                    profile,
                    purpose="null",
                    snr_db=None,
                    gpu=gpu,
                    calibration=None,
                )
        calibration = _build_calibration(args, config, bank)
        for channel in args.physical_channel:
            for offset in args.offset_fine_bins:
                profile = _profile(args, bank, channel, offset)
                for snr in sorted(args.snr_db):
                    _run_shard(
                        args,
                        config,
                        profile,
                        purpose="h1",
                        snr_db=snr,
                        gpu=gpu,
                        calibration=calibration,
                    )
        _build_report(args, config, bank, calibration)
        print(f"smoke complete: {args.output_dir}")
        return 0
    raise AssertionError(args.stage)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
