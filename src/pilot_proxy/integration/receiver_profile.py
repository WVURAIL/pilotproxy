# coding=utf-8
"""Strict receiver/channelizer profile contract for external integrations."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.detector_constants import (
    DEFAULT_DETECTOR_WINDOW_SAMPLES,
    LOCKED_INPUT_FORMAT as DEFAULT_ADAPTER_OUTPUT_FORMAT,
    LOCKED_SAMPLE_BITS_PER_COMPONENT as DEFAULT_BITS_PER_COMPONENT,
)
from pilot_proxy.detector_geometry import (
    SPECTRAL_SENSE_NORMAL,
    normalize_spectral_sense,
    spectral_sense_requires_time_reversal,
)
from pilot_proxy.dtv_units import (
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
)
from pilot_proxy.reference_channelizer import (
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_BANDWIDTH_HZ,
    REFERENCE_NUM_CHANNELS,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
)

from .schemas import (
    COMBINE_MODE_COMBINED_STREAMS,
    DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO,
    RECEIVER_PROFILE_SCHEMA_TOKEN,
    SUPPORTED_QUANTIZATION_SCALE_MODES,
)

DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_NUM_INPUT_STREAMS = 1
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_CHANNELIZER_TYPE = "pfb"
DEFAULT_CHANNELIZER_RESPONSE = "sinc_hamming"
DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT = "complex64"
DEFAULT_QUANTIZATION_SCALE_MODE = "global"
REFERENCE_PROFILE_ID = "reference_800mhz_pfb"
PROFILE_HASH_HEX_CHARS = 64
FREQUENCY_ORDER_ASCENDING_RF = "ascending_rf"
FREQUENCY_ORDER_DESCENDING_RF = "descending_rf"
SUPPORTED_FREQUENCY_ORDER = frozenset(
    {FREQUENCY_ORDER_ASCENDING_RF, FREQUENCY_ORDER_DESCENDING_RF}
)


def _exact_coarse_channel_index(value: object, *, num_channels: int) -> int:
    """Validate one receiver-grid index without truncation or booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("coarse channel index must be an integer, not a boolean.")
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise TypeError("coarse channel index must be an integer.") from exc
    if not 0 <= index < num_channels:
        raise ValueError(
            "coarse channel index out of range: "
            f"{index}, valid 0-{num_channels - 1}"
        )
    return index


def _exact_positive_integer(value: object, *, field: str) -> int:
    """Validate one positive programmatic geometry integer."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    if result < 1:
        raise ValueError(f"{field} must be positive.")
    return result

_TOP_LEVEL_REQUIRED = frozenset(
    {
        "schema_version",
        "receiver_profile_id",
        "instrument_name",
        "rf_band",
        "channelizer",
        "framing",
        "input_streams",
        "quantization",
        "detector_adapter",
        "baseband_frame",
    }
)
_TOP_LEVEL_OPTIONAL = frozenset({"metadata"})
_RF_BAND_FIELDS = frozenset({"lower_hz", "upper_hz"})
_CHANNELIZER_FIELDS = frozenset(
    {
        "type",
        "input_sample_rate_hz",
        "pfb_fft_size",
        "taps",
        "response",
        "num_coarse_channels",
        "coarse_channel_center_offset_hz",
        "frequency_axis",
    }
)
_FREQUENCY_AXIS_FIELDS = frozenset({"spectral_sense", "order"})
_FRAMING_FIELDS = frozenset({"frame_size_samples"})
_INPUT_STREAM_FIELDS = frozenset(
    {
        "num_input_streams",
        "stream_unit",
        "stream_map_required",
        "combine_default",
    }
)
_QUANTIZATION_FIELDS = frozenset(
    {
        "native_channelized_sample_format",
        "adapter_output_format",
        "bits_per_component",
        "scale_mode_default",
        "clip_sigma_default",
        "record_scale_by_stream",
        "record_clip_fraction_by_stream",
    }
)
_DETECTOR_ADAPTER_FIELDS = frozenset(
    {
        "compatible_detector_core_id",
        "detector_window_samples",
        "fine_bin_enbw_hz",
        "dtv_bandwidth_hz",
        "pilot_below_data_db",
        "pilot_capture_efficiency",
        "time_reverse_detector_windows_before_kernel",
    }
)
_BASEBAND_FRAME_REQUIRED = frozenset(
    {"channel_center_normalized", "physical_dc_normalized"}
)
_BASEBAND_FRAME_OPTIONAL = frozenset(
    {
        "channel_center_normalized_odd_channels",
        "physical_dc_normalized_odd_channels",
    }
)


def _validate_exact_keys(
    data: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    keys = {str(key) for key in data}
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")


def _require_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    out = dict(value)
    _validate_exact_keys(
        out,
        required=required,
        optional=optional,
        context=context,
    )
    return out


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _normalized_unit_interval(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    if not 0.0 <= normalized < 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1).")
    return normalized


@dataclass(frozen=True)
class ChannelizerProfile:
    """Authoritative channelizer implementation parameters."""

    type: str
    fft_size: int
    taps: int
    response: str

    def __post_init__(self) -> None:
        if not str(self.type).strip():
            raise ValueError("channelizer.type must not be empty.")
        if int(self.fft_size) <= 0:
            raise ValueError("channelizer.pfb_fft_size must be positive.")
        if int(self.taps) <= 0:
            raise ValueError("channelizer.taps must be positive.")
        if not str(self.response).strip():
            raise ValueError("channelizer.response must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "pfb_fft_size": int(self.fft_size),
            "taps": int(self.taps),
            "response": str(self.response),
        }


@dataclass(frozen=True)
class ReceiverProfile:
    """One exact receiver-to-detector integration contract."""

    schema_version: str
    receiver_profile_id: str
    instrument_name: str
    sample_rate_hz: float
    band_lower_hz: float
    band_upper_hz: float
    num_coarse_channels: int
    coarse_channel_center_offset_hz: float
    frame_size_samples: int
    num_input_streams: int
    input_stream_unit: str
    stream_map_required: bool
    combine_default: str
    spectral_sense: str
    frequency_order: str
    channelizer: ChannelizerProfile
    native_channelized_sample_format: str
    adapter_output_format: str
    bits_per_component: int
    quantization_scale_mode_default: str
    clip_sigma_default: float
    record_scale_by_stream: bool
    record_clip_fraction_by_stream: bool
    compatible_detector_core_id: str
    detector_window_samples: int
    bin_enbw_hz: float
    dtv_bandwidth_hz: float
    pilot_below_data_db: float
    pilot_capture_efficiency: float
    time_reverse_detector_windows_before_kernel: bool
    channel_center_normalized: float
    physical_dc_normalized: float
    channel_center_normalized_odd: float | None = None
    physical_dc_normalized_odd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RECEIVER_PROFILE_SCHEMA_TOKEN:
            raise ValueError(
                "unsupported receiver profile schema_version: "
                f"{self.schema_version!r}"
            )
        if not str(self.receiver_profile_id).strip():
            raise ValueError("receiver_profile_id must not be empty.")
        if not str(self.instrument_name).strip():
            raise ValueError("instrument_name must not be empty.")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("channelizer.input_sample_rate_hz must be positive.")
        if self.band_upper_hz <= self.band_lower_hz:
            raise ValueError("rf_band.upper_hz must be greater than rf_band.lower_hz.")
        if self.num_coarse_channels <= 0:
            raise ValueError("channelizer.num_coarse_channels must be positive.")
        if self.frame_size_samples <= 0:
            raise ValueError("framing.frame_size_samples must be positive.")
        if self.num_input_streams <= 0:
            raise ValueError("input_streams.num_input_streams must be positive.")
        if not str(self.input_stream_unit).strip():
            raise ValueError("input_streams.stream_unit must not be empty.")
        if self.combine_default != COMBINE_MODE_COMBINED_STREAMS:
            raise ValueError(
                "input_streams.combine_default must be "
                f"{COMBINE_MODE_COMBINED_STREAMS!r}."
            )
        object.__setattr__(
            self,
            "spectral_sense",
            normalize_spectral_sense(self.spectral_sense),
        )
        frequency_order = str(self.frequency_order).strip().lower()
        if frequency_order not in SUPPORTED_FREQUENCY_ORDER:
            raise ValueError(
                "channelizer.frequency_axis.order must be one of "
                f"{sorted(SUPPORTED_FREQUENCY_ORDER)}; got "
                f"{self.frequency_order!r}."
            )
        object.__setattr__(self, "frequency_order", frequency_order)
        if self.bits_per_component <= 0:
            raise ValueError("quantization.bits_per_component must be positive.")
        if self.quantization_scale_mode_default not in SUPPORTED_QUANTIZATION_SCALE_MODES:
            raise ValueError(
                "quantization.scale_mode_default must be one of "
                f"{sorted(SUPPORTED_QUANTIZATION_SCALE_MODES)}."
            )
        if self.clip_sigma_default <= 0.0:
            raise ValueError("quantization.clip_sigma_default must be positive.")
        if self.compatible_detector_core_id != DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO:
            raise ValueError(
                "detector_adapter.compatible_detector_core_id must be "
                f"{DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO!r}."
            )
        if self.detector_window_samples <= 0:
            raise ValueError(
                "detector_adapter.detector_window_samples must be positive."
            )
        if self.frame_size_samples % self.detector_window_samples != 0:
            raise ValueError(
                "framing.frame_size_samples must be an integer multiple of "
                "detector_adapter.detector_window_samples."
            )
        if self.bin_enbw_hz <= 0.0:
            raise ValueError("detector_adapter.fine_bin_enbw_hz must be positive.")
        if self.dtv_bandwidth_hz <= 0.0:
            raise ValueError("detector_adapter.dtv_bandwidth_hz must be positive.")
        if self.pilot_capture_efficiency <= 0.0:
            raise ValueError(
                "detector_adapter.pilot_capture_efficiency must be positive."
            )
        expected_fine_bin_hz = (
            self.coarse_channel_width_hz / float(self.detector_window_samples)
        )
        if not math.isclose(
            self.bin_enbw_hz,
            expected_fine_bin_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "detector_adapter.fine_bin_enbw_hz does not match "
                "coarse_channel_width_hz / detector_window_samples: "
                f"{self.bin_enbw_hz} != {expected_fine_bin_hz}."
            )
        object.__setattr__(
            self,
            "channel_center_normalized",
            _normalized_unit_interval(
                self.channel_center_normalized,
                "baseband_frame.channel_center_normalized",
            ),
        )
        object.__setattr__(
            self,
            "physical_dc_normalized",
            _normalized_unit_interval(
                self.physical_dc_normalized,
                "baseband_frame.physical_dc_normalized",
            ),
        )
        if self.channel_center_normalized_odd is not None:
            object.__setattr__(
                self,
                "channel_center_normalized_odd",
                _normalized_unit_interval(
                    self.channel_center_normalized_odd,
                    "baseband_frame.channel_center_normalized_odd_channels",
                ),
            )
        if self.physical_dc_normalized_odd is not None:
            object.__setattr__(
                self,
                "physical_dc_normalized_odd",
                _normalized_unit_interval(
                    self.physical_dc_normalized_odd,
                    "baseband_frame.physical_dc_normalized_odd_channels",
                ),
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def profile_status(self) -> str | None:
        value = self.metadata.get("profile_status")
        return None if value is None else str(value)

    @property
    def bandwidth_hz(self) -> float:
        return float(self.band_upper_hz - self.band_lower_hz)

    @property
    def coarse_channel_width_hz(self) -> float:
        return float(self.bandwidth_hz / int(self.num_coarse_channels))

    @property
    def center_offset_hz(self) -> float:
        return float(self.coarse_channel_center_offset_hz)

    def frame_center_normalized(self, coarse_channel_index: int | None = None) -> float:
        index = (
            None
            if coarse_channel_index is None
            else _exact_coarse_channel_index(
                coarse_channel_index,
                num_channels=self.num_coarse_channels,
            )
        )
        if (
            self.channel_center_normalized_odd is not None
            and index is not None
            and index % 2 == 1
        ):
            return float(self.channel_center_normalized_odd)
        return float(self.channel_center_normalized)

    def forbidden_dc_normalized(self, coarse_channel_index: int | None = None) -> float:
        index = (
            None
            if coarse_channel_index is None
            else _exact_coarse_channel_index(
                coarse_channel_index,
                num_channels=self.num_coarse_channels,
            )
        )
        if (
            self.physical_dc_normalized_odd is not None
            and index is not None
            and index % 2 == 1
        ):
            return float(self.physical_dc_normalized_odd)
        return float(self.physical_dc_normalized)

    def coarse_channel_center_hz(self, index: int) -> float:
        idx = _exact_coarse_channel_index(
            index,
            num_channels=self.num_coarse_channels,
        )
        if self.frequency_order == FREQUENCY_ORDER_DESCENDING_RF:
            return float(
                self.band_upper_hz
                - self.center_offset_hz
                - idx * self.coarse_channel_width_hz
            )
        return float(
            self.band_lower_hz
            + self.center_offset_hz
            + idx * self.coarse_channel_width_hz
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReceiverProfile":
        if not isinstance(data, Mapping):
            raise ValueError("receiver profile must be an object.")
        raw = dict(data)
        _validate_exact_keys(
            raw,
            required=_TOP_LEVEL_REQUIRED,
            optional=_TOP_LEVEL_OPTIONAL,
            context="receiver profile",
        )
        rf_band = _require_mapping(
            raw,
            "rf_band",
            required=_RF_BAND_FIELDS,
            context="receiver profile rf_band",
        )
        channelizer = _require_mapping(
            raw,
            "channelizer",
            required=_CHANNELIZER_FIELDS,
            context="receiver profile channelizer",
        )
        frequency_axis = _require_mapping(
            channelizer,
            "frequency_axis",
            required=_FREQUENCY_AXIS_FIELDS,
            context="receiver profile channelizer.frequency_axis",
        )
        framing = _require_mapping(
            raw,
            "framing",
            required=_FRAMING_FIELDS,
            context="receiver profile framing",
        )
        input_streams = _require_mapping(
            raw,
            "input_streams",
            required=_INPUT_STREAM_FIELDS,
            context="receiver profile input_streams",
        )
        quantization = _require_mapping(
            raw,
            "quantization",
            required=_QUANTIZATION_FIELDS,
            context="receiver profile quantization",
        )
        adapter = _require_mapping(
            raw,
            "detector_adapter",
            required=_DETECTOR_ADAPTER_FIELDS,
            context="receiver profile detector_adapter",
        )
        baseband_frame = _require_mapping(
            raw,
            "baseband_frame",
            required=_BASEBAND_FRAME_REQUIRED,
            optional=_BASEBAND_FRAME_OPTIONAL,
            context="receiver profile baseband_frame",
        )
        metadata_value = raw.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise ValueError("receiver profile metadata must be an object.")

        return cls(
            schema_version=_require_nonempty_str(
                raw["schema_version"], "schema_version"
            ),
            receiver_profile_id=_require_nonempty_str(
                raw["receiver_profile_id"], "receiver_profile_id"
            ),
            instrument_name=_require_nonempty_str(
                raw["instrument_name"], "instrument_name"
            ),
            sample_rate_hz=_require_float(
                channelizer["input_sample_rate_hz"],
                "channelizer.input_sample_rate_hz",
            ),
            band_lower_hz=_require_float(rf_band["lower_hz"], "rf_band.lower_hz"),
            band_upper_hz=_require_float(rf_band["upper_hz"], "rf_band.upper_hz"),
            num_coarse_channels=_require_int(
                channelizer["num_coarse_channels"],
                "channelizer.num_coarse_channels",
            ),
            coarse_channel_center_offset_hz=_require_float(
                channelizer["coarse_channel_center_offset_hz"],
                "channelizer.coarse_channel_center_offset_hz",
            ),
            frame_size_samples=_require_int(
                framing["frame_size_samples"], "framing.frame_size_samples"
            ),
            num_input_streams=_require_int(
                input_streams["num_input_streams"],
                "input_streams.num_input_streams",
            ),
            input_stream_unit=_require_nonempty_str(
                input_streams["stream_unit"], "input_streams.stream_unit"
            ),
            stream_map_required=_require_bool(
                input_streams["stream_map_required"],
                "input_streams.stream_map_required",
            ),
            combine_default=_require_nonempty_str(
                input_streams["combine_default"],
                "input_streams.combine_default",
            ),
            spectral_sense=_require_nonempty_str(
                frequency_axis["spectral_sense"],
                "channelizer.frequency_axis.spectral_sense",
            ),
            frequency_order=_require_nonempty_str(
                frequency_axis["order"],
                "channelizer.frequency_axis.order",
            ),
            channelizer=ChannelizerProfile(
                type=_require_nonempty_str(
                    channelizer["type"], "channelizer.type"
                ),
                fft_size=_require_int(
                    channelizer["pfb_fft_size"], "channelizer.pfb_fft_size"
                ),
                taps=_require_int(channelizer["taps"], "channelizer.taps"),
                response=_require_nonempty_str(
                    channelizer["response"], "channelizer.response"
                ),
            ),
            native_channelized_sample_format=_require_nonempty_str(
                quantization["native_channelized_sample_format"],
                "quantization.native_channelized_sample_format",
            ),
            adapter_output_format=_require_nonempty_str(
                quantization["adapter_output_format"],
                "quantization.adapter_output_format",
            ),
            bits_per_component=_require_int(
                quantization["bits_per_component"],
                "quantization.bits_per_component",
            ),
            quantization_scale_mode_default=_require_nonempty_str(
                quantization["scale_mode_default"],
                "quantization.scale_mode_default",
            ),
            clip_sigma_default=_require_float(
                quantization["clip_sigma_default"],
                "quantization.clip_sigma_default",
            ),
            record_scale_by_stream=_require_bool(
                quantization["record_scale_by_stream"],
                "quantization.record_scale_by_stream",
            ),
            record_clip_fraction_by_stream=_require_bool(
                quantization["record_clip_fraction_by_stream"],
                "quantization.record_clip_fraction_by_stream",
            ),
            compatible_detector_core_id=_require_nonempty_str(
                adapter["compatible_detector_core_id"],
                "detector_adapter.compatible_detector_core_id",
            ),
            detector_window_samples=_require_int(
                adapter["detector_window_samples"],
                "detector_adapter.detector_window_samples",
            ),
            bin_enbw_hz=_require_float(
                adapter["fine_bin_enbw_hz"],
                "detector_adapter.fine_bin_enbw_hz",
            ),
            dtv_bandwidth_hz=_require_float(
                adapter["dtv_bandwidth_hz"],
                "detector_adapter.dtv_bandwidth_hz",
            ),
            pilot_below_data_db=_require_float(
                adapter["pilot_below_data_db"],
                "detector_adapter.pilot_below_data_db",
            ),
            pilot_capture_efficiency=_require_float(
                adapter["pilot_capture_efficiency"],
                "detector_adapter.pilot_capture_efficiency",
            ),
            time_reverse_detector_windows_before_kernel=_require_bool(
                adapter["time_reverse_detector_windows_before_kernel"],
                "detector_adapter.time_reverse_detector_windows_before_kernel",
            ),
            channel_center_normalized=_require_float(
                baseband_frame["channel_center_normalized"],
                "baseband_frame.channel_center_normalized",
            ),
            physical_dc_normalized=_require_float(
                baseband_frame["physical_dc_normalized"],
                "baseband_frame.physical_dc_normalized",
            ),
            channel_center_normalized_odd=(
                None
                if "channel_center_normalized_odd_channels" not in baseband_frame
                else _require_float(
                    baseband_frame["channel_center_normalized_odd_channels"],
                    "baseband_frame.channel_center_normalized_odd_channels",
                )
            ),
            physical_dc_normalized_odd=(
                None
                if "physical_dc_normalized_odd_channels" not in baseband_frame
                else _require_float(
                    baseband_frame["physical_dc_normalized_odd_channels"],
                    "baseband_frame.physical_dc_normalized_odd_channels",
                )
            ),
            metadata=dict(metadata_value),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReceiverProfile":
        return load_receiver_profile(path)

    def to_dict(self) -> dict[str, Any]:
        channelizer = self.channelizer.to_dict()
        channelizer.update(
            {
                "input_sample_rate_hz": float(self.sample_rate_hz),
                "num_coarse_channels": int(self.num_coarse_channels),
                "coarse_channel_center_offset_hz": float(
                    self.coarse_channel_center_offset_hz
                ),
                "frequency_axis": {
                    "spectral_sense": self.spectral_sense,
                    "order": self.frequency_order,
                },
            }
        )
        baseband_frame: dict[str, Any] = {
            "channel_center_normalized": float(self.channel_center_normalized),
            "physical_dc_normalized": float(self.physical_dc_normalized),
        }
        if self.channel_center_normalized_odd is not None:
            baseband_frame["channel_center_normalized_odd_channels"] = float(
                self.channel_center_normalized_odd
            )
        if self.physical_dc_normalized_odd is not None:
            baseband_frame["physical_dc_normalized_odd_channels"] = float(
                self.physical_dc_normalized_odd
            )
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "receiver_profile_id": self.receiver_profile_id,
            "instrument_name": self.instrument_name,
            "rf_band": {
                "lower_hz": float(self.band_lower_hz),
                "upper_hz": float(self.band_upper_hz),
            },
            "channelizer": channelizer,
            "framing": {"frame_size_samples": int(self.frame_size_samples)},
            "input_streams": {
                "num_input_streams": int(self.num_input_streams),
                "stream_unit": self.input_stream_unit,
                "stream_map_required": bool(self.stream_map_required),
                "combine_default": self.combine_default,
            },
            "quantization": {
                "native_channelized_sample_format": (
                    self.native_channelized_sample_format
                ),
                "adapter_output_format": self.adapter_output_format,
                "bits_per_component": int(self.bits_per_component),
                "scale_mode_default": self.quantization_scale_mode_default,
                "clip_sigma_default": float(self.clip_sigma_default),
                "record_scale_by_stream": bool(self.record_scale_by_stream),
                "record_clip_fraction_by_stream": bool(
                    self.record_clip_fraction_by_stream
                ),
            },
            "detector_adapter": {
                "compatible_detector_core_id": self.compatible_detector_core_id,
                "detector_window_samples": int(self.detector_window_samples),
                "fine_bin_enbw_hz": float(self.bin_enbw_hz),
                "dtv_bandwidth_hz": float(self.dtv_bandwidth_hz),
                "pilot_below_data_db": float(self.pilot_below_data_db),
                "pilot_capture_efficiency": float(self.pilot_capture_efficiency),
                "time_reverse_detector_windows_before_kernel": bool(
                    self.time_reverse_detector_windows_before_kernel
                ),
            },
            "baseband_frame": baseband_frame,
        }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class ChannelSelection:
    """Mapping from an RF target frequency to a receiver coarse channel."""

    rf_hz: float
    coarse_channel_index: int
    coarse_channel_center_hz: float
    fine_bin_offset_hz: float
    spectral_sense: str
    requires_time_reversal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rf_hz": float(self.rf_hz),
            "coarse_channel_index": int(self.coarse_channel_index),
            "coarse_channel_center_hz": float(self.coarse_channel_center_hz),
            "fine_bin_offset_hz": float(self.fine_bin_offset_hz),
            "spectral_sense": str(self.spectral_sense),
            "requires_time_reversal": bool(self.requires_time_reversal),
        }


def default_reference_receiver_profile(
    *,
    frame_size_samples: int = DEFAULT_FRAME_SIZE_SAMPLES,
    num_input_streams: int = DEFAULT_NUM_INPUT_STREAMS,
) -> ReceiverProfile:
    """Return the shipped 800 MS/s, 400-800 MHz reference receiver profile."""
    frame_size = _exact_positive_integer(
        frame_size_samples,
        field="frame_size_samples",
    )
    stream_count = _exact_positive_integer(
        num_input_streams,
        field="num_input_streams",
    )
    return ReceiverProfile(
        schema_version=RECEIVER_PROFILE_SCHEMA_TOKEN,
        receiver_profile_id=REFERENCE_PROFILE_ID,
        instrument_name="reference",
        sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        band_lower_hz=REFERENCE_BAND_LOWER_HZ,
        band_upper_hz=REFERENCE_BAND_LOWER_HZ + REFERENCE_BANDWIDTH_HZ,
        num_coarse_channels=REFERENCE_NUM_CHANNELS,
        coarse_channel_center_offset_hz=(
            REFERENCE_BANDWIDTH_HZ / float(REFERENCE_NUM_CHANNELS)
        ),
        frame_size_samples=frame_size,
        num_input_streams=stream_count,
        input_stream_unit="input_stream",
        stream_map_required=stream_count > 1,
        combine_default=COMBINE_MODE_COMBINED_STREAMS,
        spectral_sense=SPECTRAL_SENSE_NORMAL,
        frequency_order=FREQUENCY_ORDER_ASCENDING_RF,
        channelizer=ChannelizerProfile(
            type=DEFAULT_CHANNELIZER_TYPE,
            fft_size=REFERENCE_PFB_FFT_SIZE,
            taps=REFERENCE_PFB_TAPS,
            response=DEFAULT_CHANNELIZER_RESPONSE,
        ),
        native_channelized_sample_format=DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT,
        adapter_output_format=DEFAULT_ADAPTER_OUTPUT_FORMAT,
        bits_per_component=DEFAULT_BITS_PER_COMPONENT,
        quantization_scale_mode_default=DEFAULT_QUANTIZATION_SCALE_MODE,
        clip_sigma_default=DEFAULT_CLIP_SIGMA,
        record_scale_by_stream=True,
        record_clip_fraction_by_stream=True,
        compatible_detector_core_id=DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO,
        detector_window_samples=DEFAULT_DETECTOR_WINDOW_SAMPLES,
        bin_enbw_hz=EFFECTIVE_BIN_BW_HZ,
        dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
        time_reverse_detector_windows_before_kernel=False,
        channel_center_normalized=0.5,
        physical_dc_normalized=0.5,
    )


def load_receiver_profile(path: str | Path) -> ReceiverProfile:
    """Load one exact receiver-profile JSON document."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("receiver profile JSON must contain an object.")
    return ReceiverProfile.from_dict(data)


def receiver_profile_hash(profile: ReceiverProfile | dict[str, Any]) -> str:
    """Return the SHA-256 of the canonical strict profile serialization."""
    canonical = (
        profile.to_dict()
        if isinstance(profile, ReceiverProfile)
        else ReceiverProfile.from_dict(dict(profile)).to_dict()
    )
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if len(digest) != PROFILE_HASH_HEX_CHARS:
        raise RuntimeError("unexpected SHA-256 digest length.")
    return digest


def receiver_frequency_to_channel(
    rf_hz: float,
    profile: ReceiverProfile,
) -> ChannelSelection:
    """Map an RF frequency to the nearest receiver coarse channel."""
    rf = _require_float(rf_hz, "rf_hz")
    if not profile.band_lower_hz <= rf <= profile.band_upper_hz:
        raise ValueError(
            f"rf_hz={rf:.3f} is outside receiver profile RF band "
            f"[{profile.band_lower_hz:.3f}, {profile.band_upper_hz:.3f}]."
        )
    if profile.frequency_order == FREQUENCY_ORDER_DESCENDING_RF:
        first_center = profile.band_upper_hz - profile.center_offset_hz
        idx = int(round((first_center - rf) / profile.coarse_channel_width_hz))
    else:
        first_center = profile.band_lower_hz + profile.center_offset_hz
        idx = int(round((rf - first_center) / profile.coarse_channel_width_hz))
    if idx < 0 or idx >= int(profile.num_coarse_channels):
        raise ValueError(
            f"rf_hz={rf:.3f} is outside receiver profile band/channel centers."
        )
    center = profile.coarse_channel_center_hz(idx)
    rf_offset = rf - center
    fine_offset = (
        -rf_offset
        if spectral_sense_requires_time_reversal(profile.spectral_sense)
        else rf_offset
    )
    return ChannelSelection(
        rf_hz=rf,
        coarse_channel_index=idx,
        coarse_channel_center_hz=center,
        fine_bin_offset_hz=fine_offset,
        spectral_sense=profile.spectral_sense,
        requires_time_reversal=spectral_sense_requires_time_reversal(
            profile.spectral_sense
        ),
    )


def validate_weight_manifest_profile_hash(
    manifest: dict[str, Any],
    profile: ReceiverProfile,
) -> bool:
    """Require the manifest to bind the exact canonical receiver profile."""
    expected = receiver_profile_hash(profile)
    if "receiver_profile_hash" not in manifest:
        raise ValueError("weight manifest does not contain receiver_profile_hash.")
    got = str(manifest["receiver_profile_hash"])
    if got != expected:
        raise ValueError(
            "weight manifest receiver profile hash does not match: "
            f"manifest={got}, expected={expected}"
        )
    return True


__all__ = [
    "ChannelSelection",
    "ChannelizerProfile",
    "ReceiverProfile",
    "default_reference_receiver_profile",
    "FREQUENCY_ORDER_ASCENDING_RF",
    "FREQUENCY_ORDER_DESCENDING_RF",
    "load_receiver_profile",
    "receiver_frequency_to_channel",
    "receiver_profile_hash",
    "validate_weight_manifest_profile_hash",
]
