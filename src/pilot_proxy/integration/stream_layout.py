# coding=utf-8
"""Input-stream layout helpers for combined and diagnostic detector modes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.detector_geometry import DetectorFrameLayout

from .detector_core import DetectorCoreProfile
from .receiver_profile import ChannelSelection, ReceiverProfile
from .schemas import (
    COMBINE_MODE_COMBINED_STREAMS,
    STREAM_MAP_SCHEMA_TOKEN,
    SUPPORTED_QUANTIZATION_SCALE_MODES,
)

UINT64_MAX = (1 << 64) - 1
COMPLEX_COMPONENT_COUNT = 2
DEFAULT_DOT_COMPONENT_BITS = 16
DEFAULT_REFERENCE_WEIGHT_TERMS = 3


@dataclass(frozen=True)
class StreamDescriptor:
    """Metadata for one flattened input stream."""

    stream_index: int
    feed_index: int
    selected_coarse_channel: int
    selected_coarse_channel_center_hz: float
    feed_id: int | str | None = None
    polarization: str | None = None
    physical_channel: int | None = None
    dtv_pilot_hz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_index": int(self.stream_index),
            "feed_index": int(self.feed_index),
            "feed_id": self.feed_id,
            "polarization": self.polarization,
            "selected_coarse_channel": int(self.selected_coarse_channel),
            "selected_coarse_channel_center_hz": float(
                self.selected_coarse_channel_center_hz
            ),
            "physical_channel": self.physical_channel,
            "dtv_pilot_hz": self.dtv_pilot_hz,
        }


@dataclass(frozen=True)
class InputStreamMap:
    """External stream metadata supplied by a receiver integration."""

    schema_version: str
    receiver_profile_id: str
    stream_unit: str
    num_streams: int
    streams: list[dict[str, Any]]
    indexing_convention: dict[str, Any] | None = None
    polarization_labels: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != STREAM_MAP_SCHEMA_TOKEN:
            raise ValueError(
                f"unsupported stream map schema_version: {self.schema_version!r}"
            )
        if self.num_streams <= 0:
            raise ValueError("num_streams must be positive.")
        object.__setattr__(self, "streams", [dict(item) for item in self.streams])
        if len(self.streams) != int(self.num_streams):
            raise ValueError(
                "stream map entries must match declared num_streams: "
                f"{len(self.streams)} != {int(self.num_streams)}"
            )
        stream_indices: list[int] = []
        for position, item in enumerate(self.streams):
            if "stream_index" not in item:
                raise ValueError(
                    f"stream map entry {position} is missing stream_index."
                )
            stream_index = item["stream_index"]
            if isinstance(stream_index, bool) or not isinstance(stream_index, int):
                raise ValueError(
                    f"stream map entry {position} stream_index must be an integer."
                )
            stream_indices.append(stream_index)
        expected_indices = list(range(int(self.num_streams)))
        if sorted(stream_indices) != expected_indices:
            raise ValueError(
                "stream_index values must contain every integer in "
                f"[0, {int(self.num_streams)}) exactly once; got "
                f"{stream_indices[:10]}{'...' if len(stream_indices) > 10 else ''}."
            )
        if self.indexing_convention is not None:
            object.__setattr__(
                self,
                "indexing_convention",
                dict(self.indexing_convention),
            )
        if self.polarization_labels is not None:
            object.__setattr__(
                self,
                "polarization_labels",
                {str(key): str(value) for key, value in self.polarization_labels.items()},
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputStreamMap":
        raw = dict(data)
        streams_raw = raw.get("streams", [])
        if not isinstance(streams_raw, SequenceABC) or isinstance(
            streams_raw,
            (str, bytes),
        ):
            raise ValueError("stream map field 'streams' must be a list of objects.")
        streams: list[dict[str, Any]] = []
        for item in streams_raw:
            if not isinstance(item, Mapping):
                raise ValueError("stream map entries must be objects.")
            streams.append(dict(item))
        return cls(
            schema_version=str(raw.get("schema_version", STREAM_MAP_SCHEMA_TOKEN)),
            receiver_profile_id=str(raw["receiver_profile_id"]),
            stream_unit=str(raw.get("stream_unit", "input_stream")),
            num_streams=int(raw.get("num_streams", len(streams))),
            streams=streams,
            indexing_convention=raw.get("indexing_convention"),
            polarization_labels=raw.get("polarization_labels"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "InputStreamMap":
        return load_stream_map(path)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "receiver_profile_id": self.receiver_profile_id,
            "stream_unit": self.stream_unit,
            "num_streams": int(self.num_streams),
            "streams": [dict(item) for item in self.streams],
        }
        if self.indexing_convention is not None:
            out["indexing_convention"] = dict(self.indexing_convention)
        if self.polarization_labels is not None:
            out["polarization_labels"] = dict(self.polarization_labels)
        return out


def validate_integration_compatibility(
    *,
    profile: ReceiverProfile,
    detector_core: DetectorCoreProfile | None = None,
    stream_map: InputStreamMap | None = None,
    require_stream_map: bool = False,
) -> None:
    """Validate that receiver, detector-core, and stream-map contracts agree.

    The individual JSON loaders validate each document in isolation. This
    check covers the cross-document identities and dimensions that cannot be
    established by any one loader.
    """
    if detector_core is not None:
        if profile.compatible_detector_core_id != detector_core.detector_core_id:
            raise ValueError(
                "receiver profile detector core is incompatible: "
                f"{profile.compatible_detector_core_id!r} != "
                f"{detector_core.detector_core_id!r}."
            )
        if int(profile.detector_window_samples) not in set(
            detector_core.supported_detector_window_samples
        ):
            raise ValueError(
                "receiver profile detector_window_samples is not supported by "
                f"the detector core: {profile.detector_window_samples} not in "
                f"{list(detector_core.supported_detector_window_samples)}."
            )
        if profile.adapter_output_format != detector_core.input_format:
            raise ValueError(
                "receiver profile adapter output format does not match the "
                f"detector core input format: {profile.adapter_output_format!r} "
                f"!= {detector_core.input_format!r}."
            )
        if int(profile.bits_per_component) != int(
            detector_core.sample_bits_per_component
        ):
            raise ValueError(
                "receiver profile component bit depth does not match the "
                f"detector core: {profile.bits_per_component} != "
                f"{detector_core.sample_bits_per_component}."
            )

    if stream_map is None:
        if require_stream_map and profile.stream_map_required:
            raise ValueError(
                f"receiver profile {profile.receiver_profile_id!r} requires a "
                "stream map."
            )
        return
    if stream_map.receiver_profile_id != profile.receiver_profile_id:
        raise ValueError(
            "stream map receiver_profile_id does not match the receiver "
            f"profile: {stream_map.receiver_profile_id!r} != "
            f"{profile.receiver_profile_id!r}."
        )
    if int(stream_map.num_streams) != int(profile.num_input_streams):
        raise ValueError(
            "stream map num_streams does not match the receiver profile: "
            f"{stream_map.num_streams} != {profile.num_input_streams}."
        )
    if stream_map.stream_unit != profile.input_stream_unit:
        raise ValueError(
            "stream map stream_unit does not match the receiver profile: "
            f"{stream_map.stream_unit!r} != {profile.input_stream_unit!r}."
        )


def build_stream_map_for_channel(
    *,
    num_input_streams: int,
    selection: ChannelSelection,
    physical_channel: int | None = None,
    dtv_pilot_hz: float | None = None,
    feed_ids: Sequence[int | str | None] | None = None,
    polarizations: Sequence[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build stream metadata for one selected coarse channel across streams."""
    count = int(num_input_streams)
    if count <= 0:
        raise ValueError("num_input_streams must be positive.")
    if feed_ids is not None and len(feed_ids) != count:
        raise ValueError("feed_ids length must match num_input_streams.")
    if polarizations is not None and len(polarizations) != count:
        raise ValueError("polarizations length must match num_input_streams.")

    out = []
    for feed_index in range(count):
        out.append(
            StreamDescriptor(
                stream_index=feed_index,
                feed_index=feed_index,
                feed_id=None if feed_ids is None else feed_ids[feed_index],
                polarization=None
                if polarizations is None
                else polarizations[feed_index],
                selected_coarse_channel=int(selection.coarse_channel_index),
                selected_coarse_channel_center_hz=float(
                    selection.coarse_channel_center_hz
                ),
                physical_channel=physical_channel,
                dtv_pilot_hz=dtv_pilot_hz,
            ).to_dict()
        )
    return out


def detector_shape_for_combined_streams(
    layout: DetectorFrameLayout,
    *,
    batch: int = 1,
) -> tuple[int, int, int]:
    """Return the packed detector shape for combined-stream detection."""
    return (
        int(batch),
        int(layout.detector_rows_per_frame),
        int(layout.detector_window_samples),
    )


def detector_shape_for_per_stream_diagnostics(
    layout: DetectorFrameLayout,
    *,
    batch: int = 1,
) -> tuple[int, int, int]:
    """Return ``packed[batch*num_streams, windows_per_stream, K]``."""
    return (
        int(batch) * int(layout.num_streams),
        int(layout.windows_per_stream),
        int(layout.detector_window_samples),
    )


def layout_from_receiver_profile(
    profile: ReceiverProfile,
    *,
    num_selected_channels: int = 1,
    combine_mode: str = COMBINE_MODE_COMBINED_STREAMS,
) -> DetectorFrameLayout:
    """Build canonical detector-frame geometry from a receiver profile."""
    return DetectorFrameLayout(
        frame_size_samples=int(profile.frame_size_samples),
        detector_window_samples=int(profile.detector_window_samples),
        num_input_streams=int(profile.num_input_streams),
        num_selected_channels=int(num_selected_channels),
        combine_mode=combine_mode,
    )


def layout_uint64_bound_check(
    *,
    frame_size_samples: int,
    num_input_streams: int,
    detector_window_samples: int,
    num_selected_channels: int = 1,
    dot_component_bits: int = DEFAULT_DOT_COMPONENT_BITS,
    num_weight_terms: int = DEFAULT_REFERENCE_WEIGHT_TERMS,
) -> dict[str, Any]:
    """Conservatively check uint64 power-sum capacity for a layout."""
    layout = DetectorFrameLayout(
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        num_input_streams=int(num_input_streams),
        num_selected_channels=int(num_selected_channels),
    )
    max_dot_component = (1 << (int(dot_component_bits) - 1)) - 1
    max_power_per_row = (
        COMPLEX_COMPONENT_COUNT * int(max_dot_component) * int(max_dot_component)
    )
    max_power_sum = int(layout.detector_rows_per_frame) * int(max_power_per_row)
    fits = max_power_sum <= UINT64_MAX
    return {
        "frame_size_samples": int(frame_size_samples),
        "detector_window_samples": int(detector_window_samples),
        "num_input_streams": int(num_input_streams),
        "num_selected_channels": int(num_selected_channels),
        "windows_per_stream": int(layout.windows_per_stream),
        "detector_rows_per_frame": int(layout.detector_rows_per_frame),
        "num_weight_terms": int(num_weight_terms),
        "dot_component_bits": int(dot_component_bits),
        "max_power_per_row": int(max_power_per_row),
        "max_power_sum_per_weight_term": int(max_power_sum),
        "uint64_max": int(UINT64_MAX),
        "power_sum_fits_uint64": bool(fits),
        "recommended_batching": "ok" if fits else "reduce_rows_or_split_batches",
    }


def quantization_metadata(
    *,
    mode: str,
    bits_per_component: int,
    clip_sigma: float,
    scale_by_stream: Sequence[float] | np.ndarray,
    clip_fraction_by_stream: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Build explicit quantization metadata for external adapters."""
    if mode not in SUPPORTED_QUANTIZATION_SCALE_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SUPPORTED_QUANTIZATION_SCALE_MODES)}; "
            f"got {mode!r}."
        )
    scales = np.asarray(scale_by_stream, dtype=np.float64)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("scale_by_stream must be a non-empty 1D array.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("scale_by_stream values must be positive and finite.")
    if clip_fraction_by_stream is None:
        clip_fraction = np.zeros(scales.shape, dtype=np.float64)
    else:
        clip_fraction = np.asarray(clip_fraction_by_stream, dtype=np.float64)
        if clip_fraction.shape != scales.shape:
            raise ValueError(
                "clip_fraction_by_stream must match scale_by_stream shape."
            )
    return {
        "mode": mode,
        "bits_per_component": int(bits_per_component),
        "clip_sigma": float(clip_sigma),
        "num_streams": int(scales.size),
        "scale_by_stream": [float(value) for value in scales],
        "clip_fraction_by_stream": [float(value) for value in clip_fraction],
    }


def load_stream_map(path: str | Path) -> InputStreamMap:
    """Load an input-stream map JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("stream map JSON must contain an object.")
    return InputStreamMap.from_dict(data)


__all__ = [
    "InputStreamMap",
    "StreamDescriptor",
    "build_stream_map_for_channel",
    "detector_shape_for_combined_streams",
    "detector_shape_for_per_stream_diagnostics",
    "layout_from_receiver_profile",
    "layout_uint64_bound_check",
    "load_stream_map",
    "quantization_metadata",
    "validate_integration_compatibility",
]
