# coding=utf-8
"""Segmented HDF5 discovery and window reads for CHIME pilot samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pilot_proxy.atsc_channels import (
    ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
    physical_channel_to_pilot_hz,
)

CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4 = "chime_offset_binary_complex_int4"
PACKED_TWOS_COMPLEMENT_COMPLEX_INT4 = "packed_twos_complement_complex_int4"
COMPLEX_FLOAT = "complex_float"
STRUCTURED_COMPLEX = "structured_complex"
REAL_IMAG_LAST_AXIS = "real_imag_last_axis"
UNKNOWN_ENCODING = "unknown"
DEFAULT_DATASET_PATH = "baseband"
CHIME_COARSE_WIDTH_HZ = 400_000_000.0 / 1024.0
ATSC_COARSE_CHANNEL_TOLERANCE_HZ = CHIME_COARSE_WIDTH_HZ / 2.0


@dataclass(frozen=True)
class ChimeSegment:
    path: Path
    physical_channel: int | None
    pilot_frequency_hz: float | None
    dataset_path: str
    num_time_samples: int
    shape: tuple[int, ...]
    dtype: str
    freq_id: int | None = None
    coarse_channel_center_hz: float | None = None
    sample_encoding: str = UNKNOWN_ENCODING


@dataclass(frozen=True)
class ChimePilotDataset:
    physical_channel: int
    pilot_frequency_hz: float
    segments: list[ChimeSegment]
    dataset_path: str
    time_axis: int
    stream_axis: int
    complex_axis: int | None = None
    sample_encoding: str = UNKNOWN_ENCODING
    freq_id: int | None = None
    coarse_channel_center_hz: float | None = None

    @property
    def total_time_samples(self) -> int:
        return int(sum(int(segment.num_time_samples) for segment in self.segments))

    @property
    def num_input_streams(self) -> int:
        if not self.segments:
            return 0
        shape = self.segments[0].shape
        return int(shape[int(self.stream_axis)])


def _walk(path: Path, filename_pattern: str) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"CHIME input directory does not exist: {root}")
    return sorted(root.rglob(filename_pattern))


def _axis_labels(attrs: Any) -> tuple[str, ...]:
    labels = attrs.get("axis", ())
    out: list[str] = []
    for label in labels:
        if isinstance(label, bytes):
            out.append(label.decode("utf-8", errors="replace"))
        else:
            out.append(str(label))
    return tuple(out)


def _find_dataset_path(h5: h5py.File, requested: str | None) -> str:
    if requested is not None:
        if requested not in h5:
            raise KeyError(f"dataset path {requested!r} not found in {h5.filename}")
        return requested
    if DEFAULT_DATASET_PATH in h5 and isinstance(h5[DEFAULT_DATASET_PATH], h5py.Dataset):
        return DEFAULT_DATASET_PATH

    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            candidates.append((int(np.prod(obj.shape)), name))

    h5.visititems(visitor)
    if not candidates:
        raise ValueError(f"no array-like HDF5 dataset found in {h5.filename}")
    return max(candidates)[1]


def _infer_axes(obj: h5py.Dataset) -> tuple[int, int, int | None]:
    labels = tuple(label.lower() for label in _axis_labels(obj.attrs))
    complex_axis = None
    scalar_dtype = np.dtype(obj.dtype)
    values_are_already_complex = bool(
        np.issubdtype(scalar_dtype, np.complexfloating) or scalar_dtype.names
    )
    if not values_are_already_complex and obj.ndim >= 3 and obj.shape[-1] == 2:
        tail_label = labels[-1] if len(labels) == obj.ndim else ""
        if tail_label in {"complex", "real_imag", "ri", ""}:
            complex_axis = obj.ndim - 1

    if "time" in labels and "input" in labels:
        time_axis = labels.index("time")
        stream_axis = labels.index("input")
    else:
        logical_axes = [axis for axis in range(obj.ndim) if axis != complex_axis]
        # Unlabelled CHIME arrays use the conventional (time, input[, RI])
        # layout.  Inferring the largest dimension as the stream axis swaps a
        # normal time-major block whenever time happens to be longest.
        time_axis = logical_axes[0] if logical_axes else 0
        stream_axis = logical_axes[1] if len(logical_axes) > 1 else time_axis
    return int(time_axis), int(stream_axis), complex_axis


def _float_attr(attrs: Any, key: str) -> float | None:
    if key not in attrs:
        return None
    values = np.asarray(attrs[key])
    if values.size != 1:
        raise ValueError(f"HDF5 attribute {key!r} must be one finite number")
    try:
        value = float(values.reshape(()).item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"HDF5 attribute {key!r} must be one finite number"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"HDF5 attribute {key!r} must be finite")
    return value


def _int_attr(attrs: Any, key: str) -> int | None:
    if key not in attrs:
        return None
    values = np.asarray(attrs[key])
    if values.size != 1:
        raise ValueError(f"HDF5 attribute {key!r} must be one integer")
    try:
        numeric = float(values.reshape(()).item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"HDF5 attribute {key!r} must be integral") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"HDF5 attribute {key!r} must be integral, got {numeric!r}")
    return int(numeric)


def _consistent_frequency_attrs(
    attrs: Any,
    keys: tuple[str, ...],
    *,
    label: str,
    mhz_keys: frozenset[str] = frozenset(),
) -> float | None:
    found: list[tuple[str, float]] = []
    for key in keys:
        value = _float_attr(attrs, key)
        if value is None:
            continue
        if key in mhz_keys and abs(value) < 1.0e6:
            value *= 1.0e6
        found.append((key, float(value)))
    if not found:
        return None
    reference = found[0][1]
    contradictory = [
        (key, value)
        for key, value in found[1:]
        if not np.isclose(value, reference, rtol=0.0, atol=1.0)
    ]
    if contradictory:
        raise ValueError(
            f"contradictory {label} HDF5 aliases: {found!r}"
        )
    return reference


def _pilot_frequency_hz_from_attrs(attrs: Any) -> float | None:
    return _consistent_frequency_attrs(
        attrs,
        ("pilot_frequency_hz", "dtv_pilot_hz"),
        label="pilot-frequency",
    )


def _coarse_center_hz_from_attrs(attrs: Any) -> float | None:
    return _consistent_frequency_attrs(
        attrs,
        (
            "coarse_channel_center_hz",
            "chime_frequency_hz",
            "frequency_hz",
            "freq",
        ),
        label="coarse-centre",
        mhz_keys=frozenset({"freq"}),
    )


def nearest_atsc_physical_channel(
    frequency_hz: float,
    *,
    tolerance_hz: float = ATSC_COARSE_CHANNEL_TOLERANCE_HZ,
) -> int | None:
    """Return the nearest UHF ATSC physical channel for a pilot-like frequency."""
    freq = float(frequency_hz)
    candidates = range(ATSC_UHF_MIN_PHYSICAL_CHANNEL, ATSC_UHF_MAX_PHYSICAL_CHANNEL + 1)
    best = min(candidates, key=lambda ch: abs(physical_channel_to_pilot_hz(ch) - freq))
    delta = abs(physical_channel_to_pilot_hz(best) - freq)
    # A pilot belongs to this receiver channel only when the coarse-channel
    # centre is at most half one 390.625-kHz F-engine bin away.  A former 3-MHz
    # threshold admitted adjacent freq_ids into the same ATSC pilot dataset.
    return int(best) if delta <= float(tolerance_hz) else None


def _physical_channel_from_attrs(
    attrs: Any,
    *,
    coarse_center_hz: float | None,
    pilot_frequency_hz: float | None,
) -> int | None:
    explicit: list[tuple[str, int]] = []
    for key in ("physical_channel", "dtv_physical_channel"):
        channel = _int_attr(attrs, key)
        if channel is not None:
            explicit.append((key, channel))
    if explicit and len({channel for _, channel in explicit}) != 1:
        raise ValueError(
            f"contradictory physical-channel HDF5 aliases: {explicit!r}"
        )
    physical_channel = explicit[0][1] if explicit else None
    pilot_channel = (
        None
        if pilot_frequency_hz is None
        else nearest_atsc_physical_channel(pilot_frequency_hz)
    )
    coarse_channel = (
        None
        if coarse_center_hz is None
        else nearest_atsc_physical_channel(coarse_center_hz)
    )
    if pilot_frequency_hz is not None and pilot_channel is None:
        raise ValueError(
            f"pilot frequency {pilot_frequency_hz:.0f} Hz is not near an ATSC pilot"
        )
    candidates = [
        channel
        for channel in (physical_channel, pilot_channel, coarse_channel)
        if channel is not None
    ]
    if candidates and len(set(candidates)) != 1:
        raise ValueError(
            "physical-channel, pilot-frequency, and coarse-centre metadata "
            f"disagree: {candidates!r}"
        )
    return candidates[0] if candidates else None


def _sample_encoding(obj: h5py.Dataset) -> str:
    dtype = np.dtype(obj.dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return COMPLEX_FLOAT
    if dtype.names:
        names = {name.lower() for name in dtype.names}
        if names & {"real", "re", "r"} and names & {"imag", "im", "i"}:
            return STRUCTURED_COMPLEX
    _, _, complex_axis = _infer_axes(obj)
    if complex_axis is not None:
        return REAL_IMAG_LAST_AXIS
    if dtype == np.dtype("uint8"):
        return CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4
    if dtype == np.dtype("int8"):
        return PACKED_TWOS_COMPLEMENT_COMPLEX_INT4
    return UNKNOWN_ENCODING


def _sort_key(path: Path) -> tuple[tuple[str, ...], int | str]:
    stem = path.stem
    numeric_stem: int | str = int(stem) if stem.isdigit() else stem
    return tuple(path.parent.parts), numeric_stem


def _read_segment(path: Path, dataset_path: str | None) -> tuple[ChimeSegment, int, int, int | None]:
    with h5py.File(path, "r") as h5:
        resolved_dataset_path = _find_dataset_path(h5, dataset_path)
        obj = h5[resolved_dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{resolved_dataset_path!r} is not a dataset in {path}")
        time_axis, stream_axis, complex_axis = _infer_axes(obj)
        data_axes = (time_axis, stream_axis)
        if (
            len(set(data_axes)) != 2
            or any(axis < 0 or axis >= obj.ndim for axis in data_axes)
        ):
            raise ValueError(
                f"CHIME dataset {resolved_dataset_path!r} in {path} must have "
                "distinct valid time and input axes"
            )
        if complex_axis is not None and (
            complex_axis in data_axes
            or complex_axis < 0
            or complex_axis >= obj.ndim
            or obj.shape[complex_axis] != 2
        ):
            raise ValueError(
                f"CHIME dataset {resolved_dataset_path!r} in {path} has an "
                "invalid real/imag axis"
            )
        logical_ndim = obj.ndim - int(complex_axis is not None)
        if logical_ndim != 2:
            raise ValueError(
                f"CHIME dataset {resolved_dataset_path!r} in {path} must have "
                "exactly time and input axes plus an optional real/imag axis; "
                f"got shape {obj.shape!r}"
            )
        freq_id = _int_attr(h5.attrs, "freq_id")
        if freq_id is not None and not 0 <= freq_id < 1024:
            raise ValueError(f"invalid CHIME freq_id {freq_id} in {path}")
        pilot_frequency_hz = _pilot_frequency_hz_from_attrs(h5.attrs)
        frequency_hz = _coarse_center_hz_from_attrs(h5.attrs)
        if frequency_hz is None and freq_id is None:
            raise ValueError(
                f"CHIME segment {path} has neither freq_id nor a finite "
                "coarse-channel centre; receiver-channel identity is unverifiable"
            )
        if frequency_hz is None and freq_id is not None:
            frequency_hz = 800_000_000.0 - freq_id * CHIME_COARSE_WIDTH_HZ
        physical_channel = _physical_channel_from_attrs(
            h5.attrs,
            coarse_center_hz=frequency_hz,
            pilot_frequency_hz=pilot_frequency_hz,
        )
        if frequency_hz is not None and freq_id is not None:
            expected_center_hz = 800_000_000.0 - freq_id * CHIME_COARSE_WIDTH_HZ
            if not np.isclose(
                frequency_hz,
                expected_center_hz,
                rtol=0.0,
                atol=1.0,
            ):
                raise ValueError(
                    f"CHIME freq_id {freq_id} in {path} implies centre "
                    f"{expected_center_hz:.0f} Hz, but metadata records "
                    f"{frequency_hz:.0f} Hz"
                )
        if frequency_hz is not None and physical_channel is not None:
            inferred_channel = nearest_atsc_physical_channel(frequency_hz)
            if inferred_channel != physical_channel:
                raise ValueError(
                    f"CHIME coarse-channel centre {frequency_hz:.0f} Hz in "
                    f"{path} is not within half a coarse bin of ATSC physical "
                    f"channel {physical_channel}"
                )
        if pilot_frequency_hz is not None and physical_channel is not None:
            expected_pilot_hz = physical_channel_to_pilot_hz(physical_channel)
            if not np.isclose(
                pilot_frequency_hz,
                expected_pilot_hz,
                rtol=0.0,
                atol=1.0,
            ):
                raise ValueError(
                    f"pilot frequency {pilot_frequency_hz:.0f} Hz does not "
                    f"match ATSC physical channel {physical_channel} pilot "
                    f"{expected_pilot_hz:.0f} Hz"
                )
        elif physical_channel is not None:
            pilot_frequency_hz = float(
                physical_channel_to_pilot_hz(physical_channel)
            )
        segment = ChimeSegment(
            path=Path(path),
            physical_channel=physical_channel,
            pilot_frequency_hz=pilot_frequency_hz,
            dataset_path=resolved_dataset_path,
            num_time_samples=int(obj.shape[time_axis]),
            shape=tuple(int(value) for value in obj.shape),
            dtype=str(obj.dtype),
            freq_id=freq_id,
            coarse_channel_center_hz=frequency_hz,
            sample_encoding=_sample_encoding(obj),
        )
    return segment, int(time_axis), int(stream_axis), complex_axis


def discover_chime_pilot_datasets(
    root: Path,
    *,
    dataset_path: str | None,
    filename_pattern: str = "*.h5",
) -> dict[int, ChimePilotDataset]:
    """Discover segmented CHIME pilot-channel datasets below the root path."""
    # Segment first by receiver-channel identity, not merely by the much wider
    # 6-MHz ATSC physical channel.  The public result remains keyed by physical
    # channel because one ATSC pilot must resolve to exactly one CHIME freq_id.
    grouped: dict[
        tuple[int, int | None, float | None],
        list[tuple[ChimeSegment, int, int, int | None]],
    ] = {}
    for path in _walk(Path(root), filename_pattern):
        segment, time_axis, stream_axis, complex_axis = _read_segment(path, dataset_path)
        if segment.physical_channel is None:
            continue
        identity = (
            int(segment.physical_channel),
            segment.freq_id,
            segment.coarse_channel_center_hz,
        )
        grouped.setdefault(identity, []).append(
            (segment, time_axis, stream_axis, complex_axis)
        )

    datasets: dict[int, ChimePilotDataset] = {}
    for identity in sorted(
        grouped,
        key=lambda value: (
            value[0],
            -1 if value[1] is None else value[1],
            float("-inf") if value[2] is None else value[2],
        ),
    ):
        physical_channel, _freq_id, _coarse_center = identity
        if physical_channel in datasets:
            previous = datasets[physical_channel]
            raise ValueError(
                "multiple CHIME coarse-channel identities map to one ATSC "
                f"physical channel {physical_channel}: "
                f"freq_id={previous.freq_id}, "
                f"center={previous.coarse_channel_center_hz!r} and "
                f"freq_id={_freq_id}, center={_coarse_center!r}. Refusing to "
                "merge neighboring receiver channels."
            )
        items = sorted(grouped[identity], key=lambda item: _sort_key(item[0].path))
        segments = [item[0] for item in items]
        first_segment, time_axis, stream_axis, complex_axis = items[0]
        for segment, seg_time_axis, seg_stream_axis, seg_complex_axis in items:
            if len(segment.shape) != len(first_segment.shape):
                raise ValueError(
                    "segments for one coarse channel use inconsistent ranks: "
                    f"{first_segment.shape!r} and {segment.shape!r}."
                )
            if segment.dataset_path != first_segment.dataset_path:
                raise ValueError(
                    "segments for one physical channel use multiple dataset paths: "
                    f"{first_segment.dataset_path!r} and {segment.dataset_path!r}"
                )
            if (seg_time_axis, seg_stream_axis, seg_complex_axis) != (
                time_axis,
                stream_axis,
                complex_axis,
            ):
                raise ValueError(
                    "segments for one physical channel use inconsistent axes."
                )
            if segment.shape[stream_axis] != first_segment.shape[stream_axis]:
                raise ValueError(
                    "segments for one physical channel use inconsistent input counts."
                )
            if segment.dtype != first_segment.dtype:
                raise ValueError(
                    "segments for one coarse channel use inconsistent dtypes: "
                    f"{first_segment.dtype!r} and {segment.dtype!r}."
                )
            if segment.sample_encoding != first_segment.sample_encoding:
                raise ValueError(
                    "segments for one coarse channel use inconsistent sample "
                    f"encodings: {first_segment.sample_encoding!r} and "
                    f"{segment.sample_encoding!r}."
                )
            for axis, (size, first_size) in enumerate(
                zip(segment.shape, first_segment.shape)
            ):
                if axis != time_axis and size != first_size:
                    raise ValueError(
                        "segments for one coarse channel use inconsistent "
                        f"non-time dimensions: {first_segment.shape!r} and "
                        f"{segment.shape!r}."
                    )
        datasets[physical_channel] = ChimePilotDataset(
            physical_channel=int(physical_channel),
            pilot_frequency_hz=float(physical_channel_to_pilot_hz(physical_channel)),
            segments=segments,
            dataset_path=first_segment.dataset_path,
            time_axis=int(time_axis),
            stream_axis=int(stream_axis),
            complex_axis=complex_axis,
            sample_encoding=first_segment.sample_encoding,
            freq_id=first_segment.freq_id,
            coarse_channel_center_hz=first_segment.coarse_channel_center_hz,
        )
    return datasets


def _structured_to_complex(arr: np.ndarray) -> np.ndarray:
    names = tuple(arr.dtype.names or ())
    lower = {name.lower(): name for name in names}
    real_name = next((lower[name] for name in ("real", "re", "r") if name in lower), None)
    imag_name = next((lower[name] for name in ("imag", "im", "i") if name in lower), None)
    if real_name is None or imag_name is None:
        raise ValueError(f"structured dtype lacks real/imag fields: {arr.dtype}")
    return np.asarray(arr[real_name]) + 1j * np.asarray(arr[imag_name])


def _normalize_read_array(
    raw: np.ndarray,
    *,
    time_axis: int,
    stream_axis: int,
    complex_axis: int | None,
) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.dtype.names:
        arr = _structured_to_complex(arr)
    elif complex_axis is not None:
        axis = int(complex_axis)
        arr = np.moveaxis(arr, axis, -1)
        arr = np.asarray(arr[..., 0]) + 1j * np.asarray(arr[..., 1])
        if axis < time_axis:
            time_axis -= 1
        if axis < stream_axis:
            stream_axis -= 1

    stream_time = np.moveaxis(arr, (int(stream_axis), int(time_axis)), (0, 1))
    if stream_time.ndim != 2:
        raise ValueError(
            "normalized CHIME block must have stream/time axes only; "
            f"got shape {stream_time.shape}"
        )
    return np.ascontiguousarray(stream_time[:, np.newaxis, :])


def _read_segment_window(
    segment: ChimeSegment,
    *,
    time_axis: int,
    stream_axis: int,
    complex_axis: int | None,
    start: int,
    stop: int,
) -> np.ndarray:
    with h5py.File(segment.path, "r") as h5:
        obj = h5[segment.dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{segment.dataset_path!r} is not a dataset in {segment.path}")
        selection: list[slice] = [slice(None)] * obj.ndim
        selection[int(time_axis)] = slice(int(start), int(stop))
        raw = obj[tuple(selection)]
    return _normalize_read_array(
        raw,
        time_axis=int(time_axis),
        stream_axis=int(stream_axis),
        complex_axis=complex_axis,
    )


def read_complex_window(
    dataset: ChimePilotDataset,
    *,
    start_sample: int,
    stop_sample: int,
) -> np.ndarray:
    """Return selected input-stream samples across segment boundaries.

    Native CHIME unsigned-byte samples remain packed in offset-binary int4 form.
    Floating-point and explicit real/imag datasets are returned as complex arrays.
    """
    start = int(start_sample)
    stop = int(stop_sample)
    if start < 0 or stop < start:
        raise ValueError("invalid sample window")
    if stop > dataset.total_time_samples:
        raise ValueError(
            "requested window extends past available samples: "
            f"stop={stop}, available={dataset.total_time_samples}"
        )
    if start == stop:
        return np.empty((dataset.num_input_streams, 1, 0), dtype=np.uint8)

    chunks: list[np.ndarray] = []
    logical_start = 0
    for segment in dataset.segments:
        logical_stop = logical_start + int(segment.num_time_samples)
        overlap_start = max(start, logical_start)
        overlap_stop = min(stop, logical_stop)
        if overlap_start < overlap_stop:
            chunks.append(
                _read_segment_window(
                    segment,
                    time_axis=dataset.time_axis,
                    stream_axis=dataset.stream_axis,
                    complex_axis=dataset.complex_axis,
                    start=overlap_start - logical_start,
                    stop=overlap_stop - logical_start,
                )
            )
        logical_start = logical_stop
        if logical_start >= stop:
            break
    if not chunks:
        raise ValueError("requested window did not overlap any segment")
    return np.ascontiguousarray(np.concatenate(chunks, axis=2))


def dataset_manifest(dataset: ChimePilotDataset) -> dict[str, Any]:
    """Return a JSON-safe manifest entry for one discovered pilot dataset."""
    return {
        "physical_channel": int(dataset.physical_channel),
        "pilot_frequency_hz": float(dataset.pilot_frequency_hz),
        "coarse_channel_center_hz": dataset.coarse_channel_center_hz,
        "freq_id": dataset.freq_id,
        "dataset_path": dataset.dataset_path,
        "time_axis": int(dataset.time_axis),
        "stream_axis": int(dataset.stream_axis),
        "complex_axis": dataset.complex_axis,
        "sample_encoding": dataset.sample_encoding,
        "num_input_streams": int(dataset.num_input_streams),
        "total_time_samples": int(dataset.total_time_samples),
        "segments": [
            {
                "path": str(segment.path),
                "num_time_samples": int(segment.num_time_samples),
                "shape": list(segment.shape),
                "dtype": segment.dtype,
                "freq_id": segment.freq_id,
                "coarse_channel_center_hz": segment.coarse_channel_center_hz,
                "sample_encoding": segment.sample_encoding,
            }
            for segment in dataset.segments
        ],
    }


__all__ = [
    "ATSC_COARSE_CHANNEL_TOLERANCE_HZ",
    "CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4",
    "CHIME_COARSE_WIDTH_HZ",
    "COMPLEX_FLOAT",
    "ChimePilotDataset",
    "ChimeSegment",
    "PACKED_TWOS_COMPLEMENT_COMPLEX_INT4",
    "REAL_IMAG_LAST_AXIS",
    "STRUCTURED_COMPLEX",
    "UNKNOWN_ENCODING",
    "dataset_manifest",
    "discover_chime_pilot_datasets",
    "nearest_atsc_physical_channel",
    "read_complex_window",
]
