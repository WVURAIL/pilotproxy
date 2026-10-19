"""Native CHORD capture reader and exact fine-detector replay primitives.

Adapted from Kotekan ``tools/verify_pilotproxy_pipeline.py`` at
``c81def5518bb86b03afa837d6c57655ef48c548d``. The same revision's
``lib/metadata/chordMetadata.cpp``, ``DataType.hpp`` and
``lib/cuda/cudaPilotProxyDetector.cpp`` define the serialized metadata ABI,
array types and product layouts used here. Only its little-endian LP64 ABI is
supported. No code is imported from a Kotekan checkout at runtime.

Geometry and file order are explicit: a recording's manifest, rather than the
small verifier's fixture constants, supplies them. Exact reference arithmetic
uses the installed package's frozen FFT and fine-decision implementation.
"""
from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.fine_decision import (
    FINE_DECISION_VERSION,
    FineRequiredMultiplier,
    fine_mask_decision,
    fine_required_multiplier_q16,
)
from pilot_proxy.fxfft import fine_power_fx

REFERENCE_KOTEKAN_COMMIT = "c81def5518bb86b03afa837d6c57655ef48c548d"
CHORD_METADATA_ABI = "chord_metadata_le_lp64_c81def5518bb"
CHORD_METADATA_SIZE = 148888
CHORD_METADATA_LIMITS = (10, 24, 24, 12288, 128, 8)
CHORD_METADATA_PREFIX = struct.Struct("<7i4xqi")
# Native int64 alignment adds padding before fpga_seq_num and dim_scaling.
CHORD_NFREQ_OFFSET = 528
CHORD_COARSE_FREQ_OFFSET = 532
CHORD_UPCHAN_FACTOR_OFFSET = 49684
CHORD_UPCHAN_INDEX_OFFSET = 98836


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class CaptureGeometry:
    """Explicit geometry for one native capture; no instrument defaults."""

    num_times: int
    k: int
    frequency_ids: tuple[int, ...]
    num_dishes: int
    num_polarizations: int = 2
    windows_per_stream: int = 128

    def __post_init__(self) -> None:
        for name in ("num_times", "k", "num_dishes", "num_polarizations", "windows_per_stream"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.k not in (64, 128):
            raise ValueError("k must be 64 or 128")
        if self.num_polarizations != 2 or self.windows_per_stream != 128:
            raise ValueError("fine replay requires 2 polarizations and 128 windows per stream")
        if self.num_streams > 2048:
            raise ValueError("num_streams exceeds the supported 2048-stream arithmetic bound")
        if self.num_times > self.block_samples or self.block_samples % self.num_times:
            raise ValueError("input frame samples must divide one complete detector block")
        ids = tuple(_integer(value, "frequency ID") for value in self.frequency_ids)
        if not ids or len(ids) > 12288 or any(value >= 12288 for value in ids):
            raise ValueError("frequency IDs must be a nonempty CHORD coarse-frequency list")
        if len(set(ids)) != len(ids):
            raise ValueError("frequency IDs must be unique")
        object.__setattr__(self, "frequency_ids", ids)

    @property
    def block_samples(self) -> int:
        return self.k * self.windows_per_stream

    @property
    def num_freq(self) -> int:
        return len(self.frequency_ids)

    @property
    def num_streams(self) -> int:
        return self.num_polarizations * self.num_dishes

    @property
    def frames_per_block(self) -> int:
        return self.block_samples // self.num_times

    @property
    def voltage_frame_size(self) -> int:
        return self.num_times * self.num_freq * self.num_streams

    @property
    def mask_frame_size(self) -> int:
        return self.num_freq

    @property
    def power_frame_size(self) -> int:
        return self.num_freq * 3 * 8


@dataclass(frozen=True)
class RawFrame:
    payload: np.ndarray
    fpga_seq_num: int
    time_downsampling_fpga: int
    frequency_ids: tuple[int, ...]
    source_path: str
    frame_offset: int
    metadata_offset: int
    payload_offset: int
    metadata_sha256: str
    frame_counter: int
    array_name: str
    data_type: int
    dimensions: tuple[int, ...]
    dimension_names: tuple[str, ...]
    dimension_scaling: tuple[int, ...]
    strides: tuple[int, ...]
    array_offset: int
    upchannel_factors: tuple[int, ...] | None
    upchannel_indices: tuple[int, ...] | None
    stream_ids: tuple[int, ...] | None
    rfi_flagged_samples: int
    lost_timesamples: int

    def provenance(self) -> dict[str, Any]:
        """JSON-safe native frame identity including exact source byte ranges."""
        return {
            "source_path": self.source_path,
            "frame_offset": self.frame_offset,
            "metadata_offset": self.metadata_offset,
            "metadata_size": CHORD_METADATA_SIZE,
            "payload_offset": self.payload_offset,
            "payload_size": int(self.payload.size),
            "metadata_sha256": self.metadata_sha256,
            "fpga_seq_num": self.fpga_seq_num,
            "time_downsampling_fpga": self.time_downsampling_fpga,
            "frequency_ids": list(self.frequency_ids),
            "frame_counter": self.frame_counter,
            "array_name": self.array_name,
            "data_type": self.data_type,
            "dimensions": list(self.dimensions),
            "dimension_names": list(self.dimension_names),
            "dimension_scaling": list(self.dimension_scaling),
            "strides": list(self.strides),
            "array_offset": self.array_offset,
            "upchannel_factors": None if self.upchannel_factors is None else list(self.upchannel_factors),
            "upchannel_indices": None if self.upchannel_indices is None else list(self.upchannel_indices),
            "stream_ids": None if self.stream_ids is None else list(self.stream_ids),
            "rfi_flagged_samples": self.rfi_flagged_samples,
            "lost_timesamples": self.lost_timesamples,
        }


def _metadata_fields(metadata: bytes) -> dict[str, Any]:
    prefix = CHORD_METADATA_PREFIX.unpack_from(metadata)
    if prefix[:6] != CHORD_METADATA_LIMITS:
        raise ValueError("unsupported CHORD metadata layout")
    frame_counter, sequence, downsampling = prefix[6:]
    if sequence < 0 or downsampling <= 0:
        raise ValueError("missing or invalid FPGA timing")
    data_type, dims = struct.unpack_from("<2i", metadata, 68)
    if not 1 <= dims <= 10:
        raise ValueError("missing or invalid array dimensions")
    nfreq = struct.unpack_from("<i", metadata, CHORD_NFREQ_OFFSET)[0]
    if not 1 <= nfreq <= 12288:
        raise ValueError("missing or invalid metadata frequency count")
    ids = struct.unpack_from(f"<{nfreq}i", metadata, CHORD_COARSE_FREQ_OFFSET)
    if any(not 0 <= value < 12288 for value in ids) or len(set(ids)) != nfreq:
        raise ValueError("missing, invalid or repeated metadata coarse_freq IDs")
    factors = struct.unpack_from(f"<{nfreq}i", metadata, CHORD_UPCHAN_FACTOR_OFFSET)
    indices = struct.unpack_from(f"<{nfreq}i", metadata, CHORD_UPCHAN_INDEX_OFFSET)
    if factors[0] == 0:
        factors = None
    elif any(value != 1 for value in factors):
        raise ValueError("upchannelized frames are outside the native coarse-voltage contract")
    if indices[0] == -1:
        indices = None
    elif any(value != 0 for value in indices):
        raise ValueError("nonzero upchannelization index is unsupported")
    stream_count = struct.unpack_from("<i", metadata, 147988)[0]
    if not -1 <= stream_count <= 128:
        raise ValueError("invalid metadata stream ID count")
    rfi_enabled, threshold_count = struct.unpack_from("<2i", metadata, 148504)
    if rfi_enabled not in (-1, 0, 1) or not -1 <= threshold_count <= 8:
        raise ValueError("invalid metadata RFI configuration")
    flagged, lost = struct.unpack_from("<2i", metadata, 148576)
    if flagged < -1 or lost < -1:
        raise ValueError("invalid metadata sample counters")

    def name(offset: int) -> str:
        try:
            return metadata[offset:offset + 24].split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("metadata array names must be ASCII") from exc

    return {
        "fpga_seq_num": sequence,
        "time_downsampling_fpga": downsampling,
        "frequency_ids": ids,
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "frame_counter": frame_counter,
        "array_name": name(44),
        "data_type": data_type,
        "dimensions": struct.unpack_from(f"<{dims}i", metadata, 76),
        "dimension_names": tuple(name(116 + index * 24) for index in range(dims)),
        "dimension_scaling": struct.unpack_from(f"<{dims}q", metadata, 360),
        "strides": struct.unpack_from(f"<{dims}q", metadata, 440),
        "array_offset": struct.unpack_from("<q", metadata, 520)[0],
        "upchannel_factors": factors,
        "upchannel_indices": indices,
        "stream_ids": None if stream_count == -1 else struct.unpack_from(f"<{stream_count}I", metadata, 147992),
        "rfi_flagged_samples": flagged,
        "lost_timesamples": lost,
    }


def read_raw_frames(paths: Sequence[str | Path], frame_size: int) -> list[RawFrame]:
    """Read explicit, ordered rawFileWrite files with native CHORD metadata.

    No glob expansion, path sorting or missing-metadata fallback is performed.
    Byte offsets include the uint32 metadata-size word preceding each frame.
    """
    frame_size = _integer(frame_size, "frame_size", 1)
    if isinstance(paths, (str, bytes, Path)):
        raise ValueError("paths must be an explicit ordered sequence of files")
    resolved = [Path(path).resolve(strict=True) for path in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("capture files must be nonempty and unique")
    frames = []
    for path in resolved:
        with path.open("rb") as stream:
            count = 0
            while True:
                offset = stream.tell()
                length = stream.read(4)
                if not length:
                    break
                if len(length) != 4:
                    raise ValueError(f"{path}: truncated metadata length at offset {offset}")
                metadata_size = struct.unpack("<I", length)[0]
                if metadata_size != CHORD_METADATA_SIZE:
                    raise ValueError(f"{path}: unsupported CHORD metadata size {metadata_size}")
                metadata = stream.read(metadata_size)
                if len(metadata) != metadata_size:
                    raise ValueError(f"{path}: truncated metadata at offset {offset + 4}")
                try:
                    fields = _metadata_fields(metadata)
                except ValueError as exc:
                    raise ValueError(f"{path}: frame at offset {offset}: {exc}") from exc
                payload_offset = stream.tell()
                payload = stream.read(frame_size)
                if len(payload) != frame_size:
                    raise ValueError(f"{path}: truncated frame at offset {payload_offset}")
                frames.append(RawFrame(
                    payload=np.frombuffer(payload, dtype=np.uint8),
                    source_path=str(path), frame_offset=offset,
                    metadata_offset=offset + 4, payload_offset=payload_offset, **fields,
                ))
                count += 1
            if not count:
                raise ValueError(f"{path}: empty capture file")
    return frames


def validate_frames(
    voltage: Sequence[RawFrame], masks: Sequence[RawFrame], powers: Sequence[RawFrame],
    expected_input_frames: int, *, geometry: CaptureGeometry,
) -> int:
    """Require complete blocks, identical frequency order and absolute timing."""
    expected_input_frames = _integer(expected_input_frames, "expected_input_frames", 1)
    if expected_input_frames % geometry.frames_per_block:
        raise ValueError("input frame count must contain whole detector blocks")
    num_blocks = expected_input_frames // geometry.frames_per_block
    if (len(voltage), len(masks), len(powers)) != (expected_input_frames, num_blocks, num_blocks):
        raise ValueError("voltage/mask/power frame counts do not match complete capture geometry")
    start = voltage[0].fpga_seq_num
    downsampling = voltage[0].time_downsampling_fpga
    layouts = (
        ("voltage", voltage, "E", 8, (geometry.num_times, geometry.num_freq, 2, geometry.num_dishes),
         ("T", "F", "P", "D"), geometry.num_times, downsampling, geometry.voltage_frame_size),
        ("mask", masks, "dtv_mask", 9, (geometry.num_freq,), ("F",), geometry.block_samples,
         geometry.block_samples * downsampling, geometry.mask_frame_size),
        ("power", powers, "dtv_powers", 6, (geometry.num_freq, 3), ("F", "W"), geometry.block_samples,
         geometry.block_samples * downsampling, geometry.power_frame_size),
    )
    for kind, frames, name, dtype, dims, names, step, expected_downsampling, frame_size in layouts:
        stride, strides = 1, []
        for extent in reversed(dims):
            strides.insert(0, stride)
            stride *= extent
        for index, frame in enumerate(frames):
            label = f"{kind} frame {index}"
            if frame.frequency_ids != geometry.frequency_ids:
                raise ValueError(f"{label}: coarse_freq IDs/order do not match capture geometry")
            if (frame.array_name, frame.data_type, frame.dimensions, frame.dimension_names,
                frame.dimension_scaling, frame.strides, frame.array_offset) != (
                name, dtype, dims, names, (1,) * len(dims), tuple(strides), 0,
            ):
                raise ValueError(f"{label}: metadata array layout does not match capture geometry")
            if frame.payload.dtype != np.dtype(np.uint8) or frame.payload.shape != (frame_size,):
                raise ValueError(f"{label}: payload size/type does not match capture geometry")
            if (frame.fpga_seq_num != start + index * step * downsampling
                or frame.time_downsampling_fpga != expected_downsampling):
                raise ValueError(f"{label}: discontinuous or mismatched FPGA timing")
    return num_blocks


def detector_rows_from_voltage(
    block_bytes: Any, freq_index: int, time_reverse_windows: bool, *, geometry: CaptureGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """[T,F,P,D] offset-binary bytes to stream-major int16 rows [streams*128,K]."""
    freq_index = _integer(freq_index, "freq_index")
    if freq_index >= geometry.num_freq:
        raise ValueError("freq_index is outside capture geometry")
    if not isinstance(time_reverse_windows, bool):
        raise ValueError("time_reverse_windows must be a boolean")
    arr = np.asarray(block_bytes)
    if arr.dtype != np.dtype(np.uint8) or arr.shape != (
        geometry.block_samples * geometry.num_freq * geometry.num_streams,
    ):
        raise ValueError("voltage block must be a complete one-dimensional uint8 block")
    arr = arr.reshape(geometry.block_samples, geometry.num_freq, 2, geometry.num_dishes)
    streams = arr[:, freq_index].transpose(1, 2, 0).reshape(geometry.num_streams, geometry.block_samples)
    rows = streams.reshape(-1, geometry.k)
    if time_reverse_windows:
        rows = rows[:, ::-1]
    real = ((rows.astype(np.int16) >> 4) & 15) - 8
    imag = (rows.astype(np.int16) & 15) - 8
    return real, imag


def expected_matched_filter_row_projections(real: np.ndarray, imag: np.ndarray, weights_packed: bytes) -> np.ndarray:
    """Exact complex projections [three terms, rows, real/imaginary]."""
    if real.dtype != np.dtype(np.int16) or imag.dtype != np.dtype(np.int16):
        raise ValueError("voltage rows must have int16 dtype")
    if real.ndim != 2 or real.shape != imag.shape or real.shape[1] not in (64, 128):
        raise ValueError("voltage rows must have matching [rows,K] shapes")
    if np.any(real < -8) or np.any(real > 7) or np.any(imag < -8) or np.any(imag > 7):
        raise ValueError("voltage rows must contain signed int4 samples")
    if len(weights_packed) != 3 * real.shape[1]:
        raise ValueError("weight bytes must contain exactly three K-sample terms")
    weights = np.frombuffer(weights_packed, dtype=np.uint8).reshape(3, real.shape[1]).astype(np.int16)
    wr = (((weights >> 4) & 15) ^ 8) - 8
    wi = ((weights & 15) ^ 8) - 8
    projections = []
    for term in range(3):
        zr = (real * wr[term] + imag * wi[term]).sum(axis=1, dtype=np.int64)
        zi = (imag * wr[term] - real * wi[term]).sum(axis=1, dtype=np.int64)
        projections.append(np.stack((zr, zi), axis=-1))
    return np.stack(projections)


@dataclass(frozen=True)
class ExpectedProducts:
    masks: np.ndarray
    powers: np.ndarray
    decisions: tuple[FineRequiredMultiplier | None, ...]


def expected_products(
    block_bytes: Any, bundle: dict[str, Any], weight_bank: bytes, *, geometry: CaptureGeometry,
) -> ExpectedProducts:
    """Compute fine masks, exact frame powers and exact Q16 keep boundaries.

    The frame powers are the all-bin sums of the exact fxfft256 fine powers,
    ``P_r = sum_f S_r[f]`` in uint64 (kernel core 3.0.0), which is what the
    stage's ``d_power_out`` carries.
    Unbound frequencies have zero mask/powers and a ``None`` decision, matching
    the submitted stage. Every bound frequency must have a fine calibration.
    Invalid rank support remains a decision with ``valid=False`` and mask zero.
    """
    profiles = {}
    for row in bundle["profiles"]:
        frequency_id = row.get("chord_channel_id")
        if frequency_id is None:
            continue
        frequency_id = _integer(frequency_id, "chord_channel_id")
        if frequency_id >= 12288 or frequency_id in profiles:
            raise ValueError("invalid or duplicate bundle chord_channel_id")
        profiles[frequency_id] = row
    reverse = bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"]
    if not isinstance(reverse, bool):
        raise ValueError("time reversal preprocessing must be a boolean")
    masks = np.zeros(geometry.num_freq, dtype=np.uint8)
    powers = np.zeros((geometry.num_freq, 3), dtype=np.uint64)
    decisions = []
    for index, frequency_id in enumerate(geometry.frequency_ids):
        profile = profiles.get(frequency_id)
        if profile is None:
            decisions.append(None)
            continue
        calibration = profile["fine_calibration"]
        if calibration.get("status") != "calibrated" or calibration.get("decision_version") != FINE_DECISION_VERSION:
            raise ValueError("bound profile requires calibrated fine_decision_v1; leave an uncalibrated channel unbound")
        offset = _integer(profile["weight_bank_offset_bytes"], "weight_bank_offset_bytes")
        nbytes = _integer(profile["weight_bank_nbytes"], "weight_bank_nbytes", 1)
        if nbytes != 3 * geometry.k or offset + nbytes > len(weight_bank):
            raise ValueError("profile weight range does not match geometry/weight bank")
        real, imag = detector_rows_from_voltage(block_bytes, index, reverse, geometry=geometry)
        projections = expected_matched_filter_row_projections(real, imag, weight_bank[offset:offset + nbytes])
        fine = fine_power_fx(projections, num_streams=geometry.num_streams)
        powers[index] = fine.sum(axis=1, dtype=np.uint64)
        words = calibration["bulk_mask_words_hex"]
        if not isinstance(words, list) or len(words) != 4 or any(not isinstance(word, str) for word in words):
            raise ValueError("bulk_mask_words_hex must contain four hexadecimal strings")
        args = {
            "anchor_bin": calibration["anchor_bin"],
            "designated_half_width": calibration["designated_half_width"],
            "bulk_mask": [int(word, 16) for word in words],
            "cfar_rank": calibration["cfar_rank"],
        }
        decision = fine_mask_decision(fine, multiplier_q16=calibration["cfar_multiplier_q16"], **args)
        required = fine_required_multiplier_q16(fine, **args)
        masks[index] = decision.mask
        decisions.append(required)
    masks.flags.writeable = False
    powers.flags.writeable = False
    return ExpectedProducts(masks, powers, tuple(decisions))
