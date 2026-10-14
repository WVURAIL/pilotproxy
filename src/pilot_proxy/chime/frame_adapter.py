# coding=utf-8
"""Normalize CHIME frame blocks into the locked detector input contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pilot_proxy.detector_geometry import (
    DetectorFrameLayout,
    apply_spectral_sense_to_detector_matrix,
    stream_time_block_to_detector_matrix,
)
from pilot_proxy.integration.packing import pack_channelized_streams_for_detector
from pilot_proxy.integration.schemas import (
    COMBINE_MODE_COMBINED_STREAMS,
    QUANTIZATION_SCALE_MODE_GLOBAL,
    QUANTIZATION_SCALE_MODE_PROVIDED,
)

from .hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    COMPLEX_FLOAT,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    REAL_IMAG_LAST_AXIS,
    STRUCTURED_COMPLEX,
)

NATIVE_CHIME_ZERO_OFFSET = 8
DEFAULT_BITS_PER_COMPONENT = 4


@dataclass(frozen=True)
class PackedChimeBlock:
    packed: np.ndarray
    input_layout: dict[str, Any]
    quantization: dict[str, Any]
    baseband_power_linear: np.ndarray


def _sign_extend_i4(nibble: np.ndarray) -> np.ndarray:
    values = np.asarray(nibble, dtype=np.int16) & 0x0F
    extended = np.asarray(np.bitwise_xor(values, 0x08) - 0x08, dtype=np.int16)
    return extended


def repack_chime_offset_binary_i4_to_twos_complement(values: np.ndarray) -> np.ndarray:
    """Convert CHIME offset-binary int4+int4 bytes to kernel-packed int8 bytes."""
    src = np.asarray(values, dtype=np.uint8)
    high = np.asarray((src >> 4) & 0x0F, dtype=np.uint8)
    low = np.asarray(src & 0x0F, dtype=np.uint8)
    real = high.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    imag = low.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    packed = np.asarray(
        np.bitwise_or(
            np.left_shift(np.bitwise_and(real, 0x0F), 4),
            np.bitwise_and(imag, 0x0F),
        ),
        dtype=np.uint8,
    )
    return np.ascontiguousarray(packed.view(np.int8))


def unpack_chime_offset_binary_i4_to_complex(values: np.ndarray) -> np.ndarray:
    """Return native CHIME-packed samples as complex signed int4 values."""
    src = np.asarray(values, dtype=np.uint8)
    high = np.asarray((src >> 4) & 0x0F, dtype=np.uint8)
    low = np.asarray(src & 0x0F, dtype=np.uint8)
    real = high.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    imag = low.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    return np.asarray(real, dtype=np.float32) + 1j * np.asarray(imag, dtype=np.float32)


def unpack_twos_complement_i4_to_complex(values: np.ndarray) -> np.ndarray:
    """Return kernel-packed two's-complement int4 samples as complex values."""
    src = np.asarray(values, dtype=np.uint8)
    real = _sign_extend_i4(np.asarray(src >> 4, dtype=np.uint8))
    imag = _sign_extend_i4(src)
    return np.asarray(real, dtype=np.float32) + 1j * np.asarray(imag, dtype=np.float32)


def _native_offset_binary_power(frame: np.ndarray) -> float:
    src = np.asarray(frame, dtype=np.uint8)
    high = np.asarray((src >> 4) & 0x0F, dtype=np.uint8)
    low = np.asarray(src & 0x0F, dtype=np.uint8)
    real = high.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    imag = low.astype(np.int16) - NATIVE_CHIME_ZERO_OFFSET
    real_i32 = np.asarray(real, dtype=np.int32)
    imag_i32 = np.asarray(imag, dtype=np.int32)
    power = real_i32 * real_i32
    power += imag_i32 * imag_i32
    return float(np.mean(power, dtype=np.float64))


def _twos_complement_power(frame: np.ndarray) -> float:
    src = np.asarray(frame, dtype=np.uint8)
    real = np.asarray(_sign_extend_i4(np.asarray(src >> 4, dtype=np.uint8)), dtype=np.int32)
    imag = np.asarray(_sign_extend_i4(src), dtype=np.int32)
    return float(np.mean(real * real + imag * imag, dtype=np.float64))


def baseband_power_by_frame(
    block: np.ndarray,
    *,
    frame_size_samples: int,
    frames_in_chunk: int,
    sample_encoding: str,
) -> np.ndarray:
    """Compute mean baseband power for each frame in a normalized block."""
    arr = np.asarray(block)
    if arr.ndim != 3 or arr.shape[1] != 1:
        raise ValueError(
            "CHIME block must have shape (num_input_streams, 1, time); "
            f"got {arr.shape}"
        )
    frame = int(frame_size_samples)
    frames = int(frames_in_chunk)
    if arr.shape[2] < frame * frames:
        raise ValueError("block does not contain the requested number of frames")

    out = np.empty(frames, dtype=np.float64)
    for index in range(frames):
        start = index * frame
        stop = start + frame
        view = arr[:, :, start:stop]
        if sample_encoding == CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4:
            out[index] = _native_offset_binary_power(view)
        elif sample_encoding == PACKED_TWOS_COMPLEMENT_COMPLEX_INT4:
            out[index] = _twos_complement_power(view)
        else:
            out[index] = float(np.mean(np.abs(view) ** 2, dtype=np.float64))
    return out


def _pack_native_quantized_block(
    block: np.ndarray,
    *,
    frame_size_samples: int,
    detector_window_samples: int,
    spectral_sense: str,
    frames_in_chunk: int,
    sample_encoding: str,
) -> np.ndarray:
    streams = np.asarray(block)[:, 0, :]
    packed_frames: list[np.ndarray] = []
    for frame_index in range(int(frames_in_chunk)):
        start = frame_index * int(frame_size_samples)
        stop = start + int(frame_size_samples)
        matrix = stream_time_block_to_detector_matrix(
            streams[:, start:stop],
            detector_window_samples=int(detector_window_samples),
        )
        matrix = apply_spectral_sense_to_detector_matrix(
            matrix,
            spectral_sense=spectral_sense,
        )
        if sample_encoding == CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4:
            packed_frames.append(repack_chime_offset_binary_i4_to_twos_complement(matrix))
        elif sample_encoding == PACKED_TWOS_COMPLEMENT_COMPLEX_INT4:
            packed_frames.append(np.ascontiguousarray(np.asarray(matrix).view(np.int8)))
        else:
            raise ValueError(f"unsupported native packed encoding: {sample_encoding!r}")
    return np.ascontiguousarray(np.stack(packed_frames, axis=0))


def pack_chime_block_for_detector(
    block: np.ndarray,
    *,
    frame_size_samples: int,
    detector_window_samples: int,
    spectral_sense: str,
    frames_in_chunk: int,
    sample_encoding: str,
    selected_coarse_channel: int,
    physical_channel: int,
    bits_per_component: int = DEFAULT_BITS_PER_COMPONENT,
    quantization_scale_mode: str = QUANTIZATION_SCALE_MODE_GLOBAL,
    scale: float | None = None,
    scale_by_stream: Sequence[float] | np.ndarray | None = None,
    clip_sigma: float = 3.0,
) -> PackedChimeBlock:
    """Pack one normalized CHIME data block into detector frame rows."""
    arr = np.asarray(block)
    if arr.ndim != 3 or arr.shape[1] != 1:
        raise ValueError(
            "CHIME block must have shape (num_input_streams, 1, time); "
            f"got {arr.shape}"
        )

    baseband_power = baseband_power_by_frame(
        arr,
        frame_size_samples=int(frame_size_samples),
        frames_in_chunk=int(frames_in_chunk),
        sample_encoding=sample_encoding,
    )
    layout = DetectorFrameLayout(
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        num_input_streams=int(arr.shape[0]),
        num_selected_channels=1,
        combine_mode=COMBINE_MODE_COMBINED_STREAMS,
    )

    if sample_encoding in {
        CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    }:
        packed = _pack_native_quantized_block(
            arr,
            frame_size_samples=int(frame_size_samples),
            detector_window_samples=int(detector_window_samples),
            spectral_sense=spectral_sense,
            frames_in_chunk=int(frames_in_chunk),
            sample_encoding=sample_encoding,
        )
        quantization = {
            "source": "native_chime",
            "mode": "passthrough_or_repack",
            "native_encoding": sample_encoding,
            "output_encoding": PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
            "bits_per_component": int(bits_per_component),
            "clip_fraction": 0.0,
            "scale": None,
        }
        return PackedChimeBlock(
            packed=packed,
            input_layout=layout.to_dict(),
            quantization=quantization,
            baseband_power_linear=baseband_power,
        )

    if sample_encoding not in {COMPLEX_FLOAT, STRUCTURED_COMPLEX, REAL_IMAG_LAST_AXIS}:
        raise ValueError(f"unsupported CHIME sample encoding: {sample_encoding!r}")

    mode = (
        QUANTIZATION_SCALE_MODE_PROVIDED
        if scale is not None or scale_by_stream is not None
        else quantization_scale_mode
    )
    packed = pack_channelized_streams_for_detector(
        feed_channel_streams=arr,
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        spectral_sense=spectral_sense,
        quantization_scale_mode=mode,
        clip_sigma=float(clip_sigma),
        combine_mode=COMBINE_MODE_COMBINED_STREAMS,
        bits_per_component=int(bits_per_component),
        num_blocks=int(frames_in_chunk),
        scale=scale,
        scale_by_stream=scale_by_stream,
        selected_channel_indices=[int(selected_coarse_channel)],
        physical_channel=int(physical_channel),
    )
    return PackedChimeBlock(
        packed=packed.packed,
        input_layout=packed.input_layout,
        quantization=packed.quantization,
        baseband_power_linear=baseband_power,
    )


__all__ = [
    "PackedChimeBlock",
    "baseband_power_by_frame",
    "pack_chime_block_for_detector",
    "repack_chime_offset_binary_i4_to_twos_complement",
    "unpack_chime_offset_binary_i4_to_complex",
    "unpack_twos_complement_i4_to_complex",
]
