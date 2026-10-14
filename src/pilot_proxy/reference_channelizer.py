# coding=utf-8
"""Reference ADC/channelizer model used by the standalone ATSC testbench."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# Included reference front end: 800 MS/s real ADC, 400-800 MHz analysis band,
# 4-tap/2048-point PFB, retaining the positive-frequency 1024 channels.
REFERENCE_ADC_SAMPLE_RATE_HZ = 800.0e6
REFERENCE_BAND_LOWER_HZ = 400.0e6
REFERENCE_BANDWIDTH_HZ = 400.0e6
REFERENCE_PFB_TAPS = 4
REFERENCE_PFB_FFT_SIZE = 2048
REFERENCE_NUM_CHANNELS = 1024
DEFAULT_OUTPUT_CHUNK_SAMPLES = 1024
DEFAULT_ADC_CHUNK_SAMPLES = 1_048_576
HALF_OPEN_INTERVAL_CENTER = 0.5
FULL_CYCLE_RADIANS = 2.0 * np.pi


@dataclass(frozen=True)
class ReferenceChannelizerSpec:
    """Parameters for the reference analysis channelizer."""

    n_tap: int = REFERENCE_PFB_TAPS
    n_sample: int = REFERENCE_PFB_FFT_SIZE
    adc_sample_rate_hz: float = REFERENCE_ADC_SAMPLE_RATE_HZ
    band_lower_hz: float = REFERENCE_BAND_LOWER_HZ
    num_channels: int = REFERENCE_NUM_CHANNELS

    @property
    def channel_width_hz(self) -> float:
        return float(self.adc_sample_rate_hz) / float(self.n_sample)

    @property
    def output_sample_rate_hz(self) -> float:
        return self.channel_width_hz


def sinc_hamming_pfb_response(
    n_tap: int = REFERENCE_PFB_TAPS,
    n_sample: int = REFERENCE_PFB_FFT_SIZE,
    *,
    sinc_scale: float = 1.0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return the sinc-Hamming prototype response for the reference PFB."""
    n_tap = int(n_tap)
    n_sample = int(n_sample)
    if n_tap <= 0 or n_sample <= 0:
        raise ValueError("n_tap and n_sample must be positive.")
    n = n_tap * n_sample
    x = n_tap * float(sinc_scale) * np.linspace(
        -HALF_OPEN_INTERVAL_CENTER,
        HALF_OPEN_INTERVAL_CENTER,
        n,
        endpoint=False,
    )
    response = np.sinc(x) * np.hamming(n)
    return np.asarray(response.reshape(n_tap, n_sample), dtype=dtype)


def reference_channel_frequencies_hz(
    spec: ReferenceChannelizerSpec | None = None,
) -> np.ndarray:
    """Return the reference coarse-channel frequency axis."""
    spec = spec or ReferenceChannelizerSpec()
    return (
        float(spec.band_lower_hz)
        + spec.channel_width_hz * (np.arange(int(spec.num_channels)) + 1)
    )


def nearest_reference_channel_index(
    frequency_hz: float,
    spec: ReferenceChannelizerSpec | None = None,
) -> int:
    """Return the nearest reference coarse-channel index."""
    freqs = reference_channel_frequencies_hz(spec)
    return int(np.argmin(np.abs(freqs - float(frequency_hz))))


def normalize_reference_channel_indices(
    channel_indices: Iterable[int],
    spec: ReferenceChannelizerSpec | None = None,
) -> np.ndarray:
    """Validate reference coarse-channel indices and return them as int64."""
    spec = spec or ReferenceChannelizerSpec()
    arr = np.asarray(list(channel_indices), dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("At least one channel index is required.")
    if np.any(arr < 0) or np.any(arr >= int(spec.num_channels)):
        raise ValueError(
            "Channel indices must be in [0, "
            f"{int(spec.num_channels) - 1}], got {arr.tolist()}."
        )
    return arr


def complex_envelope_to_real_adc_blocks(
    iq: np.ndarray,
    *,
    iq_sample_rate_hz: float,
    rf_center_hz: float,
    adc_sample_rate_hz: float,
    band_lower_hz: float,
    n_blocks: int,
    block_size: int,
    start_adc_sample: int = 0,
    chunk_samples: int = DEFAULT_ADC_CHUNK_SAMPLES,
) -> np.ndarray:
    """Interpolate a complex envelope onto the real ADC sample grid."""
    iq = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if iq.size < 2:
        raise ValueError("iq must contain at least two complex samples.")
    iq_sample_rate_hz = float(iq_sample_rate_hz)
    adc_sample_rate_hz = float(adc_sample_rate_hz)
    if iq_sample_rate_hz <= 0.0 or adc_sample_rate_hz <= 0.0:
        raise ValueError("sample rates must be positive.")
    n_blocks = int(n_blocks)
    block_size = int(block_size)
    if n_blocks <= 0 or block_size <= 0:
        raise ValueError("n_blocks and block_size must be positive.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")

    total_samples = n_blocks * block_size
    last_source_position = (
        (int(start_adc_sample) + total_samples - 1)
        * iq_sample_rate_hz
        / adc_sample_rate_hz
    )
    if last_source_position > iq.size - 1:
        required = int(np.ceil(last_source_position)) + 1
        raise ValueError(
            "input IQ file is too short for requested PFB output: "
            f"need at least {required} samples, got {iq.size}."
        )

    out = np.empty(total_samples, dtype=np.float32)
    source_axis = np.arange(iq.size, dtype=np.float64)
    source_real = np.asarray(iq.real, dtype=np.float32)
    source_imag = np.asarray(iq.imag, dtype=np.float32)
    adc_to_iq = iq_sample_rate_hz / adc_sample_rate_hz
    phase_inc = FULL_CYCLE_RADIANS * (float(rf_center_hz) - float(band_lower_hz))
    phase_inc /= adc_sample_rate_hz

    for start in range(0, total_samples, int(chunk_samples)):
        stop = min(start + int(chunk_samples), total_samples)
        adc_indices = int(start_adc_sample) + np.arange(start, stop, dtype=np.float64)
        source_pos = adc_indices * adc_to_iq
        real = np.interp(source_pos, source_axis, source_real)
        imag = np.interp(source_pos, source_axis, source_imag)
        phase = phase_inc * adc_indices
        out[start:stop] = np.asarray(
            real * np.cos(phase) - imag * np.sin(phase),
            dtype=np.float32,
        )

    return np.ascontiguousarray(out.reshape(n_blocks, block_size))


DEFAULT_GPU_ADC_CHUNK_SAMPLES = 8_388_608
DEFAULT_GPU_OUTPUT_CHUNK_SAMPLES = 4096


def complex_envelope_to_real_adc_blocks_gpu(
    iq: np.ndarray,
    *,
    iq_sample_rate_hz: float,
    rf_center_hz: float,
    adc_sample_rate_hz: float,
    band_lower_hz: float,
    n_blocks: int,
    block_size: int,
    start_adc_sample: int = 0,
    chunk_samples: int = DEFAULT_GPU_ADC_CHUNK_SAMPLES,
) -> np.ndarray:
    """GPU twin of :func:`complex_envelope_to_real_adc_blocks`.

    Same interpolation and mixing arithmetic evaluated with cupy; the result
    returns to host memory, so downstream code is unchanged. Agreement with
    the CPU path is limited only by libm-vs-CUDA trig rounding.
    """
    import cupy as cp

    iq = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if iq.size < 2:
        raise ValueError("iq must contain at least two complex samples.")
    iq_sample_rate_hz = float(iq_sample_rate_hz)
    adc_sample_rate_hz = float(adc_sample_rate_hz)
    if iq_sample_rate_hz <= 0.0 or adc_sample_rate_hz <= 0.0:
        raise ValueError("sample rates must be positive.")
    n_blocks = int(n_blocks)
    block_size = int(block_size)
    if n_blocks <= 0 or block_size <= 0:
        raise ValueError("n_blocks and block_size must be positive.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")

    total_samples = n_blocks * block_size
    last_source_position = (
        (int(start_adc_sample) + total_samples - 1)
        * iq_sample_rate_hz
        / adc_sample_rate_hz
    )
    if last_source_position > iq.size - 1:
        required = int(np.ceil(last_source_position)) + 1
        raise ValueError(
            "input IQ file is too short for requested PFB output: "
            f"need at least {required} samples, got {iq.size}."
        )

    out = cp.empty(total_samples, dtype=cp.float32)
    source_axis = cp.arange(iq.size, dtype=cp.float64)
    source_real = cp.asarray(iq.real, dtype=cp.float32)
    source_imag = cp.asarray(iq.imag, dtype=cp.float32)
    adc_to_iq = iq_sample_rate_hz / adc_sample_rate_hz
    phase_inc = FULL_CYCLE_RADIANS * (float(rf_center_hz) - float(band_lower_hz))
    phase_inc /= adc_sample_rate_hz

    for start in range(0, total_samples, int(chunk_samples)):
        stop = min(start + int(chunk_samples), total_samples)
        adc_indices = int(start_adc_sample) + cp.arange(start, stop, dtype=cp.float64)
        source_pos = adc_indices * adc_to_iq
        real = cp.interp(source_pos, source_axis, source_real)
        imag = cp.interp(source_pos, source_axis, source_imag)
        phase = phase_inc * adc_indices
        out[start:stop] = (real * cp.cos(phase) - imag * cp.sin(phase)).astype(
            cp.float32
        )

    return cp.asnumpy(out.reshape(n_blocks, block_size))


def channelize_real_blocks_to_reference_channels_gpu(
    blocks: np.ndarray,
    *,
    channel_indices: Iterable[int],
    response: np.ndarray | None = None,
    spec: ReferenceChannelizerSpec | None = None,
    output_chunk_samples: int = DEFAULT_GPU_OUTPUT_CHUNK_SAMPLES,
) -> np.ndarray:
    """GPU twin of :func:`channelize_real_blocks_to_reference_channels`."""
    import cupy as cp

    spec = spec or ReferenceChannelizerSpec()
    response = (
        sinc_hamming_pfb_response(spec.n_tap, spec.n_sample)
        if response is None
        else np.asarray(response, dtype=np.float32)
    )
    if response.shape != (int(spec.n_tap), int(spec.n_sample)):
        raise ValueError(
            "response shape must be "
            f"({int(spec.n_tap)}, {int(spec.n_sample)}), got {response.shape}."
        )

    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.ndim != 2 or blocks.shape[1] != int(spec.n_sample):
        raise ValueError(
            "blocks must have shape (n_blocks, "
            f"{int(spec.n_sample)}), got {blocks.shape}."
        )
    n_output = int(blocks.shape[0]) + 1 - int(spec.n_tap)
    if n_output <= 0:
        raise ValueError("blocks does not contain enough PFB blocks for one output.")
    if output_chunk_samples <= 0:
        raise ValueError("output_chunk_samples must be positive.")

    channel_indices_arr = normalize_reference_channel_indices(channel_indices, spec)
    rfft_bins = cp.asarray(channel_indices_arr + 1)
    blocks_gpu = cp.asarray(blocks)
    response_gpu = cp.asarray(response)
    selected = cp.empty((channel_indices_arr.size, n_output), dtype=cp.complex64)

    for start in range(0, n_output, int(output_chunk_samples)):
        stop = min(start + int(output_chunk_samples), n_output)
        ppf = cp.zeros((stop - start, int(spec.n_sample)), dtype=cp.float32)
        for tap in range(int(spec.n_tap)):
            ppf += blocks_gpu[start + tap : stop + tap] * response_gpu[tap][None, :]
        spectra = cp.fft.rfft(ppf, axis=1)
        selected[:, start:stop] = spectra[:, rfft_bins].T.astype(cp.complex64)

    return cp.asnumpy(selected)


def channelize_real_blocks_to_reference_channels(
    blocks: np.ndarray,
    *,
    channel_indices: Iterable[int],
    response: np.ndarray | None = None,
    spec: ReferenceChannelizerSpec | None = None,
    output_chunk_samples: int = DEFAULT_OUTPUT_CHUNK_SAMPLES,
) -> np.ndarray:
    """Run real ADC blocks through the reference PFB and select channels."""
    spec = spec or ReferenceChannelizerSpec()
    response = (
        sinc_hamming_pfb_response(spec.n_tap, spec.n_sample)
        if response is None
        else np.asarray(response, dtype=np.float32)
    )
    if response.shape != (int(spec.n_tap), int(spec.n_sample)):
        raise ValueError(
            "response shape must be "
            f"({int(spec.n_tap)}, {int(spec.n_sample)}), got {response.shape}."
        )

    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.ndim != 2 or blocks.shape[1] != int(spec.n_sample):
        raise ValueError(
            "blocks must have shape (n_blocks, "
            f"{int(spec.n_sample)}), got {blocks.shape}."
        )
    n_output = int(blocks.shape[0]) + 1 - int(spec.n_tap)
    if n_output <= 0:
        raise ValueError("blocks does not contain enough PFB blocks for one output.")
    if output_chunk_samples <= 0:
        raise ValueError("output_chunk_samples must be positive.")

    channel_indices_arr = normalize_reference_channel_indices(channel_indices, spec)
    rfft_bins = channel_indices_arr + 1
    selected = np.empty((channel_indices_arr.size, n_output), dtype=np.complex64)

    for start in range(0, n_output, int(output_chunk_samples)):
        stop = min(start + int(output_chunk_samples), n_output)
        ppf = np.zeros((stop - start, int(spec.n_sample)), dtype=np.float32)
        for tap in range(int(spec.n_tap)):
            ppf += blocks[start + tap : stop + tap] * response[tap][None, :]
        spectra = np.fft.rfft(ppf, axis=1)
        selected[:, start:stop] = np.asarray(
            spectra[:, rfft_bins].T,
            dtype=np.complex64,
        )

    return np.ascontiguousarray(selected)


def apply_reference_archive_phase_convention(
    streams: np.ndarray,
    *,
    start_sample_index: int = 0,
) -> np.ndarray:
    """Map centered rFFT-bin streams to lower-edge detector coordinates."""
    streams = np.asarray(streams, dtype=np.complex64)
    if streams.ndim != 2:
        raise ValueError(
            "streams must have shape (num_streams, time), "
            f"got {streams.shape}."
        )
    n = int(start_sample_index) + np.arange(streams.shape[1], dtype=np.int64)
    phase = np.asarray(1 - 2 * (n & 1), dtype=np.float32)
    return np.ascontiguousarray(np.conj(streams) * phase[None, :])


__all__ = [
    "REFERENCE_ADC_SAMPLE_RATE_HZ",
    "REFERENCE_BAND_LOWER_HZ",
    "REFERENCE_BANDWIDTH_HZ",
    "REFERENCE_NUM_CHANNELS",
    "REFERENCE_PFB_FFT_SIZE",
    "REFERENCE_PFB_TAPS",
    "ReferenceChannelizerSpec",
    "apply_reference_archive_phase_convention",
    "channelize_real_blocks_to_reference_channels",
    "complex_envelope_to_real_adc_blocks",
    "nearest_reference_channel_index",
    "normalize_reference_channel_indices",
    "reference_channel_frequencies_hz",
    "sinc_hamming_pfb_response",
]
