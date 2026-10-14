# coding=utf-8
"""Parity of the GPU synthesis twins against the CPU reference chain."""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from pilot_proxy.reference_channelizer import (  # noqa: E402
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    channelize_real_blocks_to_reference_channels,
    channelize_real_blocks_to_reference_channels_gpu,
    complex_envelope_to_real_adc_blocks,
    complex_envelope_to_real_adc_blocks_gpu,
)

IQ_RATE_HZ = 10_762_237.762237763
ADC_RATE_HZ = 800.0e6
BAND_LOWER_HZ = 400.0e6
RF_CENTER_HZ = 473.0e6
N_BLOCKS = 67


def _random_envelope(rng: np.random.Generator, n: int) -> np.ndarray:
    return (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)


def test_adc_block_synthesis_matches_cpu_reference() -> None:
    rng = np.random.default_rng(20260824)
    iq = _random_envelope(rng, 4096)
    kwargs = dict(
        iq_sample_rate_hz=IQ_RATE_HZ,
        rf_center_hz=RF_CENTER_HZ,
        adc_sample_rate_hz=ADC_RATE_HZ,
        band_lower_hz=BAND_LOWER_HZ,
        n_blocks=N_BLOCKS,
        block_size=REFERENCE_PFB_FFT_SIZE,
    )
    cpu = complex_envelope_to_real_adc_blocks(iq, **kwargs)
    gpu = complex_envelope_to_real_adc_blocks_gpu(iq, **kwargs)
    assert gpu.shape == cpu.shape and gpu.dtype == cpu.dtype
    scale = float(np.max(np.abs(cpu)))
    np.testing.assert_allclose(gpu, cpu, rtol=0.0, atol=2e-6 * scale)


def test_channelization_matches_cpu_reference() -> None:
    rng = np.random.default_rng(20260824)
    blocks = rng.standard_normal(
        (N_BLOCKS, REFERENCE_PFB_FFT_SIZE)
    ).astype(np.float32)
    spec = ReferenceChannelizerSpec(
        adc_sample_rate_hz=ADC_RATE_HZ,
        band_lower_hz=BAND_LOWER_HZ,
    )
    channels = [0, 179, 1023]
    cpu = channelize_real_blocks_to_reference_channels(
        blocks, channel_indices=channels, spec=spec
    )
    gpu = channelize_real_blocks_to_reference_channels_gpu(
        blocks, channel_indices=channels, spec=spec
    )
    assert gpu.shape == cpu.shape and gpu.dtype == cpu.dtype
    assert cpu.shape == (3, N_BLOCKS + 1 - REFERENCE_PFB_TAPS)
    scale = float(np.max(np.abs(cpu)))
    np.testing.assert_allclose(gpu, cpu, rtol=0.0, atol=2e-5 * scale)
