# coding=utf-8
from __future__ import annotations

import numpy as np

from pilot_proxy.reference_channelizer import (
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    ReferenceChannelizerSpec,
    apply_reference_archive_phase_convention,
    channelize_real_blocks_to_reference_channels,
    nearest_reference_channel_index,
    reference_channel_frequencies_hz,
    sinc_hamming_pfb_response,
)
from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.detector_geometry import stream_time_block_to_detector_matrix

PFB_TEST_TAPS = 4
PFB_TEST_SAMPLES = 8
PFB_TEST_CENTER = 0.5

FIRST_REFERENCE_CHANNEL_HZ = 400_390_625.0
REFERENCE_BAND_UPPER_EDGE_HZ = 800_000_000.0
CHANNEL_14_PILOT_HZ = 470_309_441.0
CHANNEL_14_REFERENCE_INDEX = 179

SMALL_PFB_TAPS = 2
SMALL_PFB_SAMPLES = 8
SMALL_ADC_SAMPLE_RATE_HZ = 8.0
SMALL_BAND_LOWER_HZ = 0.0
SMALL_NUM_CHANNELS = 4
SMALL_CHANNELIZER_BLOCKS = 4
SMALL_OUTPUT_CHUNK_SAMPLES = 2
INT4_COMPONENT_BITS = 4
UNITY_QUANTIZATION_SCALE = 1.0

ARCHIVE_CHANNEL_WIDTH_HZ = 390_625.0
ARCHIVE_LOCAL_HZ = -3_059.0
ARCHIVE_DETECTOR_WINDOW_SAMPLES = 128
MATCHED_DOT_THRESHOLD = 100.0
WRONG_DOT_THRESHOLD = 1.0


def test_sinc_hamming_pfb_response_matches_formula() -> None:
    n_tap = PFB_TEST_TAPS
    n_sample = PFB_TEST_SAMPLES
    response = sinc_hamming_pfb_response(n_tap, n_sample, dtype=np.float64)

    n = n_tap * n_sample
    x = n_tap * np.linspace(-PFB_TEST_CENTER, PFB_TEST_CENTER, n, endpoint=False)
    expected = (np.sinc(x) * np.hamming(n)).reshape(n_tap, n_sample)

    assert response.shape == (n_tap, n_sample)
    np.testing.assert_allclose(response, expected)


def test_reference_channel_axis_matches_weight_rom_convention() -> None:
    spec = ReferenceChannelizerSpec(
        adc_sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        band_lower_hz=REFERENCE_BAND_LOWER_HZ,
    )
    freqs = reference_channel_frequencies_hz(spec)

    assert freqs[0] == FIRST_REFERENCE_CHANNEL_HZ
    assert freqs[-1] == REFERENCE_BAND_UPPER_EDGE_HZ
    assert nearest_reference_channel_index(CHANNEL_14_PILOT_HZ, spec) == (
        CHANNEL_14_REFERENCE_INDEX
    )


def test_channelize_real_blocks_selects_reference_rfft_bins() -> None:
    spec = ReferenceChannelizerSpec(
        n_tap=SMALL_PFB_TAPS,
        n_sample=SMALL_PFB_SAMPLES,
        adc_sample_rate_hz=SMALL_ADC_SAMPLE_RATE_HZ,
        band_lower_hz=SMALL_BAND_LOWER_HZ,
        num_channels=SMALL_NUM_CHANNELS,
    )
    response = np.ones((SMALL_PFB_TAPS, SMALL_PFB_SAMPLES), dtype=np.float32)
    blocks = np.arange(
        SMALL_CHANNELIZER_BLOCKS * SMALL_PFB_SAMPLES,
        dtype=np.float32,
    ).reshape(SMALL_CHANNELIZER_BLOCKS, SMALL_PFB_SAMPLES)

    got = channelize_real_blocks_to_reference_channels(
        blocks,
        channel_indices=[0, 2],
        response=response,
        spec=spec,
        output_chunk_samples=SMALL_OUTPUT_CHUNK_SAMPLES,
    )

    ppf = np.stack(
        [
            blocks[0] + blocks[1],
            blocks[1] + blocks[2],
            blocks[2] + blocks[3],
        ]
    )
    spectra = np.fft.rfft(ppf, axis=1)

    assert got.shape == (2, 3)
    np.testing.assert_allclose(got[0], spectra[:, 1])
    np.testing.assert_allclose(got[1], spectra[:, 3])


def test_channelized_streams_pack_to_kernel_detector_order() -> None:
    streams = np.asarray(
        [
            [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j],
            [0 + 1j, 0 + 2j, 0 + 3j, 0 + 4j],
        ],
        dtype=np.complex64,
    )

    detector_matrix = stream_time_block_to_detector_matrix(
        streams,
        detector_window_samples=2,
    )
    packed = quantize_complex_numpy(
        detector_matrix,
        bits=INT4_COMPONENT_BITS,
        scale=UNITY_QUANTIZATION_SCALE,
    )

    expected_matrix = np.asarray(
        [
            [1 + 0j, 2 + 0j],
            [3 + 0j, 4 + 0j],
            [0 + 1j, 0 + 2j],
            [0 + 3j, 0 + 4j],
        ],
        dtype=np.complex64,
    )
    expected = quantize_complex_numpy(
        expected_matrix,
        bits=INT4_COMPONENT_BITS,
        scale=UNITY_QUANTIZATION_SCALE,
    )

    np.testing.assert_array_equal(packed, expected)


def test_archive_phase_convention_maps_centered_stream_to_lower_edge_norm() -> None:
    fs = ARCHIVE_CHANNEL_WIDTH_HZ
    local_hz = ARCHIVE_LOCAL_HZ
    n = np.arange(ARCHIVE_DETECTOR_WINDOW_SAMPLES, dtype=np.float64)
    centered = np.exp(2j * np.pi * local_hz * n / fs)[None, :].astype(np.complex64)

    converted = apply_reference_archive_phase_convention(centered)[0]

    lower_edge_norm = PFB_TEST_CENTER + local_hz / fs
    matched = np.exp(-2j * np.pi * lower_edge_norm * n)
    wrong = np.exp(2j * np.pi * local_hz * n / fs)

    assert abs(np.vdot(matched, converted)) > MATCHED_DOT_THRESHOLD
    assert abs(np.vdot(wrong, converted)) < WRONG_DOT_THRESHOLD
