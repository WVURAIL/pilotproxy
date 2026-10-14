# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_reference import coarse_power_ratio_cpu_reference_packed
from pilot_proxy.detector_geometry import (
    DetectorFrameLayout,
    STREAM_LAYOUT_SCHEMA_TOKEN,
    apply_spectral_sense_to_detector_matrix,
    block_time_stream_to_detector_matrix,
    build_stream_map,
    flatten_feed_channel_streams,
    stack_stream_time_blocks,
    stream_time_block_to_detector_matrix,
)

REFERENCE_FRAME_SIZE_SAMPLES = 16_384
REFERENCE_NUM_STREAMS = 2_048
REFERENCE_DETECTOR_WINDOW_SAMPLES = 128
REFERENCE_WINDOWS_PER_STREAM = 128
REFERENCE_DETECTOR_ROWS_PER_FRAME = 262_144
REFERENCE_SAMPLES_PER_RESULT = 33_554_432

SMALL_BLOCK_TIME_SAMPLES = 8
SMALL_BLOCK_STREAMS = 3
SMALL_DETECTOR_WINDOW_SAMPLES = 4
STACK_STREAMS = 2
STACK_TIME_SAMPLES = 12
STACK_DETECTOR_WINDOW_SAMPLES = 2
STACK_FRAME_SIZE_SAMPLES = 4
STACK_NUM_BLOCKS = 3
MULTIFEED_FEEDS = 2
MULTIFEED_SELECTED_CHANNELS = 2
MULTIFEED_TIME_SAMPLES = 8
PHYSICAL_CHANNEL = 14
SELECTED_CHANNEL_INDICES = [179, 180]
REFERENCE_WEIGHT_TERMS = 3
COMBINED_TEST_WINDOW = 4
COMBINED_TEST_FEEDS = 2
COMBINED_TEST_WINDOWS_PER_FEED = 3
COARSE_POWER_RATIO_SCALE = 2.0
INT4_COMPONENT_BITS = 4
COMBINED_TEST_RNG_SEED = 12345
COMBINED_PACKED_LOW = -64
COMBINED_PACKED_HIGH = 64


def test_detector_frame_geometry_derived_values() -> None:
    geometry = DetectorFrameLayout(
        frame_size_samples=REFERENCE_FRAME_SIZE_SAMPLES,
        detector_window_samples=REFERENCE_DETECTOR_WINDOW_SAMPLES,
        num_input_streams=REFERENCE_NUM_STREAMS,
    )
    assert geometry.windows_per_stream == REFERENCE_WINDOWS_PER_STREAM
    assert geometry.detector_rows_per_frame == REFERENCE_DETECTOR_ROWS_PER_FRAME
    assert geometry.samples_per_result == REFERENCE_SAMPLES_PER_RESULT
    assert geometry.to_dict() == {
        "schema_version": STREAM_LAYOUT_SCHEMA_TOKEN,
        "frame_size_samples": REFERENCE_FRAME_SIZE_SAMPLES,
        "detector_window_samples": REFERENCE_DETECTOR_WINDOW_SAMPLES,
        "windows_per_stream": REFERENCE_WINDOWS_PER_STREAM,
        "num_input_streams": REFERENCE_NUM_STREAMS,
        "num_selected_channels": 1,
        "num_streams": REFERENCE_NUM_STREAMS,
        "detector_rows_per_frame": REFERENCE_DETECTOR_ROWS_PER_FRAME,
        "samples_per_result": REFERENCE_SAMPLES_PER_RESULT,
        "combine_mode": "incoherent_power_sum_over_streams",
    }


def test_block_time_stream_to_detector_matrix_order() -> None:
    # block[time, stream]
    block = np.arange(
        SMALL_BLOCK_TIME_SAMPLES * SMALL_BLOCK_STREAMS,
        dtype=np.int8,
    ).reshape(SMALL_BLOCK_TIME_SAMPLES, SMALL_BLOCK_STREAMS)
    out = block_time_stream_to_detector_matrix(
        block,
        detector_window_samples=SMALL_DETECTOR_WINDOW_SAMPLES,
    )
    assert out.shape == (
        SMALL_BLOCK_STREAMS
        * (SMALL_BLOCK_TIME_SAMPLES // SMALL_DETECTOR_WINDOW_SAMPLES),
        SMALL_DETECTOR_WINDOW_SAMPLES,
    )
    expected = np.asarray(
        [
            block[0:4, 0],
            block[4:8, 0],
            block[0:4, 1],
            block[4:8, 1],
            block[0:4, 2],
            block[4:8, 2],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(out, expected)
    assert out.flags.c_contiguous


def test_stream_time_block_to_detector_matrix_order() -> None:
    streams = np.arange(
        SMALL_BLOCK_STREAMS * SMALL_BLOCK_TIME_SAMPLES,
        dtype=np.int8,
    ).reshape(SMALL_BLOCK_STREAMS, SMALL_BLOCK_TIME_SAMPLES)
    out = stream_time_block_to_detector_matrix(
        streams,
        detector_window_samples=SMALL_DETECTOR_WINDOW_SAMPLES,
    )
    expected = np.asarray(
        [
            streams[0, 0:4],
            streams[0, 4:8],
            streams[1, 0:4],
            streams[1, 4:8],
            streams[2, 0:4],
            streams[2, 4:8],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(out, expected)


def test_stack_stream_time_blocks() -> None:
    streams = np.arange(
        STACK_STREAMS * STACK_TIME_SAMPLES,
        dtype=np.int8,
    ).reshape(STACK_STREAMS, STACK_TIME_SAMPLES)
    stacked = stack_stream_time_blocks(
        streams,
        detector_window_samples=STACK_DETECTOR_WINDOW_SAMPLES,
        frame_size_samples=STACK_FRAME_SIZE_SAMPLES,
        block_step_samples=STACK_FRAME_SIZE_SAMPLES,
        num_blocks=STACK_NUM_BLOCKS,
    )
    assert stacked.shape == (
        STACK_NUM_BLOCKS,
        STACK_STREAMS
        * (STACK_FRAME_SIZE_SAMPLES // STACK_DETECTOR_WINDOW_SAMPLES),
        STACK_DETECTOR_WINDOW_SAMPLES,
    )
    assert np.array_equal(
        stacked[1],
        stream_time_block_to_detector_matrix(
            streams[:, 4:8],
            detector_window_samples=STACK_DETECTOR_WINDOW_SAMPLES,
        ),
    )


def test_feed_channel_streams_flatten_feed_major() -> None:
    feed_channel = np.arange(
        MULTIFEED_FEEDS * MULTIFEED_SELECTED_CHANNELS * MULTIFEED_TIME_SAMPLES,
        dtype=np.int16,
    ).reshape(MULTIFEED_FEEDS, MULTIFEED_SELECTED_CHANNELS, MULTIFEED_TIME_SAMPLES)

    flattened = flatten_feed_channel_streams(feed_channel)

    assert flattened.shape == (
        MULTIFEED_FEEDS * MULTIFEED_SELECTED_CHANNELS,
        MULTIFEED_TIME_SAMPLES,
    )
    np.testing.assert_array_equal(flattened[0], feed_channel[0, 0])
    np.testing.assert_array_equal(flattened[1], feed_channel[0, 1])
    np.testing.assert_array_equal(flattened[2], feed_channel[1, 0])
    np.testing.assert_array_equal(flattened[3], feed_channel[1, 1])


def test_multi_feed_detector_frame_layout_and_stream_map() -> None:
    layout = DetectorFrameLayout(
        frame_size_samples=REFERENCE_FRAME_SIZE_SAMPLES,
        detector_window_samples=REFERENCE_DETECTOR_WINDOW_SAMPLES,
        num_input_streams=MULTIFEED_FEEDS,
        num_selected_channels=1,
    ).to_dict()
    assert layout == {
        "schema_version": STREAM_LAYOUT_SCHEMA_TOKEN,
        "frame_size_samples": REFERENCE_FRAME_SIZE_SAMPLES,
        "detector_window_samples": REFERENCE_DETECTOR_WINDOW_SAMPLES,
        "windows_per_stream": REFERENCE_WINDOWS_PER_STREAM,
        "num_input_streams": MULTIFEED_FEEDS,
        "num_selected_channels": 1,
        "num_streams": MULTIFEED_FEEDS,
        "detector_rows_per_frame": (
            MULTIFEED_FEEDS * REFERENCE_WINDOWS_PER_STREAM
        ),
        "samples_per_result": MULTIFEED_FEEDS * REFERENCE_FRAME_SIZE_SAMPLES,
        "combine_mode": "incoherent_power_sum_over_streams",
    }

    stream_map = build_stream_map(
        num_feeds=MULTIFEED_FEEDS,
        selected_channel_indices=SELECTED_CHANNEL_INDICES,
        physical_channel=PHYSICAL_CHANNEL,
    )
    assert stream_map == [
        {
            "stream_index": 0,
            "feed_index": 0,
            "selected_channel_index": SELECTED_CHANNEL_INDICES[0],
            "physical_channel": PHYSICAL_CHANNEL,
        },
        {
            "stream_index": 1,
            "feed_index": 0,
            "selected_channel_index": SELECTED_CHANNEL_INDICES[1],
            "physical_channel": PHYSICAL_CHANNEL,
        },
        {
            "stream_index": 2,
            "feed_index": 1,
            "selected_channel_index": SELECTED_CHANNEL_INDICES[0],
            "physical_channel": PHYSICAL_CHANNEL,
        },
        {
            "stream_index": 3,
            "feed_index": 1,
            "selected_channel_index": SELECTED_CHANNEL_INDICES[1],
            "physical_channel": PHYSICAL_CHANNEL,
        },
    ]


def test_combined_feed_cpu_reference_equals_manual_power_sum() -> None:
    rng = np.random.default_rng(COMBINED_TEST_RNG_SEED)
    packed_by_feed = rng.integers(
        COMBINED_PACKED_LOW,
        COMBINED_PACKED_HIGH,
        size=(COMBINED_TEST_FEEDS, COMBINED_TEST_WINDOWS_PER_FEED, COMBINED_TEST_WINDOW),
        dtype=np.int8,
    )
    weights = rng.integers(
        COMBINED_PACKED_LOW,
        COMBINED_PACKED_HIGH,
        size=(REFERENCE_WEIGHT_TERMS, COMBINED_TEST_WINDOW),
        dtype=np.int8,
    )
    combined = packed_by_feed.reshape(
        COMBINED_TEST_FEEDS * COMBINED_TEST_WINDOWS_PER_FEED,
        COMBINED_TEST_WINDOW,
    )

    combined_fstat, combined_powers = coarse_power_ratio_cpu_reference_packed(
        combined,
        weights,
        bits=INT4_COMPONENT_BITS,
    )
    manual_powers = np.zeros(REFERENCE_WEIGHT_TERMS, dtype=np.float64)
    for feed_index in range(COMBINED_TEST_FEEDS):
        _, feed_powers = coarse_power_ratio_cpu_reference_packed(
            packed_by_feed[feed_index],
            weights,
            bits=INT4_COMPONENT_BITS,
        )
        manual_powers += feed_powers
    manual_fstat = (
        COARSE_POWER_RATIO_SCALE
        * manual_powers[0]
        / (manual_powers[1] + manual_powers[2])
    )

    np.testing.assert_allclose(combined_powers, manual_powers)
    assert combined_fstat == pytest.approx(float(manual_fstat))


def test_apply_spectral_sense_reverses_detector_window_axis() -> None:
    matrix = np.arange(
        STACK_STREAMS * STACK_FRAME_SIZE_SAMPLES,
        dtype=np.int8,
    ).reshape(STACK_STREAMS, STACK_FRAME_SIZE_SAMPLES)

    normal = apply_spectral_sense_to_detector_matrix(
        matrix,
        spectral_sense="normal",
    )
    inverted = apply_spectral_sense_to_detector_matrix(
        matrix,
        spectral_sense="inverted",
    )

    assert np.array_equal(normal, matrix)
    assert normal.flags.c_contiguous
    assert np.array_equal(inverted, matrix[:, ::-1])
    assert inverted.flags.c_contiguous


def test_detector_frame_geometry_rejects_non_divisible_frame() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        DetectorFrameLayout(
            frame_size_samples=10,
            detector_window_samples=STACK_FRAME_SIZE_SAMPLES,
            num_input_streams=STACK_STREAMS,
        )
