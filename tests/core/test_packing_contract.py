"""Observable packing contracts: exact order, scales, and invalid inputs."""

import numpy as np
import pytest

from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.integration.packing import pack_channelized_streams_for_detector
from pilot_proxy.integration.schemas import COMBINE_MODE_PER_STREAM_DIAGNOSTIC


@pytest.mark.parametrize("sense", ["normal", "inverted"])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_overlapping_blocks_keep_feed_channel_window_order(sense, diagnostic):
    rng = np.random.default_rng(291)
    data = (
        rng.integers(-4, 5, (2, 3, 20)) + 1j * rng.integers(-4, 5, (2, 3, 20))
    ).astype(np.complex64)
    scales = [1, 2, 3, 1, 2, 3]
    options = {"combine_mode": COMBINE_MODE_PER_STREAM_DIAGNOSTIC} if diagnostic else {}
    result = pack_channelized_streams_for_detector(
        data,
        frame_size_samples=8,
        detector_window_samples=4,
        num_blocks=3,
        block_step_samples=6,
        spectral_sense=sense,
        quantization_scale_mode="provided",
        scale_by_stream=scales,
        selected_channel_indices=[900, 12, 77],
        **options,
    )
    expected = []
    for start in (0, 6, 12):
        block = []
        for feed in range(2):
            for channel in range(3):
                stream = feed * 3 + channel
                rows = []
                for offset in (0, 4):
                    values = data[feed, channel, start + offset : start + offset + 4]
                    if sense == "inverted":
                        values = values[::-1]
                    # Python's round uses the specified ties-to-even rule.
                    re = np.array(
                        [
                            min(7, max(-7, round(float(x.real) * scales[stream])))
                            for x in values
                        ]
                    )
                    im = np.array(
                        [
                            min(7, max(-7, round(float(x.imag) * scales[stream])))
                            for x in values
                        ]
                    )
                    rows.append(
                        ((re & 15) * 16 + (im & 15)).astype(np.uint8).view(np.int8)
                    )
                if diagnostic:
                    expected.append(rows)
                else:
                    block.extend(rows)
        if not diagnostic:
            expected.append(block)
    np.testing.assert_array_equal(result.packed, expected)
    assert result.packed.flags.c_contiguous
    assert [row["selected_channel_index"] for row in result.stream_map] == [
        900,
        12,
        77,
    ] * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_size_samples", 8.5),
        ("detector_window_samples", 4.5),
        ("num_blocks", 1.5),
        ("block_step_samples", 8.5),
        ("num_blocks", True),
        ("detector_window_samples", True),
    ],
)
def test_packing_refuses_fractional_or_boolean_geometry(field, value):
    options = dict(
        frame_size_samples=8, detector_window_samples=4, num_blocks=1, scale=1.0
    )
    options[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        pack_channelized_streams_for_detector(
            np.ones((1, 1, 16), dtype=np.complex64), **options
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_quantization_refuses_nonfinite_samples(invalid):
    data = np.ones((2, 4), dtype=np.complex64)
    data[1, 2] = invalid
    with pytest.raises(ValueError, match="finite"):
        quantize_complex_numpy(data, 4, 1.0)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_quantization_refuses_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        quantize_complex_numpy(np.ones((2, 4), dtype=np.complex64), 4, scale)
