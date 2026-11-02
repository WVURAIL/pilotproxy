"""Tests of the packed sample order, the quantization scales and the refusal of invalid inputs."""

import numpy as np
import pytest

from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.integration.packing import (
    estimate_complex_scale,
    pack_channelized_streams_for_detector,
)
from pilot_proxy.integration.schemas import COMBINE_MODE_PER_STREAM_DIAGNOSTIC


@pytest.mark.parametrize("zone", [1, 2])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_overlapping_blocks_keep_feed_channel_window_order(zone, diagnostic):
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
        nyquist_zone=zone,
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
                    if zone == 2:
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
        ("frame_size_samples", 0),
        ("frame_size_samples", -8),
        ("detector_window_samples", 0),
        ("detector_window_samples", -4),
        ("num_blocks", 0),
        ("num_blocks", -1),
        ("block_step_samples", 0),
        ("block_step_samples", -8),
    ],
)
def test_packing_refuses_invalid_geometry(field, value):
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


@pytest.mark.parametrize("bits,limit", [(4, 7), (8, 127), (np.int64(4), 7)])
@pytest.mark.parametrize("values", [
    [2 - 10j, 6 + 10j],  # Real spread is 2, even though the imaginary spread is 10.
    [3 - 2j, 3 + 2j],  # Constant real samples require the imaginary spread.
])
def test_automatic_scale_uses_the_component_population_spread(bits, limit, values):
    values = np.asarray(values)
    scale = estimate_complex_scale(values, bits_per_component=bits, clip_sigma=2)
    assert scale == pytest.approx(limit / 4)
    amplified = estimate_complex_scale(
        7 * values, bits_per_component=bits, clip_sigma=2)
    assert amplified == pytest.approx(scale / 7)


def test_automatic_scale_uses_finite_magnitude_spread_after_component_overflow():
    values = np.asarray([1e154 + 1e154j, -1.1e154 - 1.1e154j])
    # For two values, population sigma is half their absolute difference.
    magnitude_difference = abs(abs(complex(values[1])) - abs(complex(values[0])))
    with np.errstate(over="ignore"):
        assert np.isinf(np.std(values.real))
        assert np.isinf(np.std(values.imag))
        scale = estimate_complex_scale(values, bits_per_component=4, clip_sigma=2)
    assert np.isfinite(scale) and scale > 0
    assert scale == pytest.approx(7 / magnitude_difference, rel=1e-14, abs=0)


@pytest.mark.parametrize("mode,scales,levels,clipped", [
    ("global", [7 / np.sqrt(5)] * 2, [3, 7], [0, 1]),
    ("per_stream", [7, 7 / 3], [7, 7], [0, 0]),
])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_automatic_packing_preserves_per_stream_scaling_and_clipping(
    mode, scales, levels, clipped, diagnostic,
):
    # Equal numbers of +/-1 and +/-3 have population variance (1 + 9) / 2.
    streams = np.asarray([[[1, -1] * 4], [[3, -3] * 4]], dtype=np.complex128)
    options = {"combine_mode": COMBINE_MODE_PER_STREAM_DIAGNOSTIC} if diagnostic else {}
    result = pack_channelized_streams_for_detector(
        streams, frame_size_samples=8, detector_window_samples=4,
        nyquist_zone=1, quantization_scale_mode=mode, clip_sigma=1, **options,
    )
    np.testing.assert_allclose(result.quantization["scale_by_stream"], scales)
    np.testing.assert_array_equal(result.quantization["clip_fraction_by_stream"], clipped)
    # Imaginary components are zero; the signed real nibble is the high nibble.
    rows = np.asarray([
        [16 * level, -16 * level] * 2
        for level in levels for _ in range(2)
    ], dtype=np.int8)
    expected = rows.reshape(2, 2, 4) if diagnostic else rows[np.newaxis]
    np.testing.assert_array_equal(result.packed, expected)
    assert result.packed.flags.c_contiguous


@pytest.mark.parametrize("bits", [True, np.bool_(True), 4.5, "4", 0, 1, 6, 16])
@pytest.mark.parametrize("entry", ["estimate", "pack"])
def test_automatic_scale_refuses_unsupported_or_inexact_component_bits(bits, entry):
    with pytest.raises((TypeError, ValueError), match="bit"):
        if entry == "estimate":
            estimate_complex_scale(np.asarray([-1, 1]), bits_per_component=bits, clip_sigma=3)
        else:
            pack_channelized_streams_for_detector(
                np.asarray([[[-1, 1] * 4]]), frame_size_samples=8,
                detector_window_samples=4, bits_per_component=bits,
            )


@pytest.mark.parametrize("clip_sigma", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(True), "3"])
@pytest.mark.parametrize("mode", ["global", "per_stream"])
def test_automatic_scale_refuses_invalid_clip_sigma_before_quantizing(monkeypatch, clip_sigma, mode):
    from pilot_proxy.integration import packing

    def unexpected_quantization(*args, **kwargs):
        pytest.fail("invalid automatic scale parameters reached quantization")

    monkeypatch.setattr(packing, "quantize_complex_numpy", unexpected_quantization)
    with pytest.raises(ValueError, match="clip_sigma"):
        packing.pack_channelized_streams_for_detector(
            np.asarray([[[-1, 1] * 4]]), frame_size_samples=8,
            detector_window_samples=4, quantization_scale_mode=mode,
            clip_sigma=clip_sigma,
        )


@pytest.mark.parametrize("values", [
    np.asarray([], dtype=np.complex128),
    np.asarray([complex(np.nan, -1), complex(np.nan, 1)]),
    np.asarray([complex(-1, np.inf), complex(1, np.inf)]),
])
def test_automatic_scale_refuses_missing_or_nonfinite_samples(values):
    with pytest.raises(ValueError, match="non-empty|finite"):
        estimate_complex_scale(values, bits_per_component=4, clip_sigma=3)


def test_automatic_scale_refuses_constant_samples():
    with pytest.raises(ValueError, match="positive quantization scale"):
        estimate_complex_scale(np.full(4, 2 + 3j), bits_per_component=4, clip_sigma=3)


@pytest.mark.parametrize("clip_sigma,amplitude", [
    (np.nextafter(0.0, 1.0), 1),
    (np.nextafter(0.0, 1.0), 0.25),
    (np.finfo(float).max, 2),
])
def test_automatic_scale_refuses_an_unrepresentable_scale(clip_sigma, amplitude):
    with pytest.raises(ValueError, match="positive finite quantization scale"):
        estimate_complex_scale(np.asarray([-amplitude, amplitude]),
                               bits_per_component=4, clip_sigma=clip_sigma)


@pytest.mark.parametrize("mode", ["global", "provided"])
def test_an_explicit_scale_does_not_require_an_automatic_spread(mode):
    result = pack_channelized_streams_for_detector(
        np.full((1, 1, 8), 2 + 3j), frame_size_samples=8,
        detector_window_samples=4, nyquist_zone=1,
        quantization_scale_mode=mode, scale=1, clip_sigma=0,
    )
    np.testing.assert_array_equal(result.packed, np.full((1, 2, 4), 0x23, dtype=np.int8))
    assert result.quantization["scale_by_stream"] == [1]
