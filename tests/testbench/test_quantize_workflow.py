"""Exercise IQ conversion, channel selection, and the saved detector input."""

import json

import numpy as np
import pytest

from pilot_proxy.testbench import quantize


def _tone_channel_samples(rf_center_hz, channels, count):
    """Evaluate the selected Fourier bins directly for a unit RF tone."""
    sample = np.arange(8192)
    response = np.sinc((sample - 4096) / 2048) * np.hamming(8192)
    frequency = (rf_center_hz - 400e6) / 800e6
    rows = []
    for channel in channels:
        basis = np.exp(-2j * np.pi * (channel + 1) * sample / 2048)
        rows.append([
            np.sum(
                np.cos(2 * np.pi * frequency * (start * 2048 + sample))
                * response * basis
            )
            for start in range(count)
        ])
    return np.asarray(rows)


@pytest.mark.parametrize(
    "options,channels,rf_center_hz,blocks,feeds,zone,archive_phase,scale,save",
    [
        pytest.param(
            ["--rf-center-mhz", "400.87890625", "--channel-index", "2",
             "--channel-index", "1", "--channel-index", "2"],
            [2, 1], 400878906.25, 2, 2, 1, False, None, True,
            id="automatic-scale-deduplicated-channels",
        ),
        pytest.param(
            ["--rf-center-mhz", "400.87890625", "--channel-index", "0",
             "--channel-index", "1023", "--channel-half-width", "1"],
            [0, 1, 1022, 1023], 400878906.25, 1, 2, 2, True, 0.01, True,
            id="provided-scale-edge-neighbors-archive-phase",
        ),
        pytest.param(
            ["--physical-channel", "14", "--atsc-pilot-offset-hz", "3000000"],
            [179], 470309441.0, 1, 1, 1, False, None, False,
            id="physical-channel-default-center-and-selection",
        ),
    ],
)
def test_quantize_writes_detector_input_matching_a_direct_tone_oracle(
    tmp_path, options, channels, rf_center_hz, blocks, feeds, zone,
    archive_phase, scale, save,
):
    iq_path = tmp_path / "tone.cfile"
    np.ones(2048, dtype=np.complex64).tofile(iq_path)
    output = tmp_path / "converted"
    argv = [
        "--input-iq", str(iq_path), "--output-dir", str(output),
        "--iq-sample-rate-hz", "1000000", "--frame-size-samples", "128",
        "--num-blocks", str(blocks), "--num-input-streams", str(feeds),
        "--nyquist-zone", str(zone), "--output-chunk-samples", "37",
        "--adc-chunk-samples", "10003", *options,
    ]
    if archive_phase:
        argv.append("--reference-archive-phase")
    if scale is not None:
        argv += ["--scale", str(scale)]
    if save:
        argv.append("--save-channelized")

    assert quantize.main(argv) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    packed = np.load(output / "detector_blocks_i4.npy")
    expected = _tone_channel_samples(rf_center_hz, channels, 128 * blocks)
    if archive_phase:
        expected = expected.conj() * (1 - 2 * (np.arange(128 * blocks) % 2))
    expected = np.tile(expected, (feeds, 1))

    expected_scale = scale if scale is not None else 7 / (3 * np.std(expected.real))
    assert metadata["quantization_scale"] == pytest.approx(expected_scale, rel=2e-6)
    expected_blocks = []
    for start in range(0, 128 * blocks, 128):
        values = expected[:, start:start + 128]
        if zone == 2:
            values = values[:, ::-1]
        real = np.clip(np.rint(values.real * metadata["quantization_scale"]), -7, 7)
        imag = np.clip(np.rint(values.imag * metadata["quantization_scale"]), -7, 7)
        unsigned = ((real.astype(int) & 15) << 4) | (imag.astype(int) & 15)
        expected_blocks.append(unsigned.astype(np.uint8).view(np.int8))
    np.testing.assert_array_equal(packed, expected_blocks)
    assert packed.dtype == np.int8
    assert packed.shape == (blocks, feeds * len(channels), 128)

    assert metadata["selected_channel_indices"] == channels
    assert metadata["selected_channel_rfft_bins"] == [channel + 1 for channel in channels]
    assert metadata["selected_channel_frequency_hz"] == [
        400e6 + (channel + 1) * 390625 for channel in channels
    ]
    assert metadata["rf_center_hz"] == rf_center_hz
    assert metadata["input_iq"] == str(iq_path)
    assert metadata["detector_blocks_path"] == str(output / "detector_blocks_i4.npy")
    assert metadata["detector_blocks_shape"] == list(packed.shape)
    assert metadata["nyquist_zone"] == zone
    assert metadata["reference_archive_phase"] is archive_phase
    assert metadata["input_layout"]["num_streams"] == feeds * len(channels)
    assert [row["feed_index"] for row in metadata["stream_map"]] == [
        feed for feed in range(feeds) for _ in channels
    ]
    assert [row["selected_channel_index"] for row in metadata["stream_map"]] == channels * feeds
    physical_channel = 14 if "--physical-channel" in options else None
    assert [row["physical_channel"] for row in metadata["stream_map"]] == [
        physical_channel
    ] * (feeds * len(channels))

    matrix_path = output / "detector_matrix_i4.npy"
    if blocks == 1:
        assert metadata["detector_matrix_path"] == str(matrix_path)
        np.testing.assert_array_equal(np.load(matrix_path), packed[0])
    else:
        assert metadata["detector_matrix_path"] is None
        assert not matrix_path.exists()
    streams_path = output / "channelized_streams_complex64.npy"
    if save:
        assert metadata["channelized_streams_path"] == str(streams_path)
        streams = np.load(streams_path)
        assert streams.dtype == np.complex64
        np.testing.assert_allclose(streams, expected, rtol=2e-6, atol=2e-5)
    else:
        assert metadata["channelized_streams_path"] is None
        assert not streams_path.exists()


@pytest.mark.parametrize("channel", [-1, 1024])
@pytest.mark.parametrize("half_width", [0, 1])
def test_quantize_rejects_out_of_band_centers_before_expanding_neighbors(
    tmp_path, channel, half_width,
):
    iq_path = tmp_path / "tone.cfile"
    np.ones(2048, dtype=np.complex64).tofile(iq_path)
    output = tmp_path / "converted"
    with pytest.raises(SystemExit, match=r"Channel indices must be in \[0, 1023\]"):
        quantize.main([
            "--input-iq", str(iq_path), "--output-dir", str(output),
            "--iq-sample-rate-hz", "1000000", "--frame-size-samples", "128",
            "--rf-center-mhz", "400.87890625", "--scale", "0.01",
            "--channel-index", str(channel), "--channel-half-width", str(half_width),
        ])
    assert not (output / "metadata.json").exists()
    assert not (output / "detector_blocks_i4.npy").exists()
