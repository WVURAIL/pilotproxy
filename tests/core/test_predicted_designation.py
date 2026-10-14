# coding=utf-8
"""The default designated window follows the predicted pilot line.

The anchors below are the measured carrier-line bins from the 2018--2026
CHIME survey products (argmax of the mean fine spectrum per channel, all
six channels with a persistent carrier). The prediction is pure geometry
-- coarse-grid quantization residual of the sense-flipped pilot offset --
so the measured bin differs from it by the station's own carrier offset
(up to ~340 Hz observed), which the default half-width absorbs.
"""
from __future__ import annotations

import pytest

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_geometry import (
    DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS,
    SPECTRAL_SENSE_INVERTED,
    SPECTRAL_SENSE_NORMAL,
    predicted_fine_designated_bins,
    predicted_pilot_fine_bin,
)

SAMPLE_RATE_HZ = 390625.0
CHIME_F0_HZ = 800.0e6
CHIME_BW_HZ = 400.0e6
CHIME_NCHAN = 1024
K, NFFT, FINE_BINS = 128, 16384, 256

# freq_id -> (physical channel, measured line bin in the survey products)
MEASURED_ANCHORS = {
    798: (17, 214),
    721: (22, 105),
    690: (24, 66),
    598: (30, 201),
    583: (31, 167),
    521: (35, 111),
}


def _center_hz(freq_id: int) -> float:
    return CHIME_F0_HZ - CHIME_BW_HZ * freq_id / CHIME_NCHAN


def _predict(freq_id: int, channel: int, sense: str) -> int:
    return predicted_pilot_fine_bin(
        pilot_rf_hz=physical_channel_to_pilot_hz(channel),
        coarse_center_hz=_center_hz(freq_id),
        sample_rate_hz=SAMPLE_RATE_HZ,
        detector_window_samples=K,
        nfft=NFFT,
        spectral_sense=sense,
    )


@pytest.mark.parametrize("freq_id", sorted(MEASURED_ANCHORS))
def test_measured_line_lands_in_the_default_window(freq_id):
    channel, measured = MEASURED_ANCHORS[freq_id]
    predicted = _predict(freq_id, channel, SPECTRAL_SENSE_INVERTED)
    window = predicted_fine_designated_bins(
        predicted, DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS, FINE_BINS)
    assert measured in window, (
        f"freq_id {freq_id} (ch{channel}): measured line bin {measured} "
        f"outside the default window centred on predicted bin {predicted}")


@pytest.mark.parametrize("freq_id", sorted(MEASURED_ANCHORS))
def test_normal_sense_prediction_misses_the_measured_line(freq_id):
    """The sense flip is load-bearing: the unflipped mapping must NOT fit.

    (Except where the pilot's residual happens to be small enough that both
    signs land inside the window; no surveyed channel is in that regime.)
    """
    channel, measured = MEASURED_ANCHORS[freq_id]
    predicted = _predict(freq_id, channel, SPECTRAL_SENSE_NORMAL)
    window = predicted_fine_designated_bins(
        predicted, DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS, FINE_BINS)
    assert measured not in window


def test_designated_window_wraps_modulo_grid():
    assert predicted_fine_designated_bins(1, 2, 256) == [255, 0, 1, 2, 3]
    assert predicted_fine_designated_bins(255, 1, 256) == [254, 255, 0]
    assert predicted_fine_designated_bins(62, 0, 256) == [62]
    with pytest.raises(ValueError):
        predicted_fine_designated_bins(0, 128, 256)
