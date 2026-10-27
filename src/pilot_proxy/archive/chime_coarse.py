# coding=utf-8
"""Map CHIME coarse-channel frequencies to archive channel indices.

The F-engine divides the 400--800 MHz band into 1024 channels, each
``fs = 400e6 / 1024 = 390625`` Hz wide. A baseband file holds one of these
channels. Its center frequency gives the archive ``freq_id`` (0..1023)::

    f_center = f0 - freq_id * coarse_width        (f0 = 800 MHz, top of band)
    freq_id  = round((f0 - f_center) / coarse_width)

Selection uses freq_id because that is how the inventory and filenames name
channels. Products also carry the ATSC physical-channel label for the pilot.
Keeping freq_id alongside it lets a later 6 MHz mask-expansion step find the
other coarse channels in that television band without reopening the raw data.
"""
from __future__ import annotations

import re

from pilot_proxy.config.instrument import load_instrument

_CHIME = load_instrument("chime")

# f0: top of the CHIME 400-800 MHz band. The coarse channel width is the
# F-engine spacing 400e6 / 1024, which equals the baseband sample rate fs.
# All three come from the instrument file pilot_proxy/instruments/chime.yaml.
CHIME_BAND_TOP_HZ = _CHIME.f0_hz
CHIME_N_COARSE_CHANNELS = _CHIME.n_channels
CHIME_COARSE_WIDTH_HZ = _CHIME.channel_width_hz  # == 390625.0


def chime_freq_id_from_hz(
    f_center_hz: float,
    f0_hz: float = CHIME_BAND_TOP_HZ,
    coarse_width_hz: float = CHIME_COARSE_WIDTH_HZ,
) -> int:
    """Return the coarse-channel index (``freq_id``, 0..1023) for a center freq.

    ``freq_id = round((f0 - f_center) / coarse_width)``. ``f0`` defaults to the
    CHIME band top (800 MHz) and ``coarse_width`` to 400e6/1024; pass an
    instrument's own values for non-CHIME geometries. With the defaults this is
    ``Instrument.freq_id_of_hz`` of the CHIME instrument file.
    """
    return int(round((float(f0_hz) - float(f_center_hz)) / float(coarse_width_hz)))


def source_event_key(unit_key: object, freq_id: int) -> str:
    """Identify an event by removing its file's trailing frequency-channel token.

    Combining pilots requires a common event key. The files
    ``baseband_<event>_829.h5`` and ``baseband_<event>_844.h5`` describe different
    channels of the same event, so removing the final freq_id gives them the
    same key within their source namespace.

    Keep the directory or URI prefix: independent campaigns can reuse a
    filename. Keys without the expected suffix pass through unchanged.
    """
    key = str(unit_key).replace("\\", "/")
    return re.sub(rf"_{int(freq_id)}(?=\.[^/]+$)", "", key)


__all__ = [
    "CHIME_BAND_TOP_HZ",
    "CHIME_N_COARSE_CHANNELS",
    "CHIME_COARSE_WIDTH_HZ",
    "chime_freq_id_from_hz",
    "source_event_key",
]
