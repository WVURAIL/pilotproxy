# coding=utf-8
"""CHIME coarse-channel helpers shared by the archive analyzers.

A CHIME baseband file is one *coarse* channel of the F-engine: the 400 MHz band
(0..800 MHz top) split into 1024 channels of ``fs = 400e6 / 1024 = 390625`` Hz.
A file's ``freq_id`` (0..1023) is therefore recoverable from its center
frequency, the one quantity every reader's ``probe`` already exposes::

    f_center = f0 - freq_id * coarse_width        (f0 = 800 MHz, top of band)
    freq_id  = round((f0 - f_center) / coarse_width)

The analyzers *select* on ``freq_id`` (it is what the CADC file inventory and the
on-disk filenames key on) but *label* each product with the ATSC physical
channel the pilot falls in. Recording ``freq_id`` alongside the product lets the
(deferred) 6 MHz mask-expansion step find the sibling coarse channels of a
detected pilot without re-opening the raw files.
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
    """Normalise a source unit key to an event identity, dropping the freq_id.

    A CHIME baseband file is named ``baseband_<event>_<freq_id>.h5`` (local path
    or CADC URI). The combine step needs to confirm that different pilots saw the
    *same events in the same order*, but each pilot's files carry a *different*
    freq_id token, so the raw keys never match. Removing this product's own
    freq_id token (the one immediately before the extension) collapses
    ``baseband_<event>_829.h5`` and ``baseband_<event>_844.h5`` within the same
    source namespace to the same key.  The directory/URI prefix is deliberately
    retained: two independent campaigns can reuse a filename without becoming
    the same acquisition. Keys that do not carry the suffix are returned intact.
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
